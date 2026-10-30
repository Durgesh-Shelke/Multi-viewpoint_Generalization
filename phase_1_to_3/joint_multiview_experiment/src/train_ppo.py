"""
Resumable CleanRL-style PPO Training Loop for Joint Multi-View PickCube.

Per SPEC Section 2, Section 3, Section 4:
- Architecture: NatureCNN agent from src/model.py (pure scratch, no warm-start).
- Environment: PickCube-v1 from src/env_factory.py with per-environment viewpoints.
- Hyperparameters: Loaded from configs/ppo_frozen.yaml (or custom YAML config).
- Pausable & Resumable:
  - Listens for SIGINT / SIGTERM signals and state/PAUSE file after every iteration.
  - On pause: finishes current iteration, writes atomic checkpoint via src/checkpoint.py, exits 0.
  - Second SIGINT within 10s triggers immediate exit.
  - Auto-resumes from newest valid checkpoint if present in output_dir/checkpoints.
  - Resets environment with seeds derived deterministically from (run_seed, iteration).
  - Appends to progress.csv without missing or duplicate rows.
  - Checkpoints every 5 minutes (300s) or 10 iterations, and upon completion.
  - Excludes pause wall-clock time from active_time_s.

Python 3.9 compatible only.
"""

import argparse
import csv
import hashlib
import json
import logging
import os
import random
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import psutil
import torch
import torch.nn as nn
import torch.optim as optim
import yaml

from src.checkpoint import (
    capture_rng_states,
    compute_config_hash,
    load_checkpoint_with_fallback,
    restore_rng_states,
    save_checkpoint,
)
from src.env_factory import make_joint_multiview_env, load_viewpoint_split
from src.environment import (
    EnvironmentMismatchError,
    REFERENCE_ENVIRONMENT_PATH,
    collect_environment_metadata,
    compare_environment_metadata,
    get_or_create_reference_environment,
    get_reference_environment_path,
    verify_environment,
)
from src.evaluate import evaluate_single_viewpoint
from src.fingerprint import (
    ManiSkillFingerprintMismatchError,
    REFERENCE_FINGERPRINT_PATH,
    compute_maniskill_source_fingerprint,
    get_or_create_reference_fingerprint,
    get_reference_fingerprint_path,
    verify_maniskill_source_fingerprint,
)
from src.model import Agent

logger = logging.getLogger("train_ppo")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Global stop flag and signal tracker
STOP_REQUESTED = False
LAST_SIGINT_TIME = 0.0


def sig_handler(signum: int, frame: Any) -> None:
    """Handles SIGINT and SIGTERM gracefully."""
    global STOP_REQUESTED, LAST_SIGINT_TIME
    now = time.time()
    if signum == signal.SIGINT and (now - LAST_SIGINT_TIME) < 10.0:
        logger.warning("\n[Signal] Second SIGINT received within 10s! Forcing immediate exit...")
        sys.exit(130)
    LAST_SIGINT_TIME = now
    sig_name = signal.Signals(signum).name if hasattr(signal, "Signals") else str(signum)
    logger.info("\n[Signal] Received signal %s (%s). Requesting clean pause after current iteration...", signum, sig_name)
    STOP_REQUESTED = True


# Register signal handlers
signal.signal(signal.SIGINT, sig_handler)
signal.signal(signal.SIGTERM, sig_handler)


class DictArray(object):
    """
    Vectorized dictionary observation buffer for PPO rollouts.
    Stores only the observation keys defined in element_space on the target device.
    """

    def __init__(
        self,
        buffer_shape: Tuple[int, ...],
        element_space: Optional[gym.spaces.Dict] = None,
        data_dict: Optional[Dict[str, Any]] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        self.buffer_shape = buffer_shape
        if data_dict is not None:
            self.data = data_dict
        else:
            assert isinstance(element_space, gym.spaces.Dict), f"Expected Dict space, got {type(element_space)}"
            self.data = {}
            for k, v in element_space.items():
                if isinstance(v, gym.spaces.Dict):
                    self.data[k] = DictArray(buffer_shape, v, device=device)
                else:
                    dtype = (
                        torch.float32
                        if v.dtype in (np.float32, np.float64)
                        else torch.uint8
                        if v.dtype == np.uint8
                        else torch.int16
                        if v.dtype == np.int16
                        else torch.int32
                        if v.dtype == np.int32
                        else torch.float32
                    )
                    self.data[k] = torch.zeros(buffer_shape + v.shape, dtype=dtype, device=device)

    def keys(self):
        return self.data.keys()

    def __getitem__(self, index: Any) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        if isinstance(index, str):
            return self.data[index]
        return {k: v[index] for k, v in self.data.items()}

    def __setitem__(self, index: Any, value: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> None:
        if isinstance(index, str):
            self.data[index] = value
            return
        for k in self.data.keys():
            if k in value:
                self.data[k][index] = value[k]

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.buffer_shape

    def reshape(self, shape: Tuple[int, ...]) -> "DictArray":
        t = len(self.buffer_shape)
        new_dict = {}
        for k, v in self.data.items():
            if isinstance(v, DictArray):
                new_dict[k] = v.reshape(shape)
            else:
                new_dict[k] = v.reshape(shape + v.shape[t:])
        new_buffer_shape = next(iter(new_dict.values())).shape[: len(shape)]
        return DictArray(new_buffer_shape, None, data_dict=new_dict)


def check_pause_files(pause_paths: List[str]) -> bool:
    """Checks whether any registered PAUSE file exists."""
    global STOP_REQUESTED
    if STOP_REQUESTED:
        return True
    for p in pause_paths:
        if p and os.path.isfile(p):
            logger.info("[Pause] Detected PAUSE file at: %s. Requesting clean pause...", p)
            STOP_REQUESTED = True
            return True
    return False


CSV_FIELDNAMES = [
    "iteration",
    "global_step",
    "active_time_s",
    "sps",
    "mean_reward",
    "mean_return",
    "mean_length",
    "success_rate",
    "iter_finished_episodes",
    "iter_return",
    "iter_success_rate",
    "pg_loss",
    "v_loss",
    "entropy",
    "approx_kl",
    "clipfrac",
    "lr",
]

EVAL_PER_VIEW_FIELDNAMES = [
    "step",
    "view",
    "group",
    "successes",
    "episodes",
    "wilson_lo",
    "wilson_hi",
    "mean_return",
    "mean_length",
]

EVAL_SUMMARY_FIELDNAMES = [
    "step",
    "train_mean",
    "val_mean",
]


def sanitize_eval_logs(logs_dir: str, checkpoint_step: int) -> None:
    """
    Sanitizes eval_per_view.csv and eval_summary.csv upon resume.
    Ensures that any rows with step > checkpoint_step (e.g. from an evaluation
    interrupted by a pause/crash before checkpoint was saved) are stripped.
    """
    if not logs_dir or not os.path.isdir(logs_dir):
        return
    for fname, fieldnames in [
        ("eval_per_view.csv", EVAL_PER_VIEW_FIELDNAMES),
        ("eval_summary.csv", EVAL_SUMMARY_FIELDNAMES),
    ]:
        fpath = os.path.join(logs_dir, fname)
        if not os.path.isfile(fpath):
            continue
        clean_rows = []
        needs_rewrite = False
        with open(fpath, "r", newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                try:
                    step_val = int(r["step"])
                    if step_val <= checkpoint_step:
                        clean_rows.append(r)
                    else:
                        needs_rewrite = True
                except (ValueError, KeyError):
                    clean_rows.append(r)
        if needs_rewrite:
            tmp = fpath + f".tmp.{os.getpid()}"
            with open(tmp, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                for r in clean_rows:
                    writer.writerow(r)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, fpath)
            logger.info("[Eval Sanitization] Sanitized %s to step <= %d", fname, checkpoint_step)


def run_periodic_evaluation(
    agent: nn.Module,
    global_step: int,
    modality: str,
    config: Dict[str, Any],
    logs_dir: str,
    device: str = "cuda",
    pause_paths: Optional[List[str]] = None,
) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Executes periodic evaluation across all train and val viewpoints:
    - 10 episodes per view (configurable via eval_episodes_per_view)
    - deterministic actions
    - fixed evaluation seeds identical across runs and modalities
    - isolates RNG states so training RNG is completely unchanged
    - does NOT touch or modify training environment state
    - appends to eval_per_view.csv and eval_summary.csv
    
    Returns:
        (summary_row, per_view_records) or (None, []) if paused mid-eval
    """
    # 1. Capture and isolate training RNG states
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_cpu_state = torch.get_rng_state()
    torch_cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

    agent_was_training = agent.training
    agent.eval()
    t0_periodic_eval = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    try:
        split = load_viewpoint_split()
        train_views = split.get("train", [])
        val_views = split.get("val", [])
        eval_views = [("train", v) for v in train_views] + [("val", v) for v in val_views]
        episodes_per_view = int(config.get("eval_episodes_per_view", 10))

        per_view_rows = []
        train_success_rates = []
        val_success_rates = []

        for group, vid in eval_views:
            if pause_paths and check_pause_files(pause_paths):
                logger.info("[Periodic Eval] Pause detected during periodic evaluation. Interrupted eval will be repeated on resume.")
                return None, []

            eval_seed = int((int(vid) * 1000 + 42) % (2**31 - 1))
            res = evaluate_single_viewpoint(
                agent=agent,
                viewpoint_id=vid,
                modality=modality,
                episodes=episodes_per_view,
                num_envs=min(episodes_per_view, 10),
                seed=eval_seed,
                device=device,
            )
            succ_rate = res["any_step_success_rate"]
            if group == "train":
                train_success_rates.append(succ_rate)
            elif group == "val":
                val_success_rates.append(succ_rate)

            per_view_rows.append({
                "step": global_step,
                "view": vid,
                "group": group,
                "successes": res["any_step_success_count"],
                "episodes": res["episodes"],
                "wilson_lo": res["wilson_95_lower"],
                "wilson_hi": res["wilson_95_upper"],
                "mean_return": round(res["mean_return"], 4),
                "mean_length": round(res["mean_length"], 2),
            })

        train_mean = round(float(np.mean(train_success_rates)), 2) if train_success_rates else 0.0
        val_mean = round(float(np.mean(val_success_rates)), 2) if val_success_rates else 0.0
        summary_row = {"step": global_step, "train_mean": train_mean, "val_mean": val_mean}

        # Append to CSV files
        os.makedirs(logs_dir, exist_ok=True)
        per_view_csv = os.path.join(logs_dir, "eval_per_view.csv")
        pv_exists = os.path.exists(per_view_csv)
        with open(per_view_csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=EVAL_PER_VIEW_FIELDNAMES, extrasaction="ignore")
            if not pv_exists:
                w.writeheader()
            w.writerows(per_view_rows)
            f.flush()

        summary_csv = os.path.join(logs_dir, "eval_summary.csv")
        sm_exists = os.path.exists(summary_csv)
        with open(summary_csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=EVAL_SUMMARY_FIELDNAMES, extrasaction="ignore")
            if not sm_exists:
                w.writeheader()
            w.writerow(summary_row)
            f.flush()

        eval_wall_time = time.time() - t0_periodic_eval
        peak_vram_gib = (torch.cuda.max_memory_allocated() / (1024 ** 3)) if torch.cuda.is_available() else 0.0
        logger.info(
            "[Bench] Periodic evaluation wall time: %.2fs across %d views, peak VRAM: %.3f GiB",
            eval_wall_time,
            len(eval_views),
            peak_vram_gib,
        )

        return summary_row, per_view_rows

    finally:
        # Restore agent training mode
        if agent_was_training:
            agent.train()
        # Restore training RNG states with complete fidelity
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.set_rng_state(torch_cpu_state)
        if torch_cuda_state is not None:
            torch.cuda.set_rng_state_all(torch_cuda_state)


def sanitize_csv_for_resume(csv_path: str, start_iteration: int) -> None:
    """
    Called ONCE at the start of a training/resume session before the iteration loop begins.
    If resuming at start_iteration > 1, truncates any rows in csv_path where
    iteration >= start_iteration to eliminate crash-orphaned rows.
    """
    if not os.path.exists(csv_path) or start_iteration <= 1:
        return
    clean_rows = []
    needs_rewrite = False
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                if int(r["iteration"]) < start_iteration:
                    clean_rows.append(r)
                else:
                    needs_rewrite = True
            except (ValueError, KeyError):
                clean_rows.append(r)

    if needs_rewrite:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
            writer.writeheader()
            for r in clean_rows:
                writer.writerow(r)


def log_progress_row(
    csv_path: str,
    row: Dict[str, Any],
) -> None:
    """Appends a row to progress.csv."""
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
        f.flush()


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """Loads a YAML configuration file."""
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg or {}


class ManiSkillIntegrityError(RuntimeError):
    """Raised when ManiSkill task source files have changed between start and resume."""
    pass


def compute_file_sha256(filepath: str) -> str:
    """Computes SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def get_maniskill_task_hashes() -> Dict[str, Dict[str, str]]:
    """
    Computes SHA-256 hashes for canonical ManiSkill PickCube task source files:
    - mani_skill/envs/tasks/tabletop/pick_cube.py
    - mani_skill/envs/tasks/tabletop/pick_cube_cfgs.py
    """
    import mani_skill.envs.tasks.tabletop.pick_cube as pc
    import mani_skill.envs.tasks.tabletop.pick_cube_cfgs as pcc

    pc_file = os.path.abspath(pc.__file__)
    pcc_file = os.path.abspath(pcc.__file__)

    return {
        "pick_cube.py": {
            "path": pc_file,
            "sha256": compute_file_sha256(pc_file),
        },
        "pick_cube_cfgs.py": {
            "path": pcc_file,
            "sha256": compute_file_sha256(pcc_file),
        },
    }


def write_or_verify_manifest(
    output_dir: str,
    config: Dict[str, Any],
    effective_env_settings: Dict[str, Any],
    is_resuming: bool = False,
    ref_fingerprint_path: Optional[str] = None,
    ref_environment_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Records or verifies output_dir/manifest.json.
    - Computes ManiSkill source repository fingerprint (git HEAD + diff SHA256 of mani_skill).
    - Checks fingerprint against state/reference_fingerprint.json (creating it on first run).
    - Collects runtime environment metadata (Python, torch, CUDA build, mani_skill, gymnasium, numpy, requirements SHA-256, pip freeze hash).
    - Checks environment against state/reference_environment.json (creating it on first real run).
    - If resuming, verifies that ManiSkill task SHA-256 hashes, source fingerprint, and environment versions match.
    - If changed, refuses to resume with ManiSkillIntegrityError, ManiSkillFingerprintMismatchError, or EnvironmentMismatchError.
    - Writes manifest atomically with fsync.
    """
    manifest_path = os.path.join(output_dir, "manifest.json")
    current_hashes = get_maniskill_task_hashes()
    current_fp = compute_maniskill_source_fingerprint()
    effective_ref_path = ref_fingerprint_path or get_reference_fingerprint_path()
    effective_ref_env = ref_environment_path or get_reference_environment_path()

    # 1. Verify against reference_fingerprint.json (creates if not present)
    is_match, err = verify_maniskill_source_fingerprint(
        ref_path=effective_ref_path,
        current_fp=current_fp,
    )
    if not is_match:
        logger.error(err)
        raise ManiSkillFingerprintMismatchError(err)

    # 2. Verify against reference_environment.json (creates on first real run or in EXP_STATE_DIR)
    current_env = collect_environment_metadata()
    is_env_match, env_err = verify_environment(
        ref_path=effective_ref_env,
        current_meta=current_env,
    )
    if not is_env_match:
        logger.error(env_err)
        raise EnvironmentMismatchError(env_err)

    if is_resuming and os.path.isfile(manifest_path):
        with open(manifest_path, "r") as f:
            manifest_data = json.load(f)

        # Verify commit hash against frozen export commit
        active_commit = os.environ.get("FROZEN_COMMIT")
        if not active_commit:
            commit_file = Path(__file__).resolve().parent.parent / "COMMIT_HASH"
            if commit_file.is_file():
                active_commit = commit_file.read_text().strip()
            else:
                try:
                    import subprocess
                    exp_root = Path(__file__).resolve().parent.parent
                    res = subprocess.run(
                        ["git", "rev-parse", "HEAD"],
                        cwd=str(exp_root),
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        check=True,
                    )
                    active_commit = res.stdout.strip()
                except Exception:
                    active_commit = None

        saved_commit = manifest_data.get("git_commit")
        if active_commit and saved_commit and active_commit != "unknown" and saved_commit != "unknown":
            if saved_commit != active_commit:
                err_msg = (
                    "Frozen code commit hash mismatch against saved run manifest!\n"
                    f"Manifest file: {manifest_path}\n"
                    f"Saved manifest commit: {saved_commit}\n"
                    f"Current active commit: {active_commit}\n"
                    "Refusing to resume run initialized with different code version."
                )
                logger.error(err_msg)
                raise RuntimeError(err_msg)

        saved_hashes = manifest_data.get("maniskill_task_hashes", {})
        mismatches = []
        for name, curr_info in current_hashes.items():
            if name in saved_hashes:
                saved_hash = saved_hashes[name].get("sha256")
                curr_hash = curr_info.get("sha256")
                if saved_hash != curr_hash:
                    mismatches.append(
                        f"{name} changed: original={saved_hash}, current={curr_hash}"
                    )
            else:
                mismatches.append(f"{name} missing from previous manifest.json")

        if mismatches:
            err_msg = (
                "ManiSkill source integrity violation! Task files have changed since run was initialized:\n"
                + "\n".join(f"  - {m}" for m in mismatches)
                + "\nRefusing to resume run with altered simulation physics/tasks."
            )
            logger.error(err_msg)
            raise ManiSkillIntegrityError(err_msg)

        # Also verify manifest's own saved fingerprint if present
        saved_fp_info = manifest_data.get("maniskill_source_fingerprint", {})
        saved_fp = saved_fp_info.get("fingerprint")
        if saved_fp and saved_fp != current_fp["fingerprint"]:
            err_msg = (
                f"ManiSkill source fingerprint mismatch against saved run manifest!\n"
                f"  Manifest fingerprint: {saved_fp}\n"
                f"  Current fingerprint:  {current_fp['fingerprint']}\n"
                f"Refusing to resume run with altered ManiSkill source repository."
            )
            logger.error(err_msg)
            raise ManiSkillFingerprintMismatchError(err_msg)

        # Also verify manifest's own saved environment
        saved_env = manifest_data.get("environment")
        if saved_env is None:
            saved_env = {
                "python_version": manifest_data.get("python_short_version") or manifest_data.get("python_version"),
                "torch_version": manifest_data.get("torch_version"),
                "cuda_build": manifest_data.get("cuda_build"),
                "mani_skill_version": manifest_data.get("mani_skill_version"),
                "sapien_version": manifest_data.get("sapien_version"),
                "gymnasium_version": manifest_data.get("gymnasium_version"),
                "numpy_version": manifest_data.get("numpy_version"),
            }
        manifest_env_mismatches = compare_environment_metadata(current_env, saved_env)
        if manifest_env_mismatches:
            err_msg = (
                "Environment version mismatch against saved run manifest!\n"
                f"Manifest file: {manifest_path}\n"
                "The following environment components differ:\n"
                + "\n".join(f"  - {m}" for m in manifest_env_mismatches)
                + "\nRefusing to resume run with altered environment versions."
            )
            logger.error(err_msg)
            raise EnvironmentMismatchError(err_msg)

        # Record pip freeze diff in manifest if present on resume
        if "pip_freeze_diff" in current_env and current_env["pip_freeze_diff"]:
            manifest_data["pip_freeze_diff"] = current_env["pip_freeze_diff"]
            tmp_path = manifest_path + f".tmp.{os.getpid()}"
            with open(tmp_path, "w") as f:
                json.dump(manifest_data, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, manifest_path)

        logger.info("[Manifest] ManiSkill task file hashes, source fingerprint, environment versions, and commit verified against manifest.json.")
        return manifest_data

    # New run or no previous manifest
    git_commit = "unknown"
    frozen_commit = os.environ.get("FROZEN_COMMIT")
    if frozen_commit:
        git_commit = frozen_commit
    else:
        commit_file = Path(__file__).resolve().parent.parent / "COMMIT_HASH"
        if commit_file.is_file():
            git_commit = commit_file.read_text().strip()
        else:
            try:
                import subprocess
                exp_root = Path(__file__).resolve().parent.parent
                res = subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    cwd=str(exp_root),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=True,
                )
                git_commit = res.stdout.strip()
            except Exception:
                pass

    manifest_data = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "git_commit": git_commit,
        "python_version": current_env.get("python_full_version", sys.version),
        "python_short_version": current_env.get("python_version"),
        "torch_version": current_env.get("torch_version"),
        "cuda_build": current_env.get("cuda_build"),
        "mani_skill_version": current_env.get("mani_skill_version"),
        "sapien_version": current_env.get("sapien_version"),
        "gymnasium_version": current_env.get("gymnasium_version"),
        "numpy_version": current_env.get("numpy_version"),
        "requirements_frozen_sha256": current_env.get("requirements_frozen_sha256"),
        "pip_freeze_sha256": current_env.get("pip_freeze_sha256"),
        "environment": current_env,
        "config_hash": compute_config_hash(config),
        "maniskill_task_hashes": current_hashes,
        "maniskill_source_fingerprint": current_fp,
        "effective_env_settings": effective_env_settings,
        "config": config,
    }
    if "pip_freeze_diff" in current_env and current_env["pip_freeze_diff"]:
        manifest_data["pip_freeze_diff"] = current_env["pip_freeze_diff"]

    tmp_path = manifest_path + f".tmp.{os.getpid()}"
    with open(tmp_path, "w") as f:
        json.dump(manifest_data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, manifest_path)
    logger.info("[Manifest] Recorded run manifest at: %s", manifest_path)
    return manifest_data


def train_ppo(
    config: Dict[str, Any],
    output_dir: Optional[str] = None,
    modality: Optional[str] = None,
    seed: Optional[int] = None,
    total_timesteps: Optional[int] = None,
    device: Optional[str] = None,
    pause_file: Optional[str] = None,
    ref_fingerprint_path: Optional[str] = None,
    ref_environment_path: Optional[str] = None,
    run_name: Optional[str] = None,
    target_kl: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Main PPO training entrypoint with pause and auto-resume capabilities.
    
    Args:
        config: Base configuration dictionary.
        output_dir: Directory where checkpoints and logs are saved.
        modality: Observation mode ('rgb' or 'rgbd').
        seed: Random seed for this training run.
        total_timesteps: Maximum environment steps to train for.
        device: Compute device ('cuda' or 'cpu').
        pause_file: Optional custom path to monitor for pause trigger.
        ref_fingerprint_path: Path to record or verify fingerprint.
        ref_environment_path: Path to record or verify environment settings.
        run_name: Optional identifier of the run from configs/runs.yaml.
        
    Returns:
        Summary dictionary with final status, global_step, active_time_s, checkpoint path.
    """
    global STOP_REQUESTED
    STOP_REQUESTED = False

    # Lower CPU process priority per project guidelines
    try:
        os.nice(10)
    except Exception:
        pass

    # Merge configuration overrides
    cfg = dict(config)
    if output_dir is not None:
        cfg["output_dir"] = output_dir
    if run_name is not None:
        cfg["run_name"] = run_name
    else:
        run_name = cfg.get("run_name")
    # 1. Resolve modality with single source of truth (SPEC Section 3.1)
    resolved_modality = None
    if modality is not None:
        resolved_modality = modality
    elif "modality" in config and config["modality"] is not None:
        resolved_modality = config["modality"]

    if resolved_modality is None:
        raise ValueError("Missing required parameter: 'modality'. Modality must be specified ('rgb' or 'rgbd').")

    resolved_modality = str(resolved_modality).lower().strip()
    if resolved_modality not in ("rgb", "rgbd"):
        raise ValueError(f"Invalid modality: {resolved_modality!r}. Expected 'rgb' or 'rgbd'.")

    cfg["modality"] = resolved_modality
    cfg.pop("obs_mode", None)  # Single source of truth: stored once as modality

    if seed is not None:
        cfg["seed"] = seed
    if total_timesteps is not None:
        cfg["total_timesteps"] = total_timesteps
    if device is not None:
        cfg["device"] = device
    if target_kl is not None:
        cfg["target_kl"] = target_kl

    # Defaults and required configuration values
    out_dir = cfg.get("output_dir", "runs/dev_test")
    mod = resolved_modality
    run_seed = int(cfg.get("seed", 42))
    dev_str = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    # Seed RNGs for fresh run (overwritten by checkpoint if resuming)
    random.seed(run_seed)
    np.random.seed(run_seed)
    torch.manual_seed(run_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(run_seed)

    # Enforce total_timesteps is specified and > 0
    raw_total_steps = cfg.get("total_timesteps", None)
    if raw_total_steps is None or int(raw_total_steps) <= 0:
        raise ValueError(
            "total_timesteps is None or not specified in config/arguments. "
            "Main-grid runs require an explicit budget from budget.json. "
            "Refusing to start unbudgeted run."
        )
    total_steps = int(raw_total_steps)
    cfg["total_timesteps"] = total_steps

    num_envs = int(cfg.get("num_envs", 160))
    num_steps = int(cfg.get("num_steps", 50))
    num_minibatches = int(cfg.get("num_minibatches", 16))
    update_epochs = int(cfg.get("update_epochs", 4))
    lr = float(cfg.get("lr", 2e-4))
    adam_eps = float(cfg.get("adam_eps", 1e-5))
    gamma = float(cfg.get("gamma", 0.8))
    gae_lambda = float(cfg.get("gae_lambda", 0.9))
    clip_coef = float(cfg.get("clip_coef", 0.2))
    target_kl = cfg.get("target_kl", None)
    if target_kl is not None:
        target_kl = float(target_kl)
    ent_coef = float(cfg.get("ent_coef", 0.0))
    vf_coef = float(cfg.get("vf_coef", 0.5))
    max_grad_norm = float(cfg.get("max_grad_norm", 0.5))
    norm_adv = bool(cfg.get("norm_adv", True))
    clip_vloss = bool(cfg.get("clip_vloss", False))
    anneal_lr = bool(cfg.get("anneal_lr", False))
    sim_backend = cfg.get("sim_backend", "physx_cuda")
    split_group = cfg.get("split_group", "train")
    viewpoint_ids = cfg.get("viewpoint_ids", None)
    checkpoint_interval_s = float(cfg.get("checkpoint_interval_s", 300.0))
    checkpoint_interval_iters = int(cfg.get("checkpoint_interval_iters", 10))
    rolling_keep = int(cfg.get("rolling_keep", 2))
    milestone_interval = int(cfg.get("milestone_interval", 5_000_000))

    batch_size = num_steps * num_envs
    if batch_size % num_minibatches != 0:
        raise ValueError(f"batch_size ({batch_size}) must be divisible by num_minibatches ({num_minibatches})")
    minibatch_size = batch_size // num_minibatches
    total_iterations = total_steps // batch_size

    # Establish directories
    os.makedirs(out_dir, exist_ok=True)
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    logs_dir = os.path.join(out_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "progress.csv")

    # Pause file monitoring paths
    exp_dir = Path(__file__).resolve().parent.parent
    state_dir = Path(os.environ.get("EXP_STATE_DIR", str(exp_dir / "state")))
    pause_paths = [
        str(state_dir / "PAUSE"),
        str(exp_dir / "PAUSE"),
        os.path.join(out_dir, "PAUSE"),
    ]
    if pause_file:
        pause_paths.append(pause_file)

    # 1. Initialize Vectorized Environment
    logger.info("Initializing ManiSkill environment with num_envs=%d, modality=%s, backend=%s", num_envs, mod, sim_backend)
    t0_env_init = time.time()
    env = make_joint_multiview_env(
        num_envs=num_envs,
        modality=mod,
        split_group=split_group,
        viewpoint_ids=viewpoint_ids,
        config=cfg,
        seed=run_seed,
    )
    env_init_duration = time.time() - t0_env_init
    logger.info("[Bench] Environment initialization time: %.2fs", env_init_duration)

    in_channels = 4 if mod == "rgbd" else 3
    model_cfg = cfg.get("model", {})
    state_act = model_cfg.get("state_activation", "none") if isinstance(model_cfg, dict) else "none"

    # 2. Instantiate Agent (Orthogonal Init, Pure Scratch)
    # Using sample obs from single_observation_space structure
    agent = Agent(
        env=env,
        in_channels=in_channels,
        state_dim=29,
        include_state=True,
        state_activation=state_act,
    ).to(dev_str)

    # 3. Instantiate Optimizer
    optimizer = optim.Adam(agent.parameters(), lr=lr, eps=adam_eps)

    # 4. Check for Auto-Resume Checkpoints
    ckpt_data, ckpt_path = load_checkpoint_with_fallback(ckpt_dir, current_config=cfg)
    is_resuming = (ckpt_data is not None and ckpt_path is not None)

    # Record or verify run manifest (including ManiSkill file hashes, source fingerprint, environment, and effective env settings)
    effective_settings = getattr(env, "effective_settings", {})
    effective_ref_fp = ref_fingerprint_path or get_reference_fingerprint_path()
    effective_ref_env = ref_environment_path or get_reference_environment_path()
    write_or_verify_manifest(
        output_dir=out_dir,
        config=cfg,
        effective_env_settings=effective_settings,
        is_resuming=is_resuming,
        ref_fingerprint_path=effective_ref_fp,
        ref_environment_path=effective_ref_env,
    )

    global_step = 0
    start_iteration = 1
    active_time_s = 0.0
    eval_history = []
    last_eval_step = 0
    norm_stats = None

    if ckpt_data is not None and ckpt_path is not None:
        logger.info("[Resume] Valid checkpoint found: %s", ckpt_path)
        agent.load_state_dict(ckpt_data["model_state_dict"])
        optimizer.load_state_dict(ckpt_data["optimizer_state_dict"])
        restore_rng_states(ckpt_data["rng_states"])
        global_step = int(ckpt_data["global_step"])
        start_iteration = int(ckpt_data["iteration"]) + 1
        active_time_s = float(ckpt_data.get("active_time_s", 0.0))
        eval_history = ckpt_data.get("eval_history", [])
        last_eval_step = int(ckpt_data.get("last_eval_step", 0))
        norm_stats = ckpt_data.get("norm_stats", None)
        sanitize_eval_logs(logs_dir, last_eval_step)
        logger.info(
            "[Resume] Resumed successfully at Iteration %d/%d (Global Step: %s, Active Time: %.1fs, Last Eval Step: %s)",
            start_iteration,
            total_iterations,
            f"{global_step:,}",
            active_time_s,
            f"{last_eval_step:,}",
        )
    else:
        logger.info("[Start] No prior checkpoint found. Starting fresh training from scratch.")

    # Check if an evaluation was interrupted at the resume checkpoint step
    eval_interval = int(cfg.get("eval_interval_steps", 1000000))
    if eval_interval > 0 and global_step > 0:
        current_eval_bucket = global_step // eval_interval
        last_eval_bucket = last_eval_step // eval_interval if last_eval_step > 0 else 0
        if current_eval_bucket > last_eval_bucket:
            logger.info(
                "[Resume] Evaluation at step %s was interrupted by pause. Repeating periodic evaluation...",
                f"{global_step:,}",
            )
            summary_row, per_view_records = run_periodic_evaluation(
                agent=agent,
                global_step=global_step,
                modality=mod,
                config=cfg,
                logs_dir=logs_dir,
                device=dev_str,
                pause_paths=pause_paths,
            )
            if summary_row is not None:
                eval_history.append(summary_row)
                last_eval_step = global_step
                logger.info(
                    "[Resume] Completed repeated eval at step %s | Train Mean: %.1f%% | Val Mean: %.1f%%",
                    f"{global_step:,}",
                    summary_row["train_mean"],
                    summary_row["val_mean"],
                )

    # Check if already completed
    if global_step >= total_steps or start_iteration > total_iterations:
        logger.info("[Done] Run has already reached budget (%d steps). Exiting cleanly.", global_step)
        env.close()
        return {
            "status": "completed",
            "global_step": global_step,
            "iteration": start_iteration - 1,
            "active_time_s": active_time_s,
            "checkpoint_path": ckpt_path,
        }

    # Sanitize progress.csv and eval logs prior to entering iteration loop
    sanitize_csv_for_resume(csv_path, start_iteration)
    sanitize_eval_logs(logs_dir, last_eval_step)

    # 5. Reset Environment with Deterministic Seed Derived from (run_seed, iteration)
    # Env seed formula ensures reproducible initial state for any restart iteration
    env_seed = int((run_seed * 1000003 + start_iteration * 997) % (2**31 - 1))
    next_obs, _ = env.reset(seed=env_seed)
    next_done = torch.zeros(num_envs, device=dev_str)

    # 6. Allocate Rollout Buffers
    obs_space = getattr(env, "single_observation_space", env.observation_space)
    action_space = getattr(env, "single_action_space", env.action_space)

    obs_buf = DictArray((num_steps, num_envs), obs_space, device=dev_str)
    actions_buf = torch.zeros((num_steps, num_envs) + action_space.shape, device=dev_str)
    logprobs_buf = torch.zeros((num_steps, num_envs), device=dev_str)
    rewards_buf = torch.zeros((num_steps, num_envs), device=dev_str)
    dones_buf = torch.zeros((num_steps, num_envs), device=dev_str)
    values_buf = torch.zeros((num_steps, num_envs), device=dev_str)

    last_save_time = time.time()
    last_save_iter = start_iteration - 1
    last_saved_path = ckpt_path

    # Episode metrics accumulators
    ep_returns: List[float] = []
    ep_lengths: List[int] = []
    ep_successes: List[float] = []
    recent_iter_succs: List[float] = []

    # If resuming, populate recent_iter_succs from progress.csv
    if os.path.isfile(csv_path):
        try:
            with open(csv_path, "r", newline="") as f:
                r_rows = list(csv.DictReader(f))
            for r in r_rows[-5:]:
                s_val = r.get("iter_success_rate", "")
                if s_val != "":
                    recent_iter_succs.append(float(s_val))
                else:
                    recent_iter_succs.append(0.0)
        except Exception:
            pass

    logger.info("=" * 80)
    logger.info("STARTING PPO ROLLOUT LOOP: %s", mod.upper())
    logger.info("  Total Steps:      %s (%d iterations of %d steps)", f"{total_steps:,}", total_iterations, batch_size)
    logger.info("  Start Iteration:  %d", start_iteration)
    logger.info("  Minibatch Size:   %d (%d minibatches)", minibatch_size, num_minibatches)
    logger.info("  Output Directory: %s", out_dir)
    logger.info("=" * 80)

    # Sanitize progress.csv once upon resume to eliminate crash-orphaned rows
    sanitize_csv_for_resume(csv_path, start_iteration)

    # 7. Main PPO Iteration Loop
    for iteration in range(start_iteration, total_iterations + 1):
        # Pre-iteration pause check
        if check_pause_files(pause_paths):
            logger.info("[Pause] Pause requested before iteration %d. Saving checkpoint...", iteration)
            last_saved_path = save_checkpoint(
                ckpt_dir=ckpt_dir,
                model=agent,
                optimizer=optimizer,
                global_step=global_step,
                iteration=iteration - 1,
                config=cfg,
                rng_states=capture_rng_states(),
                norm_stats=norm_stats,
                eval_history=eval_history,
                last_eval_step=last_eval_step,
                active_time_s=active_time_s,
                is_final=False,
                rolling_keep=rolling_keep,
                milestone_interval=milestone_interval,
            )
            env.close()
            logger.info("[Pause] Paused cleanly at global_step=%d. Exiting code 0.", global_step)
            sys.exit(0)

        iter_start_time = time.time()
        final_values = torch.zeros((num_steps, num_envs), device=dev_str)
        iter_finished_returns = []
        iter_finished_lengths = []
        iter_finished_successes = []

        # ----------------------------------------------------------------------
        # ROLLOUT PHASE
        # ----------------------------------------------------------------------
        agent.eval()
        for step in range(num_steps):
            global_step += num_envs
            obs_buf[step] = next_obs
            dones_buf[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values_buf[step] = value.flatten()

            actions_buf[step] = action
            logprobs_buf[step] = logprob

            next_obs, reward, term, trunc, infos = env.step(action)
            next_done = torch.logical_or(term, trunc).to(torch.float32)
            rewards_buf[step] = reward.view(-1)

            # Extract episode return & length statistics from ManiSkill info
            if "final_info" in infos:
                done_mask = infos["_final_info"]
                if done_mask.any():
                    # Bootstrap value for truncated / terminal states
                    final_obs = {k: v[done_mask] for k, v in infos["final_observation"].items()}
                    with torch.no_grad():
                        final_vals = agent.get_value(final_obs).view(-1)
                        done_indices = torch.arange(num_envs, device=dev_str)[done_mask]
                        final_values[step, done_indices] = final_vals

                    # Collect metrics if recorded by ManiSkillVectorEnv
                    if "episode" in infos["final_info"]:
                        ep_info = infos["final_info"]["episode"]
                        if "return" in ep_info:
                            rets = ep_info["return"][done_mask].cpu().tolist()
                            ep_returns.extend(rets)
                            iter_finished_returns.extend(rets)
                        if "episode_len" in ep_info:
                            lens = ep_info["episode_len"][done_mask].cpu().tolist()
                            ep_lengths.extend(lens)
                            iter_finished_lengths.extend(lens)
                        if "success_once" in ep_info:
                            succs = ep_info["success_once"][done_mask].float().cpu().tolist()
                            ep_successes.extend(succs)
                            iter_finished_successes.extend(succs)

        # ----------------------------------------------------------------------
        # ADVANTAGE ESTIMATION (GAE)
        # ----------------------------------------------------------------------
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards_buf).to(dev_str)
            lastgaelam = 0.0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    next_not_done = 1.0 - next_done
                    nextvalues = next_value
                else:
                    next_not_done = 1.0 - dones_buf[t + 1]
                    nextvalues = values_buf[t + 1]
                real_next_values = next_not_done * nextvalues + final_values[t]
                delta = rewards_buf[t] + gamma * real_next_values - values_buf[t]
                advantages[t] = lastgaelam = delta + gamma * gae_lambda * next_not_done * lastgaelam
            returns = advantages + values_buf

        # Flatten rollout buffers
        b_obs = obs_buf.reshape((-1,))
        b_logprobs = logprobs_buf.reshape(-1)
        b_actions = actions_buf.reshape((-1,) + action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values_buf.reshape(-1)

        # ----------------------------------------------------------------------
        # PPO OPTIMIZATION PHASE
        # ----------------------------------------------------------------------
        agent.train()
        b_inds = np.arange(batch_size)
        clipfracs = []
        pg_losses = []
        v_losses = []
        entropies = []
        approx_kls = []

        for epoch in range(update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                    approx_kls.append(approx_kl.item())
                    clipfracs.append(((ratio - 1.0).abs() > clip_coef).float().mean().item())

                if target_kl is not None and approx_kl.item() > target_kl:
                    break

                mb_advantages = b_advantages[mb_inds]
                if norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy surrogate loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value function loss
                newvalue = newvalue.view(-1)
                if clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -clip_coef,
                        clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                optimizer.step()

                pg_losses.append(pg_loss.item())
                v_losses.append(v_loss.item())
                entropies.append(entropy_loss.item())

            if target_kl is not None and approx_kls and approx_kls[-1] > target_kl:
                break

        # Anneal learning rate if enabled
        if anneal_lr:
            frac = 1.0 - (iteration - 1.0) / total_iterations
            lrnow = frac * lr
            optimizer.param_groups[0]["lr"] = lrnow

        # ----------------------------------------------------------------------
        # TIMING, LOGGING & CHECKPOINTING
        # ----------------------------------------------------------------------
        iter_elapsed = time.time() - iter_start_time
        active_time_s += iter_elapsed
        sps = int(batch_size / max(iter_elapsed, 1e-6))
        mean_rew = float(rewards_buf.mean().item())
        current_lr = float(optimizer.param_groups[0]["lr"])

        mean_ret = float(np.mean(ep_returns[-100:])) if ep_returns else mean_rew * num_steps
        mean_len = float(np.mean(ep_lengths[-100:])) if ep_lengths else float(num_steps)
        succ_rate = float(np.mean(ep_successes[-100:])) if ep_successes else 0.0

        iter_finished_count = len(iter_finished_returns)
        iter_mean_ret = float(np.mean(iter_finished_returns)) if iter_finished_returns else float("nan")
        iter_succ_rate = float(np.mean(iter_finished_successes)) if iter_finished_successes else float("nan")

        # Log row to progress.csv
        row_dict = {
            "iteration": iteration,
            "global_step": global_step,
            "active_time_s": round(active_time_s, 2),
            "sps": sps,
            "mean_reward": round(mean_rew, 4),
            "mean_return": round(mean_ret, 3),
            "mean_length": round(mean_len, 1),
            "success_rate": round(succ_rate, 4),
            "iter_finished_episodes": iter_finished_count,
            "iter_return": round(iter_mean_ret, 3) if not np.isnan(iter_mean_ret) else "",
            "iter_success_rate": round(iter_succ_rate, 4) if not np.isnan(iter_succ_rate) else "",
            "pg_loss": round(float(np.mean(pg_losses)), 5),
            "v_loss": round(float(np.mean(v_losses)), 5),
            "entropy": round(float(np.mean(entropies)), 5),
            "approx_kl": round(float(np.mean(approx_kls)), 5),
            "clipfrac": round(float(np.mean(clipfracs)), 4),
            "lr": current_lr,
        }
        log_progress_row(csv_path, row_dict)

        if iteration % 1 == 0 or iteration == total_iterations:
            iter_succ_str = f"{iter_succ_rate*100:5.1f}%" if not np.isnan(iter_succ_rate) else "  N/A"
            iter_ret_str = f"{iter_mean_ret:5.2f}" if not np.isnan(iter_mean_ret) else "  N/A"
            cuda_vram_gib = (torch.cuda.max_memory_allocated() / (1024 ** 3)) if torch.cuda.is_available() else 0.0
            cpu_ram_mb = psutil.Process().memory_info().rss / (1024 ** 2)
            logger.info(
                "Iter %3d/%3d | Step %8s | Iter Fin %2d (Succ %s, Ret %s) | SPS %4d | Active %5.1fs | VRAM %.2fG | RAM %.0fM",
                iteration,
                total_iterations,
                f"{global_step:,}",
                iter_finished_count,
                iter_succ_str,
                iter_ret_str,
                sps,
                active_time_s,
                cuda_vram_gib,
                cpu_ram_mb,
            )

        # Accumulate recent iteration success rates
        if not np.isnan(iter_succ_rate):
            recent_iter_succs.append(iter_succ_rate)
        else:
            recent_iter_succs.append(0.0)
        if len(recent_iter_succs) > 5:
            recent_iter_succs.pop(0)

        # Early Sanity Monitoring for Pilot (SPEC Section 4.3 rule 3)
        # At 5M and 8M steps, if mean training-episode success over last 5 iterations == 0, pause.
        is_pilot_run = cfg.get("is_pilot", False) or (run_name and run_name.startswith("pilot_"))
        sanity_checkpoints = cfg.get("sanity_checkpoint_steps", [5_000_000, 8_000_000])
        prev_step = global_step - (num_envs * num_steps)
        if is_pilot_run and len(recent_iter_succs) >= 5:
            for s_chk in sanity_checkpoints:
                if prev_step < s_chk <= global_step:
                    mean_5iter_succ = float(np.mean(recent_iter_succs[-5:]))
                    if mean_5iter_succ == 0.0:
                        alert_text = (
                            f"======================================================================\n"
                            f" EARLY SANITY ALERT (Zero Training Success Detected)\n"
                            f"======================================================================\n"
                            f" Timestamp:                      {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}\n"
                            f" Run ID:                        {run_name or out_dir}\n"
                            f" Global Step:                    {global_step:,}\n"
                            f" Sanity Checkpoint:              {s_chk:,}\n"
                            f" Last 5 Iteration Success Rates: {[round(x, 4) for x in recent_iter_succs[-5:]]}\n"
                            f" Mean Success (Last 5 Iters):    {mean_5iter_succ:.4f}\n"
                            f"----------------------------------------------------------------------\n"
                            f" Protocol Trigger: SPEC early sanity monitoring paused the pilot run\n"
                            f" because training success remained at 0.0 across the last 5 iterations.\n"
                            f" Training settings have NOT been modified.\n"
                            f" Action Required: Review learning progress. The runner is halted and will\n"
                            f" not continue until explicitly resumed via 'python src/ctl.py resume'.\n"
                            f"======================================================================\n"
                        )
                        logger.error("\n%s", alert_text)
                        alert_file = state_dir / "ALERT.txt"
                        tmp_alert = str(alert_file) + f".tmp.{os.getpid()}"
                        with open(tmp_alert, "w") as f:
                            f.write(alert_text)
                            f.flush()
                            os.fsync(f.fileno())
                        os.replace(tmp_alert, str(alert_file))

                        # Set state/PAUSE
                        pause_file = state_dir / "PAUSE"
                        with open(str(pause_file), "w") as f:
                            f.write(f"PAUSE requested by Early Sanity Monitor at step {global_step}\n")

                        # Save checkpoint atomically
                        last_saved_path = save_checkpoint(
                            ckpt_dir=ckpt_dir,
                            model=agent,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            global_step=global_step,
                            iteration=iteration,
                            config=cfg,
                            rng_states=capture_rng_states(),
                            norm_stats=norm_stats,
                            eval_history=eval_history,
                            last_eval_step=last_eval_step,
                            active_time_s=active_time_s,
                            is_final=False,
                            rolling_keep=rolling_keep,
                            milestone_interval=milestone_interval,
                        )
                        env.close()
                        logger.info("[Early Sanity Alert] Checkpoint saved. Pausing training process.")
                        return {
                            "status": "paused",
                            "reason": "early_sanity_alert",
                            "global_step": global_step,
                            "iteration": iteration,
                            "active_time_s": active_time_s,
                            "checkpoint_path": last_saved_path,
                        }

        # Periodic Evaluation Trigger (SPEC Section 5)
        eval_interval = int(cfg.get("eval_interval_steps", 1000000))
        eval_just_ran = False
        if eval_interval > 0 and global_step > 0:
            current_eval_bucket = global_step // eval_interval
            last_eval_bucket = last_eval_step // eval_interval if last_eval_step > 0 else 0
            if current_eval_bucket > last_eval_bucket:
                logger.info(
                    "[Periodic Eval] Step %s crossed %s-step interval boundary. Launching periodic evaluation...",
                    f"{global_step:,}",
                    f"{eval_interval:,}",
                )
                summary_row, per_view_records = run_periodic_evaluation(
                    agent=agent,
                    global_step=global_step,
                    modality=mod,
                    config=cfg,
                    logs_dir=logs_dir,
                    device=dev_str,
                    pause_paths=pause_paths,
                )
                if summary_row is not None:
                    eval_history.append(summary_row)
                    last_eval_step = global_step
                    eval_just_ran = True
                    logger.info(
                        "[Periodic Eval] Completed eval at step %s | Train Mean: %.1f%% | Val Mean: %.1f%%",
                        f"{global_step:,}",
                        summary_row["train_mean"],
                        summary_row["val_mean"],
                    )

        # Checkpoint Decision
        is_final = (iteration >= total_iterations)
        time_since_save = time.time() - last_save_time
        iters_since_save = iteration - last_save_iter
        pause_triggered = check_pause_files(pause_paths)

        should_checkpoint = (
            is_final
            or pause_triggered
            or eval_just_ran
            or (time_since_save >= checkpoint_interval_s)
            or (iters_since_save >= checkpoint_interval_iters)
        )

        if should_checkpoint:
            last_saved_path = save_checkpoint(
                ckpt_dir=ckpt_dir,
                model=agent,
                optimizer=optimizer,
                global_step=global_step,
                iteration=iteration,
                config=cfg,
                rng_states=capture_rng_states(),
                norm_stats=norm_stats,
                eval_history=eval_history,
                last_eval_step=last_eval_step,
                active_time_s=active_time_s,
                is_final=is_final,
                rolling_keep=rolling_keep,
                milestone_interval=milestone_interval,
            )
            last_save_time = time.time()
            last_save_iter = iteration

        # Handle post-iteration Pause Exit
        if pause_triggered:
            logger.info(
                "[Pause] Iteration %d cleanly finished. Checkpoint saved: %s. Active time: %.1fs. Exiting code 0.",
                iteration,
                last_saved_path,
                active_time_s,
            )
            env.close()
            sys.exit(0)

    # Completed full budget
    env.close()
    logger.info("[Done] Training reached target budget (%d steps). Final checkpoint: %s", global_step, last_saved_path)
    return {
        "status": "completed",
        "global_step": global_step,
        "iteration": total_iterations,
        "active_time_s": active_time_s,
        "checkpoint_path": last_saved_path,
    }


def parse_args() -> argparse.Namespace:
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description="Resumable CleanRL PPO for Joint Multi-View PickCube")
    parser.add_argument("--config", type=str, default="configs/ppo_frozen.yaml", help="Path to YAML configuration file")
    parser.add_argument("--run_name", type=str, default=None, help="Name of run from configs/runs.yaml")
    parser.add_argument("--runs_config", type=str, default="configs/runs.yaml", help="Path to runs configuration YAML")
    parser.add_argument("--modality", type=str, default=None, choices=["rgb", "rgbd"], help="Observation modality")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for checkpoints and logs")
    parser.add_argument("--total_timesteps", type=int, default=None, help="Total environment steps")
    parser.add_argument("--num_envs", type=int, default=None, help="Number of parallel environments")
    parser.add_argument("--num_steps", type=int, default=None, help="Rollout length per environment")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--device", type=str, default=None, help="Compute device ('cuda' or 'cpu')")
    parser.add_argument("--pause_file", type=str, default=None, help="Custom pause file path to monitor")
    parser.add_argument("--ref_environment_path", type=str, default=None, help="Custom path to reference_environment.json")
    parser.add_argument("--exp_name", type=str, default=None, help="Experiment name")
    parser.add_argument("--is-pilot", "--is_pilot", action="store_true", help="Flag indicating pilot run for early sanity checks")
    parser.add_argument("--target_kl", "--target-kl", type=float, default=None, help="Target KL threshold for early stopping")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_cfg = load_yaml_config(args.config)
    if args.is_pilot:
        base_cfg["is_pilot"] = True

    if args.run_name:
        runs_cfg = load_yaml_config(args.runs_config)
        runs_dict = runs_cfg.get("runs", {})
        if args.run_name not in runs_dict:
            raise KeyError(f"Run '{args.run_name}' not found in {args.runs_config}. Available: {list(runs_dict.keys())}")
        run_entry = runs_dict[args.run_name]
        if run_entry.get("total_timesteps") is None and args.total_timesteps is None:
            raise ValueError(
                f"Run '{args.run_name}' has total_timesteps: null. "
                "The budget has not been frozen yet; refusing to start unbudgeted main-grid run."
            )
        if args.modality is None:
            args.modality = run_entry.get("modality")
        if args.seed is None:
            args.seed = run_entry.get("seed")
        if args.total_timesteps is None:
            args.total_timesteps = run_entry.get("total_timesteps")
        if args.output_dir is None:
            args.output_dir = run_entry.get("output_dir")

    train_ppo(
        config=base_cfg,
        output_dir=args.output_dir,
        modality=args.modality,
        seed=args.seed,
        total_timesteps=args.total_timesteps,
        device=args.device,
        pause_file=args.pause_file,
        ref_environment_path=args.ref_environment_path,
        run_name=args.run_name,
        target_kl=args.target_kl,
    )


if __name__ == "__main__":
    main()
