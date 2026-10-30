"""
Checkpointing module per SPEC Section 2.2 and Section 2.3.

Provides:
- Atomic checkpoint saving (write to tmp + fsync + os.replace).
- Companion SHA-256 checksum verification.
- Rolling checkpoint retention (keeps last 2 rolling checkpoints).
- Milestone retention (every 5M steps preserved indefinitely).
- Checkpoint validation (checksum + torch.load load test + key validation).
- Automated fallback to previous valid checkpoint if the latest is corrupt.
- Config-hash guard (refuses resume if configuration has changed).
- Free disk space check (refuses write and raises LowDiskSpaceError if space is low).
- Full RNG state capture and restoration (Python, NumPy, PyTorch CPU & CUDA).

Python 3.9 compatible only.
"""

import difflib
import hashlib
import json
import logging
import os
import random
import re
import shutil
import tempfile
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

# Config keys that represent runtime/logging metadata and do NOT alter the experimental protocol
EXCLUDED_CONFIG_KEYS = {
    # File paths and logging directories
    "output_dir",
    "log_dir",
    "ckpt_dir",
    "run_dir",
    "tb_dir",
    "video_dir",
    "log_path",
    "checkpoint_path",
    "runs_dir",
    "eval_output_dir",
    # Timestamps and runtime markers
    "timestamp",
    "created_at",
    "start_time",
    "last_updated",
    "saved_at",
    "active_time_s",
    # Non-functional runner / hardware flags
    "exp_name",
    "device",
    "track",
    "wandb_project",
    "wandb_project_name",
    "wandb_entity",
    "wandb_group",
    "capture_video",
    "save_train_video_freq",
}


def fsync_dir(dir_path: str) -> None:
    """
    Flushes directory metadata to disk via os.fsync on the directory file descriptor.
    Guarantees directory entry changes (renames/creations/deletions) are physically durable.
    """
    if not dir_path or not os.path.isdir(dir_path):
        return
    try:
        flags = os.O_RDONLY
        if hasattr(os, "O_DIRECTORY"):
            flags |= os.O_DIRECTORY
        fd = os.open(dir_path, flags)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError as e:
        logger.debug("fsync_dir failed on '%s': %s", dir_path, e)


def filter_config_for_hashing(config: Any) -> Any:
    """
    Recursively filters out volatile and non-functional runtime keys (such as output
    directories, timestamps, and wandb settings) from the configuration dictionary so
    that only experimental protocol and algorithmic hyperparameters influence the hash.
    """
    if isinstance(config, dict):
        filtered = {}
        for k, v in config.items():
            if k in EXCLUDED_CONFIG_KEYS:
                continue
            filtered[k] = filter_config_for_hashing(v)
        return filtered
    elif isinstance(config, list):
        return [filter_config_for_hashing(item) for item in config]
    elif isinstance(config, tuple):
        return tuple(filter_config_for_hashing(item) for item in config)
    return config


class LowDiskSpaceError(RuntimeError):
    """Raised when available disk space is below safety threshold."""
    pass


class CorruptCheckpointError(RuntimeError):
    """Raised when a checkpoint file is corrupt or unreadable."""
    pass


class ConfigMismatchError(RuntimeError):
    """Raised when the checkpoint config hash does not match current configuration."""
    pass


def check_disk_space(dir_path: str, min_free_bytes: int = 500 * 1024 * 1024) -> int:
    """
    Checks free disk space in the filesystem containing dir_path.
    
    Args:
        dir_path: Directory path to check.
        min_free_bytes: Minimum required free space in bytes (default: 500 MB).
        
    Returns:
        Available free space in bytes.
        
    Raises:
        LowDiskSpaceError: If free disk space is less than min_free_bytes.
    """
    check_dir = dir_path
    while not os.path.exists(check_dir):
        parent = os.path.dirname(check_dir)
        if parent == check_dir:
            break
        check_dir = parent

    usage = shutil.disk_usage(check_dir)
    free_bytes = usage.free
    if free_bytes < min_free_bytes:
        free_mb = free_bytes / (1024.0 * 1024.0)
        req_mb = min_free_bytes / (1024.0 * 1024.0)
        raise LowDiskSpaceError(
            f"Low disk space in '{dir_path}': {free_mb:.1f} MB free, required {req_mb:.1f} MB."
        )
    return free_bytes


def compute_config_hash(config: Any) -> str:
    """
    Computes a deterministic SHA-256 hash for a configuration object.
    Filters out non-functional metadata (log paths, timestamps, wandb configs)
    while strictly hashing all experiment parameters (seed, num_envs, modality,
    total_steps, split, ppo hyperparameters).
    
    Args:
        config: Dict, string, or JSON-serializable structure.
        
    Returns:
        Hex-encoded SHA-256 hash string.
    """
    filtered = filter_config_for_hashing(config)
    if isinstance(filtered, dict):
        serialized = json.dumps(filtered, sort_keys=True, default=str)
    elif isinstance(filtered, (list, tuple)):
        serialized = json.dumps(filtered, default=str)
    elif isinstance(filtered, str):
        serialized = filtered
    else:
        serialized = str(filtered)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def verify_config_hash(checkpoint_data: dict, current_config: Any) -> bool:
    """
    Verifies that the checkpoint config hash matches the current config.
    
    Args:
        checkpoint_data: Loaded checkpoint dictionary.
        current_config: Active configuration object or dict.
        
    Returns:
        True if hashes match.
        
    Raises:
        ConfigMismatchError: If hashes differ.
    """
    saved_hash = checkpoint_data.get("config_hash")
    current_hash = compute_config_hash(current_config)

    if saved_hash != current_hash:
        diff_msg = ""
        saved_cfg = checkpoint_data.get("config")
        if saved_cfg is not None and isinstance(current_config, dict):
            filtered_saved = filter_config_for_hashing(saved_cfg)
            filtered_curr = filter_config_for_hashing(current_config)
            saved_str = json.dumps(filtered_saved, indent=2, sort_keys=True, default=str).splitlines(keepends=True)
            curr_str = json.dumps(filtered_curr, indent=2, sort_keys=True, default=str).splitlines(keepends=True)
            diff = "".join(difflib.unified_diff(saved_str, curr_str, fromfile="checkpoint_config", tofile="current_config"))
            diff_msg = f"\nConfig diff:\n{diff}"

        raise ConfigMismatchError(
            f"Config hash mismatch! Resuming with modified config is refused per SPEC Section 2.3.\n"
            f"  Checkpoint config hash: {saved_hash}\n"
            f"  Current config hash:    {current_hash}"
            f"{diff_msg}"
        )
    return True


def cleanup_orphan_tmp_files(ckpt_dir: str) -> List[str]:
    """
    Scans ckpt_dir for leftover temporary files (matching *.tmp*) from interrupted writes
    or crashes and deletes them.
    
    Returns:
        List of cleaned up file paths.
    """
    if not os.path.isdir(ckpt_dir):
        return []

    deleted = []
    for fname in os.listdir(ckpt_dir):
        if ".tmp" in fname:
            full_path = os.path.join(ckpt_dir, fname)
            try:
                if os.path.islink(full_path) or os.path.isfile(full_path):
                    os.remove(full_path)
                    deleted.append(full_path)
                    logger.info("Cleaned up orphan temporary file: %s", full_path)
                elif os.path.isdir(full_path):
                    shutil.rmtree(full_path)
                    deleted.append(full_path)
                    logger.info("Cleaned up orphan temporary directory: %s", full_path)
            except OSError as e:
                logger.warning("Failed to delete orphan temporary file '%s': %s", full_path, e)

    if deleted:
        fsync_dir(ckpt_dir)
    return deleted


def compute_file_sha256(file_path: str) -> str:
    """Computes SHA-256 checksum of a file in 64 KB chunks."""
    sha = hashlib.sha256()
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            sha.update(chunk)
    return sha.hexdigest()


def capture_rng_states() -> Dict[str, Any]:
    """Captures RNG states across Python random, NumPy, PyTorch CPU and CUDA."""
    states = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        states["torch_cuda"] = torch.cuda.get_rng_state_all()
    return states


def restore_rng_states(rng_states: Dict[str, Any]) -> None:
    """Restores RNG states for Python random, NumPy, PyTorch CPU and CUDA."""
    if "python" in rng_states and rng_states["python"] is not None:
        random.setstate(rng_states["python"])
    if "numpy" in rng_states and rng_states["numpy"] is not None:
        np.random.set_state(rng_states["numpy"])
    if "torch_cpu" in rng_states and rng_states["torch_cpu"] is not None:
        torch.set_rng_state(rng_states["torch_cpu"])
    if (
        "torch_cuda" in rng_states
        and rng_states["torch_cuda"] is not None
        and torch.cuda.is_available()
    ):
        torch.cuda.set_rng_state_all(rng_states["torch_cuda"])


def validate_checkpoint(ckpt_path: str) -> Tuple[bool, Union[dict, Exception]]:
    """
    Validates a checkpoint file:
    1. Checks file existence and non-zero size.
    2. Verifies SHA-256 checksum if a companion .sha256 file exists.
    3. Runs torch.load load test on CPU.
    4. Validates presence of critical state keys.
    
    Returns:
        (True, loaded_dict) if valid, or (False, exception) if corrupt.
    """
    if not os.path.exists(ckpt_path):
        return False, FileNotFoundError(f"Checkpoint file does not exist: {ckpt_path}")

    try:
        if os.path.getsize(ckpt_path) == 0:
            return False, CorruptCheckpointError(f"Checkpoint file is empty (0 bytes): {ckpt_path}")
    except OSError as e:
        return False, e

    # Checksum verification if companion .sha256 file exists
    sha_file = f"{ckpt_path}.sha256"
    if os.path.exists(sha_file):
        try:
            with open(sha_file, "r") as f:
                expected_sha = f.read().strip()
            actual_sha = compute_file_sha256(ckpt_path)
            if expected_sha != actual_sha:
                return False, CorruptCheckpointError(
                    f"Checksum mismatch for {ckpt_path}. Expected {expected_sha}, got {actual_sha}."
                )
        except Exception as e:
            return False, CorruptCheckpointError(f"Error checking sha256 for {ckpt_path}: {e}")
    else:
        logger.info(
            "No companion .sha256 file found for %s; accepting only if CPU load test passes.",
            ckpt_path,
        )

    # Torch load test
    try:
        try:
            data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        except TypeError:
            data = torch.load(ckpt_path, map_location="cpu")
    except Exception as e:
        return False, CorruptCheckpointError(f"torch.load failed on {ckpt_path}: {e}")

    if not isinstance(data, dict):
        return False, CorruptCheckpointError(f"Checkpoint root must be a dict, got {type(data)}")

    # Verify critical fields
    required_keys = ["model_state_dict", "optimizer_state_dict", "global_step", "iteration"]
    for k in required_keys:
        if k not in data:
            return False, CorruptCheckpointError(f"Missing required key '{k}' in checkpoint {ckpt_path}")

    return True, data


def _extract_step_and_iter(filename: str) -> Tuple[int, int]:
    """Extracts step and iteration numbers from standard checkpoint filenames."""
    step = -1
    iteration = -1
    step_match = re.search(r"step_(\d+)", filename)
    if step_match:
        step = int(step_match.group(1))
    iter_match = re.search(r"iter_(\d+)", filename)
    if iter_match:
        iteration = int(iter_match.group(1))
    return step, iteration


def get_checkpoint_candidates(ckpt_dir: str) -> List[str]:
    """
    Finds and ranks all candidate checkpoint files in ckpt_dir in descending order
    (highest step / iteration first, newest mtime as secondary sort).
    Prioritizes canonical versioned filenames over pointer symlinks like latest.pt.
    """
    if not os.path.isdir(ckpt_dir):
        return []

    versioned_candidates = []
    other_candidates = []

    for fname in os.listdir(ckpt_dir):
        if not fname.endswith(".pt") or fname.endswith(".tmp.pt") or ".tmp" in fname:
            continue

        full_path = os.path.join(ckpt_dir, fname)

        if (
            fname.startswith("ckpt_step_")
            or fname.startswith("milestone_step_")
            or fname.startswith("ckpt_final_")
        ):
            versioned_candidates.append(full_path)
        elif fname == "latest.pt" or fname.startswith("ckpt_"):
            other_candidates.append(full_path)

    # Process versioned candidates first so unique real paths preserve the versioned path
    candidates = versioned_candidates + other_candidates
    unique_candidates = []
    seen_real_paths = set()

    for c in candidates:
        try:
            real_p = os.path.realpath(c)
        except OSError:
            real_p = c
        if real_p not in seen_real_paths:
            seen_real_paths.add(real_p)
            unique_candidates.append(c)

    def sort_key(path: str) -> Tuple[int, int, int, float]:
        fname = os.path.basename(path)
        step, iteration = _extract_step_and_iter(fname)
        if step == -1:
            try:
                real_fname = os.path.basename(os.path.realpath(path))
                step, iteration = _extract_step_and_iter(real_fname)
            except OSError:
                pass

        try:
            mtime = os.path.getmtime(path)
        except OSError:
            mtime = 0.0

        is_final = 1 if "final" in fname else 0
        return (step, is_final, iteration, mtime)

    unique_candidates.sort(key=sort_key, reverse=True)
    return unique_candidates


def get_newest_checkpoint(ckpt_dir: str) -> Optional[Tuple[str, int]]:
    """
    Returns (checkpoint_path, global_step) of the newest valid checkpoint in ckpt_dir,
    or None if no valid checkpoint is found.
    """
    candidates = get_checkpoint_candidates(ckpt_dir)
    for c in candidates:
        is_valid, data = validate_checkpoint(c)
        if is_valid and isinstance(data, dict):
            return c, int(data.get("global_step", 0))
    return None


def prune_rolling_checkpoints(ckpt_dir: str, keep: int = 2) -> List[str]:
    """
    Keeps only the latest `keep` rolling checkpoints (matching `ckpt_step_*.pt`).
    Milestones (`milestone_*.pt`) and final checkpoints (`ckpt_final_*.pt`) are NEVER pruned.
    
    Returns:
        List of deleted checkpoint paths.
    """
    if not os.path.isdir(ckpt_dir) or keep <= 0:
        return []

    rolling_files = []
    for fname in os.listdir(ckpt_dir):
        if re.match(r"^ckpt_step_\d+\.pt$", fname):
            full_path = os.path.join(ckpt_dir, fname)
            step, _ = _extract_step_and_iter(fname)
            try:
                mtime = os.path.getmtime(full_path)
            except OSError:
                mtime = 0.0
            rolling_files.append((step, mtime, full_path))

    # Sort ascending by step and mtime
    rolling_files.sort(key=lambda item: (item[0], item[1]))

    deleted = []
    if len(rolling_files) > keep:
        to_delete = rolling_files[:-keep]
        for _, _, fpath in to_delete:
            try:
                os.remove(fpath)
                deleted.append(fpath)
                sha_path = f"{fpath}.sha256"
                if os.path.exists(sha_path):
                    os.remove(sha_path)
            except OSError as e:
                logger.warning("Failed to remove old rolling checkpoint '%s': %s", fpath, e)

    return deleted


def atomic_write_bytes(target_path: str, write_fn: Any) -> str:
    """
    Writes data atomically: writes to temporary file in same directory,
    flushes and fsyncs, then os.replace onto target_path.
    Also creates a companion .sha256 file atomically.
    
    Returns:
        SHA-256 checksum string of the written file.
    """
    dir_name = os.path.dirname(target_path) or "."
    base_name = os.path.basename(target_path)
    tmp_path = os.path.join(dir_name, f".{base_name}.tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}")
    tmp_sha = f"{tmp_path}.sha256"

    try:
        with open(tmp_path, "wb") as f:
            write_fn(f)
            f.flush()
            os.fsync(f.fileno())

        checksum = compute_file_sha256(tmp_path)
        with open(tmp_sha, "w") as f:
            f.write(checksum + "\n")
            f.flush()
            os.fsync(f.fileno())

        target_sha = f"{target_path}.sha256"
        os.replace(tmp_sha, target_sha)
        fsync_dir(dir_name)
        os.replace(tmp_path, target_path)
        fsync_dir(dir_name)
        return checksum
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        if os.path.exists(tmp_sha):
            try:
                os.remove(tmp_sha)
            except OSError:
                pass


def save_checkpoint(
    ckpt_dir: str,
    model: Union[torch.nn.Module, dict],
    optimizer: Union[torch.optim.Optimizer, dict],
    lr_scheduler: Optional[Union[Any, dict]] = None,
    global_step: int = 0,
    iteration: int = 0,
    config_hash: Optional[str] = None,
    config: Optional[Any] = None,
    rng_states: Optional[Dict[str, Any]] = None,
    norm_stats: Optional[Dict[str, Any]] = None,
    eval_history: Optional[Any] = None,
    last_eval_step: int = 0,
    active_time_s: float = 0.0,
    is_final: bool = False,
    is_milestone: Optional[bool] = None,
    milestone_interval: int = 5_000_000,
    rolling_keep: int = 2,
    min_free_bytes: int = 500 * 1024 * 1024,
) -> str:
    """
    Atomically saves a complete training checkpoint.
    
    Args:
        ckpt_dir: Directory to store checkpoints.
        model: PyTorch model or state dict.
        optimizer: Optimizer or state dict.
        lr_scheduler: Learning rate scheduler or state dict.
        global_step: Total environment steps trained so far.
        iteration: Current PPO iteration count.
        config_hash: SHA-256 hash string of config. Auto-derived if config provided.
        config: Full configuration dict/object.
        rng_states: Pre-captured RNG dict or None to capture automatically.
        norm_stats: Normalization statistics.
        eval_history: Periodic evaluation logs/metrics history.
        active_time_s: Cumulative active training time in seconds (excluding pause).
        is_final: Flag marking the run's final checkpoint.
        is_milestone: Explicit milestone flag. If None, derived from milestone_interval.
        milestone_interval: Step interval for milestone checkpoints (default: 5M).
        rolling_keep: Number of rolling checkpoints to preserve (default: 2).
        min_free_bytes: Free disk space threshold before writing (default: 500 MB).
        
    Returns:
        Absolute path to the saved checkpoint.
        
    Raises:
        LowDiskSpaceError: If disk space is below min_free_bytes.
    """
    os.makedirs(ckpt_dir, exist_ok=True)
    cleanup_orphan_tmp_files(ckpt_dir)
    check_disk_space(ckpt_dir, min_free_bytes=min_free_bytes)

    if config_hash is None and config is not None:
        config_hash = compute_config_hash(config)
    elif config_hash is None:
        config_hash = ""

    # Model and optimizer states
    model_sd = model.state_dict() if isinstance(model, torch.nn.Module) else model
    optim_sd = optimizer.state_dict() if isinstance(optimizer, torch.optim.Optimizer) else optimizer
    sched_sd = None
    if lr_scheduler is not None:
        sched_sd = (
            lr_scheduler.state_dict()
            if hasattr(lr_scheduler, "state_dict") and callable(lr_scheduler.state_dict)
            else lr_scheduler
        )

    if rng_states is None:
        rng_states = capture_rng_states()

    payload = {
        "model_state_dict": model_sd,
        "optimizer_state_dict": optim_sd,
        "lr_scheduler_state_dict": sched_sd,
        "global_step": int(global_step),
        "iteration": int(iteration),
        "rng_states": rng_states,
        "norm_stats": norm_stats,
        "eval_history": eval_history,
        "last_eval_step": int(last_eval_step),
        "active_time_s": float(active_time_s),
        "config_hash": config_hash,
        "config": config,
        "saved_at": time.time(),
    }

    # Decide filename
    if is_final:
        filename = f"ckpt_final_step_{global_step:09d}.pt"
    elif is_milestone is True or (
        milestone_interval > 0 and global_step > 0 and global_step % milestone_interval == 0
    ):
        filename = f"milestone_step_{global_step:09d}.pt"
    else:
        filename = f"ckpt_step_{global_step:09d}.pt"

    target_path = os.path.join(ckpt_dir, filename)

    t0_save = time.time()
    # Atomic write
    def write_payload(f_out):
        torch.save(payload, f_out)

    atomic_write_bytes(target_path, write_payload)
    save_duration = time.time() - t0_save
    try:
        ckpt_size_mb = os.path.getsize(target_path) / (1024 * 1024)
        logger.info("[Bench] Checkpoint '%s' saved in %.3fs (%.2f MB)", filename, save_duration, ckpt_size_mb)
    except OSError:
        pass

    # Atomically update latest.pt pointer/symlink
    latest_path = os.path.join(ckpt_dir, "latest.pt")
    tmp_latest_symlink = os.path.join(
        ckpt_dir, f".latest.pt.tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}"
    )
    try:
        os.symlink(filename, tmp_latest_symlink)
        os.replace(tmp_latest_symlink, latest_path)
        fsync_dir(ckpt_dir)
    except OSError:
        # Fallback to atomic copy if symlinks are not supported on filesystem
        atomic_write_bytes(latest_path, write_payload)
        fsync_dir(ckpt_dir)
    finally:
        if os.path.islink(tmp_latest_symlink) or os.path.exists(tmp_latest_symlink):
            try:
                os.remove(tmp_latest_symlink)
            except OSError:
                pass

    # Prune rolling checkpoints (milestones and final are preserved)
    if not is_final and not (
        is_milestone is True or (milestone_interval > 0 and global_step > 0 and global_step % milestone_interval == 0)
    ):
        prune_rolling_checkpoints(ckpt_dir, keep=rolling_keep)

    return os.path.abspath(target_path)


def load_checkpoint_with_fallback(
    ckpt_dir_or_path: str,
    current_config: Optional[Any] = None,
) -> Tuple[Optional[dict], Optional[str]]:
    """
    Discovers candidate checkpoints in ckpt_dir_or_path in descending order and returns
    the newest valid checkpoint.
    
    If ckpt_dir_or_path is a file and is valid, loads it directly. If corrupt,
    automatically searches its parent directory for fallback candidates.
    
    If the latest candidate is corrupt/truncated, logs a warning and automatically
    falls back to the previous valid checkpoint.
    
    If current_config is provided, verifies config hash and refuses resume if mismatch.
    
    Returns:
        (checkpoint_data, checkpoint_path) if a valid checkpoint is found,
        or (None, None) if no checkpoints exist.
        
    Raises:
        ConfigMismatchError: If the valid checkpoint has a config hash mismatch.
        CorruptCheckpointError: If checkpoints exist but ALL of them are corrupt.
    """
    if os.path.isfile(ckpt_dir_or_path):
        is_valid, data_or_err = validate_checkpoint(ckpt_dir_or_path)
        if is_valid:
            if current_config is not None:
                verify_config_hash(data_or_err, current_config)
            return data_or_err, os.path.abspath(ckpt_dir_or_path)
        else:
            logger.warning(
                "[Checkpoint] Specified checkpoint '%s' failed validation (%s). Searching parent directory for fallback...",
                ckpt_dir_or_path,
                data_or_err,
            )
            ckpt_dir = os.path.dirname(ckpt_dir_or_path)
    else:
        ckpt_dir = ckpt_dir_or_path

    cleanup_orphan_tmp_files(ckpt_dir)
    candidates = get_checkpoint_candidates(ckpt_dir)
    if not candidates:
        return None, None

    errors = []
    attempted = 0

    for candidate_path in candidates:
        attempted += 1
        is_valid, data_or_err = validate_checkpoint(candidate_path)
        if is_valid:
            if attempted > 1 or os.path.isfile(ckpt_dir_or_path):
                logger.warning(
                    "[Checkpoint] Successfully recovered by falling back to previous valid checkpoint: %s",
                    candidate_path,
                )

            # Check config guard if requested
            if current_config is not None:
                verify_config_hash(data_or_err, current_config)

            return data_or_err, os.path.abspath(candidate_path)
        else:
            logger.warning(
                "[Checkpoint] Candidate checkpoint '%s' failed validation (%s). Falling back...",
                candidate_path,
                data_or_err,
            )
            errors.append((candidate_path, data_or_err))

    err_summary = "\n".join(f" - {p}: {e}" for p, e in errors)
    raise CorruptCheckpointError(
        f"All {len(candidates)} checkpoint candidate(s) in '{ckpt_dir}' failed validation:\n{err_summary}"
    )
