"""
Evaluation Harness for Joint Multi-View RGB and RGB-D Policies.

Per SPEC Section 5 and Milestone M2a:
- Supports evaluation of checkpoints across fixed viewpoints (viewpoints.json)
  and fresh sampled viewpoints (configs/fresh_views.json).
- Vectorized evaluation using make_joint_multiview_env from src/env_factory.py.
- Deterministic evaluation seeds shared across model arms.
- Action selection matching existing benchmark evaluators (deterministic=True).
- Computes Wilson 95% score confidence intervals for binomial success rates.
- Generates structured CSV results with group labels, nearest-train distances,
  returns, episode lengths, and Wilson intervals.

Python 3.9 compatible only.
"""

import argparse
import csv
import json
import logging
import math
import os
import random
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

# Experiment workspace setup
EXP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EXP_DIR))

from src.env_factory import (
    DEFAULT_TARGET_POS,
    find_viewpoint_split_file,
    find_viewpoints_file,
    load_viewpoint_split,
    load_viewpoints,
    make_joint_multiview_env,
)
from src.model import Agent

logger = logging.getLogger(__name__)


def wilson_score_interval(
    successes: int,
    total: int,
    confidence: float = 0.95,
    z: Optional[float] = None,
) -> Tuple[float, float, float]:
    """
    Computes Wilson score confidence interval for a binomial proportion.
    
    Args:
        successes: Number of successful outcomes (k).
        total: Total number of trials (n).
        confidence: Confidence level (default 0.95).
        z: Optional explicit critical value (default 1.959963984540054 for 95%).
        
    Returns:
        (p, lower_bound, upper_bound) as floats in [0.0, 1.0].
    """
    if total <= 0:
        return 0.0, 0.0, 0.0

    p = float(successes) / float(total)

    if z is None:
        # Standard two-sided normal quantile: 95% -> 1.959963984540054
        if abs(confidence - 0.95) < 1e-4:
            z = 1.959963984540054
        else:
            # Approximate via scipy or math if needed
            try:
                from scipy.stats import norm
                z = float(norm.ppf(1.0 - (1.0 - confidence) / 2.0))
            except ImportError:
                z = 1.959963984540054

    z2 = z * z
    denom = 1.0 + z2 / float(total)
    center = (p + z2 / (2.0 * float(total))) / denom
    margin = (z / denom) * math.sqrt((p * (1.0 - p)) / float(total) + z2 / (4.0 * float(total) * float(total)))

    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)

    return p, lower, upper


def remap_legacy_weights(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Remaps legacy checkpoint state dict keys to src/model.py Agent keys.
    Supports checkpoints from train_joint_multiview.py and ppo_rgb.py.
    """
    remapped = {}
    for k, v in state_dict.items():
        new_k = k
        # ppo_rgb / sequential_viewpoint_training key mappings
        new_k = new_k.replace("feature_net.extractors.rgb.0.", "feature_net.cnn.")
        new_k = new_k.replace("feature_net.extractors.rgb.1.0.", "feature_net.fc.0.")
        new_k = new_k.replace("feature_net.extractors.state.", "feature_net.state_fc.0.")
        remapped[new_k] = v
    return remapped


def load_agent_checkpoint(
    checkpoint_path: str,
    modality: str,
    device: str = "cuda",
    in_channels: Optional[int] = None,
    action_dim: int = 4,
    state_dim: int = 29,
    state_activation: Optional[str] = None,
) -> Agent:
    """
    Loads an Agent model from a checkpoint file.
    
    Supports:
    - Experiment atomic checkpoints with "model_state_dict" or "model"
    - Legacy checkpoints (e.g. backup_original_model.pt) with weight remapping
    - Direct state dict files
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if state_activation is None:
        if isinstance(ckpt, dict) and "config" in ckpt and isinstance(ckpt["config"], dict):
            state_activation = ckpt["config"].get("model", {}).get("state_activation", "none")
        else:
            state_activation = "none"

    if in_channels is None:
        in_channels = 4 if modality.lower() == "rgbd" else 3

    agent = Agent(
        in_channels=in_channels,
        action_dim=action_dim,
        state_dim=state_dim,
        include_state=True,
        state_activation=state_activation,
    ).to(device)

    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            sd = ckpt["model_state_dict"]
        elif "model" in ckpt:
            sd = ckpt["model"]
        elif "state_dict" in ckpt:
            sd = ckpt["state_dict"]
        else:
            sd = ckpt
    else:
        sd = ckpt

    # Check if remapping is needed for legacy checkpoints
    sample_key = next(iter(sd.keys())) if sd else ""
    if "feature_net.extractors." in sample_key:
        sd = remap_legacy_weights(sd)

    agent.load_state_dict(sd, strict=False)
    agent.eval()
    return agent


def get_viewpoint_metadata_dict(
    viewpoints_file: Optional[str] = None,
    split_file: Optional[str] = None,
    fresh_views_file: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Builds a lookup table of viewpoint metadata, group labels, and nearest-train distances.
    """
    if viewpoints_file is None:
        viewpoints_file = find_viewpoints_file()

    fixed_vps = load_viewpoints(viewpoints_file)

    split = None
    try:
        if split_file is None:
            split_file = find_viewpoint_split_file()
        split = load_viewpoint_split(split_file)
    except Exception:
        pass

    # Reverse split mapping: vp_id -> group
    vp_to_group = {}
    train_ids = set()
    if split:
        for grp, ids in split.items():
            for vid in ids:
                vp_to_group[vid] = grp
        train_ids = set(split.get("train", []))

    # Calculate unit vectors for fixed viewpoints
    fixed_u = {}
    for vp in fixed_vps:
        rel = np.array(vp["rel_xyz"], dtype=float)
        fixed_u[vp["id"]] = rel / np.linalg.norm(rel)

    meta = {}
    for vp in fixed_vps:
        vid = vp["id"]
        grp = vp_to_group.get(vid, "fixed")
        # Compute distance to nearest train view
        if vid in train_ids:
            nearest_dist = 0.0
        elif train_ids:
            u_vid = fixed_u[vid]
            dists = [
                np.degrees(np.arccos(np.clip(np.dot(u_vid, fixed_u[tid]), -1.0, 1.0)))
                for tid in train_ids
            ]
            nearest_dist = float(min(dists))
        else:
            nearest_dist = 0.0

        meta[f"fixed_{vid}"] = {
            "view_id": vid,
            "type": "fixed",
            "group": grp,
            "azimuth_deg": float(vp["azimuth_deg"]),
            "elevation_deg": float(vp["elevation_deg"]),
            "nearest_train_dist_deg": round(nearest_dist, 4),
            "world_eye": vp["world_eye"],
        }

    # Load fresh views if available
    if fresh_views_file is None:
        cand_fresh = EXP_DIR / "configs" / "fresh_views.json"
        if cand_fresh.is_file():
            fresh_views_file = str(cand_fresh)

    if fresh_views_file and os.path.isfile(fresh_views_file):
        with open(fresh_views_file, "r") as f:
            fresh_vps = json.load(f)
        for vp in fresh_vps:
            vid = vp["id"]
            meta[f"fresh_{vid}"] = {
                "view_id": vid,
                "type": "fresh",
                "group": "fresh",
                "azimuth_deg": float(vp["azimuth_deg"]),
                "elevation_deg": float(vp["elevation_deg"]),
                "nearest_train_dist_deg": float(vp.get("nearest_train_dist_deg", 0.0)),
                "world_eye": vp["world_eye"],
            }

    return meta


def evaluate_single_viewpoint(
    agent: nn.Module,
    viewpoint_id: Union[int, str],
    modality: str,
    episodes: int = 200,
    num_envs: int = 160,
    base_seed: int = 42,
    seed: Optional[int] = None,
    viewpoints_file: Optional[str] = None,
    device: str = "cuda",
) -> Dict[str, Any]:
    """
    Evaluates policy on a single viewpoint across parallel environments.
    Enforces a strict per-environment episode quota so that fast-terminating
    successful episodes do not crowd out slower or failed episodes, guaranteeing
    that results do not depend on num_envs beyond statistical noise.
    
    Args:
        agent: Trained policy agent (in eval mode).
        viewpoint_id: Integer viewpoint ID (or 'train') to evaluate.
        modality: "rgb" or "rgbd".
        episodes: Total number of evaluation episodes.
        num_envs: Parallel simulation environments (clamped to episodes, default: 160).
        base_seed: Base random seed for evaluation reproducibility.
        seed: Optional explicit random seed. If None, derives eval_seed = int(viewpoint_id * 1000 + base_seed).
        viewpoints_file: Optional path to viewpoints JSON file.
        device: Device for tensors.
        
    Returns:
        Dictionary containing evaluation metrics:
        - "success_count", "episodes", "success_rate" (any-step success)
        - "wilson_95_lower", "wilson_95_upper"
        - "any_step_success_count", "any_step_success_rate", "any_step_wilson_95_lower", "any_step_wilson_95_upper"
        - "final_step_success_count", "final_step_success_rate", "final_step_wilson_95_lower", "final_step_wilson_95_upper"
        - "mean_return", "std_return"
        - "mean_length", "std_length"
    """
    # Clamp num_envs to episodes to avoid superfluous simulations
    effective_num_envs = max(1, min(num_envs, episodes))
    base_quota = episodes // effective_num_envs
    remainder = episodes % effective_num_envs
    quotas = [base_quota + (1 if i < remainder else 0) for i in range(effective_num_envs)]
    completed_per_env = [0] * effective_num_envs

    env = make_joint_multiview_env(
        num_envs=effective_num_envs,
        modality=modality,
        viewpoint_ids=[viewpoint_id],
        viewpoints_file=viewpoints_file,
    )

    if seed is not None:
        eval_seed = int(seed)
    else:
        vp_num = 0 if str(viewpoint_id).lower() in ("train", "training", "-1") else int(viewpoint_id)
        eval_seed = int(vp_num * 1000 + base_seed)

    random.seed(eval_seed)
    np.random.seed(eval_seed)
    torch.manual_seed(eval_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(eval_seed)

    obs, _ = env.reset(seed=eval_seed)

    completed = 0
    any_step_success_count = 0
    final_step_success_count = 0
    returns: List[float] = []
    episode_lens: List[float] = []

    with torch.no_grad():
        while completed < episodes:
            # Deterministic action selection matching existing evaluators
            action = agent.get_action(obs, deterministic=True)
            obs, rew, term, trunc, infos = env.step(action)

            if "final_info" in infos:
                mask = infos["_final_info"]
                for i in range(len(mask)):
                    if mask[i] and completed_per_env[i] < quotas[i]:
                        ep_info = infos["final_info"]["episode"]
                        # Any-step success (success_once)
                        any_succ = bool(
                            ep_info["success_once"][i].item()
                            if "success_once" in ep_info
                            else (ep_info["success"][i].item() if "success" in ep_info else False)
                        )
                        # Final-step success (at boundary termination/truncation)
                        final_succ = bool(
                            infos["final_info"]["success"][i].item()
                            if "success" in infos["final_info"]
                            else any_succ
                        )

                        ret_val = ep_info["return"][i]
                        ret = float(ret_val.item() if torch.is_tensor(ret_val) else ret_val)

                        len_val = ep_info["episode_len"][i]
                        ep_len = float(len_val.item() if torch.is_tensor(len_val) else len_val)

                        if any_succ:
                            any_step_success_count += 1
                        if final_succ:
                            final_step_success_count += 1

                        returns.append(ret)
                        episode_lens.append(ep_len)
                        completed_per_env[i] += 1
                        completed += 1

                        if completed >= episodes:
                            break

    env.close()

    p_any, l_any, u_any = wilson_score_interval(any_step_success_count, completed, confidence=0.95)
    p_final, l_final, u_final = wilson_score_interval(final_step_success_count, completed, confidence=0.95)
    mean_ret = float(np.mean(returns)) if returns else 0.0
    std_ret = float(np.std(returns)) if returns else 0.0
    mean_len = float(np.mean(episode_lens)) if episode_lens else 0.0
    std_len = float(np.std(episode_lens)) if episode_lens else 0.0

    return {
        "view_id": viewpoint_id,
        "episodes": completed,
        "success_count": any_step_success_count,
        "success_rate": round(p_any * 100.0, 2),
        "wilson_95_lower": round(l_any * 100.0, 2),
        "wilson_95_upper": round(u_any * 100.0, 2),
        "any_step_success_count": any_step_success_count,
        "any_step_success_rate": round(p_any * 100.0, 2),
        "any_step_wilson_95_lower": round(l_any * 100.0, 2),
        "any_step_wilson_95_upper": round(u_any * 100.0, 2),
        "final_step_success_count": final_step_success_count,
        "final_step_success_rate": round(p_final * 100.0, 2),
        "final_step_wilson_95_lower": round(l_final * 100.0, 2),
        "final_step_wilson_95_upper": round(u_final * 100.0, 2),
        "mean_return": round(mean_ret, 4),
        "std_return": round(std_ret, 4),
        "mean_length": round(mean_len, 2),
        "std_length": round(std_len, 2),
    }


def evaluate_viewpoint_set(
    agent: nn.Module,
    viewpoint_ids: List[Union[int, str]],
    modality: str,
    viewpoints_file: Optional[str] = None,
    viewpoint_type: str = "fixed",
    episodes_per_view: int = 200,
    num_envs: int = 160,
    base_seed: int = 42,
    device: str = "cuda",
    metadata_lookup: Optional[Dict[str, Dict[str, Any]]] = None,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """
    Evaluates an agent across a list of viewpoints and enriches with metadata.
    """
    if metadata_lookup is None:
        metadata_lookup = get_viewpoint_metadata_dict()

    results = []

    if verbose:
        print("\n" + "=" * 90)
        print(f" EVALUATING POLICY ({modality.upper()}) | Views: {len(viewpoint_ids)} | Episodes/View: {episodes_per_view}")
        print("=" * 90)
        print(f"{'VP ID':<6} | {'Type':<6} | {'Group':<12} | {'Azimuth':<8} | {'Elevation':<9} | {'Dist to Train':<14} | {'Success Rate (95% CI)':<24} | {'Mean Return':<10}")
        print("-" * 90)

    for vid in viewpoint_ids:
        res = evaluate_single_viewpoint(
            agent=agent,
            viewpoint_id=vid,
            modality=modality,
            episodes=episodes_per_view,
            num_envs=num_envs,
            base_seed=base_seed,
            viewpoints_file=viewpoints_file,
            device=device,
        )

        lookup_key = f"{viewpoint_type}_{vid}"
        meta = metadata_lookup.get(lookup_key, {})

        res["type"] = viewpoint_type
        res["group"] = meta.get("group", viewpoint_type)
        res["azimuth_deg"] = meta.get("azimuth_deg", 0.0)
        res["elevation_deg"] = meta.get("elevation_deg", 0.0)
        res["nearest_train_dist_deg"] = meta.get("nearest_train_dist_deg", 0.0)

        results.append(res)

        if verbose:
            ci_str = f"{res['success_rate']:5.1f}% [{res['wilson_95_lower']:4.1f}%, {res['wilson_95_upper']:4.1f}%]"
            vid_str = str(vid)
            print(
                f"{vid_str:<6} | {viewpoint_type:<6} | {res['group']:<12} | "
                f"{res['azimuth_deg']:+7.1f}° | {res['elevation_deg']:8.1f}° | "
                f"{res['nearest_train_dist_deg']:11.1f}° | {ci_str:<24} | {res['mean_return']:10.2f}"
            )

    return results


EVAL_FIELDNAMES = [
    "view_id",
    "type",
    "group",
    "azimuth_deg",
    "elevation_deg",
    "nearest_train_dist_deg",
    "episodes",
    "success_count",
    "success_rate",
    "wilson_95_lower",
    "wilson_95_upper",
    "any_step_success_count",
    "any_step_success_rate",
    "any_step_wilson_95_lower",
    "any_step_wilson_95_upper",
    "final_step_success_count",
    "final_step_success_rate",
    "final_step_wilson_95_lower",
    "final_step_wilson_95_upper",
    "mean_return",
    "std_return",
    "mean_length",
    "std_length",
]


def write_evaluation_csv(results: List[Dict[str, Any]], output_path: Union[str, Path]) -> None:
    """
    Writes evaluation results to CSV with standardized column structure.
    Reports both any-step and final-step success metrics with Wilson confidence intervals.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=EVAL_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    logger.info("Saved evaluation results to %s", output_path)


def append_evaluation_csv_row(row: Dict[str, Any], output_path: Union[str, Path]) -> None:
    """
    Appends a single viewpoint result row to CSV and flushes to disk.
    Creates header if file does not exist yet.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    exists = output_path.is_file()
    with open(output_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=EVAL_FIELDNAMES, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)
        f.flush()
        os.fsync(f.fileno())


def load_completed_view_ids(csv_path: Union[str, Path], required_episodes: int) -> Dict[Union[int, str], Dict[str, Any]]:
    """
    Loads already completed viewpoints from a CSV file.
    A viewpoint is considered completed if its episode count >= required_episodes.
    """
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        return {}
    completed = {}
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                vid_raw = row["view_id"]
                vid = int(vid_raw) if str(vid_raw).strip("-").isdigit() else vid_raw
                episodes = int(row.get("episodes", 0))
                if episodes >= required_episodes:
                    # Cast numeric fields for downstream aggregation
                    for k in ["success_count", "episodes", "any_step_success_count", "final_step_success_count"]:
                        if k in row and row[k] != "":
                            row[k] = int(row[k])
                    for k in [
                        "success_rate", "wilson_95_lower", "wilson_95_upper",
                        "any_step_success_rate", "any_step_wilson_95_lower", "any_step_wilson_95_upper",
                        "final_step_success_rate", "final_step_wilson_95_lower", "final_step_wilson_95_upper",
                        "mean_return", "std_return", "mean_length", "std_length",
                        "azimuth_deg", "elevation_deg", "nearest_train_dist_deg",
                    ]:
                        if k in row and row[k] != "":
                            row[k] = float(row[k])
                    completed[vid] = row
            except (ValueError, KeyError):
                pass
    return completed


def check_pause_files(pause_paths: Optional[List[str]]) -> bool:
    """Checks whether any registered PAUSE file exists."""
    if not pause_paths:
        return False
    for p in pause_paths:
        if p and os.path.isfile(p):
            logger.info("[Final Eval Pause] Detected PAUSE file at: %s", p)
            return True
    return False


EVAL_STOP_REQUESTED = False


def get_group_for_view(vid: Union[int, str], split: Optional[Dict[str, List[int]]] = None) -> str:
    """Helper returning viewpoint group name ('train', 'val', 'test_interp', 'test_extrap', 'unknown')."""
    if split is None:
        split = load_viewpoint_split()
    try:
        vid_int = int(vid)
    except (ValueError, TypeError):
        return "unknown"
    for grp in ("train", "val", "test_interp", "test_extrap"):
        if vid_int in split.get(grp, []):
            return grp
    return "unknown"


def run_final_evaluation(
    checkpoint_path: Union[str, Path],
    output_dir: Union[str, Path],
    modality: str,
    config: Optional[Dict[str, Any]] = None,
    viewpoints_file: Optional[str] = None,
    fresh_views_file: Optional[str] = None,
    fixed_episodes: Optional[int] = None,
    fresh_episodes: Optional[int] = None,
    fixed_view_ids: Optional[List[Union[int, str]]] = None,
    fresh_view_ids: Optional[List[Union[int, str]]] = None,
    num_envs: Optional[int] = None,
    base_seed: int = 42,
    device: Optional[str] = None,
    pause_paths: Optional[List[str]] = None,
    state_activation: Optional[str] = None,
    verbose: bool = True,
    eval_groups: Optional[Union[str, List[str]]] = None,
) -> Dict[str, Any]:
    """
    Executes final evaluation per SPEC Section 5:
    - Fixed viewpoints x 200 episodes (filtered by eval_groups, default 'all').
    - Fresh viewpoints x 100 episodes (only if 'fresh' in eval_groups).
    - Checks pause flag between viewpoints; exits cleanly if pause requested.
    - Idempotent: reads existing CSVs and skips already-completed viewpoints.
    - Saves fixed_views_eval.csv, fresh_views_eval.csv, all_views_eval.csv,
      and final_eval_summary.json.
    - Strictly enforces eval_groups: refuses test views if not permitted.
    - Returns status dict with 'completed' or 'paused'.
    """
    global EVAL_STOP_REQUESTED
    EVAL_STOP_REQUESTED = False

    output_path = Path(output_dir)
    if output_path.name == "final_eval":
        final_eval_dir = output_path
    else:
        final_eval_dir = output_path / "final_eval"
    final_eval_dir.mkdir(parents=True, exist_ok=True)

    summary_file = final_eval_dir / "final_eval_summary.json"
    cfg = config or {}

    effective_groups = eval_groups if eval_groups is not None else cfg.get("final_eval_groups", "all")
    if isinstance(effective_groups, str):
        if effective_groups.strip().lower() == "all":
            allowed_groups = {"train", "val", "test_interp", "test_extrap", "fresh"}
        else:
            allowed_groups = {g.strip() for g in effective_groups.split(",") if g.strip()}
    else:
        allowed_groups = set(effective_groups)

    split_dict = load_viewpoint_split()
    if fixed_view_ids is not None:
        req_fixed_count = len(fixed_view_ids)
    else:
        fixed_vps = load_viewpoints(viewpoints_file)
        req_fixed_count = sum(
            1 for v in fixed_vps
            if get_group_for_view(v["id"], split_dict) in allowed_groups
        )
        custom_fixed_count = cfg.get("final_eval_fixed_views_count")
        if custom_fixed_count is not None:
            req_fixed_count = min(req_fixed_count, int(custom_fixed_count))

    if "fresh" not in allowed_groups:
        req_fresh_count = 0
    else:
        if fresh_view_ids is not None:
            req_fresh_count = len(fresh_view_ids)
        else:
            req_fresh_count = int(cfg.get("final_eval_fresh_views_count", 60))

    if summary_file.is_file():
        try:
            with open(summary_file, "r") as f:
                saved_summary = json.load(f)
            if (
                saved_summary.get("status") == "completed"
                and int(saved_summary.get("fixed_views_count", 0)) >= req_fixed_count
                and int(saved_summary.get("fresh_views_count", 0)) >= req_fresh_count
                and saved_summary.get("eval_groups") == sorted(list(allowed_groups))
            ):
                if verbose:
                    logger.info("Final evaluation already fully completed in %s. Skipping.", final_eval_dir)
                return saved_summary
        except Exception:
            pass

    if pause_paths is None:
        pause_paths = [
            str(EXP_DIR / "state" / "PAUSE"),
            str(EXP_DIR / "PAUSE"),
            str(final_eval_dir / "PAUSE"),
            str(final_eval_dir.parent / "PAUSE"),
        ]

    # Handle OS termination signals cleanly
    def _sig_handler(signum, frame):
        global EVAL_STOP_REQUESTED
        logger.info("[Final Eval Signal] Caught signal %s. Requesting pause after current view...", signum)
        EVAL_STOP_REQUESTED = True

    prev_sigint = signal.getsignal(signal.SIGINT)
    prev_sigterm = signal.getsignal(signal.SIGTERM)
    try:
        signal.signal(signal.SIGINT, _sig_handler)
        signal.signal(signal.SIGTERM, _sig_handler)
    except (ValueError, TypeError):
        pass  # In sub-threads or non-main environments

    try:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # Determine episodes per view from args or config
        cfg = config or {}
        req_fixed = fixed_episodes if fixed_episodes is not None else int(cfg.get("final_eval_fixed_episodes", 200))
        req_fresh = fresh_episodes if fresh_episodes is not None else int(cfg.get("final_eval_fresh_episodes", 100))
        eval_num_envs = num_envs if num_envs is not None else int(cfg.get("final_eval_num_envs", 160))

        if state_activation is None:
            state_activation = cfg.get("model", {}).get("state_activation", "none")

        meta_lookup = get_viewpoint_metadata_dict(
            viewpoints_file=viewpoints_file,
            fresh_views_file=fresh_views_file,
        )

        # 1. Fixed Viewpoints
        fixed_csv = final_eval_dir / "fixed_views_eval.csv"
        completed_fixed = load_completed_view_ids(fixed_csv, req_fixed)

        if fixed_view_ids is None:
            fixed_vps = load_viewpoints(viewpoints_file)
            target_fixed_ids = [
                v["id"] for v in fixed_vps
                if meta_lookup.get(f"fixed_{v['id']}", {}).get("group") in allowed_groups
            ]
        else:
            for vid in fixed_view_ids:
                grp = meta_lookup.get(f"fixed_{vid}", {}).get("group", "unknown")
                if grp not in allowed_groups:
                    raise ValueError(
                        f"Protocol violation: Viewpoint {vid} belongs to group '{grp}', "
                        f"which is not permitted in final_eval_groups {sorted(list(allowed_groups))}."
                    )
            target_fixed_ids = list(fixed_view_ids)

        # 2. Fresh Viewpoints
        fresh_csv = final_eval_dir / "fresh_views_eval.csv"
        completed_fresh = load_completed_view_ids(fresh_csv, req_fresh)

        fresh_path = fresh_views_file or str(EXP_DIR / "configs" / "fresh_views.json")
        if "fresh" not in allowed_groups:
            if fresh_view_ids is not None and len(fresh_view_ids) > 0:
                raise ValueError(
                    f"Protocol violation: Fresh viewpoints are not permitted in final_eval_groups {sorted(list(allowed_groups))}."
                )
            target_fresh_ids = []
        else:
            if fresh_view_ids is None:
                with open(fresh_path, "r") as f:
                    fresh_vps = json.load(f)
                target_fresh_ids = [v["id"] for v in fresh_vps]
                fresh_views_count_cfg = cfg.get("final_eval_fresh_views_count")
                if fresh_views_count_cfg is not None:
                    target_fresh_ids = target_fresh_ids[: int(fresh_views_count_cfg)]
            else:
                target_fresh_ids = list(fresh_view_ids)

        agent = load_agent_checkpoint(
            checkpoint_path=str(checkpoint_path),
            modality=modality,
            device=device,
            state_activation=state_activation,
        )

        if verbose:
            logger.info(
                "[Final Eval] Fixed viewpoints: %d total (groups: %s), %d already completed. Need %d eps/view.",
                len(target_fixed_ids),
                sorted(list(allowed_groups)),
                len(completed_fixed),
                req_fixed,
            )

        for vid in target_fixed_ids:
            if EVAL_STOP_REQUESTED or check_pause_files(pause_paths):
                logger.info("[Final Eval Pause] Pausing before fixed view %s.", vid)
                return {
                    "status": "paused",
                    "checkpoint": str(checkpoint_path),
                    "completed_fixed": len(completed_fixed),
                    "total_fixed": len(target_fixed_ids),
                    "completed_fresh": len(load_completed_view_ids(final_eval_dir / "fresh_views_eval.csv", req_fresh)),
                    "total_fresh": 0,
                    "output_dir": str(final_eval_dir),
                }

            if vid in completed_fixed:
                continue

            if verbose:
                logger.info("[Final Eval] Evaluating fixed view %s (%d eps)...", vid, req_fixed)

            t0_v = time.time()
            res = evaluate_single_viewpoint(
                agent=agent,
                viewpoint_id=vid,
                modality=modality,
                episodes=req_fixed,
                num_envs=eval_num_envs,
                base_seed=base_seed,
                viewpoints_file=viewpoints_file,
                device=device,
            )
            v_dur = time.time() - t0_v
            if verbose:
                logger.info("[Final Eval Bench] Fixed view %s evaluated in %.2fs (%.1f eps/s)", vid, v_dur, req_fixed / max(v_dur, 1e-6))
            meta = meta_lookup.get(f"fixed_{vid}", {})
            res["type"] = "fixed"
            res["group"] = meta.get("group", "fixed")
            res["azimuth_deg"] = meta.get("azimuth_deg", 0.0)
            res["elevation_deg"] = meta.get("elevation_deg", 0.0)
            res["nearest_train_dist_deg"] = meta.get("nearest_train_dist_deg", 0.0)

            append_evaluation_csv_row(res, fixed_csv)
            completed_fixed[vid] = res

        # 2. Fresh Viewpoints
        fresh_csv = final_eval_dir / "fresh_views_eval.csv"
        completed_fresh = load_completed_view_ids(fresh_csv, req_fresh)

        fresh_path = fresh_views_file or str(EXP_DIR / "configs" / "fresh_views.json")
        if "fresh" not in allowed_groups:
            if fresh_view_ids is not None and len(fresh_view_ids) > 0:
                raise ValueError(
                    f"Protocol violation: Fresh viewpoints are not permitted in final_eval_groups {sorted(list(allowed_groups))}."
                )
            target_fresh_ids = []
        else:
            if fresh_view_ids is None:
                with open(fresh_path, "r") as f:
                    fresh_vps = json.load(f)
                target_fresh_ids = [v["id"] for v in fresh_vps]
                fresh_views_count_cfg = cfg.get("final_eval_fresh_views_count")
                if fresh_views_count_cfg is not None:
                    target_fresh_ids = target_fresh_ids[: int(fresh_views_count_cfg)]
            else:
                target_fresh_ids = list(fresh_view_ids)

        if verbose and len(target_fresh_ids) > 0:
            logger.info(
                "[Final Eval] Fresh viewpoints: %d total, %d already completed. Need %d eps/view.",
                len(target_fresh_ids),
                len(completed_fresh),
                req_fresh,
            )

        for vid in target_fresh_ids:
            if EVAL_STOP_REQUESTED or check_pause_files(pause_paths):
                logger.info("[Final Eval Pause] Pausing before fresh view %s.", vid)
                return {
                    "status": "paused",
                    "checkpoint": str(checkpoint_path),
                    "completed_fixed": len(completed_fixed),
                    "total_fixed": len(target_fixed_ids),
                    "completed_fresh": len(completed_fresh),
                    "total_fresh": len(target_fresh_ids),
                    "output_dir": str(final_eval_dir),
                }

            if vid in completed_fresh:
                continue

            if verbose:
                logger.info("[Final Eval] Evaluating fresh view %s (%d eps)...", vid, req_fresh)

            t0_fv = time.time()
            res = evaluate_single_viewpoint(
                agent=agent,
                viewpoint_id=vid,
                modality=modality,
                episodes=req_fresh,
                num_envs=eval_num_envs,
                base_seed=base_seed,
                viewpoints_file=fresh_path,
                device=device,
            )
            fv_dur = time.time() - t0_fv
            if verbose:
                logger.info("[Final Eval Bench] Fresh view %s evaluated in %.2fs (%.1f eps/s)", vid, fv_dur, req_fresh / max(fv_dur, 1e-6))
            meta = meta_lookup.get(f"fresh_{vid}", {})
            res["type"] = "fresh"
            res["group"] = "fresh"
            res["azimuth_deg"] = meta.get("azimuth_deg", 0.0)
            res["elevation_deg"] = meta.get("elevation_deg", 0.0)
            res["nearest_train_dist_deg"] = meta.get("nearest_train_dist_deg", 0.0)

            append_evaluation_csv_row(res, fresh_csv)
            completed_fresh[vid] = res

        # 3. Aggregation & Summary
        all_fixed_results = [completed_fixed[vid] for vid in target_fixed_ids if vid in completed_fixed]
        all_fresh_results = [completed_fresh[vid] for vid in target_fresh_ids if vid in completed_fresh]
        all_combined = all_fixed_results + all_fresh_results

        # Write combined all_views_eval.csv
        all_csv_path = final_eval_dir / "all_views_eval.csv"
        write_evaluation_csv(all_combined, all_csv_path)

        # Compute group statistics
        group_stats: Dict[str, List[Dict[str, Any]]] = {}
        for r in all_combined:
            grp = r.get("group", "unknown")
            group_stats.setdefault(grp, []).append(r)

        groups_summary = {}
        for grp, grp_rows in sorted(group_stats.items()):
            succ_rates = [float(r["success_rate"]) for r in grp_rows]
            returns = [float(r["mean_return"]) for r in grp_rows]
            total_eps = sum(int(r["episodes"]) for r in grp_rows)
            groups_summary[grp] = {
                "views_count": len(grp_rows),
                "total_episodes": total_eps,
                "mean_success_rate": round(float(np.mean(succ_rates)), 2) if succ_rates else 0.0,
                "mean_return": round(float(np.mean(returns)), 4) if returns else 0.0,
            }

        fixed_succ = [float(r["success_rate"]) for r in all_fixed_results]
        fresh_succ = [float(r["success_rate"]) for r in all_fresh_results]
        overall_succ = [float(r["success_rate"]) for r in all_combined]

        summary_data = {
            "status": "completed",
            "completed_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "checkpoint_path": str(checkpoint_path),
            "modality": modality,
            "eval_groups": sorted(list(allowed_groups)),
            "fixed_views_count": len(all_fixed_results),
            "fresh_views_count": len(all_fresh_results),
            "fixed_episodes_per_view": req_fixed,
            "fresh_episodes_per_view": req_fresh,
            "mean_fixed_success_rate": round(float(np.mean(fixed_succ)), 2) if fixed_succ else 0.0,
            "mean_fresh_success_rate": round(float(np.mean(fresh_succ)), 2) if fresh_succ else None,
            "mean_overall_success_rate": round(float(np.mean(overall_succ)), 2) if overall_succ else 0.0,
            "groups": groups_summary,
            "output_dir": str(final_eval_dir),
        }

        tmp_json = str(summary_file) + f".tmp.{os.getpid()}"
        with open(tmp_json, "w") as f:
            json.dump(summary_data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_json, str(summary_file))

        if verbose:
            print_group_summary(all_combined)
            logger.info(
                "[Final Eval] Full final evaluation complete! Saved summary to %s",
                summary_file,
            )

        return summary_data

    finally:
        try:
            signal.signal(signal.SIGINT, prev_sigint)
            signal.signal(signal.SIGTERM, prev_sigterm)
        except (ValueError, TypeError):
            pass


def print_group_summary(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """
    Computes and prints group aggregated statistics across viewpoints.
    """
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in results:
        grp = r.get("group", "unknown")
        groups.setdefault(grp, []).append(r)

    print("\n" + "=" * 70)
    print(" GROUP SUMMARY AGGREGATES")
    print("=" * 70)
    print(f"{'Group':<15} | {'Views':<6} | {'Total Ep':<9} | {'Mean Success %':<16} | {'Mean Return':<12}")
    print("-" * 70)

    summary = {}
    for grp, grp_results in sorted(groups.items()):
        total_ep = sum(r["episodes"] for r in grp_results)
        total_succ = sum(r["success_count"] for r in grp_results)
        mean_succ = float(np.mean([r["success_rate"] for r in grp_results]))
        mean_ret = float(np.mean([r["mean_return"] for r in grp_results]))

        summary[grp] = {
            "views_count": len(grp_results),
            "total_episodes": total_ep,
            "mean_success_rate": round(mean_succ, 2),
            "mean_return": round(mean_ret, 2),
        }
        print(f"{grp:<15} | {len(grp_results):<6d} | {total_ep:<9d} | {mean_succ:14.2f}% | {mean_ret:10.2f}")

    # Overall aggregate
    if results:
        overall_succ = float(np.mean([r["success_rate"] for r in results]))
        overall_ret = float(np.mean([r["mean_return"] for r in results]))
        print("-" * 70)
        print(f"{'ALL VIEWS':<15} | {len(results):<6d} | {sum(r['episodes'] for r in results):<9d} | {overall_succ:14.2f}% | {overall_ret:10.2f}")
    print("=" * 70 + "\n")

    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate PPO Policy across Camera Viewpoints")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to policy checkpoint (.pt)")
    parser.add_argument("--modality", type=str, required=True, choices=["rgb", "rgbd"], help="Observation modality")
    parser.add_argument(
        "--views",
        type=str,
        default="fixed",
        help="Viewpoints to evaluate: 'fixed' (all 30), 'fresh' (all 60), 'all' (30 fixed + 60 fresh), "
             "or split group ('train', 'val', 'test_interp', 'test_extrap'), or comma-separated integer IDs",
    )
    parser.add_argument("--final-eval", action="store_true", help="Run full final evaluation (30 fixed + 60 fresh views)")
    parser.add_argument(
        "--eval-groups", "--eval_groups",
        type=str,
        default=None,
        help="Viewpoint groups to evaluate: 'all' or comma-separated list e.g. 'train,val'",
    )
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save final evaluation CSVs and summary")
    parser.add_argument("--episodes-per-view", type=int, default=None, help="Episodes per viewpoint (default: 200 for fixed, 100 for fresh)")
    parser.add_argument("--num-envs", type=int, default=160, help="Parallel environments for evaluation (default: 160)")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed for evaluation (default: 42)")
    parser.add_argument("--state-activation", type=str, default="none", choices=["none", "relu"], help="State feature activation (default: 'none')")
    parser.add_argument("--output-csv", type=str, default=None, help="Path to write output CSV results")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda/cpu)")
    parser.add_argument("--viewpoints-file", type=str, default=None, help="Custom viewpoints JSON file")
    parser.add_argument("--fresh-views-file", type=str, default=None, help="Custom fresh views JSON file")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if args.final_eval:
        out_d = args.output_dir or str(EXP_DIR / "runs" / "eval")
        res = run_final_evaluation(
            checkpoint_path=args.checkpoint,
            output_dir=out_d,
            modality=args.modality,
            viewpoints_file=args.viewpoints_file,
            fresh_views_file=args.fresh_views_file,
            fixed_episodes=args.episodes_per_view,
            fresh_episodes=args.episodes_per_view,
            num_envs=args.num_envs,
            base_seed=args.seed,
            device=args.device,
            state_activation=args.state_activation,
            eval_groups=args.eval_groups,
        )
        return

    meta_lookup = get_viewpoint_metadata_dict(
        viewpoints_file=args.viewpoints_file,
        fresh_views_file=args.fresh_views_file,
    )

    agent = load_agent_checkpoint(
        checkpoint_path=args.checkpoint,
        modality=args.modality,
        device=args.device,
        state_activation=args.state_activation,
    )

    all_results = []
    mode = args.views.lower().strip()

    # Determine default episodes
    fixed_episodes = args.episodes_per_view if args.episodes_per_view is not None else 200
    fresh_episodes = args.episodes_per_view if args.episodes_per_view is not None else 100

    if mode in ("fixed", "all"):
        fixed_vps = load_viewpoints(args.viewpoints_file)
        fixed_ids = [v["id"] for v in fixed_vps]
        res = evaluate_viewpoint_set(
            agent=agent,
            viewpoint_ids=fixed_ids,
            modality=args.modality,
            viewpoints_file=args.viewpoints_file,
            viewpoint_type="fixed",
            episodes_per_view=fixed_episodes,
            num_envs=args.num_envs,
            base_seed=args.seed,
            device=args.device,
            metadata_lookup=meta_lookup,
        )
        all_results.extend(res)

    if mode in ("fresh", "all"):
        fresh_path = args.fresh_views_file or str(EXP_DIR / "configs" / "fresh_views.json")
        with open(fresh_path, "r") as f:
            fresh_vps = json.load(f)
        fresh_ids = [v["id"] for v in fresh_vps]
        res = evaluate_viewpoint_set(
            agent=agent,
            viewpoint_ids=fresh_ids,
            modality=args.modality,
            viewpoints_file=fresh_path,
            viewpoint_type="fresh",
            episodes_per_view=fresh_episodes,
            num_envs=args.num_envs,
            base_seed=args.seed,
            device=args.device,
            metadata_lookup=meta_lookup,
        )
        all_results.extend(res)

    if mode in ("train", "val", "test_interp", "test_extrap"):
        split = load_viewpoint_split()
        split_ids = split[mode]
        res = evaluate_viewpoint_set(
            agent=agent,
            viewpoint_ids=split_ids,
            modality=args.modality,
            viewpoints_file=args.viewpoints_file,
            viewpoint_type="fixed",
            episodes_per_view=fixed_episodes,
            num_envs=args.num_envs,
            base_seed=args.seed,
            device=args.device,
            metadata_lookup=meta_lookup,
        )
        all_results.extend(res)

    # Specific comma-separated IDs
    if mode not in ("fixed", "fresh", "all", "train", "val", "test_interp", "test_extrap"):
        try:
            ids = [int(x.strip()) if x.strip().isdigit() else x.strip() for x in args.views.split(",") if x.strip()]
            res = evaluate_viewpoint_set(
                agent=agent,
                viewpoint_ids=ids,
                modality=args.modality,
                viewpoints_file=args.viewpoints_file,
                viewpoint_type="fixed",
                episodes_per_view=fixed_episodes,
                num_envs=args.num_envs,
                base_seed=args.seed,
                device=args.device,
                metadata_lookup=meta_lookup,
            )
            all_results.extend(res)
        except ValueError:
            raise ValueError(f"Could not parse --views: {args.views!r}")

    print_group_summary(all_results)

    if args.output_csv:
        write_evaluation_csv(all_results, args.output_csv)
        print(f"Results successfully saved to: {args.output_csv}")


if __name__ == "__main__":
    main()
