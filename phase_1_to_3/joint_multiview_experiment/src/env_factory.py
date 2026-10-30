"""
ManiSkill PickCube Environment Factory for Joint Multi-View Training & Evaluation.

Per SPEC Section 1, 3.2, and Section 6:
- Viewpoints: Loads from viewpoints.json, assigned to parallel subcameras.
- Split: Reads viewpoint split from configs/viewpoint_split.json (never hardcoded).
- Allocation: Even distribution of num_envs across the selected viewpoint set.
- Observation Modes: RGB (128x128x3) and RGB-D (128x128x4).
- Depth Preprocessing: Metric depth int16 (mm) -> invalid (<=0) to 3000 mm -> clamp(0..3m) -> normalized [0..1].

Python 3.9 compatible only.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import gymnasium as gym
import numpy as np
import sapien.core as sapien
import torch

import mani_skill.envs
from mani_skill.utils import sapien_utils
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper, FlattenRGBDObservationWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

logger = logging.getLogger(__name__)

DEFAULT_TARGET_POS = [-0.1, 0.0, 0.1]
EXP_DIR = Path(__file__).resolve().parent.parent


def find_viewpoints_file() -> str:
    """Finds canonical viewpoints.json file."""
    candidates = [
        str(EXP_DIR / "configs" / "viewpoints.json"),
        "/home/maniskill/phase_1_to_3/sequential_viewpoint_training/viewpoints.json",
        "/home/maniskill/phase_1_to_3/RGB/checkpoints/sequential_viewpoints_checkpoints/viewpoints.json",
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    raise FileNotFoundError("Could not locate viewpoints.json in any standard path.")


def find_viewpoint_split_file() -> str:
    """Finds configs/viewpoint_split.json."""
    candidates = [
        str(EXP_DIR / "configs" / "viewpoint_split.json"),
        os.path.join(os.getcwd(), "configs", "viewpoint_split.json"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    raise FileNotFoundError("Could not locate configs/viewpoint_split.json.")


def load_viewpoints(viewpoints_file: Optional[str] = None) -> List[Dict[str, Any]]:
    """Loads all 30 Fibonacci viewpoints from JSON."""
    if viewpoints_file is None:
        viewpoints_file = find_viewpoints_file()
    with open(viewpoints_file, "r") as f:
        vps = json.load(f)
    return vps


def load_viewpoint_split(split_file: Optional[str] = None) -> Dict[str, List[int]]:
    """
    Loads train/val/test viewpoint split dictionary from configs/viewpoint_split.json.
    Ensures all IDs are integers and splits are disjoint.
    """
    if split_file is None:
        split_file = find_viewpoint_split_file()

    with open(split_file, "r") as f:
        split_data = json.load(f)

    required_keys = ["train", "val", "test_interp", "test_extrap"]
    for k in required_keys:
        if k not in split_data:
            split_data[k] = []
        else:
            split_data[k] = [int(x) for x in split_data[k]]

    # Verify disjointness
    all_sets = [set(split_data[k]) for k in required_keys]
    total_count = sum(len(s) for s in all_sets)
    union_count = len(set.union(*all_sets))
    if total_count != union_count:
        raise ValueError(f"Viewpoint sets in {split_file} are not disjoint! Total: {total_count}, Unique: {union_count}")

    return split_data


def build_viewpoint_poses(
    viewpoint_ids: List[int],
    num_envs: int,
    viewpoints: Optional[List[Dict[str, Any]]] = None,
    target: List[float] = DEFAULT_TARGET_POS,
) -> Tuple[Pose, List[int]]:
    """
    Constructs a batched ManiSkill Pose and assigned viewpoint ID list
    for num_envs evenly partitioned over viewpoint_ids.
    
    Args:
        viewpoint_ids: List of viewpoint integer IDs to assign.
        num_envs: Total number of parallel simulation environments.
        viewpoints: Full list of viewpoint metadata dictionaries.
        target: 3D target coordinates for camera look_at.
        
    Returns:
        (batched_pose, assigned_vps) where batched_pose has shape (num_envs, 7)
        and assigned_vps has length num_envs.
    """
    if viewpoints is None:
        viewpoints = load_viewpoints()

    vp_by_id = {v["id"]: v for v in viewpoints}
    num_views = len(viewpoint_ids)

    if num_envs % num_views != 0:
        raise ValueError(
            f"num_envs ({num_envs}) must be divisible by number of viewpoints ({num_views}) "
            f"for an even distribution (SPEC Section 3.2)."
        )

    envs_per_view = num_envs // num_views
    assigned_vps = []
    poses = []

    for i in range(num_envs):
        vp_id = viewpoint_ids[i // envs_per_view]
        assigned_vps.append(vp_id)

        if str(vp_id).lower() in ("train", "training", "-1") or vp_id == -1:
            vp = {
                "id": "train",
                "world_eye": [-0.615, 0.915, 0.600],
                "azimuth_deg": 0.0,
                "elevation_deg": 0.0,
            }
        elif vp_id not in vp_by_id:
            raise KeyError(f"Viewpoint ID {vp_id} not found in viewpoints.json")
        else:
            vp = vp_by_id[vp_id]

        p = sapien_utils.look_at(eye=vp["world_eye"], target=target)
        poses.append(p.raw_pose)

    batched_raw = torch.cat(poses, dim=0)
    batched_pose = Pose.create(batched_raw)
    return batched_pose, assigned_vps


def assign_viewpoints_to_envs(
    env: gym.Env,
    viewpoint_ids: List[int],
    viewpoints: Optional[List[Dict[str, Any]]] = None,
    target: List[float] = DEFAULT_TARGET_POS,
) -> List[int]:
    """
    Evenly distributes viewpoint_ids across parallel sub-cameras in the environment.
    Sets sub-camera local poses and invalidates cached poses/extrinsics.
    
    Args:
        env: ManiSkill environment (or wrapper) with base_camera sensor.
        viewpoint_ids: List of viewpoint integer IDs to assign.
        viewpoints: Full list of viewpoint dictionaries (loaded if None).
        target: 3D target coordinates for camera look_at.
        
    Returns:
        List of assigned viewpoint IDs for each environment index (length num_envs).
    """
    if viewpoints is None:
        viewpoints = load_viewpoints()

    vp_by_id = {v["id"]: v for v in viewpoints}

    cam = env.unwrapped._sensors["base_camera"]
    subcams = cam.camera._render_cameras
    num_envs = len(subcams)
    num_views = len(viewpoint_ids)

    if num_envs % num_views != 0:
        raise ValueError(
            f"num_envs ({num_envs}) must be divisible by number of viewpoints ({num_views}) "
            f"for an even distribution (SPEC Section 3.2)."
        )

    envs_per_view = num_envs // num_views
    assigned_vps = []

    for i, rcam in enumerate(subcams):
        vp_id = viewpoint_ids[i // envs_per_view]
        assigned_vps.append(vp_id)

        if str(vp_id).lower() in ("train", "training", "-1") or vp_id == -1:
            vp = {
                "id": "train",
                "world_eye": [-0.615, 0.915, 0.600],
                "azimuth_deg": 0.0,
                "elevation_deg": 0.0,
            }
        elif vp_id not in vp_by_id:
            raise KeyError(f"Viewpoint ID {vp_id} not found in viewpoints.json")
        else:
            vp = vp_by_id[vp_id]

        pose = sapien_utils.look_at(eye=vp["world_eye"], target=target)
        sp = sapien.Pose(pose.p[0].cpu().numpy(), pose.q[0].cpu().numpy())
        rcam.set_local_pose(sp)

    # Invalidate cached pose and extrinsic matrices to force recalculation
    cam.camera._cached_local_pose = None
    cam.camera._cached_extrinsic_matrix = None
    return assigned_vps


def get_simulator_camera_poses(env: gym.Env) -> Dict[str, Any]:
    """
    Reads back the actual camera poses and extrinsics from the simulator for each env.
    
    Returns:
        Dict containing:
        - "local_p": Tensor of shape (num_envs, 3)
        - "local_q": Tensor of shape (num_envs, 4)
        - "extrinsics": Tensor of shape (num_envs, 3, 4)
        - "rcam_poses": List of sapien.Pose for each render camera component
    """
    cam = env.unwrapped._sensors["base_camera"]
    cam.camera._cached_local_pose = None
    cam.camera._cached_extrinsic_matrix = None

    local_pose = cam.camera.get_local_pose()
    extrinsics = cam.camera.get_extrinsic_matrix()
    rcam_poses = [rcam.local_pose for rcam in cam.camera._render_cameras]

    return {
        "local_p": local_pose.p.detach().clone(),
        "local_q": local_pose.q.detach().clone(),
        "extrinsics": extrinsics.detach().clone(),
        "rcam_poses": rcam_poses,
    }


def verify_simulator_camera_poses(
    env: gym.Env,
    expected_viewpoint_ids: List[int],
    viewpoints: Optional[List[Dict[str, Any]]] = None,
    target: List[float] = DEFAULT_TARGET_POS,
    atol: float = 1e-4,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Verifies that camera extrinsics and poses read back from the simulator match
    viewpoints.json within numerical tolerance, and that different views differ.
    
    Args:
        env: ManiSkill environment.
        expected_viewpoint_ids: List of viewpoint IDs assigned to each environment.
        viewpoints: Full list of viewpoints metadata (loaded if None).
        target: 3D target coordinates for look_at.
        atol: Numerical absolute tolerance for floating point assertions.
        verbose: If True, prints formatted comparison table.
        
    Returns:
        Dict with max_p_err, max_q_err, and verification details.
    """
    if viewpoints is None:
        viewpoints = load_viewpoints()
    vp_by_id = {v["id"]: v for v in viewpoints}

    sim_data = get_simulator_camera_poses(env)
    sim_p = sim_data["local_p"].cpu().numpy()
    sim_q = sim_data["local_q"].cpu().numpy()
    extrinsics = sim_data["extrinsics"].cpu().numpy()

    num_envs = len(expected_viewpoint_ids)
    max_p_err = 0.0
    max_q_err = 0.0
    records = []

    if verbose:
        print("\n" + "=" * 90)
        print(f"{'Env':<4} | {'VP':<4} | {'Target Eye (viewpoints.json)':<30} | {'Simulator Pose (p)':<30} | {'p_err':<9} | {'q_err':<9}")
        print("-" * 90)

    for i in range(num_envs):
        vp_id = expected_viewpoint_ids[i]
        vp = vp_by_id[vp_id]
        expected_lookat = sapien_utils.look_at(eye=vp["world_eye"], target=target)
        exp_p = expected_lookat.p[0].cpu().numpy()
        exp_q = expected_lookat.q[0].cpu().numpy()

        p_err = float(np.max(np.abs(sim_p[i] - exp_p)))
        # Account for quaternion antipodal symmetry (q and -q represent same rotation)
        q_err = float(min(np.max(np.abs(sim_q[i] - exp_q)), np.max(np.abs(sim_q[i] + exp_q))))

        max_p_err = max(max_p_err, p_err)
        max_q_err = max(max_q_err, q_err)

        target_eye_str = f"[{vp['world_eye'][0]:.4f}, {vp['world_eye'][1]:.4f}, {vp['world_eye'][2]:.4f}]"
        sim_p_str = f"[{sim_p[i, 0]:.4f}, {sim_p[i, 1]:.4f}, {sim_p[i, 2]:.4f}]"

        if verbose:
            print(f"{i:<4} | {vp_id:02d}   | {target_eye_str:<30} | {sim_p_str:<30} | {p_err:<9.2e} | {q_err:<9.2e}")

        assert p_err <= atol, (
            f"Env {i} camera position mismatch with VP {vp_id}: "
            f"sim={sim_p[i]}, expected={exp_p}, err={p_err} > atol={atol}"
        )
        assert q_err <= atol, (
            f"Env {i} camera quaternion mismatch with VP {vp_id}: "
            f"sim={sim_q[i]}, expected={exp_q}, err={q_err} > atol={atol}"
        )

        records.append({
            "env_idx": i,
            "vp_id": vp_id,
            "sim_p": sim_p[i],
            "sim_q": sim_q[i],
            "extrinsic": extrinsics[i],
            "p_err": p_err,
            "q_err": q_err,
        })

    if verbose:
        print("=" * 90)
        print(f"Pose verification passed: max position error = {max_p_err:.2e}, max quaternion error = {max_q_err:.2e}\n")

    # Assert that distinct viewpoints have distinct camera positions
    for i in range(num_envs):
        for j in range(i + 1, num_envs):
            if expected_viewpoint_ids[i] != expected_viewpoint_ids[j]:
                dist_p = np.linalg.norm(sim_p[i] - sim_p[j])
                assert dist_p > 1e-3, (
                    f"Envs {i} and {j} have different viewpoints ({expected_viewpoint_ids[i]} vs {expected_viewpoint_ids[j]}) "
                    f"but identical camera positions in simulator! dist={dist_p}"
                )

    return {
        "max_p_err": max_p_err,
        "max_q_err": max_q_err,
        "records": records,
    }


class JointMultiViewVectorWrapper(gym.Wrapper):
    """
    Vector observation wrapper for Joint Multi-View PickCube environments.
    
    Applies standard depth preprocessing:
    - Replaces invalid depth (<= 0) with 3000.0 mm
    - Clamps metric depth to [0.0, 3.0] meters
    - Normalizes depth to [0.0, 1.0]
    
    Outputs standardized observation keys:
    - "rgb": (B, 128, 128, 3)
    - "depth": (B, 128, 128, 1) normalized [0..1] (for RGB-D)
    - "rgbd": (B, 128, 128, 4) normalized [0..1] (for RGB-D)
    - "state": (B, 29)
    """

    def __init__(
        self,
        env: gym.Env,
        modality: str,
        assigned_viewpoint_ids: Optional[List[int]] = None,
        viewpoint_metadata: Optional[Dict[int, Dict[str, Any]]] = None,
    ):
        super().__init__(env)
        if not modality or str(modality).lower().strip() not in ("rgb", "rgbd"):
            raise ValueError(f"Invalid or missing modality: {modality!r}. Must be 'rgb' or 'rgbd'.")
        self.modality = str(modality).lower().strip()
        self.assigned_viewpoint_ids = assigned_viewpoint_ids or []
        self.viewpoint_metadata = viewpoint_metadata or {}
        self.look_at_target = list(DEFAULT_TARGET_POS)
        self.camera_config = {}
        self.effective_settings = {}
        self._last_obs = None

    def _process_obs(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        processed = dict(obs)

        if self.modality == "rgbd" and "depth" in processed:
            depth = processed["depth"].float()
            # Replace invalid depth pixels (<= 0) with 3.0 m (3000 mm)
            depth = torch.where(depth <= 0, torch.full_like(depth, 3000.0), depth)
            depth_m = torch.clamp(depth / 1000.0, 0.0, 3.0)
            depth_norm = depth_m / 3.0  # (B, H, W, 1) in [0, 1]
            processed["depth"] = depth_norm

            if "rgb" in processed:
                rgb_norm = processed["rgb"].float() / 255.0
                processed["rgbd"] = torch.cat([rgb_norm, depth_norm], dim=-1)

        return processed

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        proc_obs = self._process_obs(obs)
        self._last_obs = proc_obs
        return proc_obs, info

    def step(self, action):
        obs, rew, term, trunc, info = self.env.step(action)
        proc_obs = self._process_obs(obs)
        self._last_obs = proc_obs
        if isinstance(info, dict) and "final_observation" in info and isinstance(info["final_observation"], dict):
            # Process final_observation if present without modifying in-place
            info = dict(info)
            info["final_observation"] = self._process_obs(info["final_observation"])
        return proc_obs, rew, term, trunc, info


def extract_effective_env_settings(env: gym.Env) -> Dict[str, Any]:
    """
    Reads effective runtime values from a created environment.
    Inspects wrapper hierarchy down to unwrapped PickCube and sensors.
    """
    unwrapped = env.unwrapped

    # Find vector_env wrapper
    cur_vec = getattr(env, "env", None)
    ignore_terminations = getattr(cur_vec, "ignore_terminations", False)
    record_metrics = getattr(cur_vec, "record_metrics", True)

    # Search wrapper chain for max_episode_steps
    cur = getattr(cur_vec, "_env", getattr(env, "env", None))
    max_steps = None
    while cur is not None:
        if hasattr(cur, "_max_episode_steps") and cur._max_episode_steps is not None:
            max_steps = int(cur._max_episode_steps)
            break
        cur = getattr(cur, "env", None)

    if max_steps is None:
        try:
            from mani_skill.utils.registration import REGISTERED_ENVS
            env_id = getattr(unwrapped, "env_id", "PickCube-v1")
            if env_id in REGISTERED_ENVS:
                max_steps = int(REGISTERED_ENVS[env_id].max_episode_steps)
        except Exception:
            pass

    cam = unwrapped._sensors["base_camera"]
    camera_settings = {
        "width": int(cam.camera.width),
        "height": int(cam.camera.height),
        "fov": float(cam.camera.fovx),
        "near": float(cam.camera.near),
        "far": float(cam.camera.far),
        "look_at": [float(x) for x in getattr(env, "look_at_target", DEFAULT_TARGET_POS)],
    }

    # Simulation backend
    backend_obj = getattr(unwrapped, "backend", None)
    sim_backend_val = getattr(backend_obj, "sim_backend", None)
    if sim_backend_val is None:
        sim_backend_val = getattr(unwrapped, "sim_backend", "physx_cuda")

    return {
        "env_id": getattr(unwrapped, "env_id", "PickCube-v1"),
        "reward_mode": str(unwrapped.reward_mode),
        "max_episode_steps": max_steps,
        "control_mode": str(unwrapped.control_mode),
        "sim_backend": str(sim_backend_val),
        "ignore_terminations": bool(ignore_terminations),
        "record_metrics": bool(record_metrics),
        "obs_mode": str(unwrapped.obs_mode),
        "camera": camera_settings,
    }


def validate_env_against_config(env: gym.Env, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extracts effective settings from created env and asserts exact agreement with config.
    Fails fast (raises ValueError) if the live environment disagrees with config.
    Updates config with effective settings in-place and returns effective settings dictionary.
    """
    effective = extract_effective_env_settings(env)
    mismatches = []

    # Check top-level settings if present in config
    for key in ["reward_mode", "max_episode_steps", "control_mode", "sim_backend", "ignore_terminations", "record_metrics"]:
        if key in config and config[key] is not None:
            if effective[key] != config[key]:
                mismatches.append(
                    f"Setting '{key}': config specifies {config[key]!r}, but live env has {effective[key]!r}"
                )

    # Check modality and live env's obs_mode + observation channel count
    cfg_mod = config.get("modality")
    if cfg_mod is None:
        mismatches.append("Missing required 'modality' in configuration ('rgb' or 'rgbd')")
    else:
        cfg_mod = str(cfg_mod).lower().strip()
        if cfg_mod not in ("rgb", "rgbd"):
            mismatches.append(f"Invalid modality in config: {cfg_mod!r}. Expected 'rgb' or 'rgbd'")
        else:
            # obs_mode is derived from modality: "rgbd" -> "rgbd", "rgb" -> "rgb"
            expected_obs_mode = "rgbd" if cfg_mod == "rgbd" else "rgb"
            live_obs_mode = str(effective.get("obs_mode", "")).lower()
            if live_obs_mode != expected_obs_mode:
                mismatches.append(
                    f"Live env obs_mode ({live_obs_mode!r}) does not match expected obs_mode "
                    f"({expected_obs_mode!r}) for modality {cfg_mod!r}"
                )

            # Compare channel count of a real observation (3 for rgb, 4 for rgbd)
            expected_channels = 4 if cfg_mod == "rgbd" else 3
            real_obs = getattr(env, "_last_obs", None)
            if real_obs is None:
                try:
                    real_obs, _ = env.reset(seed=0)
                except Exception as e:
                    logger.warning("Could not reset env to extract real observation for channel check: %s", e)

            if real_obs is not None:
                obs_key = "rgbd" if cfg_mod == "rgbd" else "rgb"
                if obs_key not in real_obs:
                    mismatches.append(
                        f"Real observation missing expected key '{obs_key}' for modality '{cfg_mod}'. "
                        f"Keys present: {list(real_obs.keys())}"
                    )
                else:
                    actual_channels = real_obs[obs_key].shape[-1]
                    if actual_channels != expected_channels:
                        mismatches.append(
                            f"Real observation channel count mismatch for modality '{cfg_mod}': "
                            f"expected {expected_channels} channels, got {actual_channels} "
                            f"({obs_key} shape: {real_obs[obs_key].shape})"
                        )

    # Check camera settings
    cfg_cam = config.get("camera", {})
    if isinstance(cfg_cam, dict) and cfg_cam:
        for cam_key in ["width", "height"]:
            if cam_key in cfg_cam and cfg_cam[cam_key] is not None:
                if effective["camera"][cam_key] != int(cfg_cam[cam_key]):
                    mismatches.append(
                        f"Camera '{cam_key}': config specifies {cfg_cam[cam_key]!r}, but live env has {effective['camera'][cam_key]!r}"
                    )
        for cam_key in ["fov", "near", "far"]:
            if cam_key in cfg_cam and cfg_cam[cam_key] is not None:
                if abs(float(effective["camera"][cam_key]) - float(cfg_cam[cam_key])) > 1e-4:
                    mismatches.append(
                        f"Camera '{cam_key}': config specifies {cfg_cam[cam_key]!r}, but live env has {effective['camera'][cam_key]!r}"
                    )
        if "look_at" in cfg_cam and cfg_cam["look_at"] is not None:
            cfg_la = [float(x) for x in cfg_cam["look_at"]]
            eff_la = [float(x) for x in effective["camera"]["look_at"]]
            if not np.allclose(cfg_la, eff_la, atol=1e-4):
                mismatches.append(
                    f"Camera 'look_at': config specifies {cfg_la!r}, but live env has {eff_la!r}"
                )
    elif "target_pos" in config and config["target_pos"] is not None:
        cfg_la = [float(x) for x in config["target_pos"]]
        eff_la = [float(x) for x in effective["camera"]["look_at"]]
        if not np.allclose(cfg_la, eff_la, atol=1e-4):
            mismatches.append(
                f"Camera 'look_at' (via target_pos): config specifies {cfg_la!r}, but live env has {eff_la!r}"
            )

    if mismatches:
        err_msg = "Live environment settings disagree with frozen configuration:\n" + "\n".join(f"  - {m}" for m in mismatches)
        logger.error(err_msg)
        raise ValueError(err_msg)

    # Write effective values into config
    for k, v in effective.items():
        if k == "camera":
            if "camera" not in config or not isinstance(config["camera"], dict):
                config["camera"] = {}
            for ck, cv in v.items():
                config["camera"][ck] = cv
        else:
            config[k] = v

    return effective


def make_joint_multiview_env(
    num_envs: int = 160,
    modality: Optional[str] = None,
    split_group: str = "train",
    viewpoint_ids: Optional[List[int]] = None,
    viewpoints_file: Optional[str] = None,
    split_file: Optional[str] = None,
    target_pos: Optional[List[float]] = None,
    camera_cfg: Optional[Dict[str, Any]] = None,
    reward_mode: str = "normalized_dense",
    max_episode_steps: int = 50,
    control_mode: str = "pd_ee_delta_pos",
    sim_backend: str = "physx_cuda",
    ignore_terminations: bool = False,
    record_metrics: bool = True,
    seed: Optional[int] = None,
    config: Optional[Dict[str, Any]] = None,
) -> JointMultiViewVectorWrapper:
    """
    Creates a vectorized PickCube-v1 environment with camera viewpoints distributed
    evenly across the parallel instances, validated against configuration.
    
    Args:
        num_envs: Number of parallel simulation environments.
        modality: Observation mode ('rgb' or 'rgbd').
        split_group: Name of the viewpoint split set ('train', 'val', 'test_interp', 'test_extrap').
        viewpoint_ids: Optional explicit list of viewpoint IDs. If None, loaded from split_group.
        viewpoints_file: Path to viewpoints.json.
        split_file: Path to configs/viewpoint_split.json.
        target_pos: Deprecated alias for camera look_at target coordinates.
        camera_cfg: Dict specifying camera width, height, fov, near, far, look_at.
        reward_mode: Reward mode ('normalized_dense').
        max_episode_steps: Maximum episode steps (50).
        control_mode: Control mode (default 'pd_ee_delta_pos').
        sim_backend: Simulation backend (default 'physx_cuda').
        ignore_terminations: Whether to ignore terminal states in vector wrapper (False).
        record_metrics: Whether to accumulate episode metrics (True).
        seed: Optional initialization seed.
        config: Full configuration dictionary for end-to-end setting injection & validation.
        
    Returns:
        Vectorized environment wrapped in JointMultiViewVectorWrapper with verified effective settings.
    """
    # 1. Resolve modality with single source of truth
    if modality is None:
        if config is not None and "modality" in config and config["modality"] is not None:
            modality = config["modality"]
        else:
            raise ValueError("Missing required parameter: 'modality'. Modality must be specified ('rgb' or 'rgbd').")

    modality = str(modality).lower().strip()
    if modality not in ("rgb", "rgbd"):
        raise ValueError(f"Invalid modality: {modality!r}. Expected 'rgb' or 'rgbd'.")

    # 2. Unpack other settings from config dict if provided (config does NOT override explicit modality)
    if config is not None:
        num_envs = int(config.get("num_envs", num_envs))
        split_group = str(config.get("split_group", split_group))
        if "viewpoint_ids" in config and config["viewpoint_ids"] is not None:
            viewpoint_ids = list(config["viewpoint_ids"])
        reward_mode = str(config.get("reward_mode", reward_mode))
        max_episode_steps = int(config.get("max_episode_steps", max_episode_steps))
        control_mode = str(config.get("control_mode", control_mode))
        sim_backend = str(config.get("sim_backend", sim_backend))
        ignore_terminations = bool(config.get("ignore_terminations", ignore_terminations))
        record_metrics = bool(config.get("record_metrics", record_metrics))
        if "camera" in config and isinstance(config["camera"], dict):
            camera_cfg = dict(config["camera"])
        elif "target_pos" in config and config["target_pos"] is not None:
            target_pos = config["target_pos"]

    # Resolve camera specifications
    cam_width = 128
    cam_height = 128
    cam_fov = float(np.pi / 2)
    cam_near = 0.01
    cam_far = 100.0
    cam_look_at = list(DEFAULT_TARGET_POS)

    if camera_cfg is not None:
        cam_width = int(camera_cfg.get("width", cam_width))
        cam_height = int(camera_cfg.get("height", cam_height))
        cam_fov = float(camera_cfg.get("fov", cam_fov))
        cam_near = float(camera_cfg.get("near", cam_near))
        cam_far = float(camera_cfg.get("far", cam_far))
        if "look_at" in camera_cfg and camera_cfg["look_at"] is not None:
            cam_look_at = [float(x) for x in camera_cfg["look_at"]]
    elif target_pos is not None:
        cam_look_at = [float(x) for x in target_pos]

    # Load viewpoints metadata
    all_vps = load_viewpoints(viewpoints_file)
    vp_by_id = {v["id"]: v for v in all_vps}

    # Determine viewpoint IDs
    if viewpoint_ids is None:
        split = load_viewpoint_split(split_file)
        if split_group not in split:
            raise KeyError(f"Split group '{split_group}' not found in viewpoint split file. Available: {list(split.keys())}")
        viewpoint_ids = split[split_group]

    obs_mode = "rgbd" if modality == "rgbd" else "rgb"

    # Construct batched camera poses for GPU batch renderer
    batched_pose, assigned_vps = build_viewpoint_poses(
        viewpoint_ids=viewpoint_ids,
        num_envs=num_envs,
        viewpoints=all_vps,
        target=cam_look_at,
    )

    # In ManiSkill / SAPIEN 3.0 GPU batch renderer (physx_cuda), camera poses
    # must be provided at gym.make time via sensor_configs so the CUDA pose buffer
    # and camera group are initialized with distinct viewpoint poses.
    camera_sensor_configs = {
        "base_camera": {
            "pose": batched_pose,
            "width": cam_width,
            "height": cam_height,
            "fov": cam_fov,
            "near": cam_near,
            "far": cam_far,
        }
    }

    # Create raw vectorized ManiSkill PickCube environment
    raw_env = gym.make(
        "PickCube-v1",
        num_envs=num_envs,
        obs_mode=obs_mode,
        control_mode=control_mode,
        sim_backend=sim_backend,
        reward_mode=reward_mode,
        max_episode_steps=max_episode_steps,
        sensor_configs=camera_sensor_configs,
    )

    # Sync sub-camera local poses and invalidate cached pose/extrinsic buffers
    assign_viewpoints_to_envs(
        env=raw_env,
        viewpoint_ids=viewpoint_ids,
        viewpoints=all_vps,
        target=cam_look_at,
    )

    # Apply standard ManiSkill wrappers
    wrapped_env = FlattenRGBDObservationWrapper(
        raw_env,
        rgb=True,
        depth=(modality == "rgbd"),
        state=True,
        sep_depth=(modality == "rgbd"),
    )

    if isinstance(wrapped_env.action_space, gym.spaces.Dict):
        wrapped_env = FlattenActionSpaceWrapper(wrapped_env)

    vector_env = ManiSkillVectorEnv(
        wrapped_env,
        num_envs=num_envs,
        ignore_terminations=ignore_terminations,
        record_metrics=record_metrics,
    )

    # Wrap in observation normalization and metadata tracking wrapper
    env = JointMultiViewVectorWrapper(
        env=vector_env,
        modality=modality,
        assigned_viewpoint_ids=assigned_vps,
        viewpoint_metadata=vp_by_id,
    )
    env.look_at_target = cam_look_at
    env.camera_config = {
        "width": cam_width,
        "height": cam_height,
        "fov": cam_fov,
        "near": cam_near,
        "far": cam_far,
        "look_at": cam_look_at,
    }

    # Validate live environment settings against requested configuration
    expected_cfg = config.copy() if config is not None else {}
    expected_cfg["modality"] = modality
    expected_cfg.setdefault("reward_mode", reward_mode)
    expected_cfg.setdefault("max_episode_steps", max_episode_steps)
    expected_cfg.setdefault("control_mode", control_mode)
    expected_cfg.setdefault("sim_backend", sim_backend)
    expected_cfg.setdefault("ignore_terminations", ignore_terminations)
    expected_cfg.setdefault("record_metrics", record_metrics)
    expected_cfg.setdefault("camera", env.camera_config)

    effective = validate_env_against_config(env, expected_cfg)
    env.effective_settings = effective

    if seed is not None:
        env.reset(seed=seed)

    return env


def generate_smoke_camera_grid(
    output_path: str = "results/smoke/smoke_camera_grid.png",
    num_envs: int = 8,
    num_views: int = 4,
    modality: str = "rgbd",
) -> str:
    """
    Smoke test utility:
    1. Builds an 8-environment PickCube env over 4 distinct viewpoints (2 envs per view).
    2. Verifies simulator camera extrinsics and poses post-creation against viewpoints.json.
    3. Resets environment and verifies simulator camera extrinsics and poses post-reset.
    4. Steps simulation frames.
    5. Extracts images directly from the observation sensor camera (obs sensor_data rgb, never human render camera).
    6. Annotates each tile with Viewpoint ID and Azimuth / Elevation.
    7. Stitches images into an annotated 2x4 visual grid proving distinct camera viewpoints.
    8. Saves to output_path and returns the path.
    """
    output_dir = os.path.dirname(output_path) or "."
    os.makedirs(output_dir, exist_ok=True)

    split = load_viewpoint_split()
    train_views = split["train"][:num_views]

    env = make_joint_multiview_env(
        num_envs=num_envs,
        modality=modality,
        viewpoint_ids=train_views,
        seed=42,
    )

    assigned_ids = env.assigned_viewpoint_ids
    vp_metadata = env.viewpoint_metadata

    print("\n--- [Smoke Grid] Verifying Camera Poses Post-Creation ---")
    verify_simulator_camera_poses(env, expected_viewpoint_ids=assigned_ids, verbose=True)

    obs, _ = env.reset(seed=42)

    print("--- [Smoke Grid] Verifying Camera Poses Post-Reset ---")
    verify_simulator_camera_poses(env, expected_viewpoint_ids=assigned_ids, verbose=True)

    # Step simulation frames with small delta actions to demonstrate dynamic simulation
    action = torch.zeros((num_envs, 4), device="cuda")
    for _ in range(5):
        obs, _, _, _, _ = env.step(action)

    # Extract RGB images directly from observation sensor camera (obs["rgb"]), NEVER from human render camera
    rgb_imgs = obs["rgb"].cpu().numpy()  # (8, 128, 128, 3) uint8

    tiles = []
    for i in range(num_envs):
        img_bgr = cv2.cvtColor(rgb_imgs[i], cv2.COLOR_RGB2BGR)
        # Upscale tile for clear display
        tile = cv2.resize(img_bgr, (256, 256), interpolation=cv2.INTER_NEAREST)

        vp_id = assigned_ids[i]
        meta = vp_metadata.get(vp_id, {})
        az = meta.get("azimuth_deg", 0.0)
        el = meta.get("elevation_deg", 0.0)

        # Annotate text
        text1 = f"Env {i} | VP {vp_id:02d}"
        text2 = f"Az: {az:+.1f} deg | El: {el:.1f} deg"
        cv2.putText(tile, text1, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3)
        cv2.putText(tile, text1, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 1)
        cv2.putText(tile, text2, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 0), 3)
        cv2.putText(tile, text2, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1)

        tiles.append(tile)

    # Stitch 2 rows x 4 columns
    row1 = np.hstack(tiles[:4])
    row2 = np.hstack(tiles[4:])
    grid = np.vstack([row1, row2])

    cv2.imwrite(output_path, grid)
    env.close()

    # Also copy to artifact directory if it exists
    artifact_dir = "/root/.gemini/antigravity-cli/brain/98f14b05-c8fc-47d3-aafc-73f463b9e341"
    if os.path.isdir(artifact_dir):
        import shutil
        shutil.copy2(output_path, os.path.join(artifact_dir, "smoke_camera_grid.png"))

    logger.info("Saved smoke camera grid (%dx%d) to: %s", grid.shape[1], grid.shape[0], output_path)
    return os.path.abspath(output_path)


if __name__ == "__main__":
    out_file = generate_smoke_camera_grid()
    print(f"Smoke test camera grid successfully generated at: {out_file}")
