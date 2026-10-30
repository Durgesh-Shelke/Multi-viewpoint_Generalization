"""
Teacher Rollout Collector for Multi-Teacher Policy Distillation.
Rolls out converged viewpoint specialists on their native environments to generate
high-performance multi-viewpoint demonstration datasets for student policy distillation.
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, "/home/maniskill")

import mani_skill.envs
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper, FlattenRGBDObservationWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from models import UnifiedAblationAgent

VIEWPOINTS_JSON = "/home/maniskill/phase_1_to_3/sequential_viewpoint_training/viewpoints.json"
TEACHER_CKPT_DIR = "/home/maniskill/phase_6_viewpoint_specialists/checkpoints"
DEFAULT_DATA_DIR = os.path.join(SCRIPT_DIR, "data")


def collect_single_teacher_rollouts(
    vp_id: int,
    num_steps: int = 10000,
    num_envs: int = 64,
    device: str = "cuda",
    output_dir: str = DEFAULT_DATA_DIR,
):
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, f"vp_{vp_id:02d}_rollouts.pt")
    if os.path.exists(out_file):
        print(f"[Collector] Rollouts for VP {vp_id:02d} already exist at {out_file}. Skipping.")
        return out_file

    teacher_ckpt = os.path.join(TEACHER_CKPT_DIR, f"specialist_vp_{vp_id:02d}_best.pt")
    if not os.path.exists(teacher_ckpt):
        raise FileNotFoundError(f"Teacher checkpoint not found: {teacher_ckpt}")

    # Load viewpoint coords
    with open(VIEWPOINTS_JSON) as f:
        all_vps = json.load(f)
    vp_dict = {vp["id"]: vp for vp in all_vps}
    vp = vp_dict[vp_id]
    cam_eye = vp["world_eye"]
    cam_target = [-0.1, 0.0, 0.1]

    print(f"\n[Collector] Collecting {num_steps:,} transitions for VP {vp_id:02d} (Az: {vp['azimuth_deg']:.1f}°, El: {vp['elevation_deg']:.1f}°)...")

    env_kwargs = dict(
        obs_mode="rgbd",
        render_mode="all",
        sim_backend="physx_cuda",
        sensor_cam_eye_pos=cam_eye,
        sensor_cam_target_pos=cam_target,
        control_mode="pd_ee_delta_pos",
        sensor_configs=dict(width=128, height=128),
    )
    envs = gym.make("PickCube-v1", num_envs=num_envs, reconfiguration_freq=None, **env_kwargs)
    envs = FlattenRGBDObservationWrapper(envs, rgb=True, depth=True, state=True, sep_depth=True)
    if isinstance(envs.action_space, gym.spaces.Dict):
        envs = FlattenActionSpaceWrapper(envs)
    envs = ManiSkillVectorEnv(envs, num_envs, ignore_terminations=False, record_metrics=True)

    obs, _ = envs.reset(seed=42 + vp_id)
    teacher = UnifiedAblationAgent(envs, obs, candidate="A").to(device)
    ckpt = torch.load(teacher_ckpt, map_location=device)
    teacher.load_state_dict(ckpt["model_state"])
    teacher.eval()

    num_batches = int(np.ceil(num_steps / num_envs))
    total_samples = num_batches * num_envs

    buf_rgb = torch.empty((total_samples, 128, 128, 3), dtype=torch.uint8)
    buf_depth = torch.empty((total_samples, 128, 128, 1), dtype=torch.int16)
    buf_state = torch.empty((total_samples, 29), dtype=torch.float32)
    buf_actions = torch.empty((total_samples, 4), dtype=torch.float32)

    cur_idx = 0
    t0 = time.time()
    success_count = 0
    total_episodes = 0

    with torch.no_grad():
        for b in range(num_batches):
            actions = teacher.get_action(obs, deterministic=True)
            
            # Store in CPU buffers to keep GPU VRAM minimal
            buf_rgb[cur_idx:cur_idx + num_envs] = obs["rgb"].cpu()
            buf_depth[cur_idx:cur_idx + num_envs] = obs["depth"].cpu()
            buf_state[cur_idx:cur_idx + num_envs] = obs["state"].cpu()
            buf_actions[cur_idx:cur_idx + num_envs] = actions.cpu()
            cur_idx += num_envs

            obs, _, _, _, infos = envs.step(actions)
            if "final_info" in infos and "_final_info" in infos:
                mask = infos["_final_info"]
                for i in range(len(mask)):
                    if mask[i]:
                        ep = infos["final_info"].get("episode", {})
                        k = "success_once" if "success_once" in ep else "success"
                        if k in ep:
                            val = ep[k][i]
                            if bool(val.item() if torch.is_tensor(val) else val):
                                success_count += 1
                        total_episodes += 1

    envs.close()
    elapsed = time.time() - t0
    succ_rate = (success_count / max(total_episodes, 1)) * 100.0 if total_episodes > 0 else 0.0

    # Save to disk
    data = {
        "viewpoint_id": vp_id,
        "azimuth_deg": vp["azimuth_deg"],
        "elevation_deg": vp["elevation_deg"],
        "num_samples": total_samples,
        "rollout_success_rate": succ_rate,
        "rgb": buf_rgb[:total_samples],
        "depth": buf_depth[:total_samples],
        "state": buf_state[:total_samples],
        "actions": buf_actions[:total_samples],
    }
    torch.save(data, out_file)
    print(f"[Collector] VP {vp_id:02d} Done! {total_samples:,} samples saved. Teacher Eval: {succ_rate:.1f}%. Walltime: {elapsed:.1f}s -> {out_file}")
    return out_file


def collect_all_teachers(
    viewpoints=(17, 0, 28, 18, 27, 20, 19, 6),
    num_steps: int = 10000,
    output_dir: str = DEFAULT_DATA_DIR,
):
    print("=" * 78)
    print(" MULTI-TEACHER DEMONSTRATION ROLLOUT COLLECTION")
    print(f" Viewpoints: {list(viewpoints)}")
    print(f" Steps per viewpoint: {num_steps:,} (Total: {len(viewpoints) * num_steps:,})")
    print("=" * 78)

    t0 = time.time()
    manifest = []
    for vp in viewpoints:
        path = collect_single_teacher_rollouts(vp, num_steps=num_steps, output_dir=output_dir)
        manifest.append(path)

    total_time = time.time() - t0
    print(f"\n[Collector] All {len(viewpoints)} teachers collected in {total_time:.1f}s ({total_time / 60:.1f}m)!")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--viewpoints", type=int, nargs="+", default=[17, 0, 28, 18, 27, 20, 19, 6])
    parser.add_argument("--num-steps", type=int, default=10000)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_DATA_DIR)
    args = parser.parse_args()

    collect_all_teachers(viewpoints=args.viewpoints, num_steps=args.num_steps, output_dir=args.output_dir)
