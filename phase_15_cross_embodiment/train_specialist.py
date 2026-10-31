"""
Phase 15: Single-Arm PointNet Specialist Distillation & Benchmark.

Usage:
    python3 train_specialist.py --robot_uid="panda"
    python3 train_specialist.py --robot_uid="xarm6_robotiq"
"""

import os
import sys
sys.path.append("/home/maniskill")
sys.path.append("/home/maniskill/examples/baselines/ppo")
import argparse
import time
import json
import gc
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gymnasium as gym

import mani_skill.envs
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from ppo import Agent as PrivilegedAgent
from phase_15_cross_embodiment.omniarm_models import (
    SingleArmPointNetAgent,
    extract_robot_frame_pc,
    ROBOT_PROPRIO_DIMS,
    ROBOT_ACTION_DIMS,
    ROBOT_CONTROL_MODES,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = "/home/maniskill/phase_15_cross_embodiment"
TEACHER_DIR = os.path.join(BASE_DIR, "teachers")
SPECIALIST_DIR = os.path.join(BASE_DIR, "specialists")
EVAL_DIR = os.path.join(BASE_DIR, "eval_results")
VIEWPOINTS_JSON = "/home/maniskill/phase_1_to_3/sequential_viewpoint_training/viewpoints.json"

os.makedirs(SPECIALIST_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)

ROBOT_BASE_OFFSETS = {
    "panda": (-0.615, 0.0, 0.0),
    "xarm6_robotiq": (-0.522, 0.0, 0.0),
    "so100": (-0.450, 0.0, 0.0),
    "widowxai": (-0.500, 0.0, 0.0),
}


def load_privileged_teacher(robot_uid: str, state_dim: int, action_dim: int):
    ckpt_path = os.path.join(TEACHER_DIR, f"teacher_{robot_uid}.pt")
    if not os.path.exists(ckpt_path):
        # Look in runs directory
        run_ckpt = f"/home/maniskill/runs/{robot_uid}_teacher/final_ckpt.pt"
        if os.path.exists(run_ckpt):
            ckpt_path = run_ckpt

    class DummySpace:
        def __init__(self, shape):
            self.shape = shape
    dummy_vec = SimpleNamespace(
        single_observation_space=DummySpace((state_dim,)),
        single_action_space=DummySpace((action_dim,))
    )
    teacher = PrivilegedAgent(dummy_vec).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    teacher.load_state_dict(ckpt)
    teacher.eval()
    print(f"[OK] Teacher loaded for {robot_uid} from {ckpt_path}")
    return teacher


def sample_continuous_spherical_camera(rng: np.random.RandomState, robot_base_xyz=(-0.615, 0.0, 0.0)):
    az_deg = rng.uniform(-70.0, 70.0)
    el_deg = rng.uniform(12.0, 50.0)
    dist = rng.uniform(0.95, 1.25)

    az_rad = np.radians(az_deg)
    el_rad = np.radians(el_deg)
    base_pos = np.array(robot_base_xyz)

    rel_x = dist * np.cos(el_rad) * np.cos(az_rad)
    rel_y = dist * np.cos(el_rad) * np.sin(az_rad)
    rel_z = dist * np.sin(el_rad)

    cam_eye = (base_pos + np.array([rel_x, rel_y, rel_z])).tolist()
    return cam_eye, az_deg, el_deg, dist


def collect_specialist_dataset(teacher, robot_uid: str, num_episodes: int = 1200, batch_envs: int = 32, num_points: int = 512, seed: int = 42):
    proprio_dim = ROBOT_PROPRIO_DIMS[robot_uid]
    action_dim = ROBOT_ACTION_DIMS[robot_uid]
    ctrl_mode = ROBOT_CONTROL_MODES[robot_uid]
    base_xyz = ROBOT_BASE_OFFSETS.get(robot_uid, (-0.615, 0.0, 0.0))

    print("\n" + "=" * 80)
    print(f" [STAGE 1] COLLECTING {num_episodes:,} EPISODES FOR SPECIALIST: {robot_uid.upper()}")
    print(f" Control Mode: {ctrl_mode} | Action Dim: {action_dim} | Proprio Dim: {proprio_dim}")
    print("=" * 80)

    total_samples = num_episodes * 50
    buf_pts = torch.empty((total_samples, num_points, 3), dtype=torch.float32)
    buf_proprio = torch.empty((total_samples, proprio_dim), dtype=torch.float32)
    buf_actions = torch.empty((total_samples, action_dim), dtype=torch.float32)

    rng = np.random.RandomState(seed)
    cur_idx = 0
    t0 = time.time()
    num_batches = int(np.ceil(num_episodes / batch_envs))

    for b in range(num_batches):
        cur_batch = min(batch_envs, num_episodes - b * batch_envs)
        cam_eye, az, el, dist = sample_continuous_spherical_camera(rng, base_xyz)

        env = gym.make(
            "PickCube-v1",
            robot_uids=robot_uid,
            obs_mode="pointcloud+state",
            control_mode=ctrl_mode,
            num_envs=batch_envs,
            sensor_cam_eye_pos=cam_eye,
            sensor_cam_target_pos=[-0.1, 0.0, 0.1],
            sensor_configs=dict(width=128, height=128),
        )
        vec_env = ManiSkillVectorEnv(env, batch_envs, ignore_terminations=True, record_metrics=True)
        obs, _ = vec_env.reset(seed=int(seed + b * 100))

        for step in range(50):
            state_full = obs["state"][:cur_batch]
            proprio = state_full[:, :proprio_dim]

            with torch.no_grad():
                act = teacher.get_action(state_full, deterministic=True)
                pts = extract_robot_frame_pc(obs, num_points=num_points, robot_base_xyz=base_xyz)[:cur_batch]

            buf_pts[cur_idx:cur_idx + cur_batch] = pts.cpu()
            buf_proprio[cur_idx:cur_idx + cur_batch] = proprio.cpu()
            buf_actions[cur_idx:cur_idx + cur_batch] = act.cpu()
            cur_idx += cur_batch

            if cur_batch < batch_envs:
                act_pad = torch.zeros((batch_envs, action_dim), device=DEVICE)
                act_pad[:cur_batch] = act
                obs, _, _, _, _ = vec_env.step(act_pad)
            else:
                obs, _, _, _, _ = vec_env.step(act)

        vec_env.close()
        del vec_env
        del env
        gc.collect()
        torch.cuda.empty_cache()

        if (b + 1) % 5 == 0 or b == 0 or (b + 1) == num_batches:
            print(f"  Batch {b+1:2d}/{num_batches:2d} | Viewpoint: Az={az:+5.1f}°, El={el:4.1f}° | Collected: {cur_idx:,} / {total_samples:,} samples")

    print(f"\n[OK] Dataset collection for {robot_uid} completed: {cur_idx:,} transitions in {time.time()-t0:.1f}s")
    return buf_pts[:cur_idx], buf_proprio[:cur_idx], buf_actions[:cur_idx]


def train_specialist(robot_uid: str, buf_pts, buf_proprio, buf_actions, num_epochs: int = 25, batch_size: int = 256):
    print("\n" + "=" * 80)
    print(f" [STAGE 2] TRAINING SPECIALIST: {robot_uid.upper()}")
    print(f" Transitions: {len(buf_actions):,} | Epochs: {num_epochs} | Batch Size: {batch_size}")
    print("=" * 80)

    dataset = TensorDataset(buf_pts, buf_proprio, buf_actions)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True)

    agent = SingleArmPointNetAgent(robot_uid=robot_uid).to(DEVICE)
    optimizer = optim.AdamW(agent.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)

    best_loss = float("inf")
    ckpt_path = os.path.join(SPECIALIST_DIR, f"specialist_{robot_uid}.pt")

    for epoch in range(1, num_epochs + 1):
        agent.train()
        total_loss = 0.0
        n_batches = 0
        t0 = time.time()

        for b_pts, b_proprio, b_act in loader:
            b_pts = b_pts.to(DEVICE, non_blocking=True)
            b_proprio = b_proprio.to(DEVICE, non_blocking=True)
            b_act = b_act.to(DEVICE, non_blocking=True)

            pred = agent.get_action(b_pts, b_proprio, deterministic=True)
            loss = F.mse_loss(pred, b_act)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / n_batches
        dt = time.time() - t0

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "model_state": agent.state_dict(),
                "robot_uid": robot_uid,
                "epoch": epoch,
                "best_loss": best_loss,
            }, ckpt_path)

        if epoch % 5 == 0 or epoch == 1 or epoch == num_epochs:
            print(f"  Epoch {epoch:2d}/{num_epochs:2d} | Action MSE Loss: {avg_loss:.5f} | Time: {dt:.1f}s")

    print(f"\n[OK] Specialist {robot_uid} saved to {ckpt_path} (MSE: {best_loss:.5f})")
    return ckpt_path


def evaluate_specialist(robot_uid: str, specialist_ckpt: str):
    print("\n" + "=" * 80)
    print(f" [STAGE 3] BENCHMARKING SPECIALIST: {robot_uid.upper()}")
    print("=" * 80)

    with open(VIEWPOINTS_JSON) as f:
        vps = json.load(f)
    vp_dict = {v["id"]: v for v in vps}

    ctrl_mode = ROBOT_CONTROL_MODES[robot_uid]
    proprio_dim = ROBOT_PROPRIO_DIMS[robot_uid]
    base_xyz = ROBOT_BASE_OFFSETS.get(robot_uid, (-0.615, 0.0, 0.0))

    agent = SingleArmPointNetAgent(robot_uid=robot_uid).to(DEVICE)
    ckpt = torch.load(specialist_ckpt, map_location=DEVICE)
    agent.load_state_dict(ckpt.get("model_state", ckpt))
    agent.eval()

    eval_vps = [
        (17, "VP 17 (Front-Mid Anchor)", "trained"),
        (0,  "VP 00 (Front-Low Anchor)", "trained"),
        (20, "VP 20 (Diag-Left Anchor)", "trained"),
        (19, "VP 19 (Diag-Right Anchor)", "trained"),
        (13, "VP 13 (In-Between Az: VP17-20)", "untrained"),
        (14, "VP 14 (In-Between Az: VP17-19)", "untrained"),
        (12, "VP 12 (Flank In-Between R)", "untrained"),
        (15, "VP 15 (Flank In-Between L)", "untrained"),
    ]

    results = []
    batch_envs = 50

    for vid, name, vtype in eval_vps:
        meta = vp_dict[vid]
        cam_eye = meta["world_eye"]

        env = gym.make(
            "PickCube-v1",
            robot_uids=robot_uid,
            obs_mode="pointcloud+state",
            control_mode=ctrl_mode,
            num_envs=batch_envs,
            sensor_cam_eye_pos=cam_eye,
            sensor_cam_target_pos=[-0.1, 0.0, 0.1],
            sensor_configs=dict(width=128, height=128),
        )
        eval_envs = ManiSkillVectorEnv(env, batch_envs, ignore_terminations=True, record_metrics=True)
        obs, _ = eval_envs.reset(seed=2026 + vid)
        grasped_any = torch.zeros(batch_envs, dtype=torch.bool, device=DEVICE)

        for step in range(50):
            proprio = obs["state"][:, :proprio_dim]

            with torch.no_grad():
                pts = extract_robot_frame_pc(obs, num_points=512, robot_base_xyz=base_xyz)
                act = agent.get_action(pts, proprio, deterministic=True)

            obs, rew, term, trunc, infos = eval_envs.step(act)
            if "is_grasped" in infos:
                grasped_any |= infos["is_grasped"]

        succ = infos["final_info"]["episode"]["success_once"]
        succ_rate = succ.float().mean().item() * 100.0
        grasp_rate = grasped_any.float().mean().item() * 100.0
        eval_envs.close()
        del eval_envs
        del env
        gc.collect()
        torch.cuda.empty_cache()

        results.append({
            "id": vid,
            "name": name,
            "type": vtype,
            "success_rate": succ_rate,
            "grasp_rate": grasp_rate,
        })
        tag = "[ANCHOR]" if vtype == "trained" else "[IN-BETWEEN]"
        print(f"  {tag:<12} {name:<32}: Succ={succ_rate:5.1f}% | Grasp={grasp_rate:5.1f}%")

    avg_succ = np.mean([r["success_rate"] for r in results])
    avg_grasp = np.mean([r["grasp_rate"] for r in results])
    print(f"\n[OK] {robot_uid.upper()} Overall Success: {avg_succ:.1f}% | Grasp: {avg_grasp:.1f}%")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot_uid", type=str, default="xarm6_robotiq")
    parser.add_argument("--num_episodes", type=int, default=1200)
    args = parser.parse_args()

    state_dims = {"panda": 42, "xarm6_robotiq": 48, "so100": 36, "widowxai": 40}
    action_dims = ROBOT_ACTION_DIMS

    teacher = load_privileged_teacher(args.robot_uid, state_dims[args.robot_uid], action_dims[args.robot_uid])
    buf_pts, buf_proprio, buf_actions = collect_specialist_dataset(teacher, args.robot_uid, num_episodes=args.num_episodes)
    ckpt = train_specialist(args.robot_uid, buf_pts, buf_proprio, buf_actions)
    evaluate_specialist(args.robot_uid, ckpt)


if __name__ == "__main__":
    main()
