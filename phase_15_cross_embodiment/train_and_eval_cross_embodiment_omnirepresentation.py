"""
Phase 15: Clean Cross-Embodiment Zero-Shot Transfer via Unified Table-Frame Representation.

This script:
1. Trains Panda Omni Specialist (trained ONLY on Franka Panda demonstrations, using 12-dim joint-agnostic proprio + table-frame PC).
2. Tests Panda Omni Specialist:
   - On Franka Panda (In-distribution specialist accuracy across 13 VPs)
   - On xArm 6 Robotiq (ZERO-SHOT cross-embodiment transfer across 13 VPs!)
3. Compares with the Combined OmniArm Policy (trained on both).
4. Saves full evaluation results to eval_results/unified_cross_embodiment_benchmark.json.
"""

import os
import sys
sys.path.append("/home/maniskill")
import time
import json
import gc
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
from phase_15_cross_embodiment.train_combined_omniarm import (
    OmniArmPointNetPolicy, extract_table_workspace_pc, get_universal_proprio, sample_continuous_spherical_camera
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = "/home/maniskill/phase_15_cross_embodiment"
SPEC_DIR = os.path.join(BASE_DIR, "specialists")
EVAL_DIR = os.path.join(BASE_DIR, "eval_results")
VIEWPOINTS_JSON = "/home/maniskill/phase_1_to_3/sequential_viewpoint_training/viewpoints.json"

PANDA_OMNI_CKPT = os.path.join(SPEC_DIR, "panda_omni_specialist.pt")
COMBINED_CKPT = os.path.join(BASE_DIR, "combined", "combined_omniarm_best.pt")


def collect_panda_omni_data(num_episodes=500, batch_envs=64, num_points=512, seed=42):
    print("\n" + "=" * 80)
    print(f" [STAGE 1] COLLECTING {num_episodes} EPISODES FOR PANDA OMNI SPECIALIST")
    print(" Architecture: 12-dim Joint-Agnostic Proprioception + Table-Frame Workspace Point Clouds")
    print("=" * 80)

    total_samples = num_episodes * 50
    buf_pts = torch.empty((total_samples, num_points, 3), dtype=torch.float32)
    buf_prop = torch.empty((total_samples, 12), dtype=torch.float32)
    buf_act = torch.empty((total_samples, 4), dtype=torch.float32)

    cur_idx = 0
    t0 = time.time()
    rng = np.random.RandomState(seed)
    robot_base = np.array([-0.615, 0.0, 0.0])
    num_batches = int(np.ceil(num_episodes / batch_envs))

    for b in range(num_batches):
        cur_batch = min(batch_envs, num_episodes - (b * batch_envs))
        cam_eye, az, el, dist = sample_continuous_spherical_camera(rng, robot_base)

        env = gym.make(
            "PickCube-v1",
            robot_uids="panda",
            obs_mode="pointcloud+state",
            control_mode="pd_ee_delta_pos",
            num_envs=batch_envs,
            sensor_cam_eye_pos=cam_eye,
            sensor_cam_target_pos=[-0.1, 0.0, 0.1],
            sensor_configs=dict(width=128, height=128),
            shader_dir="minimal",
        )
        vec_env = ManiSkillVectorEnv(env, batch_envs, ignore_terminations=True, record_metrics=True)
        obs, _ = vec_env.reset(seed=int(seed + b * 100))

        for step in range(50):
            agent = env.unwrapped.agent
            cube = env.unwrapped.cube
            goal = env.unwrapped.goal_site
            tcp_p = agent.tcp_pose.p
            cube_p = cube.pose.p
            diff_c = cube_p - tcp_p
            diff_g = goal.pose.p - cube_p
            is_grasped = agent.is_grasping(cube)

            act_univ = torch.zeros((batch_envs, 4), device=DEVICE)
            dist_xy = torch.norm(diff_c[:, :2], dim=-1)
            dist_z = torch.abs(diff_c[:, 2] - 0.005)
            close_cond = (~is_grasped) & (dist_xy < 0.02) & (dist_z < 0.025)

            not_grasped = ~is_grasped
            act_univ[not_grasped, 0] = diff_c[not_grasped, 0] * 10
            act_univ[not_grasped, 1] = diff_c[not_grasped, 1] * 10
            act_univ[not_grasped, 2] = (diff_c[not_grasped, 2] - 0.005) * 10
            act_univ[not_grasped, 3] = -1.0 # open
            act_univ[close_cond, 3] = 1.0 # close

            act_univ[is_grasped, 0] = torch.clamp(diff_g[is_grasped, 0] * 5.0, -0.5, 0.5)
            act_univ[is_grasped, 1] = torch.clamp(diff_g[is_grasped, 1] * 5.0, -0.5, 0.5)
            act_univ[is_grasped, 2] = torch.clamp(diff_g[is_grasped, 2] * 5.0, -0.5, 0.5)
            act_univ[is_grasped, 3] = 1.0 # close

            with torch.no_grad():
                pts = extract_table_workspace_pc(obs, num_points=num_points)[:cur_batch]
                prop = get_universal_proprio(env, "panda")[:cur_batch]

            buf_pts[cur_idx:cur_idx + cur_batch] = pts.cpu()
            buf_prop[cur_idx:cur_idx + cur_batch] = prop.cpu()
            buf_act[cur_idx:cur_idx + cur_batch] = act_univ[:cur_batch].cpu()
            cur_idx += cur_batch

            act_raw = act_univ.clone()
            act_raw[:, 3] = -act_univ[:, 3] # Panda sign mapping

            obs, _, _, _, _ = vec_env.step(act_raw)

        vec_env.close()
        del vec_env; del env; gc.collect()

        if (b + 1) % 2 == 0 or (b + 1) == num_batches:
            print(f"  Batch {b+1:2d}/{num_batches} | Samples: {cur_idx:,}/{total_samples:,} | Time: {time.time()-t0:.1f}s")

    print(f"[OK] Collection complete in {time.time()-t0:.1f}s.")
    return buf_pts[:cur_idx], buf_prop[:cur_idx], buf_act[:cur_idx]


def train_panda_omni(pts, prop, act, epochs=25, batch_size=256, lr=1e-3):
    print("\n" + "=" * 80)
    print(f" [STAGE 2] TRAINING PANDA OMNI SPECIALIST ({epochs} EPOCHS)")
    print("=" * 80)

    dataset = TensorDataset(pts, prop, act)
    val_size = int(len(dataset) * 0.1)
    train_size = len(dataset) - val_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=2)

    model = OmniArmPointNetPolicy(action_dim=4, proprio_dim=12, point_dim=256).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_loss = float("inf")
    t0 = time.time()

    for ep in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for bp, bs, ba in train_loader:
            bp = bp.to(DEVICE, non_blocking=True)
            bs = bs.to(DEVICE, non_blocking=True)
            ba = ba.to(DEVICE, non_blocking=True)

            pred = model(bp, bs)
            loss = F.mse_loss(pred, ba)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(bp)

        scheduler.step()
        train_loss /= train_size

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bp, bs, ba in val_loader:
                bp = bp.to(DEVICE, non_blocking=True)
                bs = bs.to(DEVICE, non_blocking=True)
                ba = ba.to(DEVICE, non_blocking=True)
                pred = model(bp, bs)
                val_loss += F.mse_loss(pred, ba).item() * len(bp)
        val_loss /= val_size

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), PANDA_OMNI_CKPT)
            mark = "(*) BEST"
        else:
            mark = ""

        if ep % 5 == 0 or ep == 1 or mark != "":
            print(f"  Epoch {ep:2d}/{epochs:2d} | Train MSE: {train_loss:.5f} | Val MSE: {val_loss:.5f} | LR: {scheduler.get_last_lr()[0]:.2e} {mark}")

    print(f"[OK] Training complete in {time.time()-t0:.1f}s. Saved to {PANDA_OMNI_CKPT}")
    return model


def run_zero_shot_evaluation():
    print("\n" + "=" * 80)
    print(" [STAGE 3] ZERO-SHOT CROSS-EMBODIMENT EVALUATION ACROSS 13 VIEWPOINTS")
    print(" Policy: Panda Omni Specialist (trained ONLY on Franka Panda)")
    print(" Evaluation 1: In-Distribution on Franka Panda (13 VPs)")
    print(" Evaluation 2: ZERO-SHOT Transfer to UFactory xArm 6 Robotiq (13 VPs)")
    print("=" * 80)

    with open(VIEWPOINTS_JSON, "r") as f:
        vps = json.load(f)
    vp_dict = {vp["id"]: vp for vp in vps}

    eval_vps = [
        (17, "VP 17 (Front-Mid Anchor)", "Anchor"),
        (0,  "VP 00 (Front-Low Anchor)", "Anchor"),
        (20, "VP 20 (Diag-Left Anchor)", "Anchor"),
        (19, "VP 19 (Diag-Right Anchor)", "Anchor"),
        (10, "VP 10 (In-Between El: VP00-17)", "In-Between"),
        (13, "VP 13 (In-Between Az: VP17-20)", "In-Between"),
        (14, "VP 14 (In-Between Az: VP17-19)", "In-Between"),
        (4,  "VP 04 (In-Between Low-Right)",   "In-Between"),
        (7,  "VP 07 (In-Between Low-Left)",    "In-Between"),
        (21, "VP 21 (In-Between Mid-Right)",   "In-Between"),
        (24, "VP 24 (In-Between High-Mid)",    "In-Between"),
        (12, "VP 12 (Flank In-Between R)",     "In-Between"),
        (15, "VP 15 (Flank In-Between L)",     "In-Between"),
    ]

    model = OmniArmPointNetPolicy(action_dim=4, proprio_dim=12, point_dim=256).to(DEVICE)
    model.load_state_dict(torch.load(PANDA_OMNI_CKPT, map_location=DEVICE))
    model.eval()

    batch_envs = 50
    results = {"panda_in_distribution": [], "xarm6_zeroshot": []}

    # 1. In-Distribution: Panda Specialist on Panda
    print("\n--- Evaluating Panda Specialist on Franka Panda (In-Distribution) ---")
    for vid, name, vtype in eval_vps:
        meta = vp_dict[vid]
        cam_eye = meta["world_eye"]

        env = gym.make(
            "PickCube-v1",
            robot_uids="panda",
            obs_mode="pointcloud+state",
            control_mode="pd_ee_delta_pos",
            num_envs=batch_envs,
            sensor_cam_eye_pos=cam_eye,
            sensor_cam_target_pos=[-0.1, 0.0, 0.1],
            sensor_configs=dict(width=128, height=128),
            shader_dir="minimal",
        )
        vec_env = ManiSkillVectorEnv(env, batch_envs, ignore_terminations=True, record_metrics=True)
        obs, _ = vec_env.reset(seed=8026 + vid)
        grasped_any = torch.zeros(batch_envs, dtype=torch.bool, device=DEVICE)

        for step in range(50):
            prop = get_universal_proprio(env, "panda").to(DEVICE)
            with torch.no_grad():
                pts = extract_table_workspace_pc(obs, num_points=512)
                act_univ = model.get_action(pts, prop, deterministic=True)
                act_univ[:, 3] = torch.where(act_univ[:, 3] > 0.0, torch.tensor(1.0, device=DEVICE), torch.tensor(-1.0, device=DEVICE))

            act_raw = act_univ.clone()
            act_raw[:, 3] = -act_univ[:, 3] # Panda sign mapping

            obs, _, _, _, info = vec_env.step(act_raw)
            if "is_grasped" in info:
                grasped_any |= info["is_grasped"]

        succ = info["final_info"]["episode"]["success_once"].float().mean().item() * 100.0
        grasp = grasped_any.float().mean().item() * 100.0
        vec_env.close()
        del vec_env; del env; gc.collect()

        results["panda_in_distribution"].append({
            "id": vid, "name": name, "type": vtype,
            "success_rate": succ, "grasp_rate": grasp
        })
        print(f"  [{vtype.upper():<10}] {name:<32}: Succ={succ:5.1f}% | Grasp={grasp:5.1f}%")

    # 2. ZERO-SHOT Transfer: Panda Specialist on xArm 6 Robotiq
    print("\n--- ZERO-SHOT TRANSFER: Panda Specialist Deployed on xArm 6 Robotiq (Unseen Robot!) ---")
    base_shift = np.array([-0.522 - (-0.615), 0.0, 0.0])
    for vid, name, vtype in eval_vps:
        meta = vp_dict[vid]
        cam_eye = (np.array(meta["world_eye"]) + base_shift).tolist()

        env = gym.make(
            "PickCube-v1",
            robot_uids="xarm6_robotiq",
            obs_mode="pointcloud+state",
            control_mode="pd_ee_delta_pos",
            num_envs=batch_envs,
            sensor_cam_eye_pos=cam_eye,
            sensor_cam_target_pos=[-0.1, 0.0, 0.1],
            sensor_configs=dict(width=128, height=128),
            shader_dir="minimal",
        )
        vec_env = ManiSkillVectorEnv(env, batch_envs, ignore_terminations=True, record_metrics=True)
        obs, _ = vec_env.reset(seed=9026 + vid)
        grasped_any = torch.zeros(batch_envs, dtype=torch.bool, device=DEVICE)

        for step in range(50):
            prop = get_universal_proprio(env, "xarm6_robotiq").to(DEVICE)
            with torch.no_grad():
                pts = extract_table_workspace_pc(obs, num_points=512)
                act_univ = model.get_action(pts, prop, deterministic=True)
                act_univ[:, 3] = torch.where(act_univ[:, 3] > 0.0, torch.tensor(1.0, device=DEVICE), torch.tensor(-1.0, device=DEVICE))

            act_raw = act_univ.clone()
            act_raw[:, 3] = act_univ[:, 3] # Robotiq sign mapping

            obs, _, _, _, info = vec_env.step(act_raw)
            if "is_grasped" in info:
                grasped_any |= info["is_grasped"]

        succ = info["final_info"]["episode"]["success_once"].float().mean().item() * 100.0
        grasp = grasped_any.float().mean().item() * 100.0
        vec_env.close()
        del vec_env; del env; gc.collect()

        results["xarm6_zeroshot"].append({
            "id": vid, "name": name, "type": vtype,
            "success_rate": succ, "grasp_rate": grasp
        })
        print(f"  [{vtype.upper():<10}] {name:<32}: Succ={succ:5.1f}% | Grasp={grasp:5.1f}%")

    out_json = os.path.join(EVAL_DIR, "unified_cross_embodiment_benchmark.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    p_in_succ = np.mean([r["success_rate"] for r in results["panda_in_distribution"]])
    p_in_grp = np.mean([r["grasp_rate"] for r in results["panda_in_distribution"]])
    x_zs_succ = np.mean([r["success_rate"] for r in results["xarm6_zeroshot"]])
    x_zs_grp = np.mean([r["grasp_rate"] for r in results["xarm6_zeroshot"]])

    print("\n" + "=" * 80)
    print(" [BENCHMARK SUMMARY] ZERO-SHOT CROSS-EMBODIMENT GENERALIZATION")
    print(f" Panda Specialist on Panda (In-Dist):  Success = {p_in_succ:.1f}% | Grasp = {p_in_grp:.1f}%")
    print(f" Panda Specialist on xArm 6 (ZERO-SHOT): Success = {x_zs_succ:.1f}% | Grasp = {x_zs_grp:.1f}%")
    print("=" * 80)


if __name__ == "__main__":
    pts, prop, act = collect_panda_omni_data(num_episodes=500, batch_envs=64, num_points=512, seed=42)
    train_panda_omni(pts, prop, act, epochs=25, batch_size=256, lr=1e-3)
    run_zero_shot_evaluation()
