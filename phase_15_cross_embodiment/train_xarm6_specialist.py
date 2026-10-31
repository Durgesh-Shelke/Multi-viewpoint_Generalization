"""
Phase 15: Cross-Embodiment Specialist Training & Full 13-Viewpoint Benchmark.

This script:
1. Collects 500 episodes (25,000 transitions) for xArm 6 with decisive lifting.
2. Trains PointNet xArm 6 Specialist policy (35-dim proprioception + 512-point PointNet).
3. Evaluates both Panda Specialist and xArm 6 Specialist across the exact 13-viewpoint benchmark suite (50 envs per viewpoint = 1,300 total tests).
4. Generates specialists_accuracy_benchmark.json and visualization plot.
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
from phase_13_pointcloud_generalization.pointnet_models import PointNetAgent, extract_robot_frame_pc

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = "/home/maniskill/phase_15_cross_embodiment"
SPEC_DIR = os.path.join(BASE_DIR, "specialists")
EVAL_DIR = os.path.join(BASE_DIR, "eval_results")
os.makedirs(SPEC_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)

VIEWPOINTS_JSON = "/home/maniskill/phase_1_to_3/sequential_viewpoint_training/viewpoints.json"
PANDA_CKPT = "/home/maniskill/phase_13_pointcloud_generalization/student/pointnet_student_best.pt"
XARM6_CKPT = os.path.join(SPEC_DIR, "xarm6_specialist_best.pt")


def sample_continuous_spherical_camera(rng: np.random.RandomState, robot_base=np.array([-0.522, 0.0, 0.0])):
    az_deg = rng.uniform(-70.0, 70.0)
    el_deg = rng.uniform(12.0, 50.0)
    dist = rng.uniform(0.95, 1.25)
    az_rad = np.radians(az_deg)
    el_rad = np.radians(el_deg)
    rel_x = dist * np.cos(el_rad) * np.cos(az_rad)
    rel_y = dist * np.cos(el_rad) * np.sin(az_rad)
    rel_z = dist * np.sin(el_rad)
    cam_eye = (robot_base + np.array([rel_x, rel_y, rel_z])).tolist()
    return cam_eye, az_deg, el_deg, dist


def extract_robot_frame_pc_xarm6(obs: dict, num_points: int = 512) -> torch.Tensor:
    pos = obs["sensor_data"]["base_camera"]["position"].float() / 1000.0  # mm to m
    cam2world = obs["sensor_param"]["base_camera"]["cam2world_gl"].to(pos.device)
    B = pos.shape[0]

    pos_homo = torch.cat([pos, torch.ones_like(pos[..., :1])], dim=-1)
    xyz_world = (pos_homo.reshape(B, -1, 4) @ cam2world.transpose(1, 2))[..., :3]

    robot_base = torch.tensor([-0.522, 0.0, 0.0], device=pos.device)
    xyz_robot = xyz_world - robot_base

    # Workspace bounding box in xArm 6 frame (base at -0.522)
    mask = (
        (xyz_robot[..., 0] >= 0.25) & (xyz_robot[..., 0] <= 0.80) &
        (xyz_robot[..., 1] >= -0.35) & (xyz_robot[..., 1] <= 0.35) &
        (xyz_robot[..., 2] >= 0.005) & (xyz_robot[..., 2] <= 0.40)
    )

    out_pts = torch.zeros((B, num_points, 3), device=pos.device, dtype=torch.float32)
    for b in range(B):
        pts_b = xyz_robot[b, mask[b]]
        n = pts_b.shape[0]
        if n >= num_points:
            perm = torch.randperm(n, device=pos.device)[:num_points]
            out_pts[b] = pts_b[perm]
        elif n > 0:
            idx = torch.randint(0, n, (num_points,), device=pos.device)
            out_pts[b] = pts_b[idx]
    return out_pts


def collect_xarm6_data(num_episodes=500, batch_envs=64, num_points=512, seed=42):
    print("\n" + "=" * 80)
    print(f" [STAGE 1] COLLECTING {num_episodes} EPISODES FOR XARM6 SPECIALIST")
    print(f" Teacher: 100% Markovian Oracle Heuristic with Decisive Lifting")
    print(f" Camera: Continuous Spherical Coverage (Az: [-70, 70], El: [12, 50])")
    print("=" * 80)

    total_samples = num_episodes * 50
    buf_pts = torch.empty((total_samples, num_points, 3), dtype=torch.float32)
    buf_proprio = torch.empty((total_samples, 35), dtype=torch.float32)
    buf_actions = torch.empty((total_samples, 4), dtype=torch.float32)

    rng = np.random.RandomState(seed)
    cur_idx = 0
    t0 = time.time()
    num_batches = int(np.ceil(num_episodes / batch_envs))

    # 4 nominal anchor cameras for xArm6 (shifted to xArm6 base)
    base_shift = np.array([-0.522 - (-0.615), 0.0, 0.0])
    anchor_cameras = [
        ([-0.1774 + base_shift[0], -0.0147, 0.5768], -4.74, 31.8, 1.0942),  # VP 17
        ([0.4542 + base_shift[0], 0.0, 0.2326], 0.0, 12.27, 1.0942),        # VP 00
        ([0.0886 + base_shift[0], -0.5361, 0.6272], -37.2, 34.98, 1.0942),   # VP 20
        ([0.0076 + base_shift[0], 0.6728, 0.6070], 47.79, 33.69, 1.0942),   # VP 19
    ]

    for b in range(num_batches):
        cur_batch = min(batch_envs, num_episodes - (b * batch_envs))
        if b < len(anchor_cameras):
            base_eye, az, el, dist = anchor_cameras[b]
            jitter = rng.normal(0, 0.015, size=3)
            cam_eye = (np.array(base_eye) + jitter).tolist()
        else:
            cam_eye, az, el, dist = sample_continuous_spherical_camera(rng)

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

            act = torch.zeros((batch_envs, 4), device=DEVICE)
            dist_xy = torch.norm(diff_c[:, :2], dim=-1)
            dist_z = torch.abs(diff_c[:, 2] - 0.01)
            close_cond = (~is_grasped) & (dist_xy < 0.02) & (dist_z < 0.025)

            not_grasped = ~is_grasped
            act[not_grasped, 0] = diff_c[not_grasped, 0] * 10
            act[not_grasped, 1] = diff_c[not_grasped, 1] * 10
            act[not_grasped, 2] = (diff_c[not_grasped, 2] - 0.01) * 10
            act[not_grasped, 3] = -1.0 # open
            act[close_cond, 3] = 1.0 # close

            # Decisive lifting towards goal
            act[is_grasped, 0] = torch.clamp(diff_g[is_grasped, 0] * 10.0, -1.0, 1.0)
            act[is_grasped, 1] = torch.clamp(diff_g[is_grasped, 1] * 10.0, -1.0, 1.0)
            act[is_grasped, 2] = torch.clamp(diff_g[is_grasped, 2] * 10.0, -1.0, 1.0)
            act[is_grasped, 3] = 1.0 # keep closed

            with torch.no_grad():
                pts = extract_robot_frame_pc_xarm6(obs, num_points=num_points)[:cur_batch]
                proprio_35 = obs["state"][:cur_batch, :35]

            buf_pts[cur_idx:cur_idx + cur_batch] = pts.cpu()
            buf_proprio[cur_idx:cur_idx + cur_batch] = proprio_35.cpu()
            buf_actions[cur_idx:cur_idx + cur_batch] = act[:cur_batch].cpu()
            cur_idx += cur_batch

            obs, _, _, _, _ = vec_env.step(act)

        vec_env.close()
        del vec_env
        del env
        gc.collect()

        if (b + 1) % 2 == 0 or (b + 1) == num_batches:
            elapsed = time.time() - t0
            print(f"  Batch {b+1:2d}/{num_batches} | Samples: {cur_idx:,}/{total_samples:,} | Time: {elapsed:.1f}s")

    print(f"[OK] Collection complete in {time.time() - t0:.1f}s. Total dataset size: {cur_idx:,} samples.")
    return buf_pts[:cur_idx], buf_proprio[:cur_idx], buf_actions[:cur_idx]


def train_xarm6_policy(pts_data, prop_data, act_data, epochs=25, batch_size=256, lr=1e-3):
    print("\n" + "=" * 80)
    print(f" [STAGE 2] TRAINING XARM6 POINTNET SPECIALIST ({epochs} EPOCHS)")
    print("=" * 80)

    dataset = TensorDataset(pts_data, prop_data, act_data)
    val_size = int(len(dataset) * 0.1)
    train_size = len(dataset) - val_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=2)

    model = PointNetAgent(action_dim=4, proprio_dim=35, point_dim=256).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_loss = float("inf")
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

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), XARM6_CKPT)
            mark = "(*) BEST"
        else:
            mark = ""

        if ep % 5 == 0 or ep == 1 or mark != "":
            print(f"  Epoch {ep:2d}/{epochs:2d} | Train MSE: {train_loss:.5f} | Val MSE: {val_loss:.5f} | LR: {scheduler.get_last_lr()[0]:.2e} {mark}")

    print(f"[OK] Training finished in {time.time() - t0:.1f}s. Best checkpoint saved to {XARM6_CKPT}")
    return model


def evaluate_specialist_suite():
    print("\n" + "=" * 80)
    print(" [STAGE 3] RIGOROUS 13-VIEWPOINT BENCHMARK FOR BOTH SPECIALISTS")
    print(" Explicit User Constraint: Testing specialists BEFORE training student model")
    print(" Viewpoint Suite: 4 Anchors (Trained) + 9 Novel In-Between Viewpoints (Interpolated)")
    print(" Environments per Viewpoint: 50 | Total Tests per Specialist: 650 (1,300 total)")
    print("=" * 80)

    with open(VIEWPOINTS_JSON, "r") as f:
        vps = json.load(f)
    vp_dict = {vp["id"]: vp for vp in vps}

    eval_vps = [
        # 4 Anchors
        (17, "VP 17 (Front-Mid Anchor)", "Anchor"),
        (0,  "VP 00 (Front-Low Anchor)", "Anchor"),
        (20, "VP 20 (Diag-Left Anchor)", "Anchor"),
        (19, "VP 19 (Diag-Right Anchor)", "Anchor"),
        # 9 In-Between Viewpoints
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

    # Load Panda Specialist (proprio_dim=29)
    panda_policy = PointNetAgent(action_dim=4, proprio_dim=29, point_dim=256).to(DEVICE)
    p_ckpt = torch.load(PANDA_CKPT, map_location=DEVICE)
    panda_policy.load_state_dict(p_ckpt.get("model_state", p_ckpt))
    panda_policy.eval()

    # Load xArm 6 Specialist (proprio_dim=35)
    xarm6_policy = PointNetAgent(action_dim=4, proprio_dim=35, point_dim=256).to(DEVICE)
    xarm6_policy.load_state_dict(torch.load(XARM6_CKPT, map_location=DEVICE))
    xarm6_policy.eval()

    benchmark_results = {"panda": [], "xarm6": []}
    batch_envs = 50

    # 1. Evaluate Franka Panda Specialist
    print("\n--- Evaluating Specialist 1: Franka Panda (7 DoF) ---")
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
        obs, _ = vec_env.reset(seed=2026 + vid)
        grasped_any = torch.zeros(batch_envs, dtype=torch.bool, device=DEVICE)

        for step in range(50):
            proprio = obs["state"][:, :29]
            with torch.no_grad():
                pts = extract_robot_frame_pc(obs, num_points=512)
                act = panda_policy.get_action(pts, proprio, deterministic=True)
            obs, _, _, _, info = vec_env.step(act)
            if "is_grasped" in info:
                grasped_any |= info["is_grasped"]

        succ = info["final_info"]["episode"]["success_once"].float().mean().item() * 100.0
        grasp = grasped_any.float().mean().item() * 100.0
        vec_env.close()
        del vec_env; del env; gc.collect()

        benchmark_results["panda"].append({
            "id": vid, "name": name, "type": vtype,
            "azimuth": meta["azimuth_deg"], "elevation": meta["elevation_deg"],
            "success_rate": succ, "grasp_rate": grasp
        })
        print(f"  [{vtype.upper():<10}] {name:<32}: Succ={succ:5.1f}% | Grasp={grasp:5.1f}%")

    # 2. Evaluate xArm 6 Specialist
    print("\n--- Evaluating Specialist 2: UFactory xArm 6 Robotiq (6 DoF) ---")
    for vid, name, vtype in eval_vps:
        meta = vp_dict[vid]
        base_shift = np.array([-0.522 - (-0.615), 0.0, 0.0])
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
        obs, _ = vec_env.reset(seed=3026 + vid)
        grasped_any = torch.zeros(batch_envs, dtype=torch.bool, device=DEVICE)

        for step in range(50):
            proprio = obs["state"][:, :35]
            with torch.no_grad():
                pts = extract_robot_frame_pc_xarm6(obs, num_points=512)
                act = xarm6_policy.get_action(pts, proprio, deterministic=True)
                # Firm clamping torque saturation for Robotiq 4-bar linkage
                act[:, 3] = torch.where(act[:, 3] > 0.0, torch.tensor(1.0, device=DEVICE), torch.tensor(-1.0, device=DEVICE))
            obs, _, _, _, info = vec_env.step(act)
            if "is_grasped" in info:
                grasped_any |= info["is_grasped"]

        succ = info["final_info"]["episode"]["success_once"].float().mean().item() * 100.0
        grasp = grasped_any.float().mean().item() * 100.0
        vec_env.close()
        del vec_env; del env; gc.collect()

        benchmark_results["xarm6"].append({
            "id": vid, "name": name, "type": vtype,
            "azimuth": meta["azimuth_deg"], "elevation": meta["elevation_deg"],
            "success_rate": succ, "grasp_rate": grasp
        })
        print(f"  [{vtype.upper():<10}] {name:<32}: Succ={succ:5.1f}% | Grasp={grasp:5.1f}%")

    # Compute Summary Statistics
    panda_anchor_succ = np.mean([r["success_rate"] for r in benchmark_results["panda"] if r["type"] == "Anchor"])
    panda_interp_succ = np.mean([r["success_rate"] for r in benchmark_results["panda"] if r["type"] == "In-Between"])
    panda_overall_succ = np.mean([r["success_rate"] for r in benchmark_results["panda"]])
    panda_overall_grasp = np.mean([r["grasp_rate"] for r in benchmark_results["panda"]])

    xarm_anchor_succ = np.mean([r["success_rate"] for r in benchmark_results["xarm6"] if r["type"] == "Anchor"])
    xarm_interp_succ = np.mean([r["success_rate"] for r in benchmark_results["xarm6"] if r["type"] == "In-Between"])
    xarm_overall_succ = np.mean([r["success_rate"] for r in benchmark_results["xarm6"]])
    xarm_overall_grasp = np.mean([r["grasp_rate"] for r in benchmark_results["xarm6"]])

    summary = {
        "panda": {
            "anchor_success": float(panda_anchor_succ),
            "inbetween_success": float(panda_interp_succ),
            "overall_success": float(panda_overall_succ),
            "overall_grasp": float(panda_overall_grasp),
        },
        "xarm6": {
            "anchor_success": float(xarm_anchor_succ),
            "inbetween_success": float(xarm_interp_succ),
            "overall_success": float(xarm_overall_succ),
            "overall_grasp": float(xarm_overall_grasp),
        },
        "benchmark_records": benchmark_results,
    }

    out_json = os.path.join(EVAL_DIR, "specialists_accuracy_benchmark.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print(" [BENCHMARK SUMMARY] SPECIALISTS ACCURACY ACROSS 13 VIEWPOINTS")
    print(f" Franka Panda Specialist: Anchor={panda_anchor_succ:.1f}% | In-Between={panda_interp_succ:.1f}% | Overall={panda_overall_succ:.1f}% | Grasp={panda_overall_grasp:.1f}%")
    print(f" xArm 6 Specialist:       Anchor={xarm_anchor_succ:.1f}% | In-Between={xarm_interp_succ:.1f}% | Overall={xarm_overall_succ:.1f}% | Grasp={xarm_overall_grasp:.1f}%")
    print("=" * 80)

    # Plot Comparison Chart
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(eval_vps))
    width = 0.35

    panda_scores = [r["success_rate"] for r in benchmark_results["panda"]]
    xarm_scores = [r["success_rate"] for r in benchmark_results["xarm6"]]

    rects1 = ax.bar(x - width/2, panda_scores, width, label=f"Franka Panda Specialist (Mean: {panda_overall_succ:.1f}%)", color="#1f77b4", alpha=0.9)
    rects2 = ax.bar(x + width/2, xarm_scores, width, label=f"xArm 6 Robotiq Specialist (Mean: {xarm_overall_succ:.1f}%)", color="#2ca02c", alpha=0.9)

    ax.axvline(3.5, color="red", linestyle="--", linewidth=1.5, label="Boundary: Trained Anchors vs Novel In-Betweens")
    ax.set_ylabel("Success Rate (%)", fontsize=12)
    ax.set_title("Cross-Embodiment Specialist Accuracy Suite: 13 Viewpoints (4 Anchors + 9 In-Betweens)", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"VP {r['id']}\n({r['type'][:3]})" for r in benchmark_results["panda"]], fontsize=10)
    ax.set_ylim(0, 110)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.legend(loc="lower right", fontsize=11)

    plt.tight_layout()
    plot_path = os.path.join(EVAL_DIR, "specialists_accuracy_benchmark.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"[OK] Comparison plot saved to {plot_path}")


if __name__ == "__main__":
    pts, prop, act = collect_xarm6_data(num_episodes=500, batch_envs=64, num_points=512, seed=42)
    train_xarm6_policy(pts, prop, act, epochs=25, batch_size=256, lr=1e-3)
    evaluate_specialist_suite()
