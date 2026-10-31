"""
Phase 13: 3D Point-Cloud / Base-Frame Back-Projection Distillation & Benchmark.

Executes Track 4 of the Continuous Generalization Roadmap:
1. Samples 1,920 episodes across continuous spherical camera viewpoints.
2. Back-projects depth into 3D Cartesian points in the robot base frame.
3. Distills privileged teacher actions into PointNetAgent.
4. Benchmarks on the exact standard 13-viewpoint suite (4 anchors + 9 in-between viewpoints).
"""

import os
import sys
sys.path.append("/home/maniskill")
sys.path.append("/home/maniskill/examples/baselines/ppo")
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
from phase_13_pointcloud_generalization.pointnet_models import PointNetAgent, extract_robot_frame_pc

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEACHER_CKPT = "/home/maniskill/phase_12_privileged_state_distillation/teacher/privileged_state_teacher.pt"
PHASE13_DIR = "/home/maniskill/phase_13_pointcloud_generalization"
STUDENT_DIR = os.path.join(PHASE13_DIR, "student")
EVAL_DIR = os.path.join(PHASE13_DIR, "eval_results")
VIEWPOINTS_JSON = "/home/maniskill/phase_1_to_3/sequential_viewpoint_training/viewpoints.json"

os.makedirs(STUDENT_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)


def load_privileged_teacher():
    class DummySpace:
        def __init__(self, shape):
            self.shape = shape
    dummy_vec = SimpleNamespace(
        single_observation_space=DummySpace((42,)),
        single_action_space=DummySpace((4,))
    )
    teacher = PrivilegedAgent(dummy_vec).to(DEVICE)
    ckpt = torch.load(TEACHER_CKPT, map_location=DEVICE)
    teacher.load_state_dict(ckpt)
    teacher.eval()
    print(f"[OK] Privileged State Teacher loaded successfully from {TEACHER_CKPT}")
    return teacher


def sample_continuous_spherical_camera(rng: np.random.RandomState):
    """
    Samples a continuous camera pose over the frontal hemisphere:
    - Azimuth in [-70 deg, +70 deg]
    - Elevation in [12 deg, 50 deg]
    - Distance to base in [0.95m, 1.25m]
    """
    az_deg = rng.uniform(-70.0, 70.0)
    el_deg = rng.uniform(12.0, 50.0)
    dist = rng.uniform(0.95, 1.25)

    az_rad = np.radians(az_deg)
    el_rad = np.radians(el_deg)

    # Robot base position in ManiSkill PickCube-v1 world frame
    base_pos = np.array([-0.615, 0.0, 0.0])

    rel_x = dist * np.cos(el_rad) * np.cos(az_rad)
    rel_y = dist * np.cos(el_rad) * np.sin(az_rad)
    rel_z = dist * np.sin(el_rad)

    cam_eye = (base_pos + np.array([rel_x, rel_y, rel_z])).tolist()
    return cam_eye, az_deg, el_deg, dist


def collect_pointcloud_dataset(teacher, num_episodes: int = 1920, batch_envs: int = 32, num_points: int = 512, seed: int = 42):
    print("\n" + "=" * 80)
    print(f" [STAGE 1] COLLECTING {num_episodes:,} EPISODES IN 3D POINT-CLOUD FORMAT")
    print(f" Teacher: Privileged State Oracle (100% success rate)")
    print(f" Points per Frame: {num_points} (Robot Base Frame Metric Coordinates)")
    print("=" * 80)

    total_samples = num_episodes * 50
    buf_pts = torch.empty((total_samples, num_points, 3), dtype=torch.float32)
    buf_proprio = torch.empty((total_samples, 29), dtype=torch.float32)
    buf_actions = torch.empty((total_samples, 4), dtype=torch.float32)

    rng = np.random.RandomState(seed)
    cur_idx = 0
    t0 = time.time()
    num_batches = int(np.ceil(num_episodes / batch_envs))

    # Nominal anchor cameras
    anchor_cameras = [
        ([-0.1774, -0.0147, 0.5768], -4.74, 31.8, 1.0942),  # VP 17
        ([0.4542, 0.0, 0.2326], 0.0, 12.27, 1.0942),        # VP 00
        ([0.0886, -0.5361, 0.6272], -37.2, 34.98, 1.0942),   # VP 20
        ([0.0076, 0.6728, 0.6070], 47.79, 33.69, 1.0942),   # VP 19
    ]

    for b in range(num_batches):
        cur_batch = min(batch_envs, num_episodes - (b * batch_envs))

        if b < len(anchor_cameras):
            base_eye, az, el, dist = anchor_cameras[b]
            jitter = rng.normal(0, 0.015, size=3)
            cam_eye = (np.array(base_eye) + jitter).tolist()
            tag = f"Anchor VP (nominal Az={az:+.1f}°)"
        else:
            cam_eye, az, el, dist = sample_continuous_spherical_camera(rng)
            tag = "Continuous Random"

        env = gym.make(
            "PickCube-v1",
            obs_mode="pointcloud+state",
            control_mode="pd_ee_delta_pos",
            num_envs=batch_envs,
            sensor_cam_eye_pos=cam_eye,
            sensor_cam_target_pos=[-0.1, 0.0, 0.1],
            sensor_configs=dict(width=128, height=128),
        )
        vec_env = ManiSkillVectorEnv(env, batch_envs, ignore_terminations=True, record_metrics=True)
        obs, _ = vec_env.reset(seed=int(seed + b * 100))

        for step in range(50):
            state_42 = obs["state"][:cur_batch]
            proprio_29 = state_42[:, :29]

            # Privileged teacher computes optimal action from 42-dim ground-truth state
            with torch.no_grad():
                act = teacher.get_action(state_42, deterministic=True)
                pts = extract_robot_frame_pc(obs, num_points=num_points)[:cur_batch]

            buf_pts[cur_idx:cur_idx + cur_batch] = pts.cpu()
            buf_proprio[cur_idx:cur_idx + cur_batch] = proprio_29.cpu()
            buf_actions[cur_idx:cur_idx + cur_batch] = act.cpu()
            cur_idx += cur_batch

            if cur_batch < batch_envs:
                act_pad = torch.zeros((batch_envs, 4), device=DEVICE)
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
            print(f"  Batch {b+1:2d}/{num_batches:2d} | [{tag}] Az={az:+5.1f}°, El={el:4.1f}°, Dist={dist:.2f}m | Collected: {cur_idx:,} / {total_samples:,} samples")

    print(f"\n[OK] 3D Point-Cloud dataset collection completed: {cur_idx:,} transitions in {time.time()-t0:.1f}s")
    return buf_pts[:cur_idx], buf_proprio[:cur_idx], buf_actions[:cur_idx]


def train_pointnet_student(buf_pts, buf_proprio, buf_actions, num_epochs=25, batch_size=256):
    print("\n" + "=" * 80)
    print(" [STAGE 2] TRAINING PHASE 13 PointNet STUDENT POLICY")
    print(f" Dataset: {len(buf_actions):,} transitions | Epochs: {num_epochs} | Batch Size: {batch_size}")
    print(" Architecture: Robot-Frame 3D PointNet + Proprioception Fusion")
    print("=" * 80)

    dataset = TensorDataset(buf_pts, buf_proprio, buf_actions)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True, num_workers=0)

    student = PointNetAgent(action_dim=4, proprio_dim=29, point_dim=256).to(DEVICE)
    optimizer = optim.AdamW(student.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)

    best_loss = float("inf")
    best_ckpt_path = os.path.join(STUDENT_DIR, "pointnet_student_best.pt")

    for epoch in range(1, num_epochs + 1):
        student.train()
        total_loss = 0.0
        n_batches = 0
        t0 = time.time()

        for batch_pts, batch_proprio, batch_actions in loader:
            batch_pts = batch_pts.to(DEVICE, non_blocking=True)
            batch_proprio = batch_proprio.to(DEVICE, non_blocking=True)
            batch_actions = batch_actions.to(DEVICE, non_blocking=True)

            pred_action = student.get_action(batch_pts, batch_proprio, deterministic=True)
            loss = F.mse_loss(pred_action, batch_actions)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / n_batches
        dt = time.time() - t0

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "model_state": student.state_dict(),
                "epoch": epoch,
                "best_loss": best_loss,
            }, best_ckpt_path)

        if epoch % 5 == 0 or epoch == 1 or epoch == num_epochs:
            print(f"  Epoch {epoch:2d}/{num_epochs:2d} | Action MSE Loss: {avg_loss:.5f} | lr: {scheduler.get_last_lr()[0]:.2e} | Time: {dt:.1f}s")

    print(f"\n[OK] Distillation finished! Best model saved to {best_ckpt_path} (MSE: {best_loss:.5f})")
    return best_ckpt_path


def evaluate_pointnet_student(student_ckpt: str, num_points: int = 512):
    print("\n" + "=" * 80)
    print(" [STAGE 3] BENCHMARKING PHASE 13 PointNet ON IN-BETWEEN & ANCHOR VIEWPOINTS")
    print(f" Checkpoint: {student_ckpt}")
    print("=" * 80)

    with open(VIEWPOINTS_JSON) as f:
        vps = json.load(f)
    vp_dict = {v["id"]: v for v in vps}

    student = PointNetAgent(action_dim=4, proprio_dim=29, point_dim=256).to(DEVICE)
    ckpt = torch.load(student_ckpt, map_location=DEVICE)
    student.load_state_dict(ckpt.get("model_state", ckpt))
    student.eval()

    eval_vps = [
        # 4 Anchors
        (17, "VP 17 (Front-Mid Anchor)", "trained"),
        (0,  "VP 00 (Front-Low Anchor)", "trained"),
        (20, "VP 20 (Diag-Left Anchor)", "trained"),
        (19, "VP 19 (Diag-Right Anchor)", "trained"),
        # The 9 In-Between Viewpoints
        (10, "VP 10 (In-Between El: VP00-17)", "untrained"),
        (13, "VP 13 (In-Between Az: VP17-20)", "untrained"),
        (14, "VP 14 (In-Between Az: VP17-19)", "untrained"),
        (4,  "VP 04 (In-Between Low-Right)",   "untrained"),
        (7,  "VP 07 (In-Between Low-Left)",    "untrained"),
        (21, "VP 21 (In-Between Mid-Right)",   "untrained"),
        (24, "VP 24 (In-Between High-Mid)",    "untrained"),
        (12, "VP 12 (Flank In-Between R)",     "untrained"),
        (15, "VP 15 (Flank In-Between L)",     "untrained"),
    ]

    results = []
    batch_envs = 50

    for vid, name, vtype in eval_vps:
        meta = vp_dict[vid]
        cam_eye = meta["world_eye"]

        env = gym.make(
            "PickCube-v1",
            obs_mode="pointcloud+state",
            control_mode="pd_ee_delta_pos",
            num_envs=batch_envs,
            sensor_cam_eye_pos=cam_eye,
            sensor_cam_target_pos=[-0.1, 0.0, 0.1],
            sensor_configs=dict(width=128, height=128),
        )
        eval_envs = ManiSkillVectorEnv(env, batch_envs, ignore_terminations=True, record_metrics=True)

        obs, _ = eval_envs.reset(seed=2026 + vid)
        grasped_any = torch.zeros(batch_envs, dtype=torch.bool, device=DEVICE)

        for step in range(50):
            proprio_29 = obs["state"][:, :29]

            with torch.no_grad():
                pts = extract_robot_frame_pc(obs, num_points=num_points)
                act = student.get_action(pts, proprio_29, deterministic=True)

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

        res_item = {
            "id": vid,
            "name": name,
            "type": vtype,
            "azimuth": meta["azimuth_deg"],
            "elevation": meta["elevation_deg"],
            "grasp_rate": grasp_rate,
            "success_rate": succ_rate,
        }
        results.append(res_item)
        tag = "[ANCHOR]" if vtype == "trained" else "[IN-BETWEEN]"
        print(f"  {tag:<12} {name:<32} (Az={meta['azimuth_deg']:5.1f}°, El={meta['elevation_deg']:4.1f}°): Succ={succ_rate:5.1f}% | Grasp={grasp_rate:5.1f}%")

    # Compute Averages
    anchor_succ = np.mean([r["success_rate"] for r in results if r["type"] == "trained"])
    interp_succ = np.mean([r["success_rate"] for r in results if r["type"] == "untrained"])
    interp_grasp = np.mean([r["grasp_rate"] for r in results if r["type"] == "untrained"])

    print("\n" + "=" * 80)
    print(f" PHASE 13 FINAL SUMMARY:")
    print(f"   Anchor Viewpoint Success:     {anchor_succ:5.1f}%")
    print(f"   In-Between Viewpoint Success: {interp_succ:5.1f}%  (Phase 10: 2.9% | Phase 12: 12.9%)")
    print(f"   In-Between Grasp Rate:        {interp_grasp:5.1f}%  (Phase 10: 19.8% | Phase 12: 53.1%)")
    print("=" * 80)

    # Save benchmark JSON
    json_path = os.path.join(EVAL_DIR, "phase13_pointnet_eval_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    # Plot Comparison Figure
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(results))
    width = 0.35

    grasps = [r["grasp_rate"] for r in results]
    succs = [r["success_rate"] for r in results]
    labels = [f"VP{r['id']:02d}\n({r['azimuth']:+.0f}°)" for r in results]
    colors = ["#2e7d32" if r["type"] == "trained" else "#1565c0" for r in results]

    ax.bar(x - width/2, grasps, width, label="Grasp Initiation Rate (%)", color="#90caf9", edgecolor="black", alpha=0.85)
    ax.bar(x + width/2, succs, width, label="Pick-and-Lift Success Rate (%)", color=colors, edgecolor="black")

    ax.set_ylabel("Rate (%)", fontsize=12, fontweight="bold")
    ax.set_title(f"Phase 13 3D Point-Cloud Distillation: In-Between Viewpoint Generalization\n(In-Between Avg Success: {interp_succ:.1f}% | Anchor Avg: {anchor_succ:.1f}%)", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", fontsize=10)

    plot_path = os.path.join(EVAL_DIR, "phase13_pointnet_comparison.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"\n[OK] Benchmark plot saved to {plot_path}")

    return results


def main():
    teacher = load_privileged_teacher()
    buf_pts, buf_proprio, buf_actions = collect_pointcloud_dataset(
        teacher, num_episodes=1920, batch_envs=32, num_points=512
    )
    best_ckpt = train_pointnet_student(
        buf_pts, buf_proprio, buf_actions, num_epochs=25, batch_size=256
    )
    evaluate_pointnet_student(best_ckpt, num_points=512)


if __name__ == "__main__":
    main()
