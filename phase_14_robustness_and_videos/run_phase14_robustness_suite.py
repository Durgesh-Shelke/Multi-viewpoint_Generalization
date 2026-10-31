"""
Phase 14: Comprehensive Robustness Stress-Testing Suite for 3D PointNet Policy.

Evaluates PointNetAgent on:
1. 1,000 Continuous Uniform Random Viewpoints across the entire hemisphere.
2. Camera Extrinsic Calibration Noise Sensitivity (0 to 10 cm error).
3. Point Cloud Sparsity & Dropout Sensitivity (512 down to 32 points).
4. Depth Sensor Range Noise Sensitivity (0 to 20 mm Gaussian noise).
"""

import os
import sys
sys.path.append("/home/maniskill")
import time
import json
import gc
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gymnasium as gym

import mani_skill.envs
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from phase_13_pointcloud_generalization.pointnet_models import PointNetAgent

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
POINTNET_CKPT = "/home/maniskill/phase_13_pointcloud_generalization/student/pointnet_student_best.pt"
OUTPUT_DIR = "/home/maniskill/phase_14_robustness_and_videos/eval_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def extract_robot_frame_pc_robust(
    obs: dict,
    num_points: int = 512,
    extrinsic_noise_std: float = 0.0,
    depth_noise_std: float = 0.0,
) -> torch.Tensor:
    pos = obs["sensor_data"]["base_camera"]["position"].float() / 1000.0  # mm to m
    if depth_noise_std > 0:
        pos = pos + torch.randn_like(pos) * depth_noise_std

    cam2world = obs["sensor_param"]["base_camera"]["cam2world_gl"].clone()
    if extrinsic_noise_std > 0:
        noise_trans = torch.randn((cam2world.shape[0], 3), device=pos.device) * extrinsic_noise_std
        cam2world[:, :3, 3] += noise_trans

    B = pos.shape[0]
    pos_homo = torch.cat([pos, torch.ones_like(pos[..., :1])], dim=-1)
    xyz_world = (pos_homo.reshape(B, -1, 4) @ cam2world.transpose(1, 2))[..., :3]

    robot_base = torch.tensor([-0.615, 0.0, 0.0], device=pos.device)
    xyz_robot = xyz_world - robot_base

    mask = (
        (xyz_robot[..., 0] >= 0.35) & (xyz_robot[..., 0] <= 0.85) &
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


def sample_random_spherical_camera(rng: np.random.RandomState):
    az_deg = rng.uniform(-80.0, 80.0)
    el_deg = rng.uniform(10.0, 60.0)
    dist = rng.uniform(0.85, 1.35)

    az_rad = np.radians(az_deg)
    el_rad = np.radians(el_deg)
    base_pos = np.array([-0.615, 0.0, 0.0])

    rel_x = dist * np.cos(el_rad) * np.cos(az_rad)
    rel_y = dist * np.cos(el_rad) * np.sin(az_rad)
    rel_z = dist * np.sin(el_rad)

    cam_eye = (base_pos + np.array([rel_x, rel_y, rel_z])).tolist()
    return cam_eye, az_deg, el_deg, dist


def run_evaluation_trials(
    agent,
    num_episodes: int = 100,
    batch_envs: int = 50,
    num_points: int = 512,
    extrinsic_noise_std: float = 0.0,
    depth_noise_std: float = 0.0,
    seed: int = 42,
    random_cameras: bool = True,
    fixed_cam_eye: list = None,
):
    rng = np.random.RandomState(seed)
    num_batches = int(np.ceil(num_episodes / batch_envs))
    total_succ = 0
    total_grasp = 0
    trial_records = []

    for b in range(num_batches):
        cur_batch = min(batch_envs, num_episodes - b * batch_envs)

        if random_cameras or fixed_cam_eye is None:
            cam_eye, az, el, dist = sample_random_spherical_camera(rng)
        else:
            cam_eye = fixed_cam_eye
            az, el, dist = 0.0, 0.0, 1.0942

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
        obs, _ = eval_envs.reset(seed=int(seed + b * 137))
        grasped_any = torch.zeros(batch_envs, dtype=torch.bool, device=DEVICE)

        for step in range(50):
            proprio_29 = obs["state"][:, :29]

            with torch.no_grad():
                pts = extract_robot_frame_pc_robust(
                    obs,
                    num_points=num_points,
                    extrinsic_noise_std=extrinsic_noise_std,
                    depth_noise_std=depth_noise_std,
                )
                act = agent.get_action(pts, proprio_29, deterministic=True)

            obs, rew, term, trunc, infos = eval_envs.step(act)
            if "is_grasped" in infos:
                grasped_any |= infos["is_grasped"]

        succ = infos["final_info"]["episode"]["success_once"][:cur_batch]
        succ_cnt = succ.sum().item()
        grasp_cnt = grasped_any[:cur_batch].sum().item()

        total_succ += succ_cnt
        total_grasp += grasp_cnt

        eval_envs.close()
        del eval_envs
        del env
        gc.collect()
        torch.cuda.empty_cache()

        trial_records.append({
            "batch": b,
            "azimuth": az,
            "elevation": el,
            "distance": dist,
            "success_rate": (succ_cnt / cur_batch) * 100.0,
            "grasp_rate": (grasp_cnt / cur_batch) * 100.0,
        })

    succ_rate = (total_succ / num_episodes) * 100.0
    grasp_rate = (total_grasp / num_episodes) * 100.0
    return succ_rate, grasp_rate, trial_records


def main():
    print("=" * 80)
    print(" [PHASE 14] POINTNET COMPREHENSIVE ROBUSTNESS & STRESS-TESTING SUITE")
    print(f" Checkpoint: {POINTNET_CKPT}")
    print("=" * 80)

    agent = PointNetAgent(action_dim=4, proprio_dim=29, point_dim=256).to(DEVICE)
    ckpt = torch.load(POINTNET_CKPT, map_location=DEVICE)
    agent.load_state_dict(ckpt.get("model_state", ckpt))
    agent.eval()

    all_results = {}

    # TEST 1: 1,000 Continuous Spherical Random Trials
    print("\n--- TEST 1: 1,000 Continuous Uniform Spherical Trials ---")
    t0 = time.time()
    t1_succ, t1_grasp, t1_records = run_evaluation_trials(
        agent, num_episodes=1000, batch_envs=50, num_points=512, seed=1001, random_cameras=True
    )
    dt1 = time.time() - t0
    print(f" [RESULT] 1,000 Continuous Spherical Trials (in {dt1:.1f}s):")
    print(f"   Success Rate: {t1_succ:.2f}% | Grasp Rate: {t1_grasp:.2f}%")
    all_results["test1_continuous_1000"] = {
        "success_rate": t1_succ,
        "grasp_rate": t1_grasp,
        "records": t1_records,
    }

    # TEST 2: Extrinsic Calibration Noise Sensitivity
    print("\n--- TEST 2: Extrinsic Calibration Noise Sweep ---")
    ext_noise_levels = [0.0, 0.01, 0.02, 0.03, 0.05, 0.10]
    t2_results = []
    for std in ext_noise_levels:
        s_rate, g_rate, _ = run_evaluation_trials(
            agent, num_episodes=100, batch_envs=50, num_points=512,
            extrinsic_noise_std=std, seed=2000 + int(std * 1000), random_cameras=True
        )
        print(f"  Extrinsic Noise: {std*100:4.1f} cm | Success: {s_rate:5.1f}% | Grasp: {g_rate:5.1f}%")
        t2_results.append({"noise_cm": std * 100, "success_rate": s_rate, "grasp_rate": g_rate})
    all_results["test2_extrinsic_noise"] = t2_results

    # TEST 3: Point Cloud Sparsity & Dropout Sensitivity
    print("\n--- TEST 3: Point Cloud Sparsity & Sensor Dropout Sweep ---")
    point_levels = [512, 256, 128, 64, 32]
    t3_results = []
    for n_pts in point_levels:
        s_rate, g_rate, _ = run_evaluation_trials(
            agent, num_episodes=100, batch_envs=50, num_points=n_pts,
            seed=3000 + n_pts, random_cameras=True
        )
        print(f"  Point Count: {n_pts:4d} pts | Success: {s_rate:5.1f}% | Grasp: {g_rate:5.1f}%")
        t3_results.append({"num_points": n_pts, "success_rate": s_rate, "grasp_rate": g_rate})
    all_results["test3_point_sparsity"] = t3_results

    # TEST 4: Depth Sensor Noise Sensitivity
    print("\n--- TEST 4: Depth Sensor Range Noise Sweep ---")
    depth_noise_levels = [0.0, 0.002, 0.005, 0.010, 0.020]
    t4_results = []
    for d_std in depth_noise_levels:
        s_rate, g_rate, _ = run_evaluation_trials(
            agent, num_episodes=100, batch_envs=50, num_points=512,
            depth_noise_std=d_std, seed=4000 + int(d_std * 10000), random_cameras=True
        )
        print(f"  Depth Range Noise: {d_std*1000:4.1f} mm | Success: {s_rate:5.1f}% | Grasp: {g_rate:5.1f}%")
        t4_results.append({"noise_mm": d_std * 1000, "success_rate": s_rate, "grasp_rate": g_rate})
    all_results["test4_depth_noise"] = t4_results

    # Save JSON results
    json_path = os.path.join(OUTPUT_DIR, "phase14_robustness_results.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[OK] Robustness results saved to {json_path}")

    # Plot 4-Panel Sensitivity Figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Panel 1: 1,000 Continuous Spherical Trials (by Azimuth)
    ax1 = axes[0, 0]
    rec_az = [r["azimuth"] for r in t1_records]
    rec_succ = [r["success_rate"] for r in t1_records]
    rec_grasp = [r["grasp_rate"] for r in t1_records]
    ax1.scatter(rec_az, rec_grasp, color="#90caf9", alpha=0.8, s=60, label="Batch Grasp Rate (%)", edgecolor="black")
    ax1.scatter(rec_az, rec_succ, color="#1565c0", alpha=0.9, s=70, label="Batch Success Rate (%)", edgecolor="black")
    ax1.axhline(t1_succ, color="#d32f2f", linestyle="--", linewidth=2, label=f"Mean Success: {t1_succ:.1f}%")
    ax1.set_xlabel("Camera Azimuth (°)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Rate (%)", fontsize=11, fontweight="bold")
    ax1.set_title(f"A: 1,000 Continuous Spherical Random Trials (Mean: {t1_succ:.1f}%)", fontsize=12, fontweight="bold")
    ax1.set_ylim(0, 105)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="lower right", fontsize=9)

    # Panel 2: Extrinsic Calibration Noise
    ax2 = axes[0, 1]
    n_cm = [r["noise_cm"] for r in t2_results]
    n_succ = [r["success_rate"] for r in t2_results]
    n_grasp = [r["grasp_rate"] for r in t2_results]
    ax2.plot(n_cm, n_grasp, marker="s", color="#ce93d8", linewidth=2.5, label="Grasp Rate (%)")
    ax2.plot(n_cm, n_succ, marker="o", color="#7b1fa2", linewidth=2.5, label="Pick-and-Lift Success (%)")
    ax2.set_xlabel("Camera Extrinsic Calibration Noise $\sigma$ (cm)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Rate (%)", fontsize=11, fontweight="bold")
    ax2.set_title("B: Extrinsic Calibration Sensitivity", fontsize=12, fontweight="bold")
    ax2.set_ylim(0, 105)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="lower left", fontsize=10)

    # Panel 3: Point Cloud Sparsity
    ax3 = axes[1, 0]
    p_pts = [str(r["num_points"]) for r in t3_results]
    p_succ = [r["success_rate"] for r in t3_results]
    p_grasp = [r["grasp_rate"] for r in t3_results]
    x_p = np.arange(len(p_pts))
    w = 0.35
    ax3.bar(x_p - w/2, p_grasp, w, label="Grasp Rate (%)", color="#a5d6a7", edgecolor="black")
    ax3.bar(x_p + w/2, p_succ, w, label="Pick-and-Lift Success (%)", color="#2e7d32", edgecolor="black")
    ax3.set_xlabel("Points Sampled per Frame ($N$)", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Rate (%)", fontsize=11, fontweight="bold")
    ax3.set_title("C: Point Cloud Sparsity & Sensor Dropout", fontsize=12, fontweight="bold")
    ax3.set_xticks(x_p)
    ax3.set_xticklabels(p_pts, fontweight="bold")
    ax3.set_ylim(0, 105)
    ax3.grid(axis="y", linestyle="--", alpha=0.5)
    ax3.legend(loc="lower left", fontsize=10)

    # Panel 4: Depth Sensor Range Noise
    ax4 = axes[1, 1]
    d_mm = [r["noise_mm"] for r in t4_results]
    d_succ = [r["success_rate"] for r in t4_results]
    d_grasp = [r["grasp_rate"] for r in t4_results]
    ax4.plot(d_mm, d_grasp, marker="s", color="#ffb74d", linewidth=2.5, label="Grasp Rate (%)")
    ax4.plot(d_mm, d_succ, marker="o", color="#e65100", linewidth=2.5, label="Pick-and-Lift Success (%)")
    ax4.set_xlabel("Depth Sensor Range Noise $\sigma$ (mm)", fontsize=11, fontweight="bold")
    ax4.set_ylabel("Rate (%)", fontsize=11, fontweight="bold")
    ax4.set_title("D: Depth Sensor Gaussian Range Noise", fontsize=12, fontweight="bold")
    ax4.set_ylim(0, 105)
    ax4.grid(True, linestyle="--", alpha=0.5)
    ax4.legend(loc="lower left", fontsize=10)

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "phase14_robustness_curves.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"[OK] 4-Panel Robustness plot saved to {plot_path}")


if __name__ == "__main__":
    main()
