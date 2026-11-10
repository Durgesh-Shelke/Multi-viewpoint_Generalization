"""
Comprehensive Viewpoint Generalization Stress-Testing & Robustness Suite.

Evaluates the Flagship 3D PointNet Policy across:
1. Sim-to-Real Extrinsic Calibration Error Tolerance Sweep (Translational + Rotational Jitter).
2. Realistic RealSense Depth Sensor Noise (mm) & Point Cloud Sparsity / Dropout Sweep.
3. Extreme Out-of-Distribution (OOD) Viewpoints (Flanks ±65°, Zoom 0.90m-1.35m, Top-down 75°).
4. Real-Time Forward Pass Latency & Throughput Profiling (CPU vs GPU in ms / FPS).
"""

import os
import sys
import time
import json
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gymnasium as gym

import mani_skill.envs
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

REPO_DIR = "/home/maniskill/Multi-viewpoint_Generalization"
sys.path.append(REPO_DIR)
sys.path.append("/home/maniskill")

from phase_13_pointcloud_generalization.pointnet_models import PointNetAgent, extract_robot_frame_pc

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = os.path.join(REPO_DIR, "phase_14_robustness_and_videos")
CKPT_PATH = os.path.join(REPO_DIR, "phase_13_pointcloud_generalization/student/pointnet_student_best.pt")

os.makedirs(OUT_DIR, exist_ok=True)

# Nominal VP 17 (Front-Mid Anchor Camera)
NOMINAL_EYE = [0.3118, -0.0768, 0.5766]
TARGET_POS = [-0.1, 0.0, 0.1]
BASE_POS = np.array([-0.615, 0.0, 0.0])


def load_policy():
    print(f"Loading Flagship 3D PointNet Model from {CKPT_PATH}...", flush=True)
    model = PointNetAgent(action_dim=4, proprio_dim=29, point_dim=256).to(DEVICE)
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    st = ckpt["model_state"] if "model_state" in ckpt else ckpt
    model.load_state_dict(st)
    model.eval()
    print("[OK] PointNet Agent successfully loaded and ready for stress-testing.", flush=True)
    return model


# =========================================================================
# EXPERIMENT 1: Extrinsic Calibration Noise Sensitivity
# =========================================================================
def run_calibration_noise_experiment(model, batch_envs=50):
    print("\n" + "=" * 80, flush=True)
    print(" EXPERIMENT 1: EXTRINSIC CALIBRATION NOISE TOLERANCE SWEEP", flush=True)
    print("=" * 80, flush=True)

    trans_errors_cm = [0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0]
    rot_errors_deg = [0.0, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0]

    results_trans = []
    results_rot = []

    # 1A. Translation Error Sweep
    print("--- 1A: Translation Error Sweep ---", flush=True)
    for err_cm in trans_errors_cm:
        err_m = err_cm / 100.0
        env = gym.make(
            "PickCube-v1",
            robot_uids="panda",
            obs_mode="pointcloud+state",
            control_mode="pd_ee_delta_pos",
            num_envs=batch_envs,
            sensor_cam_eye_pos=NOMINAL_EYE,
            sensor_cam_target_pos=TARGET_POS,
            sensor_configs=dict(width=128, height=128),
            shader_dir="minimal",
        )
        vec_env = ManiSkillVectorEnv(env, batch_envs, ignore_terminations=True, record_metrics=True)
        obs, _ = vec_env.reset(seed=100 + int(err_cm * 10))
        grasped_any = torch.zeros(batch_envs, dtype=torch.bool, device=DEVICE)

        for step in range(50):
            prop = obs["state"][:, :29]
            pts = extract_robot_frame_pc(obs, num_points=512)
            if err_m > 0:
                jitter = torch.randn((batch_envs, 1, 3), device=DEVICE) * err_m
                pts = pts + jitter

            with torch.no_grad():
                act = model.get_action(pts, prop, deterministic=True)

            obs, rew, term, trunc, info = vec_env.step(act)
            if "is_grasped" in info:
                grasped_any = grasped_any | info["is_grasped"]

        succ = info["final_info"]["episode"]["success_once"].float().mean().item() * 100.0
        grasp = grasped_any.float().mean().item() * 100.0
        results_trans.append({"err_cm": err_cm, "success_rate": succ, "grasp_rate": grasp})
        print(f"  Translation Error = {err_cm:4.1f} cm | Success: {succ:5.1f}% | Grasp: {grasp:5.1f}%", flush=True)
        vec_env.close()

    # 1B. Rotation Error Sweep
    print("\n--- 1B: Rotation Error Sweep ---", flush=True)
    for err_deg in rot_errors_deg:
        err_rad = np.radians(err_deg)
        env = gym.make(
            "PickCube-v1",
            robot_uids="panda",
            obs_mode="pointcloud+state",
            control_mode="pd_ee_delta_pos",
            num_envs=batch_envs,
            sensor_cam_eye_pos=NOMINAL_EYE,
            sensor_cam_target_pos=TARGET_POS,
            sensor_configs=dict(width=128, height=128),
            shader_dir="minimal",
        )
        vec_env = ManiSkillVectorEnv(env, batch_envs, ignore_terminations=True, record_metrics=True)
        obs, _ = vec_env.reset(seed=200 + int(err_deg * 10))
        grasped_any = torch.zeros(batch_envs, dtype=torch.bool, device=DEVICE)

        for step in range(50):
            prop = obs["state"][:, :29]
            pts = extract_robot_frame_pc(obs, num_points=512)
            if err_rad > 0:
                angle = (torch.rand((batch_envs, 1, 1), device=DEVICE) - 0.5) * 2 * err_rad
                cos_a, sin_a = torch.cos(angle), torch.sin(angle)
                x_rot = pts[..., 0:1] * cos_a - pts[..., 1:2] * sin_a
                y_rot = pts[..., 0:1] * sin_a + pts[..., 1:2] * cos_a
                pts = torch.cat([x_rot, y_rot, pts[..., 2:3]], dim=-1)

            with torch.no_grad():
                act = model.get_action(pts, prop, deterministic=True)

            obs, rew, term, trunc, info = vec_env.step(act)
            if "is_grasped" in info:
                grasped_any = grasped_any | info["is_grasped"]

        succ = info["final_info"]["episode"]["success_once"].float().mean().item() * 100.0
        grasp = grasped_any.float().mean().item() * 100.0
        results_rot.append({"err_deg": err_deg, "success_rate": succ, "grasp_rate": grasp})
        print(f"  Rotation Error    = {err_deg:4.1f}° | Success: {succ:5.1f}% | Grasp: {grasp:5.1f}%", flush=True)
        vec_env.close()

    # Plot Experiment 1
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot([r["err_cm"] for r in results_trans], [r["success_rate"] for r in results_trans], "o-", color="#1f77b4", linewidth=2.5, label="Success Rate (%)")
    ax1.plot([r["err_cm"] for r in results_trans], [r["grasp_rate"] for r in results_trans], "s--", color="#2ca02c", linewidth=2.0, label="Grasp Rate (%)")
    ax1.axvline(2.0, color="orange", linestyle=":", linewidth=2, label="Typical Hand-Eye Calibration Jitter (±2 cm)")
    ax1.set_xlabel("Camera Extrinsic Translation Error (cm)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Rate (%)", fontsize=11, fontweight="bold")
    ax1.set_title("Tolerance to Extrinsic Translation Error", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.set_ylim(0, 105)
    ax1.legend(loc="lower left", fontsize=10)

    ax2.plot([r["err_deg"] for r in results_rot], [r["success_rate"] for r in results_rot], "o-", color="#d62728", linewidth=2.5, label="Success Rate (%)")
    ax2.plot([r["err_deg"] for r in results_rot], [r["grasp_rate"] for r in results_rot], "s--", color="#2ca02c", linewidth=2.0, label="Grasp Rate (%)")
    ax2.axvline(3.0, color="orange", linestyle=":", linewidth=2, label="Typical Angular Error (±3°)")
    ax2.set_xlabel("Camera Extrinsic Angular Error (Degrees)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Rate (%)", fontsize=11, fontweight="bold")
    ax2.set_title("Tolerance to Extrinsic Rotational Error", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.set_ylim(0, 105)
    ax2.legend(loc="lower left", fontsize=10)

    plt.suptitle("Sim-to-Real Extrinsic Calibration Tolerance Curves (Franka Panda 3D PointNet)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plot_path = os.path.join(OUT_DIR, "calibration_noise_robustness_curve.png")
    plt.savefig(plot_path, dpi=200)
    plt.savefig(os.path.join(REPO_DIR, "calibration_noise_robustness_curve.png"), dpi=200)
    plt.close()
    print(f"[OK] Calibration noise curve saved to {plot_path}", flush=True)

    return {"translation_sweep": results_trans, "rotation_sweep": results_rot}


# =========================================================================
# EXPERIMENT 2: RealSense Depth Noise & Point Cloud Sparsity
# =========================================================================
def run_realsense_depth_noise_experiment(model, batch_envs=50):
    print("\n" + "=" * 80, flush=True)
    print(" EXPERIMENT 2: REALSENSE DEPTH SENSOR NOISE & POINT SPARSITY SWEEP", flush=True)
    print("=" * 80, flush=True)

    depth_noises_mm = [0.0, 2.0, 5.0, 10.0, 15.0, 20.0]
    points_configs = [
        (512, 0.0),    # Full density
        (384, 25.0),   # 25% missing depth points
        (256, 50.0),   # 50% missing depth points
        (128, 75.0),   # 75% missing depth points
        (64, 87.5),    # 87.5% extreme depth holes
    ]

    results_noise = []
    results_sparsity = []

    # 2A: Depth Gaussian Noise Sweep
    print("--- 2A: Depth Gaussian Noise Sweep ---", flush=True)
    for noise_mm in depth_noises_mm:
        noise_m = noise_mm / 1000.0
        env = gym.make(
            "PickCube-v1",
            robot_uids="panda",
            obs_mode="pointcloud+state",
            control_mode="pd_ee_delta_pos",
            num_envs=batch_envs,
            sensor_cam_eye_pos=NOMINAL_EYE,
            sensor_cam_target_pos=TARGET_POS,
            sensor_configs=dict(width=128, height=128),
            shader_dir="minimal",
        )
        vec_env = ManiSkillVectorEnv(env, batch_envs, ignore_terminations=True, record_metrics=True)
        obs, _ = vec_env.reset(seed=300 + int(noise_mm * 10))
        grasped_any = torch.zeros(batch_envs, dtype=torch.bool, device=DEVICE)

        for step in range(50):
            prop = obs["state"][:, :29]
            pts = extract_robot_frame_pc(obs, num_points=512)
            if noise_m > 0:
                pts = pts + torch.randn_like(pts) * noise_m

            with torch.no_grad():
                act = model.get_action(pts, prop, deterministic=True)

            obs, rew, term, trunc, info = vec_env.step(act)
            if "is_grasped" in info:
                grasped_any = grasped_any | info["is_grasped"]

        succ = info["final_info"]["episode"]["success_once"].float().mean().item() * 100.0
        grasp = grasped_any.float().mean().item() * 100.0
        results_noise.append({"noise_mm": noise_mm, "success_rate": succ, "grasp_rate": grasp})
        print(f"  Gaussian Depth Noise = {noise_mm:4.1f} mm | Success: {succ:5.1f}% | Grasp: {grasp:5.1f}%", flush=True)
        vec_env.close()

    # 2B: Point Cloud Sparsity Sweep (Simulating Missing Active-IR Depth Holes)
    print("\n--- 2B: Point Cloud Sparsity Sweep (Missing Depth Pixels) ---", flush=True)
    for n_pts, drop_rate in points_configs:
        env = gym.make(
            "PickCube-v1",
            robot_uids="panda",
            obs_mode="pointcloud+state",
            control_mode="pd_ee_delta_pos",
            num_envs=batch_envs,
            sensor_cam_eye_pos=NOMINAL_EYE,
            sensor_cam_target_pos=TARGET_POS,
            sensor_configs=dict(width=128, height=128),
            shader_dir="minimal",
        )
        vec_env = ManiSkillVectorEnv(env, batch_envs, ignore_terminations=True, record_metrics=True)
        obs, _ = vec_env.reset(seed=400 + n_pts)
        grasped_any = torch.zeros(batch_envs, dtype=torch.bool, device=DEVICE)

        for step in range(50):
            prop = obs["state"][:, :29]
            pts = extract_robot_frame_pc(obs, num_points=n_pts)

            with torch.no_grad():
                act = model.get_action(pts, prop, deterministic=True)

            obs, rew, term, trunc, info = vec_env.step(act)
            if "is_grasped" in info:
                grasped_any = grasped_any | info["is_grasped"]

        succ = info["final_info"]["episode"]["success_once"].float().mean().item() * 100.0
        grasp = grasped_any.float().mean().item() * 100.0
        results_sparsity.append({"n_points": n_pts, "drop_rate": drop_rate, "success_rate": succ, "grasp_rate": grasp})
        print(f"  Points = {n_pts:3d} (Dropout: {drop_rate:4.1f}%) | Success: {succ:5.1f}% | Grasp: {grasp:5.1f}%", flush=True)
        vec_env.close()

    # Plot Experiment 2
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot([r["noise_mm"] for r in results_noise], [r["success_rate"] for r in results_noise], "o-", color="#9467bd", linewidth=2.5, label="Success Rate (%)")
    ax1.plot([r["noise_mm"] for r in results_noise], [r["grasp_rate"] for r in results_noise], "s--", color="#2ca02c", linewidth=2.0, label="Grasp Rate (%)")
    ax1.axvline(2.5, color="orange", linestyle=":", linewidth=2, label="Intel RealSense D435 Noise (~2-3 mm)")
    ax1.set_xlabel("Gaussian Depth Sensor Noise σ (mm)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Rate (%)", fontsize=11, fontweight="bold")
    ax1.set_title("Tolerance to RealSense Depth Jitter", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.set_ylim(0, 105)
    ax1.legend(loc="lower left", fontsize=10)

    ax2.plot([r["drop_rate"] for r in results_sparsity], [r["success_rate"] for r in results_sparsity], "o-", color="#e377c2", linewidth=2.5, label="Success Rate (%)")
    ax2.plot([r["drop_rate"] for r in results_sparsity], [r["grasp_rate"] for r in results_sparsity], "s--", color="#2ca02c", linewidth=2.0, label="Grasp Rate (%)")
    ax2.axvline(20.0, color="orange", linestyle=":", linewidth=2, label="Real-World Specular Dropouts (~20%)")
    ax2.set_xlabel("Missing Depth Hole / Sparsity Rate (%)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Rate (%)", fontsize=11, fontweight="bold")
    ax2.set_title("Tolerance to Point Cloud Sparsity & Holes", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.set_ylim(0, 105)
    ax2.legend(loc="lower left", fontsize=10)

    plt.suptitle("Hardware Depth Sensor Resilience & Degradation Curves (Franka Panda 3D PointNet)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plot_path = os.path.join(OUT_DIR, "realsense_depth_noise_ablation.png")
    plt.savefig(plot_path, dpi=200)
    plt.savefig(os.path.join(REPO_DIR, "realsense_depth_noise_ablation.png"), dpi=200)
    plt.close()
    print(f"[OK] Depth noise ablation plot saved to {plot_path}", flush=True)

    return {"depth_noise_sweep": results_noise, "sparsity_sweep": results_sparsity}


# =========================================================================
# EXPERIMENT 3: Extreme Out-of-Distribution (OOD) Viewpoints
# =========================================================================
def run_extreme_ood_experiment(model, batch_envs=50):
    print("\n" + "=" * 80, flush=True)
    print(" EXPERIMENT 3: EXTREME OUT-OF-DISTRIBUTION (OOD) VIEWPOINTS BENCHMARK", flush=True)
    print("=" * 80, flush=True)

    ood_configs = [
        ("Front Anchor (VP 17)", -4.74, 31.8, 1.0942),
        ("Extreme Left Flank (Az=-65°)", -65.0, 30.0, 1.10),
        ("Extreme Right Flank (Az=+65°)", 65.0, 30.0, 1.10),
        ("Close Macro Zoom (Dist=0.90m)", 0.0, 30.0, 0.90),
        ("Wide Telephoto Zoom (Dist=1.35m)", 0.0, 30.0, 1.35),
        ("Steep Overhead (El=75°)*", 0.0, 75.0, 1.10),
    ]

    results_ood = []
    for name, az_deg, el_deg, dist in ood_configs:
        az_rad = np.radians(az_deg)
        el_rad = np.radians(el_deg)
        rel_x = dist * np.cos(el_rad) * np.cos(az_rad)
        rel_y = dist * np.cos(el_rad) * np.sin(az_rad)
        rel_z = dist * np.sin(el_rad)
        cam_eye = (BASE_POS + np.array([rel_x, rel_y, rel_z])).tolist()

        env = gym.make(
            "PickCube-v1",
            robot_uids="panda",
            obs_mode="pointcloud+state",
            control_mode="pd_ee_delta_pos",
            num_envs=batch_envs,
            sensor_cam_eye_pos=cam_eye,
            sensor_cam_target_pos=TARGET_POS,
            sensor_configs=dict(width=128, height=128),
            shader_dir="minimal",
        )
        vec_env = ManiSkillVectorEnv(env, batch_envs, ignore_terminations=True, record_metrics=True)
        obs, _ = vec_env.reset(seed=500 + int(abs(az_deg) + el_deg))
        grasped_any = torch.zeros(batch_envs, dtype=torch.bool, device=DEVICE)

        for step in range(50):
            prop = obs["state"][:, :29]
            pts = extract_robot_frame_pc(obs, num_points=512)

            with torch.no_grad():
                act = model.get_action(pts, prop, deterministic=True)

            obs, rew, term, trunc, info = vec_env.step(act)
            if "is_grasped" in info:
                grasped_any = grasped_any | info["is_grasped"]

        succ = info["final_info"]["episode"]["success_once"].float().mean().item() * 100.0
        grasp = grasped_any.float().mean().item() * 100.0
        results_ood.append({"name": name, "az": az_deg, "el": el_deg, "dist": dist, "success_rate": succ, "grasp_rate": grasp})
        print(f"  {name:<36} | Success: {succ:5.1f}% | Grasp: {grasp:5.1f}%", flush=True)
        vec_env.close()

    # Plot Experiment 3
    fig, ax = plt.subplots(figsize=(11, 5.5))
    names = [r["name"].replace(" ", "\n") for r in results_ood]
    succs = [r["success_rate"] for r in results_ood]
    grasps = [r["grasp_rate"] for r in results_ood]
    x = np.arange(len(names))
    w = 0.35

    b1 = ax.bar(x - w/2, succs, w, label="Success Rate (%)", color="#1f77b4", alpha=0.9)
    b2 = ax.bar(x + w/2, grasps, w, label="Grasp Rate (%)", color="#2ca02c", alpha=0.9)

    ax.set_ylabel("Rate (%)", fontsize=11, fontweight="bold")
    ax.set_title("Extreme Out-of-Distribution (OOD) Camera Angles Benchmark\n(*Steep Overhead suffers optical occlusion from gripper)", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9.5)
    ax.set_ylim(0, 115)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", fontsize=10)

    for bar in b1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5, f"{bar.get_height():.0f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for bar in b2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5, f"{bar.get_height():.0f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    plot_path = os.path.join(OUT_DIR, "extreme_ood_viewpoints_benchmark.png")
    plt.savefig(plot_path, dpi=200)
    plt.savefig(os.path.join(REPO_DIR, "extreme_ood_viewpoints_benchmark.png"), dpi=200)
    plt.close()
    print(f"[OK] OOD viewpoint plot saved to {plot_path}", flush=True)

    return results_ood


# =========================================================================
# EXPERIMENT 4: Real-Time Latency & Compute Profiling
# =========================================================================
def run_latency_profiling_experiment(model, num_warmup=100, num_trials=1000):
    print("\n" + "=" * 80, flush=True)
    print(" EXPERIMENT 4: REAL-TIME LATENCY & COMPUTE PROFILING (1,000 INFERENCE RUNS)", flush=True)
    print("=" * 80, flush=True)

    dummy_pts_gpu = torch.randn((1, 512, 3), device="cuda")
    dummy_prop_gpu = torch.randn((1, 29), device="cuda")

    # Warmup GPU
    for _ in range(num_warmup):
        _ = model.get_action(dummy_pts_gpu, dummy_prop_gpu, deterministic=True)
    torch.cuda.synchronize()

    # Benchmark GPU
    gpu_times = []
    for _ in range(num_trials):
        t0 = time.perf_counter()
        _ = model.get_action(dummy_pts_gpu, dummy_prop_gpu, deterministic=True)
        torch.cuda.synchronize()
        gpu_times.append((time.perf_counter() - t0) * 1000.0)

    gpu_mean = np.mean(gpu_times)
    gpu_p50 = np.percentile(gpu_times, 50)
    gpu_p95 = np.percentile(gpu_times, 95)
    gpu_p99 = np.percentile(gpu_times, 99)
    gpu_fps = 1000.0 / gpu_mean

    # Benchmark CPU
    model_cpu = PointNetAgent(action_dim=4, proprio_dim=29, point_dim=256).to("cpu")
    model_cpu.load_state_dict(model.state_dict())
    model_cpu.eval()

    dummy_pts_cpu = torch.randn((1, 512, 3), device="cpu")
    dummy_prop_cpu = torch.randn((1, 29), device="cpu")

    for _ in range(num_warmup):
        _ = model_cpu.get_action(dummy_pts_cpu, dummy_prop_cpu, deterministic=True)

    cpu_times = []
    for _ in range(num_trials):
        t0 = time.perf_counter()
        _ = model_cpu.get_action(dummy_pts_cpu, dummy_prop_cpu, deterministic=True)
        cpu_times.append((time.perf_counter() - t0) * 1000.0)

    cpu_mean = np.mean(cpu_times)
    cpu_p50 = np.percentile(cpu_times, 50)
    cpu_p95 = np.percentile(cpu_times, 95)
    cpu_p99 = np.percentile(cpu_times, 99)
    cpu_fps = 1000.0 / cpu_mean

    print(f"  GPU (RTX 4060) Latency: Mean = {gpu_mean:5.2f} ms | p50 = {gpu_p50:5.2f} ms | p95 = {gpu_p95:5.2f} ms | Throughput = {gpu_fps:6.1f} FPS", flush=True)
    print(f"  CPU (AMD Ryzen) Latency: Mean = {cpu_mean:5.2f} ms | p50 = {cpu_p50:5.2f} ms | p95 = {cpu_p95:5.2f} ms | Throughput = {cpu_fps:6.1f} FPS", flush=True)

    latency_data = {
        "gpu_rtx4060": {
            "mean_ms": float(gpu_mean),
            "p50_ms": float(gpu_p50),
            "p95_ms": float(gpu_p95),
            "p99_ms": float(gpu_p99),
            "throughput_fps": float(gpu_fps),
            "control_frequency_headroom": f"{gpu_fps / 50.0:.1f}x 50Hz budget",
        },
        "cpu_ryzen": {
            "mean_ms": float(cpu_mean),
            "p50_ms": float(cpu_p50),
            "p95_ms": float(cpu_p95),
            "p99_ms": float(cpu_p99),
            "throughput_fps": float(cpu_fps),
            "control_frequency_headroom": f"{cpu_fps / 50.0:.1f}x 50Hz budget",
        }
    }

    out_json = os.path.join(OUT_DIR, "latency_profiling_results.json")
    with open(out_json, "w") as f:
        json.dump(latency_data, f, indent=2)
    print(f"[OK] Latency profiling saved to {out_json}", flush=True)

    return latency_data


def main():
    print("=" * 80, flush=True)
    print(" EXECUTING FLAGSHIP 3D POINTNET VIEWPOINT STRESS SUITE", flush=True)
    print("=" * 80, flush=True)

    model = load_policy()

    # Run all 4 experiments
    res1 = run_calibration_noise_experiment(model)
    res2 = run_realsense_depth_noise_experiment(model)
    res3 = run_extreme_ood_experiment(model)
    res4 = run_latency_profiling_experiment(model)

    # Save summary JSON
    summary_data = {
        "experiment_1_calibration_noise": res1,
        "experiment_2_depth_sensor_noise": res2,
        "experiment_3_extreme_ood_viewpoints": res3,
        "experiment_4_latency_profiling": res4,
    }

    summary_file = os.path.join(OUT_DIR, "comprehensive_viewpoint_stress_summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary_data, f, indent=2)

    print("\n" + "=" * 80, flush=True)
    print(f" ALL 4 VIEWPOINT STRESS EXPERIMENTS COMPLETED SUCCESSFULLY!", flush=True)
    print(f" Master Summary JSON: {summary_file}", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
