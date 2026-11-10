"""
Consistent StackCube-v1 Training & Viewpoint Benchmark Suite.

Scales StackCube-v1 to the 96,000 transitions / 25 epochs consistent regime:
- Control Mode: pd_ee_delta_pose (7-DoF delta pose)
- Teacher: Verified 100.0% PPO Oracle (/root/.maniskill/demos/StackCube-v1/rl/ppo_pd_ee_delta_pose_ckpt.pt)
- Dataset: 1,920 episodes (96,000 transitions) across continuous spherical dome
- Optimization: 25 epochs with AdamW + Cosine Annealing (lr: 5e-4 to 1e-5)
- Perception: 512-point PointNet in robot base frame + 25-dim proprioception
- Evaluation: Standardized 13 viewpoints with 50 evaluation episodes per viewpoint
"""

import os
import sys
import time
import json
import gc
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import gymnasium as gym
import mani_skill.envs
from mani_skill.utils import sapien_utils
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

REPO_DIR = "/home/maniskill/Multi-viewpoint_Generalization"
sys.path.append(REPO_DIR)
sys.path.append(os.path.join(REPO_DIR, "phase_13_pointcloud_generalization"))
sys.path.append(os.path.join(REPO_DIR, "phase_18_multitask_extended"))

from pointnet_models import PointNetAgent, extract_robot_frame_pc
from train_4tasks_suite import load_ppo_teacher, generate_master_7task_plot

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = os.path.join(REPO_DIR, "phase_18_multitask_extended")
CKPT_DIR = os.path.join(OUT_DIR, "checkpoints")
EVAL_DIR = os.path.join(OUT_DIR, "eval_results")
PLOT_DIR = os.path.join(OUT_DIR, "plots")
VIEWPOINTS_JSON = os.path.join(REPO_DIR, "interpolated_viewpoints_results.json")

os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

BASE_POS = np.array([-0.615, 0.0, 0.0])
TARGET_POS = [0.0, 0.0, 0.05]
TEACHER_CKPT = "/root/.maniskill/demos/StackCube-v1/rl/ppo_pd_ee_delta_pose_ckpt.pt"


def sample_continuous_spherical_camera(rng: np.random.RandomState):
    az_deg = rng.uniform(-70.0, 70.0)
    el_deg = rng.uniform(12.0, 50.0)
    dist = rng.uniform(0.95, 1.25)

    az_rad = np.radians(az_deg)
    el_rad = np.radians(el_deg)
    rel_x = dist * np.cos(el_rad) * np.cos(az_rad)
    rel_y = dist * np.cos(el_rad) * np.sin(az_rad)
    rel_z = dist * np.sin(el_rad)

    cam_eye = (BASE_POS + np.array([rel_x, rel_y, rel_z])).tolist()
    return cam_eye, az_deg, el_deg, dist


def load_viewpoint_catalog():
    with open(VIEWPOINTS_JSON, "r") as f:
        data = json.load(f)
    return data["catalog_results"]


# =========================================================================
# STAGE 1: Data Collection across Continuous Spherical Viewpoints (96K)
# =========================================================================
def collect_dataset(num_episodes: int = 1920, batch_envs: int = 32, max_steps: int = 50, seed: int = 42):
    print("\n" + "=" * 80, flush=True)
    print(f" [STAGE 1] COLLECTING {num_episodes:,} EPISODES ({num_episodes * max_steps:,} TRANSITIONS) FOR StackCube-v1", flush=True)
    print(" Continuous Random Viewpoint Sampling across Frontal Hemisphere Dome", flush=True)
    print(" Control Mode: pd_ee_delta_pose (7-dim) | Proprio Dim: 25 | State Dim: 48", flush=True)
    print("=" * 80, flush=True)

    teacher = load_ppo_teacher(TEACHER_CKPT, state_dim=48, action_dim=7)

    total_samples = num_episodes * max_steps
    buf_pts = torch.empty((total_samples, 512, 3), dtype=torch.float32)
    buf_proprio = torch.empty((total_samples, 25), dtype=torch.float32)
    buf_actions = torch.empty((total_samples, 7), dtype=torch.float32)

    rng = np.random.RandomState(seed)
    cur_idx = 0
    t0 = time.time()
    num_batches = int(np.ceil(num_episodes / batch_envs))

    anchor_cameras = [
        ([0.3118, -0.0768, 0.5766], -4.74, 31.8, 1.0942),  # VP 17
        ([0.4542, 0.0, 0.2326], 0.0, 12.27, 1.0942),       # VP 00
        ([0.0992, -0.542, 0.6272], -37.2, 34.98, 1.0942),  # VP 20
        ([-0.0033, 0.6743, 0.607], 47.79, 33.69, 1.0942),  # VP 19
    ]

    for b in range(num_batches):
        cur_batch = min(batch_envs, num_episodes - (b * batch_envs))

        if b < len(anchor_cameras):
            base_eye, az, el, dist = anchor_cameras[b]
            jitter = rng.normal(0, 0.015, size=3)
            cam_eye = (np.array(base_eye) + jitter).tolist()
            tag = "Anchor View"
        else:
            cam_eye, az, el, dist = sample_continuous_spherical_camera(rng)
            tag = "Continuous Dome"

        cam_pose = sapien_utils.look_at(eye=cam_eye, target=TARGET_POS)
        sensor_cfg = dict(base_camera=dict(pose=cam_pose, width=128, height=128))

        env = gym.make(
            "StackCube-v1",
            robot_uids="panda",
            obs_mode="pointcloud+state",
            control_mode="pd_ee_delta_pose",
            num_envs=batch_envs,
            sensor_configs=sensor_cfg,
            shader_dir="minimal",
        )
        vec_env = ManiSkillVectorEnv(env, batch_envs, ignore_terminations=True, record_metrics=True)
        obs, _ = vec_env.reset(seed=int(seed + b * 100))

        for step in range(max_steps):
            state_vec = obs["state"][:, :48].to(DEVICE)
            proprio_25 = obs["state"][:, :25]

            with torch.no_grad():
                act = teacher.get_action(state_vec, deterministic=True)
                pts = extract_robot_frame_pc(obs, num_points=512)[:cur_batch]

            buf_pts[cur_idx:cur_idx + cur_batch] = pts.cpu()
            buf_proprio[cur_idx:cur_idx + cur_batch] = proprio_25[:cur_batch].cpu()
            buf_actions[cur_idx:cur_idx + cur_batch] = act[:cur_batch].cpu()
            cur_idx += cur_batch

            if cur_batch < batch_envs:
                act_pad = torch.zeros((batch_envs, 7), device=DEVICE)
                act_pad[:cur_batch] = act[:cur_batch]
                obs, _, _, _, _ = vec_env.step(act_pad)
            else:
                obs, _, _, _, _ = vec_env.step(act)

        vec_env.close()
        del vec_env
        del env
        gc.collect()
        torch.cuda.empty_cache()

        if (b + 1) % 10 == 0 or b == 0 or (b + 1) == num_batches:
            dt = time.time() - t0
            print(f"  Batch {b+1:2d}/{num_batches:2d} | [{tag}] Az={az:+5.1f}°, El={el:4.1f}°, Dist={dist:.2f}m | Samples: {cur_idx:,} / {total_samples:,} ({dt:.1f}s)", flush=True)

    print(f"[OK] Consistent dataset for StackCube-v1 collected: {cur_idx:,} samples in {time.time()-t0:.1f}s", flush=True)
    return buf_pts[:cur_idx], buf_proprio[:cur_idx], buf_actions[:cur_idx]


# =========================================================================
# STAGE 2: PointNet Policy Training (25 Epochs, Cosine Annealing)
# =========================================================================
def train_pointnet(buf_pts, buf_proprio, buf_actions, num_epochs: int = 25, batch_size: int = 256):
    print("\n" + "=" * 80, flush=True)
    print(f" [STAGE 2] TRAINING 3D PointNet POLICY FOR StackCube-v1 (CONSISTENT REGIME)", flush=True)
    print(f" Transitions: {len(buf_actions):,} | Epochs: {num_epochs} | Batch Size: {batch_size} | Action Dim: 7", flush=True)
    print("=" * 80, flush=True)

    dataset = TensorDataset(buf_pts, buf_proprio, buf_actions)
    bs = min(batch_size, len(buf_actions))
    loader = DataLoader(dataset, batch_size=bs, shuffle=True, drop_last=(len(buf_actions) > bs), pin_memory=True, num_workers=0)

    student = PointNetAgent(action_dim=7, proprio_dim=25, point_dim=256).to(DEVICE)
    optimizer = optim.AdamW(student.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)

    best_loss = float("inf")
    ckpt_path = os.path.join(CKPT_DIR, "pointnet_stackcube-v1_best.pt")

    for epoch in range(1, num_epochs + 1):
        student.train()
        total_loss = 0.0
        n_batches = 0
        t0 = time.time()

        for batch_pts, batch_proprio, batch_act in loader:
            batch_pts = batch_pts.to(DEVICE, non_blocking=True)
            batch_proprio = batch_proprio.to(DEVICE, non_blocking=True)
            batch_act = batch_act.to(DEVICE, non_blocking=True)

            pred_action = student.get_action(batch_pts, batch_proprio, deterministic=True)
            loss = F.mse_loss(pred_action, batch_act)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        if n_batches > 0:
            scheduler.step()
            avg_loss = total_loss / n_batches
        else:
            avg_loss = 0.0
        dt = time.time() - t0

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(student.state_dict(), ckpt_path)

        if epoch % 5 == 0 or epoch == 1 or epoch == num_epochs:
            cur_lr = scheduler.get_last_lr()[0]
            print(f"  Epoch {epoch:2d}/{num_epochs:2d} | Train MSE Loss: {avg_loss:.6f} | Best: {best_loss:.6f} | lr: {cur_lr:.2e} | ({dt:.1f}s)", flush=True)

    print(f"[OK] Training complete. Best checkpoint saved to {ckpt_path} (MSE: {best_loss:.6f})", flush=True)
    return ckpt_path


# =========================================================================
# STAGE 3: 13-Viewpoint Continuous Benchmark
# =========================================================================
def evaluate_13_viewpoints(ckpt_path: str, num_eval_episodes: int = 50, max_steps: int = 50):
    print("\n" + "=" * 80, flush=True)
    print(" [STAGE 3] BENCHMARKING StackCube-v1 ACROSS 13 STANDARDIZED VIEWPOINTS", flush=True)
    print(f" 4 Anchor Cameras + 9 Interpolated In-Between Angles | {num_eval_episodes} Episodes/View", flush=True)
    print("=" * 80, flush=True)

    viewpoint_catalog = load_viewpoint_catalog()
    policy = PointNetAgent(action_dim=7, proprio_dim=25, point_dim=256).to(DEVICE)
    policy.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    policy.eval()

    eval_results = []
    t_bench = time.time()

    for item in viewpoint_catalog:
        vid = item["id"]
        label = item["label"]
        vtype = item["type"]
        cam_eye = item["cam_eye"]
        az = item["az"]
        el = item["el"]

        cam_pose = sapien_utils.look_at(eye=cam_eye, target=TARGET_POS)
        sensor_cfg = dict(base_camera=dict(pose=cam_pose, width=128, height=128))

        env = gym.make(
            "StackCube-v1",
            robot_uids="panda",
            obs_mode="pointcloud+state",
            control_mode="pd_ee_delta_pose",
            num_envs=num_eval_episodes,
            sensor_configs=sensor_cfg,
            shader_dir="minimal",
        )
        vec_env = ManiSkillVectorEnv(env, num_eval_episodes, ignore_terminations=True, record_metrics=True)
        obs, _ = vec_env.reset(seed=1000 + vid)

        for step in range(max_steps):
            proprio = obs["state"][:, :25].to(DEVICE)
            with torch.no_grad():
                pts = extract_robot_frame_pc(obs, num_points=512)
                act = policy.get_action(pts, proprio, deterministic=True)
            obs, rew, term, trunc, info = vec_env.step(act)

        succ = info["final_info"]["episode"]["success_once"].float().mean().item() * 100.0

        vec_env.close()
        del vec_env
        del env
        gc.collect()
        torch.cuda.empty_cache()

        res_entry = {
            "vp_id": vid,
            "vp_name": label,
            "vp_type": "anchor" if vtype == "trained" else "in-between",
            "azimuth_deg": az,
            "elevation_deg": el,
            "cam_eye": cam_eye,
            "success_rate": succ,
        }
        eval_results.append(res_entry)
        tag = "[ANCHOR]" if vtype == "trained" else "[IN-BETWEEN]"
        print(f"  {tag:<12} {label:<32} (Az={az:+5.1f}°, El={el:4.1f}°): Success = {succ:5.1f}%", flush=True)

    anchor_sr = float(np.mean([r["success_rate"] for r in eval_results if r["vp_type"] == "anchor"]))
    interp_sr = float(np.mean([r["success_rate"] for r in eval_results if r["vp_type"] != "anchor"]))
    overall_sr = float(np.mean([r["success_rate"] for r in eval_results]))
    gen_gap = anchor_sr - interp_sr

    print("-" * 80, flush=True)
    print(" Summary for StackCube-v1 (Consistent Regime):", flush=True)
    print(f"  Anchor Views Success Rate:        {anchor_sr:5.1f}%", flush=True)
    print(f"  In-Between Views Success Rate:    {interp_sr:5.1f}%", flush=True)
    print(f"  Overall 13-View Success Rate:     {overall_sr:5.1f}%", flush=True)
    print(f"  Generalization Gap (Anchor - IB): {gen_gap:5.1f}%", flush=True)
    print(f" Benchmark completed in {time.time()-t_bench:.1f}s", flush=True)
    print("=" * 80, flush=True)

    summary = {
        "env_id": "StackCube-v1",
        "control_mode": "pd_ee_delta_pose",
        "overall_success_rate": overall_sr,
        "anchor_success_rate": anchor_sr,
        "inbetween_success_rate": interp_sr,
        "generalization_gap": gen_gap,
        "results": eval_results,
    }

    # Save JSON
    json_path = os.path.join(EVAL_DIR, "stackcube-v1_viewpoint_benchmark.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[OK] Evaluation JSON saved to {json_path}", flush=True)

    generate_task_plot(eval_results, anchor_sr, interp_sr, overall_sr, gen_gap)
    return summary


def generate_task_plot(results, anchor_sr, interp_sr, overall_sr, gen_gap):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [3, 1]})

    v_ids = [r["vp_id"] for r in results]
    labels = [f"VP {r['vp_id']}\n{r['vp_name'].split('(')[0].strip()}" for r in results]
    rates = [r["success_rate"] for r in results]
    types = [r["vp_type"] for r in results]
    colors = ["#1f77b4" if t == "anchor" else "#ff7f0e" for t in types]

    bars = ax1.bar(range(len(v_ids)), rates, color=colors, alpha=0.9, width=0.65)
    ax1.set_xticks(range(len(v_ids)))
    ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax1.set_ylabel("Success Rate (%)", fontsize=11, fontweight="bold")
    ax1.set_ylim(0, 115)
    ax1.grid(axis="y", linestyle=":", alpha=0.6)
    ax1.set_title("StackCube-v1 — Continuous Viewpoint Invariance (Consistent 96K Regime)\n4 Trained Anchors (Blue) vs 9 Continuous In-Between Angles (Orange)", fontsize=11, fontweight="bold")

    for bar in bars:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + 1.5, f"{h:.0f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#1f77b4", label="Anchor Cameras (Trained)"),
        Patch(facecolor="#ff7f0e", label="In-Between Angles (Novel OOD)")
    ]
    ax1.legend(handles=legend_elements, loc="upper right", fontsize=9)

    # Summary bar plot
    cats = ["Anchor\n(4 Views)", "In-Between\n(9 Views)", "Overall\n(13 Views)"]
    vals = [anchor_sr, interp_sr, overall_sr]
    cat_colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    sbars = ax2.bar(cats, vals, color=cat_colors, width=0.55, alpha=0.9)
    ax2.set_ylim(0, 115)
    ax2.set_ylabel("Success Rate (%)", fontsize=10, fontweight="bold")
    ax2.set_title(f"Generalization Gap: {gen_gap:+.1f}%", fontsize=10, fontweight="bold", color="#d62728" if gen_gap > 10 else "#2ca02c")
    ax2.grid(axis="y", linestyle=":", alpha=0.6)

    for bar in sbars:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, h + 1.5, f"{h:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    plot_path = os.path.join(PLOT_DIR, "stackcube-v1_viewpoint_benchmark.png")
    plt.savefig(plot_path, dpi=200)
    plt.savefig(os.path.join(REPO_DIR, "stackcube-v1_viewpoint_benchmark.png"), dpi=200)
    plt.close()
    print(f"[OK] StackCube benchmark plot saved to {plot_path}", flush=True)


def update_master_suite(summary):
    summary_file = os.path.join(EVAL_DIR, "suite_4tasks_summary.json")
    if os.path.exists(summary_file):
        with open(summary_file, "r") as f:
            all_summary = json.load(f)
    else:
        all_summary = {}

    all_summary["StackCube-v1"] = {
        "env_id": "StackCube-v1",
        "overall_sr": summary["overall_success_rate"],
        "anchor_sr": summary["anchor_success_rate"],
        "inbetween_sr": summary["inbetween_success_rate"],
        "gen_gap": summary["generalization_gap"],
    }

    with open(summary_file, "w") as f:
        json.dump(all_summary, f, indent=2)

    generate_master_7task_plot(all_summary)


def main():
    t0 = time.time()
    buf_pts, buf_proprio, buf_actions = collect_dataset(num_episodes=1920, batch_envs=32, max_steps=50)
    ckpt_path = train_pointnet(buf_pts, buf_proprio, buf_actions, num_epochs=25, batch_size=256)
    del buf_pts, buf_proprio, buf_actions
    gc.collect()
    torch.cuda.empty_cache()

    summary = evaluate_13_viewpoints(ckpt_path, num_eval_episodes=50, max_steps=50)
    update_master_suite(summary)
    print(f"\n[ALL DONE] StackCube pipeline completed in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
