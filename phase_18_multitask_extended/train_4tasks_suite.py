"""
Phase 18: Multi-Task Viewpoint Generalization Extended Suite (4 Tasks)
----------------------------------------------------------------------
Tasks:
  1. StackCube-v1       (Multi-Object Precision Stacking - PPO Teacher)
  2. PokeCube-v1        (Planar Impulsive Interaction - PPO Teacher)
  3. PegInsertionSide-v1(High-Tolerance Sub-Centimeter Assembly - Demos)
  4. PlugCharger-v1     (Peg-in-Hole Precision Insertion - Motion Planning Demos)

Outputs:
  - Checkpoints: phase_18_multitask_extended/checkpoints/
  - Eval Results: phase_18_multitask_extended/eval_results/
  - Plots: phase_18_multitask_extended/plots/
"""

import os
import sys
import time
import json
import gc
import h5py
import argparse
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
sys.path.append("/home/maniskill")
sys.path.append("/home/maniskill/examples/baselines/ppo")

from phase_13_pointcloud_generalization.pointnet_models import PointNetAgent, extract_robot_frame_pc

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

# PPO Actor helper
class MLPTeacher(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.actor_mean = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, action_dim)
        )
    def forward(self, x):
        return self.actor_mean(x)
    def get_action(self, x, deterministic=True):
        return self.actor_mean(x)


def load_ppo_teacher(ckpt_path: str, state_dim: int, action_dim: int):
    teacher = MLPTeacher(state_dim, action_dim).to(DEVICE)
    raw = torch.load(ckpt_path, map_location=DEVICE)
    weights = {k: v for k, v in raw.items() if k.startswith("actor_mean.")}
    teacher.load_state_dict(weights)
    teacher.eval()
    print(f"[OK] Loaded PPO Teacher: state_dim={state_dim}, action_dim={action_dim} from {ckpt_path}", flush=True)
    return teacher


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
# DATA COLLECTION: Interactive PPO Teacher (StackCube & PokeCube)
# =========================================================================
def collect_dataset_from_teacher(env_id: str, control_mode: str, teacher: nn.Module, state_dim: int, action_dim: int, target_pos, num_episodes: int = 1280, batch_envs: int = 32, max_steps: int = 50, seed: int = 42):
    print("\n" + "=" * 80, flush=True)
    print(f" [STAGE 1] COLLECTING {num_episodes:,} EPISODES FOR {env_id} VIA PPO TEACHER", flush=True)
    print(f" Control Mode: {control_mode} | Action Dim: {action_dim} | State Dim: {state_dim}", flush=True)
    print("=" * 80, flush=True)

    total_samples = num_episodes * max_steps
    buf_pts = torch.empty((total_samples, 512, 3), dtype=torch.float32)
    buf_proprio = torch.empty((total_samples, 29), dtype=torch.float32)
    buf_actions = torch.empty((total_samples, action_dim), dtype=torch.float32)

    rng = np.random.RandomState(seed)
    cur_idx = 0
    t0 = time.time()
    num_batches = int(np.ceil(num_episodes / batch_envs))

    anchor_cameras = [
        ([0.3118, -0.0768, 0.5766], -4.74, 31.8, 1.0942),
        ([0.4542, 0.0, 0.2326], 0.0, 12.27, 1.0942),
        ([0.0992, -0.542, 0.6272], -37.2, 34.98, 1.0942),
        ([-0.0033, 0.6743, 0.607], 47.79, 33.69, 1.0942),
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

        cam_pose = sapien_utils.look_at(eye=cam_eye, target=target_pos)
        sensor_cfg = dict(base_camera=dict(pose=cam_pose, width=128, height=128))

        env = gym.make(
            env_id,
            robot_uids="panda",
            obs_mode="pointcloud+state",
            control_mode=control_mode,
            num_envs=batch_envs,
            sensor_configs=sensor_cfg,
        )
        vec_env = ManiSkillVectorEnv(env, batch_envs, ignore_terminations=True, record_metrics=True)
        obs, _ = vec_env.reset(seed=int(seed + b * 100))

        for step in range(max_steps):
            state_vec = obs["state"][:, :state_dim].to(DEVICE)
            proprio_29 = obs["state"][:, :29]

            with torch.no_grad():
                act = teacher.get_action(state_vec, deterministic=True)
                pts = extract_robot_frame_pc(obs, num_points=512)[:cur_batch]

            buf_pts[cur_idx:cur_idx + cur_batch] = pts.cpu()
            buf_proprio[cur_idx:cur_idx + cur_batch] = proprio_29[:cur_batch].cpu()
            buf_actions[cur_idx:cur_idx + cur_batch] = act[:cur_batch].cpu()
            cur_idx += cur_batch

            if cur_batch < batch_envs:
                act_pad = torch.zeros((batch_envs, action_dim), device=DEVICE)
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
            print(f"  Batch {b+1:2d}/{num_batches:2d} | [{tag}] Az={az:+5.1f}°, El={el:4.1f}°, Dist={dist:.2f}m | Samples: {cur_idx:,} / {total_samples:,} ({time.time()-t0:.1f}s)", flush=True)

    print(f"[OK] Dataset for {env_id} collected: {cur_idx:,} samples in {time.time()-t0:.1f}s", flush=True)
    return buf_pts[:cur_idx], buf_proprio[:cur_idx], buf_actions[:cur_idx]


# =========================================================================
# DATA COLLECTION: Demonstration Replay (PegInsertion & PlugCharger)
# =========================================================================
def collect_dataset_from_demos(env_id: str, control_mode: str, demo_h5: str, demo_json: str, action_dim: int, target_pos, num_episodes: int = 400, max_steps: int = 50, seed: int = 42):
    print("\n" + "=" * 80, flush=True)
    print(f" [STAGE 1] COLLECTING {num_episodes:,} EPISODES FOR {env_id} VIA DEMONSTRATION REPLAY", flush=True)
    print(f" Control Mode: {control_mode} | Action Dim: {action_dim} | Demos: {demo_h5}", flush=True)
    print("=" * 80, flush=True)

    with open(demo_json, "r") as f:
        meta = json.load(f)
    ep_metas = meta["episodes"][:num_episodes]

    total_samples = num_episodes * max_steps
    buf_pts = torch.empty((total_samples, 512, 3), dtype=torch.float32)
    buf_proprio = torch.empty((total_samples, 29), dtype=torch.float32)
    buf_actions = torch.empty((total_samples, action_dim), dtype=torch.float32)

    rng = np.random.RandomState(seed)
    cur_idx = 0
    t0 = time.time()

    with h5py.File(demo_h5, "r") as f_h5:
        for i, ep_info in enumerate(ep_metas):
            ep_id = ep_info["episode_id"]
            ep_seed = ep_info.get("episode_seed", ep_id)
            ep_key = f"traj_{ep_id}"
            if ep_key not in f_h5:
                continue

            actions_seq = torch.tensor(f_h5[ep_key]["actions"][:max_steps], dtype=torch.float32, device=DEVICE)
            T = actions_seq.shape[0]

            cam_eye, az, el, dist = sample_continuous_spherical_camera(rng)
            cam_pose = sapien_utils.look_at(eye=cam_eye, target=target_pos)
            sensor_cfg = dict(base_camera=dict(pose=cam_pose, width=128, height=128))

            env = gym.make(
                env_id,
                obs_mode="pointcloud+state",
                control_mode=control_mode,
                num_envs=1,
                sensor_configs=sensor_cfg,
            )
            obs, _ = env.reset(seed=ep_seed)

            for step in range(T):
                proprio_29 = obs["state"][0, :29]
                with torch.no_grad():
                    pts = extract_robot_frame_pc(obs, num_points=512)[0]

                act = actions_seq[step]
                buf_pts[cur_idx] = pts.cpu()
                buf_proprio[cur_idx] = proprio_29.cpu()
                buf_actions[cur_idx] = act.cpu()
                cur_idx += 1

                obs, _, _, _, _ = env.step(act[None, :])

            env.close()
            del env

            if (i + 1) % 50 == 0 or i == 0 or (i + 1) == len(ep_metas):
                print(f"  Ep {i+1:3d}/{len(ep_metas):3d} | Az={az:+5.1f}°, El={el:4.1f}°, Dist={dist:.2f}m | Samples: {cur_idx:,} / {total_samples:,} ({time.time()-t0:.1f}s)", flush=True)

    print(f"[OK] Replay dataset for {env_id} collected: {cur_idx:,} samples in {time.time()-t0:.1f}s", flush=True)
    return buf_pts[:cur_idx], buf_proprio[:cur_idx], buf_actions[:cur_idx]


# =========================================================================
# POINTNET TRAINING
# =========================================================================
def train_pointnet_policy(task_name: str, action_dim: int, buf_pts, buf_proprio, buf_actions, num_epochs=20, batch_size=256):
    print("\n" + "=" * 80, flush=True)
    print(f" [STAGE 2] TRAINING 3D PointNet POLICY FOR {task_name.upper()}", flush=True)
    print(f" Action Dim: {action_dim} | Transitions: {len(buf_actions):,} | Epochs: {num_epochs}", flush=True)
    print("=" * 80, flush=True)

    dataset = TensorDataset(buf_pts, buf_proprio, buf_actions)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True, num_workers=0)

    student = PointNetAgent(action_dim=action_dim, proprio_dim=29, point_dim=256).to(DEVICE)
    optimizer = optim.AdamW(student.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)

    best_loss = float("inf")
    ckpt_path = os.path.join(CKPT_DIR, f"pointnet_{task_name.lower()}_best.pt")

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
        epoch_loss = total_loss / max(1, n_batches)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(student.state_dict(), ckpt_path)

        if epoch % 5 == 0 or epoch == 1 or epoch == num_epochs:
            print(f"  Epoch {epoch:2d}/{num_epochs:2d} | MSE Loss: {epoch_loss:.6f} | Best: {best_loss:.6f} | Time: {time.time()-t0:.2f}s", flush=True)

    print(f"[OK] Training complete. Best model saved to {ckpt_path}", flush=True)
    return ckpt_path


# =========================================================================
# EVALUATION: 13 CONTINUOUS VIEWPOINTS BENCHMARK
# =========================================================================
def evaluate_13_viewpoints(env_id: str, control_mode: str, action_dim: int, target_pos, ckpt_path: str, num_eval_episodes=20, max_steps=50):
    print("\n" + "=" * 80, flush=True)
    print(f" [STAGE 3] EVALUATING 13 CONTINUOUS VIEWPOINTS FOR {env_id}", flush=True)
    print("=" * 80, flush=True)

    viewpoint_catalog = load_viewpoint_catalog()
    policy = PointNetAgent(action_dim=action_dim, proprio_dim=29, point_dim=256).to(DEVICE)
    policy.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    policy.eval()

    eval_results = []
    t0 = time.time()

    for vp_info in viewpoint_catalog:
        vp_id = vp_info["id"]
        vp_type = "anchor" if vp_info.get("type") in ["trained", "anchor"] else "in-between"
        vp_name = vp_info.get("label", f"VP {vp_id}")
        cam_eye = vp_info["cam_eye"]
        az = vp_info.get("az", vp_info.get("azimuth_deg", 0.0))
        el = vp_info.get("el", vp_info.get("elevation_deg", 0.0))

        cam_pose = sapien_utils.look_at(eye=cam_eye, target=target_pos)
        sensor_cfg = dict(base_camera=dict(pose=cam_pose, width=128, height=128))

        env = gym.make(
            env_id,
            obs_mode="pointcloud+state",
            control_mode=control_mode,
            num_envs=num_eval_episodes,
            sensor_configs=sensor_cfg,
        )
        vec_env = ManiSkillVectorEnv(env, num_eval_episodes, ignore_terminations=True, record_metrics=True)
        obs, _ = vec_env.reset(seed=1000 + vp_id)

        success_once = torch.zeros(num_eval_episodes, dtype=torch.bool, device=DEVICE)
        for step in range(max_steps):
            proprio_29 = obs["state"][:, :29].to(DEVICE)
            with torch.no_grad():
                pts = extract_robot_frame_pc(obs, num_points=512)
                act = policy.get_action(pts, proprio_29, deterministic=True)
            obs, _, _, _, info = vec_env.step(act)
            success_once = success_once | info["success"]

        success_rate = success_once.float().mean().item() * 100.0
        vec_env.close()
        del vec_env
        del env
        gc.collect()
        torch.cuda.empty_cache()

        eval_results.append({
            "vp_id": vp_id,
            "vp_name": vp_name,
            "vp_type": vp_type,
            "azimuth_deg": az,
            "elevation_deg": el,
            "cam_eye": cam_eye,
            "success_rate": success_rate
        })

        tag = "[ANCHOR]" if vp_type == "anchor" else "[IN-BETWEEN]"
        print(f"  {tag:<13} VP {vp_id:02d} ({vp_name:<28}) (Az={az:+5.1f}°, El={el:4.1f}°): Success = {success_rate:5.1f}%", flush=True)

    anchor_srs = [r["success_rate"] for r in eval_results if r["vp_type"] == "anchor"]
    inbetween_srs = [r["success_rate"] for r in eval_results if r["vp_type"] != "anchor"]
    overall_sr = float(np.mean([r["success_rate"] for r in eval_results]))
    anchor_sr = float(np.mean(anchor_srs))
    inbetween_sr = float(np.mean(inbetween_srs))
    gen_gap = anchor_sr - inbetween_sr

    print("-" * 80, flush=True)
    print(f" Summary for {env_id}:", flush=True)
    print(f"  Anchor Views Success Rate:        {anchor_sr:5.1f}%", flush=True)
    print(f"  In-Between Views Success Rate:    {inbetween_sr:5.1f}%", flush=True)
    print(f"  Overall 13-View Success Rate:     {overall_sr:5.1f}%", flush=True)
    print(f"  Generalization Gap (Drop):        {gen_gap:5.1f}%", flush=True)
    print(f" Benchmark completed in {time.time()-t0:.1f}s", flush=True)
    print("=" * 80, flush=True)

    json_path = os.path.join(EVAL_DIR, f"{env_id.lower()}_viewpoint_benchmark.json")
    with open(json_path, "w") as f:
        json.dump({
            "env_id": env_id,
            "control_mode": control_mode,
            "overall_success_rate": overall_sr,
            "anchor_success_rate": anchor_sr,
            "inbetween_success_rate": inbetween_sr,
            "generalization_gap": gen_gap,
            "results": eval_results
        }, f, indent=2)

    plot_task_benchmark(env_id, eval_results, anchor_sr, inbetween_sr, overall_sr, gen_gap)
    return {
        "env_id": env_id,
        "overall_sr": overall_sr,
        "anchor_sr": anchor_sr,
        "inbetween_sr": inbetween_sr,
        "gen_gap": gen_gap
    }


def plot_task_benchmark(env_id: str, results: list, anchor_sr: float, inbetween_sr: float, overall_sr: float, gen_gap: float):
    names = [f"VP{r['vp_id']:02d}\n({r['azimuth_deg']:+.0f}°)" for r in results]
    srs = [r["success_rate"] for r in results]
    types = [r["vp_type"] for r in results]
    colors = ["#2b5c8f" if t == "anchor" else "#00a86b" for t in types]

    fig, ax = plt.subplots(figsize=(13, 5.5), dpi=300)
    bars = ax.bar(range(len(names)), srs, color=colors, width=0.62, edgecolor="black", linewidth=1.1, zorder=3)

    for bar, val in zip(bars, srs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5, f"{val:.1f}%",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.axhline(overall_sr, color="#e63946", linestyle="--", linewidth=2.0, zorder=4,
               label=f"Overall Continuous Mean: {overall_sr:.1f}%")
    ax.axhline(anchor_sr, color="#2b5c8f", linestyle=":", linewidth=1.8, zorder=4,
               label=f"Anchor Mean (4 Views): {anchor_sr:.1f}%")
    ax.axhline(inbetween_sr, color="#00a86b", linestyle="-.", linewidth=1.8, zorder=4,
               label=f"In-Between Mean (9 Views): {inbetween_sr:.1f}%")

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=8.5, fontweight="semibold")
    ax.set_ylim(0, 105)
    ax.set_ylabel("Task Success Rate (%)", fontsize=11, fontweight="bold")
    ax.set_title(f"Continuous Viewpoint Generalization on {env_id}\n3D PointNet Metric Policy across 13 Novel Spherical Angles (Gap: {gen_gap:+.1f}%)",
                 fontsize=12, fontweight="bold", pad=12)
    ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
    ax.legend(loc="lower left", fontsize=10, framealpha=0.95)

    plt.tight_layout()
    plot_path = os.path.join(PLOT_DIR, f"{env_id.lower()}_viewpoint_benchmark.png")
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Benchmark plot saved to {plot_path}", flush=True)


# =========================================================================
# SYNTHESIS PLOT: ALL 7 TASKS MASTER COMPARISON
# =========================================================================
def generate_master_7task_plot(all_tasks_summary: dict):
    print("\n" + "=" * 80, flush=True)
    print(" GENERATING MASTER 7-TASK MULTI-TASK GENERALIZATION PLOT", flush=True)
    print("=" * 80, flush=True)

    tasks = list(all_tasks_summary.keys())
    anchors = [all_tasks_summary[t]["anchor_sr"] for t in tasks]
    inbetweens = [all_tasks_summary[t]["inbetween_sr"] for t in tasks]
    overalls = [all_tasks_summary[t]["overall_sr"] for t in tasks]
    gaps = [all_tasks_summary[t]["gen_gap"] for t in tasks]

    x = np.arange(len(tasks))
    width = 0.26

    fig, ax = plt.subplots(figsize=(14, 6.2), dpi=300)

    rects1 = ax.bar(x - width, anchors, width, label="Anchor Views (4 Cameras)", color="#2b5c8f", edgecolor="black", linewidth=1.1, zorder=3)
    rects2 = ax.bar(x, inbetweens, width, label="Continuous In-Between (9 Cameras)", color="#00a86b", edgecolor="black", linewidth=1.1, zorder=3)
    rects3 = ax.bar(x + width, overalls, width, label="Continuous Hemisphere Overall", color="#f4a261", edgecolor="black", linewidth=1.1, zorder=3)

    for i, rect in enumerate(rects3):
        h = rect.get_height()
        gap = gaps[i]
        ax.text(rect.get_x() + rect.get_width() / 2, h + 2.0, f"Gap:\n{gap:+.1f}%",
                ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#1d3557")

    ax.set_ylabel("Task Success Rate (%)", fontsize=12, fontweight="bold")
    ax.set_title("Universal Multi-Task Continuous Viewpoint Generalization (3D PointNet Metric Space)\nEmpirical Evaluation across 7 Tabletop Manipulation Tasks",
                 fontsize=13, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, fontsize=10, fontweight="bold")
    ax.set_ylim(0, 115)
    ax.legend(loc="upper right", fontsize=10.5, framealpha=0.95)
    ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)

    plt.tight_layout()
    master_path = os.path.join(PLOT_DIR, "multitask_7task_viewpoint_generalization_master.png")
    fig.savefig(master_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Master 7-Task Comparison Plot saved to {master_path}", flush=True)

    # Copy to root repo
    os.system(f"cp {master_path} {REPO_DIR}/multitask_7task_viewpoint_generalization_master.png")


# =========================================================================
# MAIN DRIVER
# =========================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, choices=["StackCube-v1", "PokeCube-v1", "PegInsertionSide-v1", "PlugCharger-v1", "all", "synthesize"], default="all")
    args = parser.parse_args()

    # Pre-existing Phase 13 & Phase 17 results
    all_summary = {
        "PickCube-v1": {"anchor_sr": 92.5, "inbetween_sr": 88.3, "overall_sr": 90.6, "gen_gap": 4.2},
        "PushCube-v1": {"anchor_sr": 75.0, "inbetween_sr": 75.3, "overall_sr": 75.2, "gen_gap": -0.3},
        "PullCube-v1": {"anchor_sr": 44.0, "inbetween_sr": 48.2, "overall_sr": 46.9, "gen_gap": -4.2},
    }

    # Load previously computed results if available
    summary_file = os.path.join(EVAL_DIR, "suite_4tasks_summary.json")
    if os.path.exists(summary_file):
        with open(summary_file, "r") as f:
            all_summary.update(json.load(f))

    if args.task in ["StackCube-v1", "all"]:
        print("\n" + "#" * 80, flush=True)
        print(" EXECUTING PIPELINE FOR TASK: StackCube-v1", flush=True)
        print("#" * 80, flush=True)
        ckpt_path = os.path.join(CKPT_DIR, "pointnet_stackcube-v1_joint_best.pt")
        if not os.path.exists(ckpt_path):
            teacher = load_ppo_teacher("/root/.maniskill/demos/StackCube-v1/rl/ppo_pd_joint_delta_pos_ckpt.pt", state_dim=48, action_dim=8)
            pts, proprio, acts = collect_dataset_from_teacher("StackCube-v1", "pd_joint_delta_pos", teacher, state_dim=48, action_dim=8, target_pos=[0.0, 0.0, 0.05], num_episodes=1280, batch_envs=32)
            ckpt_path = train_pointnet_policy("StackCube-v1", action_dim=8, buf_pts=pts, buf_proprio=proprio, buf_actions=acts, num_epochs=20)
        else:
            print(f"[REUSING CHECKPOINT] Found trained model at {ckpt_path}", flush=True)
        res = evaluate_13_viewpoints("StackCube-v1", "pd_joint_delta_pos", action_dim=8, target_pos=[0.0, 0.0, 0.05], ckpt_path=ckpt_path)
        all_summary["StackCube-v1"] = res

    if args.task in ["PokeCube-v1", "all"]:
        print("\n" + "#" * 80, flush=True)
        print(" EXECUTING PIPELINE FOR TASK: PokeCube-v1", flush=True)
        print("#" * 80, flush=True)
        ckpt_path = os.path.join(CKPT_DIR, "pointnet_pokecube-v1_best.pt")
        if not os.path.exists(ckpt_path):
            teacher = load_ppo_teacher("/root/.maniskill/demos/PokeCube-v1/rl/ppo_pd_joint_delta_pos_ckpt.pt", state_dim=54, action_dim=8)
            pts, proprio, acts = collect_dataset_from_teacher("PokeCube-v1", "pd_joint_delta_pos", teacher, state_dim=54, action_dim=8, target_pos=[0.0, 0.0, 0.05], num_episodes=1280, batch_envs=32)
            ckpt_path = train_pointnet_policy("PokeCube-v1", action_dim=8, buf_pts=pts, buf_proprio=proprio, buf_actions=acts, num_epochs=20)
        else:
            print(f"[REUSING CHECKPOINT] Found trained model at {ckpt_path}", flush=True)
        res = evaluate_13_viewpoints("PokeCube-v1", "pd_joint_delta_pos", action_dim=8, target_pos=[0.0, 0.0, 0.05], ckpt_path=ckpt_path)
        all_summary["PokeCube-v1"] = res

    if args.task in ["PegInsertionSide-v1", "all"]:
        print("\n" + "#" * 80, flush=True)
        print(" EXECUTING PIPELINE FOR TASK: PegInsertionSide-v1", flush=True)
        print("#" * 80, flush=True)
        ckpt_path = os.path.join(CKPT_DIR, "pointnet_peginsertionside-v1_best.pt")
        if not os.path.exists(ckpt_path):
            pts, proprio, acts = collect_dataset_from_demos(
                "PegInsertionSide-v1", "pd_joint_delta_pos",
                demo_h5="/root/.maniskill/demos/PegInsertionSide-v1/rl/trajectory.h5",
                demo_json="/root/.maniskill/demos/PegInsertionSide-v1/rl/trajectory.json",
                action_dim=8, target_pos=[0.0, 0.0, 0.1], num_episodes=400, max_steps=50
            )
            ckpt_path = train_pointnet_policy("PegInsertionSide-v1", action_dim=8, buf_pts=pts, buf_proprio=proprio, buf_actions=acts, num_epochs=20)
        else:
            print(f"[REUSING CHECKPOINT] Found trained model at {ckpt_path}", flush=True)
        res = evaluate_13_viewpoints("PegInsertionSide-v1", "pd_joint_delta_pos", action_dim=8, target_pos=[0.0, 0.0, 0.1], ckpt_path=ckpt_path)
        all_summary["PegInsertionSide-v1"] = res

    if args.task in ["PlugCharger-v1", "all"]:
        print("\n" + "#" * 80, flush=True)
        print(" EXECUTING PIPELINE FOR TASK: PlugCharger-v1", flush=True)
        print("#" * 80, flush=True)
        ckpt_path = os.path.join(CKPT_DIR, "pointnet_plugcharger-v1_best.pt")
        if not os.path.exists(ckpt_path):
            pts, proprio, acts = collect_dataset_from_demos(
                "PlugCharger-v1", "pd_joint_pos",
                demo_h5="/root/.maniskill/demos/PlugCharger-v1/motionplanning/trajectory.h5",
                demo_json="/root/.maniskill/demos/PlugCharger-v1/motionplanning/trajectory.json",
                action_dim=8, target_pos=[0.0, 0.0, 0.1], num_episodes=300, max_steps=120
            )
            ckpt_path = train_pointnet_policy("PlugCharger-v1", action_dim=8, buf_pts=pts, buf_proprio=proprio, buf_actions=acts, num_epochs=20)
        else:
            print(f"[REUSING CHECKPOINT] Found trained model at {ckpt_path}", flush=True)
        res = evaluate_13_viewpoints("PlugCharger-v1", "pd_joint_pos", action_dim=8, target_pos=[0.0, 0.0, 0.1], ckpt_path=ckpt_path, max_steps=120)
        all_summary["PlugCharger-v1"] = res

    with open(summary_file, "w") as f:
        json.dump(all_summary, f, indent=2)

    generate_master_7task_plot(all_summary)


if __name__ == "__main__":
    main()
