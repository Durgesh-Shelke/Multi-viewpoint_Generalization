"""
Phase 17 Consistent: High-Performance Multi-Task Viewpoint Generalization Suite.

Equates the training regime of PushCube-v1 and PullCube-v1 to the PickCube-v1 benchmark:
- 1,920 episodes (96,000 transitions) across continuous spherical dome.
- 25 epochs with AdamW + Cosine Annealing (lr: 5e-4 to 1e-5).
- 100% PPO Teacher Oracles using 7-dim end-effector delta pose (pd_ee_delta_pose).
- Proprioception: 28-dim [qpos(9), qvel(9), tcp_pose(7), goal_pos(3)].
- Metric 3D PointNet (512 points) in robot base frame.
- Comprehensive 13-viewpoint benchmark (4 anchors + 9 in-between views).
"""

import os
import sys
import time
import json
import gc
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
OUT_DIR = os.path.join(REPO_DIR, "phase_17_multitask_replication")
CKPT_DIR = os.path.join(OUT_DIR, "checkpoints")
EVAL_DIR = os.path.join(OUT_DIR, "eval_results")
PLOT_DIR = os.path.join(OUT_DIR, "plots")
VIEWPOINTS_JSON = os.path.join(REPO_DIR, "interpolated_viewpoints_results.json")

os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

BASE_POS = np.array([-0.615, 0.0, 0.0])
TARGET_POS = [-0.1, 0.0, 0.1]


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

    def get_action(self, x, deterministic: bool = True):
        return self.actor_mean(x)


def load_ppo_teacher(ckpt_path: str, state_dim: int = 35, action_dim: int = 7):
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
# STAGE 1: Data Collection across Continuous Spherical Viewpoints
# =========================================================================
def collect_consistent_dataset(env_id: str, teacher: nn.Module, state_dim: int = 35, action_dim: int = 7,
                              num_episodes: int = 1920, batch_envs: int = 32, max_steps: int = 50, seed: int = 42):
    print("\n" + "=" * 80, flush=True)
    print(f" [STAGE 1] COLLECTING {num_episodes:,} EPISODES ({num_episodes * max_steps:,} TRANSITIONS) FOR {env_id}", flush=True)
    print(" Continuous Random Viewpoint Sampling across Frontal Hemisphere Dome", flush=True)
    print(" Control Mode: pd_ee_delta_pose (7-dim) | Proprio Dim: 28", flush=True)
    print("=" * 80, flush=True)

    total_samples = num_episodes * max_steps
    buf_pts = torch.empty((total_samples, 512, 3), dtype=torch.float32)
    buf_proprio = torch.empty((total_samples, 28), dtype=torch.float32)
    buf_actions = torch.empty((total_samples, action_dim), dtype=torch.float32)

    rng = np.random.RandomState(seed)
    cur_idx = 0
    t0 = time.time()
    num_batches = int(np.ceil(num_episodes / batch_envs))

    # 4 nominal anchor cameras for guaranteed anchor distribution coverage
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
            env_id,
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
            state_vec = obs["state"][:, :state_dim].to(DEVICE)
            proprio_28 = obs["state"][:, :28]

            with torch.no_grad():
                act = teacher.get_action(state_vec, deterministic=True)
                pts = extract_robot_frame_pc(obs, num_points=512)[:cur_batch]

            buf_pts[cur_idx:cur_idx + cur_batch] = pts.cpu()
            buf_proprio[cur_idx:cur_idx + cur_batch] = proprio_28[:cur_batch].cpu()
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
            dt = time.time() - t0
            print(f"  Batch {b+1:2d}/{num_batches:2d} | [{tag}] Az={az:+5.1f}°, El={el:4.1f}°, Dist={dist:.2f}m | Samples: {cur_idx:,} / {total_samples:,} ({dt:.1f}s)", flush=True)

    print(f"[OK] Consistent dataset for {env_id} collected: {cur_idx:,} samples in {time.time()-t0:.1f}s", flush=True)
    return buf_pts[:cur_idx], buf_proprio[:cur_idx], buf_actions[:cur_idx]


# =========================================================================
# STAGE 2: PointNet Student Policy Training (25 Epochs, Cosine Annealing)
# =========================================================================
def train_pointnet_student(task_name: str, buf_pts, buf_proprio, buf_actions, action_dim: int = 7, proprio_dim: int = 28,
                           num_epochs: int = 25, batch_size: int = 256):
    print("\n" + "=" * 80, flush=True)
    print(f" [STAGE 2] TRAINING 3D PointNet POLICY FOR {task_name.upper()} (CONSISTENT REGIME)", flush=True)
    print(f" Transitions: {len(buf_actions):,} | Epochs: {num_epochs} | Batch Size: {batch_size} | Action Dim: {action_dim}", flush=True)
    print("=" * 80, flush=True)

    dataset = TensorDataset(buf_pts, buf_proprio, buf_actions)
    bs = min(batch_size, len(buf_actions))
    loader = DataLoader(dataset, batch_size=bs, shuffle=True, drop_last=(len(buf_actions) > bs), pin_memory=True, num_workers=0)

    student = PointNetAgent(action_dim=action_dim, proprio_dim=proprio_dim, point_dim=256).to(DEVICE)
    optimizer = optim.AdamW(student.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)

    best_loss = float("inf")
    ckpt_path = os.path.join(CKPT_DIR, f"pointnet_{task_name.lower()}_consistent_best.pt")

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
            torch.save({
                "model_state": student.state_dict(),
                "epoch": epoch,
                "best_loss": best_loss,
                "task": task_name,
                "action_dim": action_dim,
                "proprio_dim": proprio_dim,
                "control_mode": "pd_ee_delta_pose",
            }, ckpt_path)

        if epoch % 5 == 0 or epoch == 1 or epoch == num_epochs:
            cur_lr = scheduler.get_last_lr()[0]
            print(f"  Epoch {epoch:2d}/{num_epochs:2d} | Train MSE Loss: {avg_loss:.6f} | Best: {best_loss:.6f} | lr: {cur_lr:.2e} | ({dt:.1f}s)", flush=True)

    print(f"[OK] Training complete. Best checkpoint saved to {ckpt_path} (MSE: {best_loss:.6f})", flush=True)
    return ckpt_path


# =========================================================================
# STAGE 3: Comprehensive 13-Viewpoint Continuous Benchmark
# =========================================================================
def evaluate_policy_across_viewpoints(env_id: str, ckpt_path: str, action_dim: int = 7, proprio_dim: int = 28, batch_envs: int = 50):
    print("\n" + "=" * 80, flush=True)
    print(f" [STAGE 3] BENCHMARKING {env_id} ACROSS 13 STANDARDIZED VIEWPOINTS", flush=True)
    print(" 4 Anchor Cameras + 9 Interpolated Continuous In-Between Angles", flush=True)
    print(" Control Mode: pd_ee_delta_pose", flush=True)
    print("=" * 80, flush=True)

    catalog = load_viewpoint_catalog()

    # Load student model
    model = PointNetAgent(action_dim=action_dim, proprio_dim=proprio_dim, point_dim=256).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    eval_results = []
    t_bench = time.time()

    for item in catalog:
        vid = item["id"]
        label = item["label"]
        vtype = item["type"]
        cam_eye = item["cam_eye"]
        az = item["az"]
        el = item["el"]

        cam_pose = sapien_utils.look_at(eye=cam_eye, target=TARGET_POS)
        sensor_cfg = dict(base_camera=dict(pose=cam_pose, width=128, height=128))

        env = gym.make(
            env_id,
            robot_uids="panda",
            obs_mode="pointcloud+state",
            control_mode="pd_ee_delta_pose",
            num_envs=batch_envs,
            sensor_configs=sensor_cfg,
            shader_dir="minimal",
        )
        vec_env = ManiSkillVectorEnv(env, batch_envs, ignore_terminations=True, record_metrics=True)
        obs, _ = vec_env.reset(seed=3026 + vid)

        for step in range(50):
            proprio = obs["state"][:, :proprio_dim]
            with torch.no_grad():
                pts = extract_robot_frame_pc(obs, num_points=512)
                act = model.get_action(pts, proprio, deterministic=True)

            obs, rew, term, trunc, info = vec_env.step(act)

        succ = info["final_info"]["episode"]["success_once"].float().mean().item() * 100.0

        vec_env.close()
        del vec_env
        del env
        gc.collect()
        torch.cuda.empty_cache()

        res_entry = {
            "id": vid,
            "label": label,
            "type": vtype,
            "cam_eye": cam_eye,
            "azimuth": az,
            "elevation": el,
            "success_rate": succ,
        }
        eval_results.append(res_entry)
        tag = "[ANCHOR]" if vtype == "trained" else "[IN-BETWEEN]"
        print(f"  {tag:<12} {label:<32} (Az={az:+5.1f}°, El={el:4.1f}°): Success = {succ:5.1f}%", flush=True)

    # Compute aggregate stats
    anchor_sr = float(np.mean([r["success_rate"] for r in eval_results if r["type"] == "trained"]))
    interp_sr = float(np.mean([r["success_rate"] for r in eval_results if r["type"] != "trained"]))
    overall_sr = float(np.mean([r["success_rate"] for r in eval_results]))
    gen_gap = anchor_sr - interp_sr

    print("-" * 80, flush=True)
    print(f" Summary for {env_id} (Consistent Regime):", flush=True)
    print(f"  Anchor Views Success Rate:        {anchor_sr:5.1f}%", flush=True)
    print(f"  In-Between Views Success Rate:    {interp_sr:5.1f}%", flush=True)
    print(f"  Overall 13-View Success Rate:     {overall_sr:5.1f}%", flush=True)
    print(f"  Generalization Gap (Anchor - IB): {gen_gap:5.1f}%", flush=True)
    print(f" Benchmark completed in {time.time()-t_bench:.1f}s", flush=True)
    print("=" * 80, flush=True)

    summary = {
        "env_id": env_id,
        "control_mode": "pd_ee_delta_pose",
        "action_dim": action_dim,
        "proprio_dim": proprio_dim,
        "anchor_success_rate": anchor_sr,
        "interp_success_rate": interp_sr,
        "overall_success_rate": overall_sr,
        "generalization_gap": gen_gap,
        "detailed_results": eval_results,
    }

    # Save JSON
    task_slug = env_id.lower().replace("-v1", "")
    json_path = os.path.join(EVAL_DIR, f"{task_slug}-v1_viewpoint_benchmark_consistent.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[OK] Evaluation JSON saved to {json_path}", flush=True)

    # Generate task plot
    generate_task_benchmark_plot(env_id, eval_results, anchor_sr, interp_sr, overall_sr, gen_gap)
    return summary


def generate_task_benchmark_plot(env_id: str, results, anchor_sr, interp_sr, overall_sr, gen_gap):
    task_slug = env_id.lower().replace("-v1", "")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [3, 1]})

    v_ids = [r["id"] for r in results]
    labels = [f"VP {r['id']}\n{r['label'].split('(')[0].strip()}" for r in results]
    rates = [r["success_rate"] for r in results]
    types = [r["type"] for r in results]
    colors = ["#1f77b4" if t == "trained" else "#ff7f0e" for t in types]

    bars = ax1.bar(range(len(v_ids)), rates, color=colors, alpha=0.9, width=0.65)
    ax1.set_xticks(range(len(v_ids)))
    ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax1.set_ylabel("Success Rate (%)", fontsize=11, fontweight="bold")
    ax1.set_ylim(0, 115)
    ax1.grid(axis="y", linestyle=":", alpha=0.6)
    ax1.set_title(f"{env_id} — Continuous Viewpoint Invariance (Consistent 96K Regime)\n4 Trained Anchors (Blue) vs 9 Continuous In-Between Angles (Orange)", fontsize=11, fontweight="bold")

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
    plot_path = os.path.join(PLOT_DIR, f"{task_slug}-v1_viewpoint_benchmark_consistent.png")
    plt.savefig(plot_path, dpi=200)
    plt.savefig(os.path.join(REPO_DIR, f"{task_slug}-v1_viewpoint_benchmark_consistent.png"), dpi=200)
    plt.close()
    print(f"[OK] Task benchmark plot saved to {plot_path}", flush=True)


def update_master_synthesis():
    print("\n" + "#" * 80, flush=True)
    print(" SYNTHESIZING MASTER MULTI-TASK VIEWPOINT GENERALIZATION BENCHMARK", flush=True)
    print("#" * 80, flush=True)

    tasks_data = {}

    # PickCube flagship reference
    tasks_data["PickCube-v1\n(Prehensile Lift)"] = {
        "anchor_sr": 93.8,
        "interp_sr": 89.6,
        "overall_sr": 90.6,
        "gen_gap": 4.2,
    }

    # Load PushCube consistent
    push_json = os.path.join(EVAL_DIR, "pushcube-v1_viewpoint_benchmark_consistent.json")
    if os.path.exists(push_json):
        with open(push_json, "r") as f:
            d = json.load(f)
        tasks_data["PushCube-v1\n(Planar Sliding)"] = {
            "anchor_sr": d.get("anchor_success_rate", 0.0),
            "interp_sr": d.get("interp_success_rate", d.get("inbetween_success_rate", 0.0)),
            "overall_sr": d.get("overall_success_rate", 0.0),
            "gen_gap": d.get("generalization_gap", 0.0),
        }

    # Load PullCube consistent
    pull_json = os.path.join(EVAL_DIR, "pullcube-v1_viewpoint_benchmark_consistent.json")
    if os.path.exists(pull_json):
        with open(pull_json, "r") as f:
            d = json.load(f)
        tasks_data["PullCube-v1\n(Non-Prehensile Pull)"] = {
            "anchor_sr": d.get("anchor_success_rate", 0.0),
            "interp_sr": d.get("interp_success_rate", d.get("inbetween_success_rate", 0.0)),
            "overall_sr": d.get("overall_success_rate", 0.0),
            "gen_gap": d.get("generalization_gap", 0.0),
        }

    # Load PokeCube from Phase 18
    poke_json = os.path.join(REPO_DIR, "phase_18_multitask_extended/eval_results/pokecube-v1_viewpoint_benchmark.json")
    if os.path.exists(poke_json):
        with open(poke_json, "r") as f:
            d = json.load(f)
        tasks_data["PokeCube-v1\n(Impulsive Poke)"] = {
            "anchor_sr": d.get("anchor_success_rate", 0.0),
            "interp_sr": d.get("interp_success_rate", d.get("inbetween_success_rate", 0.0)),
            "overall_sr": d.get("overall_success_rate", 0.0),
            "gen_gap": d.get("generalization_gap", 0.0),
        }

    # Load LiftPegUpright from Phase 18
    lift_json = os.path.join(REPO_DIR, "phase_18_multitask_extended/eval_results/liftpegupright-v1_viewpoint_benchmark.json")
    if os.path.exists(lift_json):
        with open(lift_json, "r") as f:
            d = json.load(f)
        tasks_data["LiftPegUpright-v1\n(Dexterous Reorientation)"] = {
            "anchor_sr": d.get("anchor_success_rate", 0.0),
            "interp_sr": d.get("interp_success_rate", d.get("inbetween_success_rate", 0.0)),
            "overall_sr": d.get("overall_success_rate", 0.0),
            "gen_gap": d.get("generalization_gap", 0.0),
        }

    # Generate master plot
    tasks = list(tasks_data.keys())
    anchor_scores = [tasks_data[t]["anchor_sr"] for t in tasks]
    interp_scores = [tasks_data[t]["interp_sr"] for t in tasks]
    overall_scores = [tasks_data[t]["overall_sr"] for t in tasks]

    x = np.arange(len(tasks))
    w = 0.25

    fig, ax = plt.subplots(figsize=(13.5, 6))
    b1 = ax.bar(x - w, anchor_scores, w, label="Anchor Cameras (4 Views)", color="#1f77b4", alpha=0.9)
    b2 = ax.bar(x, interp_scores, w, label="Novel In-Between Cameras (9 Views)", color="#ff7f0e", alpha=0.9)
    b3 = ax.bar(x + w, overall_scores, w, label="Overall Continuous Benchmark", color="#2ca02c", alpha=0.9)

    ax.set_ylabel("Success Rate (%)", fontsize=11, fontweight="bold")
    ax.set_title("Multi-Task Viewpoint Generalization Mastery\nConsistent 96,000 Transitions Regime across Manipulation Archetypes", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, fontsize=10, fontweight="bold")
    ax.set_ylim(0, 115)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", fontsize=10)

    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 1.5, f"{h:.1f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    plt.tight_layout()
    master_plot_path = os.path.join(PLOT_DIR, "multitask_viewpoint_generalization_consistent_master.png")
    plt.savefig(master_plot_path, dpi=200)
    plt.savefig(os.path.join(REPO_DIR, "multitask_viewpoint_generalization_consistent_master.png"), dpi=200)
    plt.close()
    print(f"[OK] Master consistent multi-task comparison plot saved to {master_plot_path}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="all", choices=["all", "PushCube-v1", "PullCube-v1", "synthesize"])
    parser.add_argument("--episodes", type=int, default=1920)
    parser.add_argument("--epochs", type=int, default=25)
    args = parser.parse_args()

    if args.task == "synthesize":
        update_master_synthesis()
        return

    task_cfgs = {
        "PushCube-v1": {
            "env_id": "PushCube-v1",
            "name": "PushCube",
            "state_dim": 35,
            "action_dim": 7,
            "proprio_dim": 28,
            "teacher_ckpt": "/root/.maniskill/demos/PushCube-v1/rl/ppo_pd_ee_delta_pose_ckpt.pt",
        },
        "PullCube-v1": {
            "env_id": "PullCube-v1",
            "name": "PullCube",
            "state_dim": 35,
            "action_dim": 7,
            "proprio_dim": 28,
            "teacher_ckpt": "/root/.maniskill/demos/PullCube-v1/rl/ppo_pd_ee_delta_pose_ckpt.pt",
        },
    }

    tasks_to_run = [args.task] if args.task in task_cfgs else ["PushCube-v1", "PullCube-v1"]

    for tname in tasks_to_run:
        cfg = task_cfgs[tname]
        teacher = load_ppo_teacher(cfg["teacher_ckpt"], state_dim=cfg["state_dim"], action_dim=cfg["action_dim"])

        # 1. Collect dataset (96,000 transitions)
        buf_pts, buf_proprio, buf_actions = collect_consistent_dataset(
            cfg["env_id"], teacher, state_dim=cfg["state_dim"], action_dim=cfg["action_dim"],
            num_episodes=args.episodes, batch_envs=32, max_steps=50
        )

        # 2. Train student (25 epochs)
        ckpt_student = train_pointnet_student(
            cfg["name"], buf_pts, buf_proprio, buf_actions,
            action_dim=cfg["action_dim"], proprio_dim=cfg["proprio_dim"],
            num_epochs=args.epochs, batch_size=256
        )

        del buf_pts, buf_proprio, buf_actions, teacher
        gc.collect()
        torch.cuda.empty_cache()

        # 3. Benchmark across 13 Viewpoints
        evaluate_policy_across_viewpoints(
            cfg["env_id"], ckpt_student,
            action_dim=cfg["action_dim"], proprio_dim=cfg["proprio_dim"],
            batch_envs=50
        )

    update_master_synthesis()


if __name__ == "__main__":
    main()
