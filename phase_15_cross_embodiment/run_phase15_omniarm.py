"""
Phase 15: Master Cross-Embodiment Distillation & Evaluation Pipeline.

Trains the Universal OmniArm PointNet policy across multiple robot embodiments:
- Franka Panda (4-dim EE delta)
- UFactory xArm 6 + Robotiq (4-dim EE delta)

Benchmarks the unified model across all 13 camera viewpoints on every robot arm.
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
from phase_15_cross_embodiment.omniarm_models import (
    UniversalOmniArmAgent,
    extract_robot_frame_pc,
    EMBODIMENT_NAMES,
    ROBOT_PROPRIO_DIMS,
    ROBOT_ACTION_DIMS,
    ROBOT_CONTROL_MODES,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = "/home/maniskill/phase_15_cross_embodiment"
TEACHER_DIR = os.path.join(BASE_DIR, "teachers")
OMNIARM_DIR = os.path.join(BASE_DIR, "omniarm")
EVAL_DIR = os.path.join(BASE_DIR, "eval_results")
VIEWPOINTS_JSON = "/home/maniskill/phase_1_to_3/sequential_viewpoint_training/viewpoints.json"

os.makedirs(OMNIARM_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)

ROBOT_BASE_OFFSETS = {
    "panda": (-0.615, 0.0, 0.0),
    "xarm6_robotiq": (-0.522, 0.0, 0.0),
}


def load_privileged_teacher(robot_uid: str):
    state_dims = {"panda": 42, "xarm6_robotiq": 48}
    act_dims = {"panda": 4, "xarm6_robotiq": 4}

    ckpt_path = os.path.join(TEACHER_DIR, f"teacher_{robot_uid}.pt")
    if not os.path.exists(ckpt_path):
        ckpt_path = f"/home/maniskill/runs/{robot_uid}_teacher/final_ckpt.pt"

    class DummySpace:
        def __init__(self, shape):
            self.shape = shape
    dummy_vec = SimpleNamespace(
        single_observation_space=DummySpace((state_dims[robot_uid],)),
        single_action_space=DummySpace((act_dims[robot_uid],))
    )
    teacher = PrivilegedAgent(dummy_vec).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    teacher.load_state_dict(ckpt)
    teacher.eval()
    print(f"[OK] Teacher for {robot_uid} loaded from {ckpt_path}")
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


def collect_multi_arm_dataset(active_robots=["panda", "xarm6_robotiq"], num_episodes_per_robot=1000, batch_envs=32, num_points=512, seed=42):
    print("\n" + "=" * 80)
    print(f" [STAGE 1] COLLECTING MULTI-EMBODIMENT DATASET ACROSS {len(active_robots)} ROBOT ARMS")
    print(f" Robots: {active_robots} | Episodes per Robot: {num_episodes_per_robot:,}")
    print("=" * 80)

    dataset_by_robot = {}
    rng = np.random.RandomState(seed)

    for robot_idx, robot_uid in enumerate(active_robots):
        teacher = load_privileged_teacher(robot_uid)
        ctrl_mode = ROBOT_CONTROL_MODES[robot_uid]
        proprio_dim = ROBOT_PROPRIO_DIMS[robot_uid]
        action_dim = ROBOT_ACTION_DIMS[robot_uid]
        base_xyz = ROBOT_BASE_OFFSETS.get(robot_uid, (-0.615, 0.0, 0.0))

        total_samples = num_episodes_per_robot * 50
        buf_pts = torch.empty((total_samples, num_points, 3), dtype=torch.float32)
        buf_proprio = torch.empty((total_samples, proprio_dim), dtype=torch.float32)
        buf_actions = torch.empty((total_samples, action_dim), dtype=torch.float32)

        cur_idx = 0
        t0 = time.time()
        num_batches = int(np.ceil(num_episodes_per_robot / batch_envs))

        for b in range(num_batches):
            cur_batch = min(batch_envs, num_episodes_per_robot - b * batch_envs)
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
            obs, _ = vec_env.reset(seed=int(seed + robot_idx * 1000 + b * 100))

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
                print(f"  [{robot_uid:<15}] Batch {b+1:2d}/{num_batches:2d} | Collected: {cur_idx:,} / {total_samples:,} samples")

        dataset_by_robot[robot_uid] = {
            "pts": buf_pts[:cur_idx],
            "proprio": buf_proprio[:cur_idx],
            "actions": buf_actions[:cur_idx],
        }
        print(f"  [OK] Collected {cur_idx:,} transitions for {robot_uid} in {time.time()-t0:.1f}s")

    return dataset_by_robot


def train_universal_omniarm(dataset_by_robot, num_epochs=25, batch_size=256):
    print("\n" + "=" * 80)
    print(" [STAGE 2] TRAINING UNIVERSAL OMNIARM POINTNET POLICY")
    print(f" Multi-Robot Joint Distillation | Epochs: {num_epochs} | Batch Size: {batch_size}")
    print("=" * 80)

    model = UniversalOmniArmAgent().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)

    loaders = {}
    for r, d in dataset_by_robot.items():
        ds = TensorDataset(d["pts"], d["proprio"], d["actions"])
        loaders[r] = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True)

    best_loss = float("inf")
    ckpt_path = os.path.join(OMNIARM_DIR, "omniarm_universal_best.pt")

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        n_steps = 0
        t0 = time.time()

        # Interleaved multi-embodiment training
        iterators = {r: iter(loaders[r]) for r in loaders}
        max_batches = max(len(loaders[r]) for r in loaders)

        for _ in range(max_batches):
            for r in loaders:
                try:
                    b_pts, b_proprio, b_act = next(iterators[r])
                except StopIteration:
                    iterators[r] = iter(loaders[r])
                    b_pts, b_proprio, b_act = next(iterators[r])

                b_pts = b_pts.to(DEVICE, non_blocking=True)
                b_proprio = b_proprio.to(DEVICE, non_blocking=True)
                b_act = b_act.to(DEVICE, non_blocking=True)

                pred = model.get_action(b_pts, b_proprio, robot_uid=r, deterministic=True)
                loss = F.mse_loss(pred, b_act)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()
                n_steps += 1

        scheduler.step()
        avg_loss = total_loss / n_steps
        dt = time.time() - t0

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "model_state": model.state_dict(),
                "epoch": epoch,
                "best_loss": best_loss,
            }, ckpt_path)

        if epoch % 5 == 0 or epoch == 1 or epoch == num_epochs:
            print(f"  Epoch {epoch:2d}/{num_epochs:2d} | Joint Multi-Arm Loss: {avg_loss:.5f} | Time: {dt:.1f}s")

    print(f"\n[OK] Universal OmniArm model saved to {ckpt_path} (MSE: {best_loss:.5f})")
    return ckpt_path


def evaluate_universal_omniarm(omniarm_ckpt: str, active_robots=["panda", "xarm6_robotiq"]):
    print("\n" + "=" * 80)
    print(" [STAGE 3] BENCHMARKING UNIVERSAL OMNIARM POLICY ACROSS ALL ROBOT ARMS")
    print(f" Checkpoint: {omniarm_ckpt}")
    print("=" * 80)

    with open(VIEWPOINTS_JSON) as f:
        vps = json.load(f)
    vp_dict = {v["id"]: v for v in vps}

    model = UniversalOmniArmAgent().to(DEVICE)
    ckpt = torch.load(omniarm_ckpt, map_location=DEVICE)
    model.load_state_dict(ckpt.get("model_state", ckpt))
    model.eval()

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

    all_benchmarks = {}
    batch_envs = 50

    for robot_uid in active_robots:
        print(f"\n>>> Evaluating on Robot: {robot_uid.upper()} <<<")
        ctrl_mode = ROBOT_CONTROL_MODES[robot_uid]
        proprio_dim = ROBOT_PROPRIO_DIMS[robot_uid]
        base_xyz = ROBOT_BASE_OFFSETS.get(robot_uid, (-0.615, 0.0, 0.0))

        robot_results = []
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
                    act = model.get_action(pts, proprio, robot_uid=robot_uid, deterministic=True)

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

            res = {
                "id": vid,
                "name": name,
                "type": vtype,
                "success_rate": succ_rate,
                "grasp_rate": grasp_rate,
            }
            robot_results.append(res)
            tag = "[ANCHOR]" if vtype == "trained" else "[IN-BETWEEN]"
            print(f"  {tag:<12} {name:<32}: Succ={succ_rate:5.1f}% | Grasp={grasp_rate:5.1f}%")

        avg_s = np.mean([r["success_rate"] for r in robot_results])
        avg_g = np.mean([r["grasp_rate"] for r in robot_results])
        print(f"  ==> {robot_uid.upper()} Overall Success: {avg_s:.1f}% | Grasp: {avg_g:.1f}% <==")
        all_benchmarks[robot_uid] = {
            "overall_success": avg_s,
            "overall_grasp": avg_g,
            "details": robot_results,
        }

    # Save benchmark JSON
    json_path = os.path.join(EVAL_DIR, "phase15_omniarm_results.json")
    with open(json_path, "w") as f:
        json.dump(all_benchmarks, f, indent=2)

    # Plot Multi-Arm Cross-Embodiment Comparison
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(eval_vps))
    width = 0.35

    s_panda = [r["success_rate"] for r in all_benchmarks["panda"]["details"]]
    s_xarm = [r["success_rate"] for r in all_benchmarks["xarm6_robotiq"]["details"]]
    labels = [f"VP{r['id']:02d}\n({vps[r['id']]['azimuth_deg']:+.0f}°)" for r in all_benchmarks["panda"]["details"]]

    ax.bar(x - width/2, s_panda, width, label=f"Franka Panda (Mean: {all_benchmarks['panda']['overall_success']:.1f}%)", color="#1565c0", edgecolor="black")
    ax.bar(x + width/2, s_xarm, width, label=f"xArm 6 Robotiq (Mean: {all_benchmarks['xarm6_robotiq']['overall_success']:.1f}%)", color="#2e7d32", edgecolor="black")

    ax.set_ylabel("Pick-and-Lift Success Rate (%)", fontsize=12, fontweight="bold")
    ax.set_title("Phase 15: Universal OmniArm PointNet Policy Evaluated Across Different Robot Arms", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", fontsize=11)

    plot_path = os.path.join(EVAL_DIR, "phase15_cross_embodiment_comparison.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"\n[OK] Cross-embodiment plot saved to {plot_path}")

    return all_benchmarks


def main():
    dataset = collect_multi_arm_dataset(
        active_robots=["panda", "xarm6_robotiq"],
        num_episodes_per_robot=1000,
        batch_envs=32,
        num_points=512,
    )
    best_ckpt = train_universal_omniarm(dataset, num_epochs=25, batch_size=256)
    evaluate_universal_omniarm(best_ckpt, active_robots=["panda", "xarm6_robotiq"])


if __name__ == "__main__":
    main()
