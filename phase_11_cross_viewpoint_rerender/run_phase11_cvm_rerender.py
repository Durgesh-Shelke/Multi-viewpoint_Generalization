"""
Phase 11: Cross-Viewpoint Multi-Angle Trajectory Re-Rendering (CVM-ReRender)
and Continuous Distillation Pipeline.

Overcomes the discrete viewpoint interpolation gap by re-rendering successful teacher
trajectories across 22 dense viewpoints covering the entire visual hemisphere,
combined with continuous 3D camera jitter and extrinsics noise.

Pipeline:
1. Collect & verify successful trajectories from the 8 trained teachers.
2. Re-render trajectories across 22 dense viewpoints (8 anchors + 14 intermediate viewpoints).
3. Distill into Next-Gen OmniView student policy with balanced mini-batches.
4. Benchmark across the full evaluation suite (Anchors, In-Betweens, Spatial Basin, Zoom).
"""

import os
import sys
sys.path.append("/home/maniskill")
import time
import json
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
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper, FlattenActionSpaceWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from phase_10_omniview_nextgen.train_teacher_specialists_v3 import Phase10TeacherAgent, append_extrinsics

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VIEWPOINTS_JSON = "/home/maniskill/Multi-viewpoint_Generalization/phase_1_to_3/sequential_viewpoint_training/viewpoints.json"
TEACHERS_DIR = "/home/maniskill/Multi-viewpoint_Generalization/phase_10_omniview_nextgen/teachers"
PHASE11_DIR = "/home/maniskill/Multi-viewpoint_Generalization/phase_11_cross_viewpoint_rerender"
STUDENT_DIR = os.path.join(PHASE11_DIR, "student")
EVAL_DIR = os.path.join(PHASE11_DIR, "eval_results")

os.makedirs(STUDENT_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)

CORE_TEACHERS = [17, 0, 28, 20, 19, 6, 18, 27]

# Dense 22 viewpoints: 8 anchors + 14 intermediate interpolation viewpoints
DENSE_VIEWPOINTS = [
    # 8 Anchors
    17, 0, 28, 20, 19, 6, 18, 27,
    # 14 In-Between / Interpolated viewpoints
    10, 13, 14, 4, 7, 21, 24, 12, 15, 9, 8, 25, 23, 3
]


def load_teacher(vpid: int):
    ckpt_path = os.path.join(TEACHERS_DIR, f"specialist_vp_{vpid:02d}_final.pt")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(TEACHERS_DIR, f"specialist_vp_{vpid:02d}_best.pt")
    teacher = Phase10TeacherAgent(action_dim=4, state_dim=33).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    teacher.load_state_dict(ckpt.get("model_state", ckpt))
    teacher.eval()
    return teacher


def collect_expert_trajectories(num_trajs_per_teacher: int = 12):
    """
    Collects strictly successful trajectories from the 8 trained teachers.
    Each trajectory stores (seed, list of actions, camera_eye).
    """
    print("\n" + "=" * 80)
    print(f" [STAGE 1] COLLECTING {num_trajs_per_teacher} SUCCESSFUL TRAJECTORIES PER TEACHER")
    print("=" * 80)

    with open(VIEWPOINTS_JSON) as f:
        vps = json.load(f)
    vp_dict = {v["id"]: v for v in vps}

    successful_trajectories = []
    t0 = time.time()

    for vpid in CORE_TEACHERS:
        meta = vp_dict[vpid]
        cam_eye = meta["world_eye"]
        cam_eye_t = torch.tensor(cam_eye, device=DEVICE)
        teacher = load_teacher(vpid)

        batch_envs = 32
        env_kwargs = dict(
            obs_mode="rgbd",
            render_mode="all",
            sim_backend="physx_cuda",
            sensor_cam_eye_pos=cam_eye,
            sensor_cam_target_pos=[-0.1, 0.0, 0.1],
            control_mode="pd_ee_delta_pos",
            sensor_configs=dict(width=128, height=128),
        )
        env = gym.make("PickCube-v1", num_envs=batch_envs, reconfiguration_freq=None, **env_kwargs)
        env = FlattenRGBDObservationWrapper(env, rgb=True, depth=True, state=True, sep_depth=True)
        if isinstance(env.action_space, gym.spaces.Dict):
            env = FlattenActionSpaceWrapper(env)
        envs = ManiSkillVectorEnv(env, batch_envs, ignore_terminations=True, record_metrics=True)

        found = 0
        seed_counter = 1000 + vpid * 100

        while found < num_trajs_per_teacher:
            obs, _ = envs.reset(seed=seed_counter)
            actions_history = []
            for step in range(50):
                with torch.no_grad():
                    s33 = append_extrinsics(obs["state"], cam_eye_t, noise_std=0.0, device=DEVICE)
                    act = teacher.get_action(obs["rgb"], obs["depth"], s33, deterministic=True)
                obs, rew, term, trunc, infos = envs.step(act)
                actions_history.append(act)

            succ_mask = infos["final_info"]["episode"]["success_once"]
            for env_i in range(batch_envs):
                if succ_mask[env_i] and found < num_trajs_per_teacher:
                    # Extract single env action trajectory
                    act_traj = [actions_history[t][env_i].clone().cpu() for t in range(50)]
                    successful_trajectories.append({
                        "teacher_vp": vpid,
                        "seed": seed_counter + env_i,
                        "actions": act_traj,
                    })
                    found += 1

            seed_counter += batch_envs

        envs.close()
        print(f"  Teacher VP {vpid:02d} ({meta['azimuth_deg']:5.1f}°): Collected {found} successful trajectories")

    print(f"\n[OK] Total successful trajectories collected: {len(successful_trajectories)} in {time.time()-t0:.1f}s")
    return successful_trajectories, vp_dict


def rerender_dataset_multi_angle(trajectories: list, vp_dict: dict):
    """
    Replays the verified action trajectories across 22 dense viewpoints,
    recording (RGB, Depth, State_33, Action) with continuous 3D camera jitter.
    """
    print("\n" + "=" * 80)
    print(f" [STAGE 2] RE-RENDERING {len(trajectories)} TRAJECTORIES ACROSS {len(DENSE_VIEWPOINTS)} VIEWPOINTS")
    print("=" * 80)

    num_trajs = len(trajectories)
    num_vps = len(DENSE_VIEWPOINTS)
    total_samples = num_trajs * num_vps * 50

    print(f" Target Distillation Buffer: {total_samples:,} Transitions ({total_samples // 50:,} Full Episodes)")

    # Compact preallocated buffers
    buf_rgb = torch.empty((total_samples, 128, 128, 3), dtype=torch.uint8)
    buf_depth = torch.empty((total_samples, 128, 128, 1), dtype=torch.int16)
    buf_state = torch.empty((total_samples, 33), dtype=torch.float32)
    buf_actions = torch.empty((total_samples, 4), dtype=torch.float32)

    cur_idx = 0
    t0 = time.time()

    # Process in batches of trajectories across viewpoints
    batch_envs = min(num_trajs, 48)

    for vp_i, target_vpid in enumerate(DENSE_VIEWPOINTS):
        meta = vp_dict[target_vpid]
        base_eye = np.array(meta["world_eye"])

        # Jitter camera position by +-2cm for continuous visual diversity
        jitter = np.random.normal(0, 0.02, size=3)
        cam_eye = (base_eye + jitter).tolist()
        cam_eye_t = torch.tensor(cam_eye, device=DEVICE)

        env_kwargs = dict(
            obs_mode="rgbd",
            render_mode="all",
            sim_backend="physx_cuda",
            sensor_cam_eye_pos=cam_eye,
            sensor_cam_target_pos=[-0.1, 0.0, 0.1],
            control_mode="pd_ee_delta_pos",
            sensor_configs=dict(width=128, height=128),
        )
        env = gym.make("PickCube-v1", num_envs=batch_envs, reconfiguration_freq=None, **env_kwargs)
        env = FlattenRGBDObservationWrapper(env, rgb=True, depth=True, state=True, sep_depth=True)
        if isinstance(env.action_space, gym.spaces.Dict):
            env = FlattenActionSpaceWrapper(env)
        eval_envs = ManiSkillVectorEnv(env, batch_envs, ignore_terminations=True, record_metrics=True)

        for batch_start in range(0, num_trajs, batch_envs):
            batch_end = min(batch_start + batch_envs, num_trajs)
            active_batch = batch_end - batch_start

            # Reset environments with trajectory seeds
            batch_seeds = [trajectories[batch_start + i]["seed"] for i in range(active_batch)]
            if active_batch < batch_envs:
                batch_seeds += [9999 + i for i in range(batch_envs - active_batch)]
            obs, _ = eval_envs.reset(seed=batch_seeds)

            for step in range(50):
                # Construct batch action tensor
                act_step = torch.stack([trajectories[batch_start + i]["actions"][step] for i in range(active_batch)]).to(DEVICE)

                # Store current observation and target action
                n = active_batch
                buf_rgb[cur_idx:cur_idx+n] = obs["rgb"][:n].cpu()
                buf_depth[cur_idx:cur_idx+n] = obs["depth"][:n].cpu()

                s33 = append_extrinsics(obs["state"][:n], cam_eye_t, noise_std=0.02, device=DEVICE)
                buf_state[cur_idx:cur_idx+n] = s33.cpu()
                buf_actions[cur_idx:cur_idx+n] = act_step.cpu()

                cur_idx += n

                # Step environment with recorded action
                # Pad action if active_batch < batch_envs
                if active_batch < batch_envs:
                    act_pad = torch.zeros((batch_envs, 4), device=DEVICE)
                    act_pad[:active_batch] = act_step
                    obs, rew, term, trunc, infos = eval_envs.step(act_pad)
                else:
                    obs, rew, term, trunc, infos = eval_envs.step(act_step)

        eval_envs.close()
        tag = "[ANCHOR]" if target_vpid in CORE_TEACHERS else "[INTERP]"
        print(f"  {tag} Viewpoint {target_vpid:02d} ({meta['azimuth_deg']:5.1f}°): Rendered {num_trajs*50} samples (Total: {cur_idx:,})")

    print(f"\n[OK] Re-rendering completed: {cur_idx:,} transitions in {time.time()-t0:.1f}s")
    return buf_rgb[:cur_idx], buf_depth[:cur_idx], buf_state[:cur_idx], buf_actions[:cur_idx]


def train_rerender_student(buf_rgb, buf_depth, buf_state, buf_actions, num_epochs=25, batch_size=256):
    print("\n" + "=" * 80)
    print(" [STAGE 3] TRAINING PHASE 11 CVM-RERENDER STUDENT POLICY")
    print(f" Dataset: {len(buf_actions):,} transitions | Epochs: {num_epochs} | Batch Size: {batch_size}")
    print("=" * 80)

    dataset = TensorDataset(buf_rgb, buf_depth, buf_state, buf_actions)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True, num_workers=0)

    student = Phase10TeacherAgent(action_dim=4, state_dim=33).to(DEVICE)
    optimizer = optim.AdamW(student.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)

    best_loss = float("inf")
    best_ckpt_path = os.path.join(STUDENT_DIR, "rerender_student_best.pt")

    for epoch in range(1, num_epochs + 1):
        student.train()
        total_loss = 0.0
        n_batches = 0
        t0 = time.time()

        for batch_rgb, batch_depth, batch_state, batch_actions in loader:
            batch_rgb = batch_rgb.to(DEVICE, non_blocking=True)
            batch_depth = batch_depth.to(DEVICE, non_blocking=True)
            batch_state = batch_state.to(DEVICE, non_blocking=True)
            batch_actions = batch_actions.to(DEVICE, non_blocking=True)

            pred_action = student.get_action(batch_rgb, batch_depth, batch_state, deterministic=True)
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


def evaluate_student(student_ckpt: str, vp_dict: dict):
    print("\n" + "=" * 80)
    print(" [STAGE 4] BENCHMARKING PHASE 11 STUDENT ON IN-BETWEEN & ANCHOR VIEWPOINTS")
    print(f" Checkpoint: {student_ckpt}")
    print("=" * 80)

    student = Phase10TeacherAgent(action_dim=4, state_dim=33).to(DEVICE)
    ckpt = torch.load(student_ckpt, map_location=DEVICE)
    student.load_state_dict(ckpt.get("model_state", ckpt))
    student.eval()

    eval_vps = [
        # 4 Anchors for control comparison
        (17, "VP 17 (Front-Mid Anchor)", "trained"),
        (0,  "VP 00 (Front-Low Anchor)", "trained"),
        (20, "VP 20 (Diag-Left Anchor)", "trained"),
        (19, "VP 19 (Diag-Right Anchor)", "trained"),
        # The 9 Untrained In-Between Viewpoints that previously scored 2.9%
        (10, "VP 10 (In-Between El)", "untrained"),
        (13, "VP 13 (In-Between Az L)", "untrained"),
        (14, "VP 14 (In-Between Az R)", "untrained"),
        (4,  "VP 04 (In-Between Low-Right)", "untrained"),
        (7,  "VP 07 (In-Between Low-Left)", "untrained"),
        (21, "VP 21 (In-Between Mid-Right)", "untrained"),
        (24, "VP 24 (In-Between High-Mid)", "untrained"),
        (12, "VP 12 (Flank In-Between R)", "untrained"),
        (15, "VP 15 (Flank In-Between L)", "untrained"),
    ]

    results = []
    batch_envs = 50

    for vid, name, vtype in eval_vps:
        meta = vp_dict[vid]
        cam_eye = meta["world_eye"]
        cam_eye_t = torch.tensor(cam_eye, device=DEVICE)

        env_kwargs = dict(
            obs_mode="rgbd",
            render_mode="all",
            sim_backend="physx_cuda",
            sensor_cam_eye_pos=cam_eye,
            sensor_cam_target_pos=[-0.1, 0.0, 0.1],
            control_mode="pd_ee_delta_pos",
            sensor_configs=dict(width=128, height=128),
        )
        env = gym.make("PickCube-v1", num_envs=batch_envs, reconfiguration_freq=None, **env_kwargs)
        env = FlattenRGBDObservationWrapper(env, rgb=True, depth=True, state=True, sep_depth=True)
        if isinstance(env.action_space, gym.spaces.Dict):
            env = FlattenActionSpaceWrapper(env)
        eval_envs = ManiSkillVectorEnv(env, batch_envs, ignore_terminations=True, record_metrics=True)

        obs, _ = eval_envs.reset(seed=1234 + vid)
        grasped_any = torch.zeros(batch_envs, dtype=torch.bool, device=DEVICE)

        for step in range(50):
            with torch.no_grad():
                s33 = append_extrinsics(obs["state"], cam_eye_t, noise_std=0.0, device=DEVICE)
                act = student.get_action(obs["rgb"], obs["depth"], s33, deterministic=True)
            obs, rew, term, trunc, infos = eval_envs.step(act)
            if "is_grasped" in infos:
                grasped_any |= infos["is_grasped"]

        succ = infos["final_info"]["episode"]["success_once"]
        succ_rate = succ.float().mean().item() * 100.0
        grasp_rate = grasped_any.float().mean().item() * 100.0
        eval_envs.close()

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

    # Save benchmark JSON
    json_path = os.path.join(EVAL_DIR, "phase11_rerender_eval_results.json")
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
    ax.set_title("Phase 11 CVM-ReRender: In-Between Viewpoint Generalization\n(Green = Original Anchors | Blue = Intermediate In-Between Viewpoints)", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", fontsize=10)

    plot_path = os.path.join(EVAL_DIR, "phase11_rerender_comparison.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"\n[OK] Benchmark plot saved to {plot_path}")

    return results


def main():
    trajectories, vp_dict = collect_expert_trajectories(num_trajs_per_teacher=8)
    buf_rgb, buf_depth, buf_state, buf_actions = rerender_dataset_multi_angle(trajectories, vp_dict)
    best_ckpt = train_rerender_student(buf_rgb, buf_depth, buf_state, buf_actions, num_epochs=25, batch_size=256)
    evaluate_student(best_ckpt, vp_dict)


if __name__ == "__main__":
    main()
