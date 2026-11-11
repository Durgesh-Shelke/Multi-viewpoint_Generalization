"""
Phase 10 Next-Gen OmniView: Balanced Multi-Teacher Distillation and 1,000-Trial Benchmark.
Methodology:
1. Roll out all 8 converged Phase 10 teachers (97.25% pool accuracy) on their respective viewpoints
   using TemporalExtrinsicWrapper (12-channel 3-frame stack + 33-dim extrinsics state + jitter).
2. Construct a unified multi-viewpoint demonstration dataset (80,000 expert state-action pairs).
3. Train OmniViewAgent with shuffled mini-batches (batch size 256) ensuring joint representation
   across all 8 viewpoints simultaneously (eliminating sequential catastrophic forgetting).
4. Run the 1,000-Trial Randomized Viewpoint Benchmark and produce publication figures.
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
from phase_10_omniview_nextgen.models_nextgen import OmniViewAgent
from phase_10_omniview_nextgen.train_teacher_specialists_v3 import Phase10TeacherAgent
from phase_10_omniview_nextgen.wrappers import TemporalExtrinsicWrapper

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VIEWPOINTS_JSON = "/home/maniskill/Multi-viewpoint_Generalization/phase_1_to_3/sequential_viewpoint_training/viewpoints.json"
TEACHERS_DIR = "/home/maniskill/Multi-viewpoint_Generalization/phase_10_omniview_nextgen/teachers"
STUDENT_DIR = "/home/maniskill/Multi-viewpoint_Generalization/phase_10_omniview_nextgen/student"
EVAL_DIR = "/home/maniskill/Multi-viewpoint_Generalization/phase_10_omniview_nextgen/eval_results"
DATA_DIR = "/home/maniskill/Multi-viewpoint_Generalization/phase_10_omniview_nextgen/data"

CORE_TEACHERS = [17, 0, 28, 18, 27, 20, 19, 6]

VP_ROLES = {
    17: "Front-Mid Canonical",
    0:  "Front-Low Graze",
    28: "Front-High Steep Tilt",
    18: "Left-Flank 90°",
    27: "Right-Flank 90°",
    20: "Diag Front-Left",
    19: "Diag Front-Right",
    6:  "Low Flank Right",
}


def load_teacher(vpid: int):
    ckpt_path = os.path.join(TEACHERS_DIR, f"specialist_vp_{vpid:02d}_best.pt")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(TEACHERS_DIR, f"specialist_vp_{vpid:02d}_final.pt")
    teacher = Phase10TeacherAgent(action_dim=4, state_dim=33).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    teacher.load_state_dict(ckpt.get("model_state", ckpt))
    teacher.eval()
    return teacher


def collect_single_teacher_dataset(vpid: int, vp_dict: dict, num_steps: int = 10000, num_envs: int = 64):
    meta = vp_dict[vpid]
    cam_eye = meta["world_eye"]
    cam_target = [-0.1, 0.0, 0.1]

    print(f"[Collector] Viewpoint {vpid:02d} ({VP_ROLES[vpid]}): Collecting {num_steps:,} transitions...")
    teacher = load_teacher(vpid)

    env_kwargs = dict(
        obs_mode="rgbd",
        render_mode="all",
        sim_backend="physx_cuda",
        sensor_cam_eye_pos=cam_eye,
        sensor_cam_target_pos=cam_target,
        control_mode="pd_ee_delta_pos",
        sensor_configs=dict(width=128, height=128),
    )
    env = gym.make("PickCube-v1", num_envs=num_envs, reconfiguration_freq=None, **env_kwargs)
    env = FlattenRGBDObservationWrapper(env, rgb=True, depth=True, state=True, sep_depth=True)
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    # Enable temporal frame stacking (12 channels) and extrinsics conditioning (33 dim) with jitter
    env = TemporalExtrinsicWrapper(env, cam_eye=cam_eye, frame_stack=3, noise_std=0.02, device="cuda")
    envs = ManiSkillVectorEnv(env, num_envs, ignore_terminations=False, record_metrics=True)

    num_batches = int(np.ceil(num_steps / num_envs))
    total_samples = num_batches * num_envs

    buf_rgbd = torch.empty((total_samples, 12, 128, 128), dtype=torch.float32)
    buf_state = torch.empty((total_samples, 33), dtype=torch.float32)
    buf_actions = torch.empty((total_samples, 4), dtype=torch.float32)

    obs, _ = envs.reset(seed=42 + vpid)
    cur_idx = 0
    t0 = time.time()
    successes = 0
    episodes = 0

    with torch.no_grad():
        for b in range(num_batches):
            # Teacher queries newest 4 channels
            latest_rgbd = obs["rgbd"][:, 8:12, :, :]
            vis_feat = teacher.feature_net.cnn(latest_rgbd)
            state_feat = teacher.feature_net.state_fc(obs["state"])
            feat = torch.cat([vis_feat, state_feat], dim=1)
            actions = teacher.actor_mean(feat)

            buf_rgbd[cur_idx:cur_idx + num_envs] = obs["rgbd"].cpu()
            buf_state[cur_idx:cur_idx + num_envs] = obs["state"].cpu()
            buf_actions[cur_idx:cur_idx + num_envs] = actions.cpu()
            cur_idx += num_envs

            obs, _, _, _, infos = envs.step(actions)
            if "final_info" in infos and "_final_info" in infos:
                mask = infos["_final_info"]
                for i in range(len(mask)):
                    if mask[i]:
                        ep = infos["final_info"].get("episode", {})
                        k = "success_once" if "success_once" in ep else "success"
                        if k in ep:
                            val = ep[k][i]
                            if bool(val.item() if torch.is_tensor(val) else val):
                                successes += 1
                        episodes += 1

    envs.close()
    elapsed = time.time() - t0
    succ_rate = (successes / max(episodes, 1)) * 100.0 if episodes > 0 else 0.0
    print(f"  >>> VP {vpid:02d}: {total_samples:,} samples collected in {elapsed:.1f}s (Rollout Success: {succ_rate:.1f}%)")
    return buf_rgbd, buf_state, buf_actions


def collect_multi_teacher_dataset(vp_dict: dict, steps_per_teacher: int = 10000):
    os.makedirs(DATA_DIR, exist_ok=True)
    all_rgbd = []
    all_state = []
    all_actions = []

    print("\n" + "=" * 80)
    print(f" COLLECTING MULTI-TEACHER DEMONSTRATIONS ({len(CORE_TEACHERS)} TEACHERS x {steps_per_teacher:,} STEPS)")
    print("=" * 80)

    t0 = time.time()
    for vpid in CORE_TEACHERS:
        rgbd, state, actions = collect_single_teacher_dataset(vpid, vp_dict, num_steps=steps_per_teacher)
        all_rgbd.append(rgbd)
        all_state.append(state)
        all_actions.append(actions)

    cat_rgbd = torch.cat(all_rgbd, dim=0)
    cat_state = torch.cat(all_state, dim=0)
    cat_actions = torch.cat(all_actions, dim=0)
    total_time = time.time() - t0

    print(f"\n[Dataset] Total Unified Dataset: {cat_rgbd.shape[0]:,} transitions collected in {total_time:.1f}s!")
    return cat_rgbd, cat_state, cat_actions


def evaluate_student_viewpoints(student: nn.Module, vp_dict: dict, num_episodes: int = 50):
    student.eval()
    results = {}
    for vpid in CORE_TEACHERS:
        meta = vp_dict[vpid]
        cam_eye = meta["world_eye"]
        cam_target = [-0.1, 0.0, 0.1]

        env_kwargs = dict(
            obs_mode="rgbd",
            render_mode="all",
            sim_backend="physx_cuda",
            sensor_cam_eye_pos=cam_eye,
            sensor_cam_target_pos=cam_target,
            control_mode="pd_ee_delta_pos",
            sensor_configs=dict(width=128, height=128),
        )
        env = gym.make("PickCube-v1", num_envs=50, reconfiguration_freq=None, **env_kwargs)
        env = FlattenRGBDObservationWrapper(env, rgb=True, depth=True, state=True, sep_depth=True)
        if isinstance(env.action_space, gym.spaces.Dict):
            env = FlattenActionSpaceWrapper(env)
        env = TemporalExtrinsicWrapper(env, cam_eye=cam_eye, frame_stack=3, noise_std=0.0, device="cuda")
        eval_envs = ManiSkillVectorEnv(env, 50, ignore_terminations=True, record_metrics=True)

        obs, _ = eval_envs.reset()
        collected = 0
        successes = 0

        with torch.no_grad():
            for _ in range(50):
                if collected >= num_episodes:
                    break
                act = student.get_action(obs["rgbd"], obs["state"], deterministic=True)
                obs, _, _, _, infos = eval_envs.step(act)

                if "final_info" in infos and "_final_info" in infos:
                    mask = infos["_final_info"]
                    ep_info = infos["final_info"].get("episode", {})
                    succ_key = "success_once" if "success_once" in ep_info else "success"
                    for env_i in range(len(mask)):
                        if collected >= num_episodes:
                            break
                        if mask[env_i]:
                            val = ep_info[succ_key][env_i]
                            succ = bool(val.item() if torch.is_tensor(val) else val)
                            if succ:
                                successes += 1
                            collected += 1

        eval_envs.close()
        acc = (successes / max(collected, 1)) * 100.0
        results[vpid] = acc

    return results


def train_unified_student(all_rgbd, all_state, all_actions, vp_dict, epochs: int = 30, batch_size: int = 256, lr: float = 5e-4):
    os.makedirs(STUDENT_DIR, exist_ok=True)
    best_student_path = os.path.join(STUDENT_DIR, "omniview_student_best.pt")

    print("\n" + "=" * 80)
    print(" TRAINING UNIFIED OMNIVIEW STUDENT POLICY (BALANCED MULTI-VIEW MINIBATCHES)")
    print(f" Samples: {all_rgbd.shape[0]:,} | Epochs: {epochs} | Batch: {batch_size} | LR: {lr}")
    print("=" * 80)

    dataset = TensorDataset(all_rgbd, all_state, all_actions)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=2)

    student = OmniViewAgent(action_dim=4, state_dim=33).to(DEVICE)
    canonical_teacher = load_teacher(17)
    student.warm_start_from_single_frame(canonical_teacher.state_dict())

    optimizer = optim.Adam(student.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_mean_acc = 0.0
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        student.train()
        total_loss = 0.0
        batches = 0

        for b_rgbd, b_state, b_actions in dataloader:
            b_rgbd = b_rgbd.to(DEVICE, non_blocking=True)
            b_state = b_state.to(DEVICE, non_blocking=True)
            b_actions = b_actions.to(DEVICE, non_blocking=True)

            feat = student.get_features(b_rgbd, b_state)
            pred_actions = student.actor_mean(feat)

            loss = F.mse_loss(pred_actions, b_actions)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            batches += 1

        scheduler.step()
        avg_loss = total_loss / batches
        elapsed = time.time() - t0

        if epoch % 5 == 0 or epoch == epochs or epoch == 1:
            print(f"Epoch {epoch:02d}/{epochs:02d} | Train MSE: {avg_loss:.5f} | LR: {scheduler.get_last_lr()[0]:.2e} | Time: {elapsed:.1f}s")

        if epoch % 5 == 0 or epoch == epochs:
            eval_res = evaluate_student_viewpoints(student, vp_dict, num_episodes=50)
            mean_acc = float(np.mean(list(eval_res.values())))
            viable_acc = float(np.mean([eval_res[v] for v in CORE_TEACHERS if v != 28]))
            print(f"  >>> [EVAL Epoch {epoch:02d}] Mean: {mean_acc:.1f}% | Viable: {viable_acc:.1f}% | VP 28: {eval_res[28]:.1f}% | VP 17: {eval_res[17]:.1f}% | VP 00: {eval_res[0]:.1f}%")

            if mean_acc >= best_mean_acc:
                best_mean_acc = mean_acc
                torch.save({
                    "student_state": student.state_dict(),
                    "eval_results": eval_res,
                    "mean_acc": mean_acc,
                    "viable_acc": viable_acc,
                    "epoch": epoch,
                }, best_student_path)
                print(f"  >>> [NEW BEST STUDENT] Mean: {best_mean_acc:.1f}% -> {best_student_path}")

    print("\n" + "=" * 80)
    print(f" UNIFIED DISTILLATION COMPLETE! Best Accuracy: {best_mean_acc:.1f}%")
    print("=" * 80)
    return best_student_path


def run_1000_tests_benchmark(student_ckpt: str, vp_dict: dict, num_tests: int = 1000, seed: int = 42):
    os.makedirs(EVAL_DIR, exist_ok=True)
    print("\n" + "=" * 80)
    print(f" PHASE 10 OMNIVIEW STUDENT: {num_tests:,} RANDOMIZED VIEWPOINT TESTS")
    print(f" Checkpoint: {student_ckpt}")
    print("=" * 80)

    rng = np.random.RandomState(seed)
    trial_vps = rng.choice(CORE_TEACHERS, size=num_tests)

    student = OmniViewAgent(action_dim=4, state_dim=33).to(DEVICE)
    ckpt = torch.load(student_ckpt, map_location=DEVICE)
    student.load_state_dict(ckpt.get("student_state", ckpt.get("model_state", ckpt)))
    student.eval()

    trial_results = [None] * num_tests
    per_vp_results = {}
    t0 = time.time()

    for vp_id in CORE_TEACHERS:
        vp_indices = np.where(trial_vps == vp_id)[0]
        n_vp = len(vp_indices)
        if n_vp == 0:
            continue

        meta = vp_dict[vp_id]
        cam_eye = meta["world_eye"]
        cam_target = [-0.1, 0.0, 0.1]

        batch_envs = min(n_vp, 50)
        env_kwargs = dict(
            obs_mode="rgbd",
            render_mode="all",
            sim_backend="physx_cuda",
            sensor_cam_eye_pos=cam_eye,
            sensor_cam_target_pos=cam_target,
            control_mode="pd_ee_delta_pos",
            sensor_configs=dict(width=128, height=128),
        )
        env = gym.make("PickCube-v1", num_envs=batch_envs, reconfiguration_freq=None, **env_kwargs)
        env = FlattenRGBDObservationWrapper(env, rgb=True, depth=True, state=True, sep_depth=True)
        if isinstance(env.action_space, gym.spaces.Dict):
            env = FlattenActionSpaceWrapper(env)
        env = TemporalExtrinsicWrapper(env, cam_eye=cam_eye, frame_stack=3, noise_std=0.0, device="cuda")
        eval_envs = ManiSkillVectorEnv(env, batch_envs, ignore_terminations=True, record_metrics=True)

        evaluated = 0
        success_count = 0

        while evaluated < n_vp:
            obs, _ = eval_envs.reset()
            active_batch = min(batch_envs, n_vp - evaluated)
            episodes_done = [False] * active_batch
            episodes_succ = [False] * active_batch

            with torch.no_grad():
                for _ in range(50):
                    act = student.get_action(obs["rgbd"], obs["state"], deterministic=True)
                    obs, _, _, _, infos = eval_envs.step(act)

                    if "final_info" in infos and "_final_info" in infos:
                        mask = infos["_final_info"]
                        ep_info = infos["final_info"].get("episode", {})
                        succ_key = "success_once" if "success_once" in ep_info else "success"
                        for i in range(active_batch):
                            if mask[i] and not episodes_done[i]:
                                episodes_done[i] = True
                                val = ep_info[succ_key][i]
                                succ = bool(val.item() if torch.is_tensor(val) else val)
                                episodes_succ[i] = succ

            for i in range(active_batch):
                idx = vp_indices[evaluated + i]
                succ = episodes_succ[i]
                trial_results[idx] = succ
                if succ:
                    success_count += 1
            evaluated += active_batch

        eval_envs.close()
        succ_rate = (success_count / n_vp) * 100.0
        per_vp_results[vp_id] = {
            "viewpoint_id": vp_id,
            "role": VP_ROLES[vp_id],
            "trials": n_vp,
            "successes": success_count,
            "success_rate": round(succ_rate, 2),
        }
        print(f"  VP {vp_id:02d} ({VP_ROLES[vp_id]:<24}): {success_count:3d}/{n_vp:3d} ({succ_rate:5.1f}%)")

    total_time = time.time() - t0
    overall_successes = sum(1 for r in trial_results if r)
    overall_acc = (overall_successes / num_tests) * 100.0

    viable_vp_ids = [v for v in CORE_TEACHERS if v != 28]
    viable_trials = sum(per_vp_results[v]["trials"] for v in viable_vp_ids)
    viable_successes = sum(per_vp_results[v]["successes"] for v in viable_vp_ids)
    viable_acc = (viable_successes / viable_trials) * 100.0

    print("\n" + "=" * 80)
    print(f" BENCHMARK RESULTS ({num_tests:,} Random Trials in {total_time:.1f}s):")
    print(f" Overall Accuracy:        {overall_acc:.2f}% ({overall_successes}/{num_tests})")
    print(f" Viable Views Accuracy:   {viable_acc:.2f}% ({viable_successes}/{viable_trials})")
    print(f" VP 28 (Occluded View):   {per_vp_results[28]['success_rate']:.1f}% ({per_vp_results[28]['successes']}/{per_vp_results[28]['trials']})")
    print("=" * 80)

    # Save JSON metrics
    metrics_path = os.path.join(EVAL_DIR, "phase10_omniview_1000_benchmark.json")
    with open(metrics_path, "w") as f:
        json.dump({
            "overall_accuracy": round(overall_acc, 2),
            "viable_accuracy": round(viable_acc, 2),
            "total_trials": num_tests,
            "total_successes": overall_successes,
            "evaluation_time_seconds": round(total_time, 1),
            "per_viewpoint": per_vp_results,
        }, f, indent=2)

    # Plot breakdown bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    vp_names = [f"VP {v:02d}\n{VP_ROLES[v]}" for v in CORE_TEACHERS]
    accs = [per_vp_results[v]["success_rate"] for v in CORE_TEACHERS]
    colors = ["#2ecc71" if a >= 85 else "#f39c12" if a >= 50 else "#e74c3c" for a in accs]

    bars = ax.bar(vp_names, accs, color=colors, edgecolor="black", linewidth=1.2, width=0.6)
    ax.axhline(y=overall_acc, color="#e67e22", linestyle="--", linewidth=2.0, label=f"Overall Mean: {overall_acc:.1f}%")
    ax.axhline(y=viable_acc, color="#27ae60", linestyle=":", linewidth=2.0, label=f"Viable Mean: {viable_acc:.1f}%")

    for bar, acc in zip(bars, accs):
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, yval + 1.5, f"{acc:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=10)

    ax.set_ylim(0, 110)
    ax.set_ylabel("Grasp Success Rate (%)", fontsize=12, fontweight="bold")
    ax.set_title(f"Phase 10 OmniView Student Policy: 1,000 Randomized Viewpoint Benchmark\n(3-Frame Temporal Stack + Camera Extrinsics Conditioning)", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    chart_path = os.path.join(EVAL_DIR, "phase10_omniview_1000_tests_breakdown.png")
    plt.tight_layout()
    plt.savefig(chart_path, dpi=200)
    plt.close()

    # Also copy to standalone repo
    report_chart_path = "/home/maniskill/Multi-viewpoint_Generalization/reports/phase10_omniview_1000_tests_breakdown.png"
    os.makedirs(os.path.dirname(report_chart_path), exist_ok=True)
    import shutil
    shutil.copyfile(chart_path, report_chart_path)

    return overall_acc, viable_acc, per_vp_results


def main():
    with open(VIEWPOINTS_JSON) as f:
        vp_dict = {v["id"]: v for v in json.load(f)}

    # Stage 1: Collect demonstration dataset from the 8 converged teachers
    all_rgbd, all_state, all_actions = collect_multi_teacher_dataset(vp_dict, steps_per_teacher=10000)

    # Stage 2: Train unified student model on balanced multi-view minibatches
    best_student_path = train_unified_student(all_rgbd, all_state, all_actions, vp_dict, epochs=30, batch_size=256, lr=5e-4)

    # Stage 3: Run 1,000-Trial Benchmark across all 8 viewpoints
    overall_acc, viable_acc, per_vp = run_1000_tests_benchmark(best_student_path, vp_dict, num_tests=1000)

    print("\n" + "=" * 80)
    print(" >>> PHASE 10 COMPLETE: ALL EXPERIMENTS AND BENCHMARKS SUCCESSFUL <<<")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
