"""
Evaluation Harness: 1,000 Randomized Viewpoint Tests for Phase 10 OmniView Student Policy.
Evaluates across the 8 strategic viewpoints with 3-frame temporal stacking and camera extrinsics.
Generates:
1. Per-viewpoint accuracy breakdown table
2. Breakdown visualization chart (PNG)
3. Full trial logs (CSV & JSON)
"""

import os
import sys
sys.path.append("/home/maniskill")
import json
import time
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mani_skill.envs
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper, FlattenActionSpaceWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from phase_10_omniview_nextgen.models_nextgen import OmniViewAgent
from phase_10_omniview_nextgen.wrappers import TemporalExtrinsicWrapper

VIEWPOINTS_JSON = "/home/maniskill/Multi-viewpoint_Generalization/phase_1_to_3/sequential_viewpoint_training/viewpoints.json"
STUDENT_CKPT = "/home/maniskill/Multi-viewpoint_Generalization/phase_10_omniview_nextgen/student/omniview_student_best.pt"
EVAL_DIR = "/home/maniskill/Multi-viewpoint_Generalization/phase_10_omniview_nextgen/eval_results"

CORE_VP_IDS = [17, 0, 28, 18, 27, 20, 19, 6]

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


def run_1000_benchmark(num_tests: int = 1000, seed: int = 42, device_str: str = "cuda"):
    os.makedirs(EVAL_DIR, exist_ok=True)
    device = torch.device(device_str)

    print("\n" + "=" * 80)
    print(f" PHASE 10 OMNIVIEW STUDENT: {num_tests:,} RANDOMIZED VIEWPOINT TESTS")
    print(f" Checkpoint: {STUDENT_CKPT}")
    print(f" Architecture: 3-Frame Temporal Stack (12-ch) + Extrinsics Conditioning (33-dim state)")
    print("=" * 80 + "\n")

    with open(VIEWPOINTS_JSON) as f:
        all_vps = json.load(f)
    vp_dict = {vp["id"]: vp for vp in all_vps}

    rng = np.random.RandomState(seed)
    trial_vps = rng.choice(CORE_VP_IDS, size=num_tests)

    # Load Student Model
    student = OmniViewAgent(action_dim=4, state_dim=33).to(device)
    ckpt = torch.load(STUDENT_CKPT, map_location=device)
    student.load_state_dict(ckpt.get("student_state", ckpt.get("model_state", ckpt)))
    student.eval()

    trial_results = [None] * num_tests
    per_vp_results = {}
    t0 = time.time()

    for vp_id in CORE_VP_IDS:
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
        env = TemporalExtrinsicWrapper(env, cam_eye=cam_eye, frame_stack=3, noise_std=0.0, device=device_str)
        eval_envs = ManiSkillVectorEnv(env, batch_envs, ignore_terminations=True, record_metrics=True)

        collected = 0
        vp_successes = 0

        while collected < n_vp:
            obs, _ = eval_envs.reset(seed=int(seed * 1000 + vp_id * 100 + collected))
            for step in range(50):
                with torch.no_grad():
                    act = student.get_action(obs["rgbd"], obs["state"], deterministic=True)
                obs, _, _, _, infos = eval_envs.step(act)

            if "final_info" in infos and "_final_info" in infos:
                mask = infos["_final_info"]
                ep_info = infos["final_info"].get("episode", {})
                succ_key = "success_once" if "success_once" in ep_info else "success"
                for env_i in range(len(mask)):
                    if collected >= n_vp:
                        break
                    if mask[env_i]:
                        trial_idx = vp_indices[collected]
                        val = ep_info[succ_key][env_i]
                        succ = bool(val.item() if torch.is_tensor(val) else val)
                        trial_results[trial_idx] = {
                            "trial_id": int(trial_idx),
                            "viewpoint_id": int(vp_id),
                            "viewpoint_name": VP_ROLES[vp_id],
                            "success": succ,
                        }
                        if succ:
                            vp_successes += 1
                        collected += 1

        eval_envs.close()
        acc = (vp_successes / n_vp) * 100.0
        per_vp_results[vp_id] = {
            "viewpoint_id": vp_id,
            "role": VP_ROLES[vp_id],
            "azimuth": meta["azimuth_deg"],
            "elevation": meta["elevation_deg"],
            "trials": n_vp,
            "successes": vp_successes,
            "accuracy": acc,
        }
        print(f"  VP {vp_id:02d} ({VP_ROLES[vp_id]:<22}): {vp_successes:3d}/{n_vp:3d} ({acc:5.1f}%)")

    total_time = time.time() - t0
    total_successes = sum(r["successes"] for r in per_vp_results.values())
    total_trials = sum(r["trials"] for r in per_vp_results.values())
    overall_acc = (total_successes / total_trials) * 100.0

    viable_successes = sum(r["successes"] for vid, r in per_vp_results.items() if vid != 28)
    viable_trials = sum(r["trials"] for vid, r in per_vp_results.items() if vid != 28)
    viable_acc = (viable_successes / viable_trials) * 100.0 if viable_trials > 0 else 0.0

    print("\n" + "=" * 80)
    print(f" BENCHMARK RESULTS ({total_trials:,} Random Trials in {total_time:.1f}s):")
    print(f" Overall Accuracy:        {overall_acc:.2f}% ({total_successes}/{total_trials})")
    print(f" Viable Views Accuracy:   {viable_acc:.2f}% ({viable_successes}/{viable_trials})")
    print(f" VP 28 (Occluded View):   {per_vp_results[28]['accuracy']:.1f}% ({per_vp_results[28]['successes']}/{per_vp_results[28]['trials']})")
    print("=" * 80)

    # Save summary JSON
    summary_path = os.path.join(EVAL_DIR, "random_1000_tests_summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "overall_accuracy": overall_acc,
            "viable_accuracy": viable_acc,
            "vp_28_accuracy": per_vp_results[28]["accuracy"],
            "total_time_seconds": total_time,
            "per_viewpoint": per_vp_results,
        }, f, indent=2)

    # Generate Chart
    plt.figure(figsize=(12, 6))
    names = [f"VP {v}\n{VP_ROLES[v]}" for v in CORE_VP_IDS]
    accs = [per_vp_results[v]["accuracy"] for v in CORE_VP_IDS]
    colors = ["#2ecc71" if acc >= 85 else "#f39c12" if acc >= 50 else "#e74c3c" for acc in accs]

    bars = plt.bar(names, accs, color=colors, width=0.6, edgecolor="black", linewidth=1.2)
    plt.axhline(y=overall_acc, color="#2980b9", linestyle="--", linewidth=2, label=f"Overall Mean: {overall_acc:.1f}%")
    plt.axhline(y=viable_acc, color="#27ae60", linestyle=":", linewidth=2, label=f"Viable Mean: {viable_acc:.1f}%")

    for bar, acc in zip(bars, accs):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5, f"{acc:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=10)

    plt.ylim(0, 110)
    plt.ylabel("Grasp & Lift Success Rate (%)", fontsize=12, fontweight="bold")
    plt.title(f"Phase 10 OmniView Policy: 1,000 Randomized Viewpoint Benchmark\n(Temporal Stacking + Camera Extrinsics Conditioning)", fontsize=13, fontweight="bold")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.legend(loc="lower right", fontsize=11)
    plt.tight_layout()

    plot_path = os.path.join(EVAL_DIR, "random_1000_tests_accuracy_breakdown.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Chart saved to: {plot_path}")

    # Copy to reports in standalone repo
    repo_reports = "/home/maniskill/Multi-viewpoint_Generalization/reports"
    os.makedirs(repo_reports, exist_ok=True)
    os.system(f"cp {plot_path} {repo_reports}/phase10_omniview_1000_tests_breakdown.png")
    print(f"Copied benchmark plot to {repo_reports}/phase10_omniview_1000_tests_breakdown.png")

    return overall_acc, viable_acc, per_vp_results


if __name__ == "__main__":
    run_1000_benchmark()
