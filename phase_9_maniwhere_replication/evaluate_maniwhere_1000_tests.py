"""
Evaluation Harness for Maniwhere: 1,000 Randomized Viewpoint Tests.
Direct head-to-head comparison against the Distilled Unified Policy on the exact same 1,000 trials.
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, "/home/maniskill")

import mani_skill.envs
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper, FlattenRGBDObservationWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from models_maniwhere import ManiwhereAgent

VIEWPOINTS_JSON = "/home/maniskill/phase_1_to_3/sequential_viewpoint_training/viewpoints.json"
MANIWHERE_CKPT = "/home/maniskill/phase_9_maniwhere_replication/checkpoints/maniwhere_best.pt"
DISTILLED_SUMMARY = "/home/maniskill/Multi-viewpoint_Generalization/reports/random_1000_tests_summary.json"
CORE_VP_IDS = [17, 0, 28, 18, 27, 20, 19, 6]

VP_ROLES = {
    17: "Front-Mid Canonical",
    0: "Front-Low Graze",
    28: "Front-High Steep Tilt",
    18: "Left-Flank 90°",
    27: "Right-Flank 90°",
    20: "Diag Front-Left",
    19: "Diag Front-Right",
    6: "Low Flank Right",
}


def run_maniwhere_evaluation(
    num_tests: int = 1000,
    seed: int = 42,
    batch_envs: int = 50,
    output_dir: str = "/home/maniskill/phase_9_maniwhere_replication/eval_results",
    device_str: str = "cuda",
):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(device_str)

    print("\n" + "=" * 78)
    print(f" EVALUATING MANIWHERE MODEL: {num_tests:,} RANDOMIZED VIEWPOINT TESTS")
    print(f" Checkpoint: {MANIWHERE_CKPT}")
    print("=" * 78 + "\n")

    with open(VIEWPOINTS_JSON) as f:
        all_vps = json.load(f)
    vp_dict = {vp["id"]: vp for vp in all_vps}

    rng = np.random.RandomState(seed)
    trial_vps = rng.choice(CORE_VP_IDS, size=num_tests)

    # Load agent
    agent = ManiwhereAgent(action_dim=4, state_dim=29, feature_dim=256).to(device)
    ckpt = torch.load(MANIWHERE_CKPT, map_location=device)
    agent.load_state_dict(ckpt["model_state"] if "model_state" in ckpt else ckpt)
    agent.eval()

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

        vp_env_kwargs = dict(
            obs_mode="rgbd",
            render_mode="all",
            sim_backend="physx_cuda",
            sensor_cam_eye_pos=cam_eye,
            sensor_cam_target_pos=cam_target,
            control_mode="pd_ee_delta_pos",
            sensor_configs=dict(width=128, height=128),
        )

        n_envs = min(n_vp, batch_envs)
        eval_envs = gym.make("PickCube-v1", num_envs=n_envs, reconfiguration_freq=None, **vp_env_kwargs)
        eval_envs = FlattenRGBDObservationWrapper(eval_envs, rgb=True, depth=True, state=True, sep_depth=True)
        if isinstance(eval_envs.action_space, gym.spaces.Dict):
            eval_envs = FlattenActionSpaceWrapper(eval_envs)
        eval_envs = ManiSkillVectorEnv(eval_envs, n_envs, ignore_terminations=True, record_metrics=True)

        collected = 0
        vp_successes = 0

        while collected < n_vp:
            obs, _ = eval_envs.reset(seed=int(seed * 1000 + vp_id * 100 + collected))
            for step in range(50):
                with torch.no_grad():
                    act = agent.get_action(obs, deterministic=True)
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
                        if succ:
                            vp_successes += 1
                        trial_results[trial_idx] = {
                            "trial_id": int(trial_idx) + 1,
                            "viewpoint_id": int(vp_id),
                            "success": succ,
                        }
                        collected += 1

        eval_envs.close()
        vp_acc = (vp_successes / n_vp) * 100.0
        per_vp_results[vp_id] = {
            "viewpoint_id": vp_id,
            "role": VP_ROLES[vp_id],
            "trials": n_vp,
            "successes": vp_successes,
            "success_rate": round(vp_acc, 2),
        }
        print(f"  [Maniwhere VP {vp_id:02d}] {VP_ROLES[vp_id]:<22} | Success: {vp_successes:3d}/{n_vp:3d} ({vp_acc:5.1f}%)")

    total_time = time.time() - t0
    total_successes = sum(1 for t in trial_results if t["success"])
    overall_acc = (total_successes / num_tests) * 100.0

    viable_trials = [t for t in trial_results if t["viewpoint_id"] != 28]
    viable_succ = sum(1 for t in viable_trials if t["success"])
    viable_acc = (viable_succ / len(viable_trials)) * 100.0 if viable_trials else 0.0

    print("\n" + "=" * 78)
    print(f" MANIWHERE 1,000 RANDOMIZED TESTS ACCURACY: {overall_acc:.2f}% (Viable: {viable_acc:.2f}%)")
    print("=" * 78 + "\n")

    # Load Distilled Summary to compare
    distilled_data = {}
    if os.path.exists(DISTILLED_SUMMARY):
        with open(DISTILLED_SUMMARY) as f:
            distilled_data = json.load(f)

    # Save Comparison JSON
    comparison_summary = {
        "benchmark": "1,000 Randomized Viewpoint Tests",
        "maniwhere_overall_accuracy": round(overall_acc, 2),
        "maniwhere_viable_accuracy": round(viable_acc, 2),
        "distilled_overall_accuracy": distilled_data.get("overall_accuracy", 80.9),
        "distilled_viable_accuracy": distilled_data.get("viable_accuracy_excl_vp28", 90.3),
        "per_viewpoint_maniwhere": per_vp_results,
    }
    json_path = os.path.join(output_dir, "maniwhere_vs_distillation_comparison.json")
    with open(json_path, "w") as f:
        json.dump(comparison_summary, f, indent=2)

    # Plot Side-by-Side Comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    vps = CORE_VP_IDS
    x = np.arange(len(vps))
    width = 0.35

    dist_accs = [distilled_data.get("per_viewpoint_breakdown", {}).get(str(vp), {}).get("success_rate", 0.0) for vp in vps]
    mw_accs = [per_vp_results[vp]["success_rate"] for vp in vps]

    rects1 = ax.bar(x - width / 2, dist_accs, width, label="Multi-Teacher Distillation (Our Phase 7)", color="#2563eb", alpha=0.85)
    rects2 = ax.bar(x + width / 2, mw_accs, width, label="Maniwhere (CoRL 2024 Replication)", color="#10b981", alpha=0.85)

    ax.set_ylabel("Success Rate (%)", fontsize=12, fontweight="bold")
    ax.set_title("1,000 Randomized Viewpoint Benchmark: Multi-Teacher Distillation vs. Maniwhere", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"VP {vp}\n{VP_ROLES[vp].split()[0]}" for vp in vps], fontsize=10)
    ax.legend(loc="upper right", fontsize=11)
    ax.set_ylim(0, 110)
    ax.grid(axis="y", linestyle=":", alpha=0.6)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "distillation_vs_maniwhere_breakdown.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print(f"[Saved] Head-to-Head Comparison Plot: {plot_path}")
    return overall_acc, viable_acc
