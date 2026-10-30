"""
Cross-Viewpoint Transfer Matrix Evaluation.
Evaluates every trained specialist model across all 8 core viewpoints (8x8 evaluation matrix).
Produces the generalization transfer heatmap and JSON summary.
"""

import os
import sys
import json
import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import gymnasium as gym

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, "/home/maniskill")

import mani_skill.envs
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper, FlattenRGBDObservationWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from models import UnifiedAblationAgent

CORE_VP_IDS = [17, 0, 28, 18, 27, 20, 19, 6]


def evaluate_model_on_viewpoint(agent, eye, target, num_episodes=100, num_envs=50, max_steps=50, device="cuda"):
    env_kwargs = dict(
        obs_mode="rgbd",
        render_mode="all",
        sim_backend="physx_cuda",
        sensor_cam_eye_pos=eye,
        sensor_cam_target_pos=target,
        control_mode="pd_ee_delta_pos",
        sensor_configs=dict(width=128, height=128),
    )
    env = gym.make("PickCube-v1", num_envs=num_envs, reconfiguration_freq=None, **env_kwargs)
    env = FlattenRGBDObservationWrapper(env, rgb=True, depth=True, state=True, sep_depth=True)
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    env = ManiSkillVectorEnv(env, num_envs, ignore_terminations=True, record_metrics=True)

    agent.eval()
    eval_obs, _ = env.reset()
    eval_successes = 0
    eval_eps = 0
    returns = []
    lengths = []

    with torch.no_grad():
        for _ in range(max_steps):
            act = agent.get_action(eval_obs, deterministic=True)
            eval_obs, _, _, _, eval_infos = env.step(act)
            if "final_info" in eval_infos:
                mask = eval_infos.get("_final_info", None)
                if mask is not None:
                    for i in range(len(mask)):
                        if mask[i]:
                            ep_info = eval_infos["final_info"].get("episode", {})
                            succ_key = "success_once" if "success_once" in ep_info else "success"
                            succ = False
                            if succ_key in ep_info:
                                val = ep_info[succ_key][i]
                                succ = bool(val.item() if torch.is_tensor(val) else val)
                            if succ:
                                eval_successes += 1
                            if "r" in ep_info:
                                r_val = ep_info["r"][i]
                                returns.append(float(r_val.item() if torch.is_tensor(r_val) else r_val))
                            if "l" in ep_info:
                                l_val = ep_info["l"][i]
                                lengths.append(float(l_val.item() if torch.is_tensor(l_val) else l_val))
                            eval_eps += 1

    env.close()
    succ_rate = (eval_successes / max(eval_eps, 1)) * 100.0 if eval_eps > 0 else 0.0
    return succ_rate, float(np.mean(returns)) if returns else 0.0, float(np.mean(lengths)) if lengths else 0.0


def run_matrix_eval(output_dir="/home/maniskill/phase_6_viewpoint_specialists", episodes_per_cell=200):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    eval_dir = os.path.join(output_dir, "eval_results")
    vps_file = "/home/maniskill/phase_1_to_3/sequential_viewpoint_training/viewpoints.json"

    with open(vps_file) as f:
        all_vps = json.load(f)
    vp_dict = {vp["id"]: vp for vp in all_vps}

    target = [-0.1, 0.0, 0.1]
    matrix = np.zeros((len(CORE_VP_IDS), len(CORE_VP_IDS)))

    # Dummy env to initialize agent architecture
    dummy_kwargs = dict(
        obs_mode="rgbd",
        control_mode="pd_ee_delta_pos",
        sim_backend="physx_cuda",
        sensor_configs=dict(width=128, height=128),
    )
    dummy_env = gym.make("PickCube-v1", num_envs=1, **dummy_kwargs)
    dummy_env = FlattenRGBDObservationWrapper(dummy_env, rgb=True, depth=True, state=True, sep_depth=True)
    if isinstance(dummy_env.action_space, gym.spaces.Dict):
        dummy_env = FlattenActionSpaceWrapper(dummy_env)
    dummy_env = ManiSkillVectorEnv(dummy_env, 1, ignore_terminations=False)
    sample_obs, _ = dummy_env.reset()

    agent = UnifiedAblationAgent(dummy_env, sample_obs, candidate="A").to(device)
    dummy_env.close()

    print(f"\n{'='*78}")
    print(f" COMPUTING 8x8 CROSS-VIEWPOINT TRANSFER MATRIX ({episodes_per_cell} episodes per cell)")
    print(f"{'='*78}\n")

    matrix_dict = {}

    for row_idx, model_vpid in enumerate(CORE_VP_IDS):
        ckpt_path = os.path.join(ckpt_dir, f"specialist_vp_{model_vpid:02d}_best.pt")
        if not os.path.exists(ckpt_path):
            print(f"[Warning] Checkpoint for VP {model_vpid:02d} not found at {ckpt_path}. Skipping row.")
            continue

        ckpt = torch.load(ckpt_path, map_location=device)
        agent.load_state_dict(ckpt["model_state"])
        matrix_dict[f"model_vp_{model_vpid:02d}"] = {}

        for col_idx, test_vpid in enumerate(CORE_VP_IDS):
            test_vp = vp_dict[test_vpid]
            eye = test_vp["world_eye"]

            succ, ret, length = evaluate_model_on_viewpoint(
                agent=agent,
                eye=eye,
                target=target,
                num_episodes=episodes_per_cell,
                num_envs=50,
                device=device,
            )
            matrix[row_idx, col_idx] = succ
            matrix_dict[f"model_vp_{model_vpid:02d}"][f"test_vp_{test_vpid:02d}"] = {
                "success_rate": round(succ, 1),
                "mean_return": round(ret, 2),
                "mean_length": round(length, 1),
            }
            print(f" Model VP {model_vpid:02d} on Test VP {test_vpid:02d} ({test_vp['azimuth_deg']:+.0f}°, {test_vp['elevation_deg']:.0f}°): {succ:5.1f}%")

    # Save JSON and CSV
    matrix_json_path = os.path.join(eval_dir, "cross_viewpoint_transfer_matrix.json")
    with open(matrix_json_path, "w") as f:
        json.dump(matrix_dict, f, indent=2)

    labels = [f"VP {v:02d}" for v in CORE_VP_IDS]
    csv_path = os.path.join(eval_dir, "cross_viewpoint_transfer_matrix.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Model"] + labels)
        for idx, row in enumerate(matrix):
            writer.writerow([labels[idx]] + [round(val, 1) for val in row])

    # Plot Heatmap with pure Matplotlib
    plt.figure(figsize=(10, 8), dpi=150)
    plt.gcf().patch.set_facecolor("#0f172a")
    ax = plt.subplot(1, 1, 1)
    ax.set_facecolor("#0f172a")

    im = ax.imshow(matrix, cmap="viridis", vmin=0, vmax=100)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Grasp Success Rate (%)", color="#cbd5e1", labelpad=10)
    cbar.ax.yaxis.set_tick_params(color="#94a3b8")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="#94a3b8")

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, color="#cbd5e1")
    ax.set_yticklabels(labels, color="#cbd5e1")

    # Annotate values inside cells
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = matrix[i, j]
            color = "#ffffff" if val < 50 else "#000000"
            ax.text(j, i, f"{val:.1f}%", ha="center", va="center", color=color, fontsize=9, weight="bold")

    plt.title("8x8 Cross-Viewpoint Transfer Matrix\n(Rows: Trained Model | Columns: Evaluated Viewpoint)", color="#f8fafc", fontsize=12, pad=15)
    plt.xlabel("Evaluation Viewpoint", color="#cbd5e1", labelpad=10)
    plt.ylabel("Trained Specialist Model", color="#cbd5e1", labelpad=10)
    plt.tight_layout()

    heatmap_path = os.path.join(eval_dir, "cross_viewpoint_transfer_heatmap.png")
    plt.savefig(heatmap_path, facecolor=plt.gcf().get_facecolor(), edgecolor="none")
    plt.close()
    print(f"\n[Done] Saved Cross-Viewpoint Transfer Heatmap to: {heatmap_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="/home/maniskill/phase_6_viewpoint_specialists")
    parser.add_argument("--episodes", type=int, default=200)
    args = parser.parse_args()
    run_matrix_eval(args.output_dir, args.episodes)
