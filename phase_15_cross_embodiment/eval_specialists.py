"""
Phase 15: Evaluation of Robot Specialists Before Student Training.

Evaluates:
1. Franka Panda Specialist (panda_specialist_best.pt) on Franka Panda across 13 viewpoints.
2. UFactory xArm 6 Robotiq Specialist (xarm6_specialist_best.pt) on xArm 6 Robotiq across 13 viewpoints.
"""

import os
import sys
sys.path.append("/home/maniskill")
import time
import json
import gc
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gymnasium as gym

import mani_skill.envs
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from phase_15_cross_embodiment.run_cross_embodiment_master import (
    UniversalCartesianPointNetAgent,
    extract_table_frame_pc,
    get_universal_proprio,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = "/home/maniskill/phase_15_cross_embodiment"
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")
EVAL_DIR = os.path.join(BASE_DIR, "eval_results")
VIEWPOINTS_JSON = "/home/maniskill/phase_1_to_3/sequential_viewpoint_training/viewpoints.json"

os.makedirs(EVAL_DIR, exist_ok=True)


def evaluate_specialist(model_path: str, robot_uid: str, specialist_name: str, num_envs_per_vp: int = 50):
    print("\n" + "=" * 80)
    print(f" BENCHMARKING SPECIALIST: {specialist_name.upper()} ON {robot_uid.upper()}")
    print(f" Checkpoint: {model_path}")
    print(f" Test Suite: 13 Viewpoints (4 Anchors + 9 Novel In-Betweens) | {num_envs_per_vp} envs/vp")
    print("=" * 80)

    agent = UniversalCartesianPointNetAgent(action_dim=4, proprio_dim=8, point_dim=256).to(DEVICE)
    ckpt = torch.load(model_path, map_location=DEVICE)
    agent.load_state_dict(ckpt.get("model_state", ckpt))
    agent.eval()

    with open(VIEWPOINTS_JSON) as f:
        vps = json.load(f)
    vp_dict = {v["id"]: v for v in vps}

    eval_vps = [
        # 4 Anchors
        (17, "VP 17 (Front-Mid Anchor)", "anchor"),
        (0,  "VP 00 (Front-Low Anchor)", "anchor"),
        (20, "VP 20 (Diag-Left Anchor)", "anchor"),
        (19, "VP 19 (Diag-Right Anchor)", "anchor"),
        # 9 Novel In-Betweens
        (10, "VP 10 (In-Between Elevation)", "inbetween"),
        (13, "VP 13 (In-Between Az: VP17-20)", "inbetween"),
        (14, "VP 14 (In-Between Az: VP17-19)", "inbetween"),
        (8,  "VP 08 (Low-El Interpolated)", "inbetween"),
        (9,  "VP 09 (Low-El Interpolated)", "inbetween"),
        (12, "VP 12 (Flank In-Between R)", "inbetween"),
        (15, "VP 15 (Flank In-Between L)", "inbetween"),
        (21, "VP 21 (High-El Steep Angle)", "inbetween"),
        (22, "VP 22 (High-El Steep Angle)", "inbetween"),
    ]

    results = []
    t0 = time.time()

    for vid, name, vtype in eval_vps:
        meta = vp_dict[vid]
        cam_eye = meta["world_eye"]

        env = gym.make(
            "PickCube-v1",
            robot_uids=robot_uid,
            obs_mode="pointcloud+state",
            control_mode="pd_ee_delta_pos",
            shader_dir="minimal",
            num_envs=num_envs_per_vp,
            sensor_cam_eye_pos=cam_eye,
            sensor_cam_target_pos=[-0.1, 0.0, 0.1],
            sensor_configs=dict(width=128, height=128),
        )
        eval_envs = ManiSkillVectorEnv(env, num_envs_per_vp, ignore_terminations=True, record_metrics=True)
        obs, _ = eval_envs.reset(seed=2026 + vid)
        grasped_any = torch.zeros(num_envs_per_vp, dtype=torch.bool, device=DEVICE)

        for step in range(50):
            proprio = get_universal_proprio(env, robot_uid)
            with torch.no_grad():
                pts = extract_table_frame_pc(obs, num_points=512)
                act_universal = agent.get_action(pts, proprio, deterministic=True)

            act_env = act_universal.clone()
            if robot_uid == "xarm6_robotiq":
                act_env[:, 3] = -act_universal[:, 3]

            obs, rew, term, trunc, infos = eval_envs.step(act_env)
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

        results.append({
            "id": vid,
            "name": name,
            "type": vtype,
            "success_rate": succ_rate,
            "grasp_rate": grasp_rate,
        })
        v_tag = "[ANCHOR]" if vtype == "anchor" else "[IN-BETWEEN]"
        print(f"  {v_tag:<13} {name:<32}: Succ={succ_rate:5.1f}% | Grasp={grasp_rate:5.1f}%")

    anchor_succ = np.mean([r["success_rate"] for r in results if r["type"] == "anchor"])
    inbetween_succ = np.mean([r["success_rate"] for r in results if r["type"] == "inbetween"])
    overall_succ = np.mean([r["success_rate"] for r in results])
    overall_grasp = np.mean([r["grasp_rate"] for r in results])

    print(f"\n================================================================================")
    print(f" {specialist_name.upper()} RESULTS SUMMARY (Time: {time.time()-t0:.1f}s):")
    print(f" Overall Success Rate:            {overall_succ:5.1f}%")
    print(f" Novel In-Between Viewpoints (9): {inbetween_succ:5.1f}%")
    print(f" Anchor Viewpoints (4):           {anchor_succ:5.1f}%")
    print(f" Overall Grasp Rate:              {overall_grasp:5.1f}%")
    print(f"================================================================================\n")

    return {
        "specialist_name": specialist_name,
        "robot_uid": robot_uid,
        "model_path": model_path,
        "overall_success": overall_succ,
        "anchor_success": anchor_succ,
        "inbetween_success": inbetween_succ,
        "overall_grasp": overall_grasp,
        "details": results,
    }


def main():
    panda_ckpt = os.path.join(CKPT_DIR, "panda_specialist_best.pt")
    xarm6_ckpt = os.path.join(CKPT_DIR, "xarm6_specialist_best.pt")

    assert os.path.exists(panda_ckpt), f"Panda checkpoint not found: {panda_ckpt}"
    assert os.path.exists(xarm6_ckpt), f"xArm6 checkpoint not found: {xarm6_ckpt}"

    all_results = {}

    # 1. Evaluate Panda Specialist on Panda
    all_results["panda_specialist"] = evaluate_specialist(
        panda_ckpt, robot_uid="panda", specialist_name="Franka Panda Specialist", num_envs_per_vp=50
    )

    # 2. Evaluate xArm 6 Robotiq Specialist on xArm 6
    all_results["xarm6_specialist"] = evaluate_specialist(
        xarm6_ckpt, robot_uid="xarm6_robotiq", specialist_name="xArm 6 Robotiq Specialist", num_envs_per_vp=50
    )

    # Save benchmark JSON
    json_path = os.path.join(EVAL_DIR, "specialists_accuracy_benchmark.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"[OK] Specialist accuracy results saved to {json_path}")

    # Generate Comparative Bar Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    specialists = ["panda_specialist", "xarm6_specialist"]
    labels = ["Franka Panda Specialist\n(7 DoF, 4D Action)", "UFactory xArm 6 Robotiq\n(6 DoF, 4D Action)"]
    x = np.arange(len(specialists))
    w = 0.22

    anchors = [all_results[s]["anchor_success"] for s in specialists]
    inbetweens = [all_results[s]["inbetween_success"] for s in specialists]
    overalls = [all_results[s]["overall_success"] for s in specialists]
    grasps = [all_results[s]["overall_grasp"] for s in specialists]

    ax.bar(x - 1.5*w, grasps, w, label="Grasp Rate (%)", color="#b0bec5", edgecolor="black", alpha=0.85)
    ax.bar(x - 0.5*w, anchors, w, label="Anchor Viewpoints (4)", color="#1976d2", edgecolor="black")
    ax.bar(x + 0.5*w, inbetweens, w, label="Novel In-Betweens (9)", color="#ff9800", edgecolor="black")
    rects = ax.bar(x + 1.5*w, overalls, w, label="Overall Success (%)", color="#2e7d32", edgecolor="black")

    for r in rects:
        h = r.get_height()
        ax.annotate(f"{h:.1f}%", xy=(r.get_x() + r.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_ylabel("Rate (%)", fontsize=12, fontweight="bold")
    ax.set_title("Pre-Student Accuracy Validation: Robot Specialist Policies\n(Evaluated on 13 Viewpoints: 4 Anchors + 9 Novel In-Betweens)", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", fontsize=10.5)

    plot_path = os.path.join(EVAL_DIR, "specialists_accuracy_benchmark.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"[OK] Comparison plot saved to {plot_path}")


if __name__ == "__main__":
    main()
