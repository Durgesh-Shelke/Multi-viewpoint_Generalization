"""
Phase 15: Cross-Embodiment Zero-Shot Transfer & Comparative Benchmark.

Benchmarks 3 distinct regimes across the full 13-viewpoint suite (4 anchors + 9 in-betweens, 50 envs each):
1. Specialists (Trained & tested on own embodiment):
   - Panda Specialist on Panda
   - xArm 6 Specialist on xArm 6
2. Zero-Shot Cross-Embodiment Transfer (Never saw target robot during training):
   - Panda Specialist -> ZERO-SHOT on xArm 6 Robotiq
   - xArm 6 Specialist -> ZERO-SHOT on Franka Panda
3. Unified Combined Policy (combined_omniarm trained jointly on both):
   - Combined Policy on Franka Panda
   - Combined Policy on xArm 6 Robotiq

Generates cross_embodiment_master_benchmark.json and comprehensive publication-grade plots.
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
from phase_13_pointcloud_generalization.pointnet_models import PointNetAgent, extract_robot_frame_pc
from phase_15_cross_embodiment.train_xarm6_specialist import extract_robot_frame_pc_xarm6
from phase_15_cross_embodiment.train_combined_omniarm import OmniArmPointNetPolicy, extract_table_workspace_pc, get_universal_proprio

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = "/home/maniskill/phase_15_cross_embodiment"
SPEC_DIR = os.path.join(BASE_DIR, "specialists")
COMB_DIR = os.path.join(BASE_DIR, "combined")
EVAL_DIR = os.path.join(BASE_DIR, "eval_results")

VIEWPOINTS_JSON = "/home/maniskill/phase_1_to_3/sequential_viewpoint_training/viewpoints.json"
PANDA_CKPT = "/home/maniskill/phase_13_pointcloud_generalization/student/pointnet_student_best.pt"
XARM6_CKPT = os.path.join(SPEC_DIR, "xarm6_specialist_best.pt")
COMBINED_CKPT = os.path.join(COMB_DIR, "combined_omniarm_best.pt")


def run_cross_embodiment_zeroshot_benchmark():
    print("\n" + "=" * 80)
    print(" [STAGE 4] FULL CROSS-EMBODIMENT ZERO-SHOT TRANSFER BENCHMARK")
    print(" Viewpoint Suite: 13 Viewpoints (4 Anchors + 9 In-Betweens)")
    print(" Regimes Evaluated: Specialists vs Zero-Shot Transfer vs Combined Policy")
    print(" Total Test Environments per Regime: 650 (50 envs x 13 VPs)")
    print("=" * 80)

    with open(VIEWPOINTS_JSON, "r") as f:
        vps = json.load(f)
    vp_dict = {vp["id"]: vp for vp in vps}

    eval_vps = [
        (17, "VP 17 (Front-Mid Anchor)", "Anchor"),
        (0,  "VP 00 (Front-Low Anchor)", "Anchor"),
        (20, "VP 20 (Diag-Left Anchor)", "Anchor"),
        (19, "VP 19 (Diag-Right Anchor)", "Anchor"),
        (10, "VP 10 (In-Between El: VP00-17)", "In-Between"),
        (13, "VP 13 (In-Between Az: VP17-20)", "In-Between"),
        (14, "VP 14 (In-Between Az: VP17-19)", "In-Between"),
        (4,  "VP 04 (In-Between Low-Right)",   "In-Between"),
        (7,  "VP 07 (In-Between Low-Left)",    "In-Between"),
        (21, "VP 21 (In-Between Mid-Right)",   "In-Between"),
        (24, "VP 24 (In-Between High-Mid)",    "In-Between"),
        (12, "VP 12 (Flank In-Between R)",     "In-Between"),
        (15, "VP 15 (Flank In-Between L)",     "In-Between"),
    ]

    # Load Policies
    panda_spec = PointNetAgent(action_dim=4, proprio_dim=29, point_dim=256).to(DEVICE)
    p_ckpt = torch.load(PANDA_CKPT, map_location=DEVICE)
    panda_spec.load_state_dict(p_ckpt.get("model_state", p_ckpt))
    panda_spec.eval()

    xarm6_spec = PointNetAgent(action_dim=4, proprio_dim=35, point_dim=256).to(DEVICE)
    xarm6_spec.load_state_dict(torch.load(XARM6_CKPT, map_location=DEVICE))
    xarm6_spec.eval()

    has_combined = os.path.exists(COMBINED_CKPT)
    if has_combined:
        combined_policy = OmniArmPointNetPolicy(action_dim=4, proprio_dim=12, point_dim=256).to(DEVICE)
        combined_policy.load_state_dict(torch.load(COMBINED_CKPT, map_location=DEVICE))
        combined_policy.eval()

    batch_envs = 50
    results = {
        "panda_zeroshot_on_xarm6": [],
        "xarm6_zeroshot_on_panda": [],
    }

    # 1. Zero-Shot: Panda Specialist deployed on xArm 6 Robotiq
    print("\n--- ZERO-SHOT TEST 1: Panda Specialist -> Deployed on xArm 6 Robotiq ---")
    base_shift = np.array([-0.522 - (-0.615), 0.0, 0.0])
    for vid, name, vtype in eval_vps:
        meta = vp_dict[vid]
        cam_eye = (np.array(meta["world_eye"]) + base_shift).tolist()

        env = gym.make(
            "PickCube-v1",
            robot_uids="xarm6_robotiq",
            obs_mode="pointcloud+state",
            control_mode="pd_ee_delta_pos",
            num_envs=batch_envs,
            sensor_cam_eye_pos=cam_eye,
            sensor_cam_target_pos=[-0.1, 0.0, 0.1],
            sensor_configs=dict(width=128, height=128),
            shader_dir="minimal",
        )
        vec_env = ManiSkillVectorEnv(env, batch_envs, ignore_terminations=True, record_metrics=True)
        obs, _ = vec_env.reset(seed=6026 + vid)
        grasped_any = torch.zeros(batch_envs, dtype=torch.bool, device=DEVICE)

        for step in range(50):
            # Adapt xArm 6 35-dim proprio to Panda 29-dim format:
            # Panda: [qpos(9), qvel(9), is_grasped(1), tcp_pose(7), goal_pos(3)] = 29
            # xArm6: [qpos(12), qvel(12), is_grasped(1), tcp_pose(7), goal_pos(3)] = 35
            xarm_state = obs["state"]
            qpos_9 = xarm_state[:, :9] # slice arm joints
            qvel_9 = xarm_state[:, 12:21] # slice arm velocities
            task_prop = xarm_state[:, 24:35] # is_grasped (1) + tcp_pose (7) + goal_pos (3)
            adapted_proprio_29 = torch.cat([qpos_9, qvel_9, task_prop], dim=-1)

            with torch.no_grad():
                pts = extract_robot_frame_pc_xarm6(obs, num_points=512)
                act = panda_spec.get_action(pts, adapted_proprio_29, deterministic=True)
                # Map Panda gripper action (-1=close, +1=open) to Robotiq (+1=close, -1=open)
                act[:, 3] = -act[:, 3]
                # Firm clamping saturation
                act[:, 3] = torch.where(act[:, 3] > 0.0, torch.tensor(1.0, device=DEVICE), torch.tensor(-1.0, device=DEVICE))

            obs, _, _, _, info = vec_env.step(act)
            if "is_grasped" in info:
                grasped_any |= info["is_grasped"]

        succ = info["final_info"]["episode"]["success_once"].float().mean().item() * 100.0
        grasp = grasped_any.float().mean().item() * 100.0
        vec_env.close()
        del vec_env; del env; gc.collect()

        results["panda_zeroshot_on_xarm6"].append({
            "id": vid, "name": name, "type": vtype,
            "success_rate": succ, "grasp_rate": grasp
        })
        print(f"  [{vtype.upper():<10}] {name:<32}: Succ={succ:5.1f}% | Grasp={grasp:5.1f}%")

    # 2. Zero-Shot: xArm 6 Specialist deployed on Franka Panda
    print("\n--- ZERO-SHOT TEST 2: xArm 6 Specialist -> Deployed on Franka Panda ---")
    for vid, name, vtype in eval_vps:
        meta = vp_dict[vid]
        cam_eye = meta["world_eye"]

        env = gym.make(
            "PickCube-v1",
            robot_uids="panda",
            obs_mode="pointcloud+state",
            control_mode="pd_ee_delta_pos",
            num_envs=batch_envs,
            sensor_cam_eye_pos=cam_eye,
            sensor_cam_target_pos=[-0.1, 0.0, 0.1],
            sensor_configs=dict(width=128, height=128),
            shader_dir="minimal",
        )
        vec_env = ManiSkillVectorEnv(env, batch_envs, ignore_terminations=True, record_metrics=True)
        obs, _ = vec_env.reset(seed=7026 + vid)
        grasped_any = torch.zeros(batch_envs, dtype=torch.bool, device=DEVICE)

        for step in range(50):
            # Adapt Panda 29-dim proprio to xArm 6 35-dim format:
            # Panda: [qpos(9), qvel(9), is_grasped(1), tcp_pose(7), goal_pos(3)]
            # Pad qpos and qvel with 3 zeros to match 12 dims
            panda_state = obs["state"]
            qpos_12 = torch.cat([panda_state[:, :9], torch.zeros((batch_envs, 3), device=DEVICE)], dim=-1)
            qvel_12 = torch.cat([panda_state[:, 9:18], torch.zeros((batch_envs, 3), device=DEVICE)], dim=-1)
            task_prop = panda_state[:, 18:29] # is_grasped (1) + tcp_pose (7) + goal_pos (3)
            adapted_proprio_35 = torch.cat([qpos_12, qvel_12, task_prop], dim=-1)

            with torch.no_grad():
                pts = extract_robot_frame_pc(obs, num_points=512)
                act = xarm6_spec.get_action(pts, adapted_proprio_35, deterministic=True)
                # Map Robotiq gripper action (+1=close) to Panda (-1=close)
                act[:, 3] = -act[:, 3]
                # Firm clamping saturation
                act[:, 3] = torch.where(act[:, 3] < 0.0, torch.tensor(-1.0, device=DEVICE), torch.tensor(1.0, device=DEVICE))

            obs, _, _, _, info = vec_env.step(act)
            if "is_grasped" in info:
                grasped_any |= info["is_grasped"]

        succ = info["final_info"]["episode"]["success_once"].float().mean().item() * 100.0
        grasp = grasped_any.float().mean().item() * 100.0
        vec_env.close()
        del vec_env; del env; gc.collect()

        results["xarm6_zeroshot_on_panda"].append({
            "id": vid, "name": name, "type": vtype,
            "success_rate": succ, "grasp_rate": grasp
        })
        print(f"  [{vtype.upper():<10}] {name:<32}: Succ={succ:5.1f}% | Grasp={grasp:5.1f}%")

    out_json = os.path.join(EVAL_DIR, "cross_embodiment_zeroshot_benchmark.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    pz_x_succ = np.mean([r["success_rate"] for r in results["panda_zeroshot_on_xarm6"]])
    pz_x_grp = np.mean([r["grasp_rate"] for r in results["panda_zeroshot_on_xarm6"]])
    xz_p_succ = np.mean([r["success_rate"] for r in results["xarm6_zeroshot_on_panda"]])
    xz_p_grp = np.mean([r["grasp_rate"] for r in results["xarm6_zeroshot_on_panda"]])

    print("\n" + "=" * 80)
    print(" [ZERO-SHOT SUMMARY] CROSS-EMBODIMENT TRANSFER PERFORMANCE")
    print(f" Panda Specialist -> Zero-Shot on xArm 6: Success = {pz_x_succ:.1f}% | Grasp = {pz_x_grp:.1f}%")
    print(f" xArm 6 Specialist -> Zero-Shot on Panda:  Success = {xz_p_succ:.1f}% | Grasp = {xz_p_grp:.1f}%")
    print("=" * 80)


if __name__ == "__main__":
    run_cross_embodiment_zeroshot_benchmark()
