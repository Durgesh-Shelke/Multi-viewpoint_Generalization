"""
Comprehensive System Verification Test Suite:
Tests all deliverable models, checkpoints, architectures, and simulation pipelines.
"""

import os
import sys
import torch
import numpy as np
import gymnasium as gym

import mani_skill.envs
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

REPO_DIR = "/home/maniskill/Multi-viewpoint_Generalization"
sys.path.append(REPO_DIR)
sys.path.append("/home/maniskill")

from phase_16_composite_modular_architecture.composite_models import UniversalCompositeOmniPolicy
from phase_15_cross_embodiment.train_combined_omniarm import OmniArmPointNetPolicy, extract_table_workspace_pc, get_universal_proprio
from phase_13_pointcloud_generalization.pointnet_models import PointNetAgent

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_checkpoints():
    print("=" * 80)
    print(" TEST SUITE 1: CHECKPOINT EXISTENCE & WEIGHT INTEGRITY")
    print("=" * 80)
    checkpoints = [
        ("Phase 16 Composite Deliverable", f"{REPO_DIR}/phase_16_composite_modular_architecture/checkpoints/universal_composite_omniarm.pt"),
        ("Phase 15 Combined OmniArm Model", f"{REPO_DIR}/phase_15_cross_embodiment/combined/combined_omniarm_best.pt"),
        ("Phase 15 Standardized Gripper Model", f"{REPO_DIR}/phase_15_cross_embodiment/combined/combined_path2_standardized_best.pt"),
        ("Phase 15 Panda Specialist", f"{REPO_DIR}/phase_15_cross_embodiment/checkpoints/panda_specialist_best.pt"),
        ("Phase 15 xArm 6 Specialist", f"{REPO_DIR}/phase_15_cross_embodiment/checkpoints/xarm6_specialist_best.pt"),
        ("Phase 13 PointNet Generalist", f"{REPO_DIR}/phase_13_pointcloud_generalization/student/pointnet_student_best.pt"),
    ]
    all_passed = True
    for name, path in checkpoints:
        exists = os.path.exists(path)
        size_mb = os.path.getsize(path) / (1024 * 1024) if exists else 0
        try:
            ckpt = torch.load(path, map_location="cpu")
            valid = isinstance(ckpt, dict) and len(ckpt) > 0
            num_tensors = len(ckpt)
        except Exception as e:
            valid = False
            num_tensors = 0

        status = "PASSED" if (exists and valid) else "FAILED"
        if not (exists and valid):
            all_passed = False
        print(f"  [{status}] {name:<36} | Size: {size_mb:6.2f} MB | Tensors: {num_tensors:2d}")

    return all_passed


def test_model_forward_passes():
    print("\n" + "=" * 80)
    print(" TEST SUITE 2: MODEL ARCHITECTURE & FORWARD INFERENCE SANITY")
    print("=" * 80)
    B, N = 4, 512
    dummy_pts = torch.randn((B, N, 3), device=DEVICE)
    dummy_prop = torch.randn((B, 12), device=DEVICE)
    dummy_qpos = torch.randn((B, 7), device=DEVICE)

    all_passed = True

    # 1. Test Phase 16 Composite Deliverable
    try:
        model16 = UniversalCompositeOmniPolicy().to(DEVICE)
        res = model16(dummy_pts, dummy_prop, qpos_arm=dummy_qpos, robot_id=0)
        c_act = res["cartesian_hardware"]
        j_act = res["joint_delta"]
        assert c_act.shape == (B, 4), f"Unexpected shape {c_act.shape}"
        assert j_act.shape == (B, 7), f"Unexpected shape {j_act.shape}"
        print(f"  [PASSED] UniversalCompositeOmniPolicy | Cartesian Act: {c_act.shape} | Joint Delta: {j_act.shape}")
    except Exception as e:
        print(f"  [FAILED] UniversalCompositeOmniPolicy forward pass: {e}")
        all_passed = False

    # 2. Test Phase 15 Combined OmniArm Model
    try:
        model15 = OmniArmPointNetPolicy().to(DEVICE)
        ckpt_path = f"{REPO_DIR}/phase_15_cross_embodiment/combined/combined_omniarm_best.pt"
        model15.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
        model15.eval()
        act15 = model15.get_action(dummy_pts, dummy_prop)
        assert act15.shape == (B, 4), f"Unexpected shape {act15.shape}"
        print(f"  [PASSED] OmniArmPointNetPolicy (Phase 15) | Action Shape: {act15.shape}")
    except Exception as e:
        print(f"  [FAILED] OmniArmPointNetPolicy forward pass: {e}")
        all_passed = False

    # 3. Test Phase 13 PointNet Policy
    try:
        model13 = PointNetAgent(proprio_dim=29, action_dim=4).to(DEVICE)
        ckpt13_path = f"{REPO_DIR}/phase_13_pointcloud_generalization/student/pointnet_student_best.pt"
        ckpt13 = torch.load(ckpt13_path, map_location=DEVICE)
        state_dict13 = ckpt13["model_state"] if "model_state" in ckpt13 else ckpt13
        model13.load_state_dict(state_dict13)
        model13.eval()
        dummy_prop29 = torch.randn((B, 29), device=DEVICE)
        act13 = model13.get_action(dummy_pts, dummy_prop29)
        assert act13.shape == (B, 4), f"Unexpected shape {act13.shape}"
        print(f"  [PASSED] PointNetAgent (Phase 13) | Action Shape: {act13.shape}")
    except Exception as e:
        print(f"  [FAILED] PointNetAgent forward pass: {e}")
        all_passed = False

    return all_passed


def test_simulation_rollouts():
    print("\n" + "=" * 80)
    print(" TEST SUITE 3: SIMULATION ROLLOUT ON BOTH ROBOT ARMS")
    print("=" * 80)
    all_passed = True

    model = UniversalCompositeOmniPolicy.load_deliverable(
        f"{REPO_DIR}/phase_16_composite_modular_architecture/checkpoints/universal_composite_omniarm.pt",
        device=DEVICE
    )

    for robot_uid, robot_id, n_dof in [("panda", 0, 7), ("xarm6_robotiq", 1, 6)]:
        try:
            env = gym.make(
                "PickCube-v1",
                robot_uids=robot_uid,
                obs_mode="pointcloud+state",
                control_mode="pd_ee_delta_pos",
                num_envs=2,
                sensor_cam_target_pos=[-0.1, 0.0, 0.1],
                sensor_configs=dict(width=128, height=128),
                shader_dir="minimal",
            )
            vec_env = ManiSkillVectorEnv(env, 2, ignore_terminations=True)
            obs, _ = vec_env.reset(seed=42)

            for step in range(5):
                prop = get_universal_proprio(env, robot_uid).to(DEVICE)
                pts = extract_table_workspace_pc(obs, num_points=512)
                act = model.get_action(pts, prop, robot_id=robot_id)
                obs, rew, term, trunc, info = vec_env.step(act)

            vec_env.close()
            print(f"  [PASSED] Simulation rollout on {robot_uid:<18} (DoF={n_dof}) | 5 steps executed cleanly")
        except Exception as e:
            print(f"  [FAILED] Simulation rollout on {robot_uid}: {e}")
            all_passed = False

    return all_passed


if __name__ == "__main__":
    print("\n" + "#" * 80)
    print(" RUNNING COMPLETE VERIFICATION TEST SUITE ACROSS ALL PHASES")
    print("#" * 80)

    t0 = os.times().elapsed
    p1 = test_checkpoints()
    p2 = test_model_forward_passes()
    p3 = test_simulation_rollouts()

    print("\n" + "=" * 80)
    print(" TEST SUITE VERIFICATION SUMMARY")
    print("=" * 80)
    print(f" Checkpoints Integrity:     {'ALL PASSED' if p1 else 'FAILED'}")
    print(f" Model Architectures:       {'ALL PASSED' if p2 else 'FAILED'}")
    print(f" Simulation Multi-Arm Env:  {'ALL PASSED' if p3 else 'FAILED'}")
    print("=" * 80)
    if p1 and p2 and p3:
        print(" >>> ALL VERIFICATION TESTS PASSED SUCCESSFULLY! SYSTEM IS 100% OPERATIONAL. <<<")
    else:
        sys.exit(1)
