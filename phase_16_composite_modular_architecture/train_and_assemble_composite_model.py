"""
Phase 16: Train, Assemble, and Benchmark Universal Composite OmniPolicy.

1. Instantiates UniversalCompositeOmniPolicy.
2. Copies pre-trained ViewpointPerceptionEngine weights from Phase 15.
3. Collects kinematic trajectory pairs to train EmbodimentKinematicEngine (Neural IK).
4. Assembles and exports the single deliverable: universal_composite_omniarm.pt.
5. Benchmarks the single composite model on both Franka Panda and xArm 6 across 13 viewpoints suite.
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
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gymnasium as gym

import mani_skill.envs
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from phase_16_composite_modular_architecture.composite_models import UniversalCompositeOmniPolicy
from phase_15_cross_embodiment.train_combined_omniarm import extract_table_workspace_pc, get_universal_proprio

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = "/home/maniskill/phase_16_composite_modular_architecture"
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")
EVAL_DIR = os.path.join(BASE_DIR, "eval_results")
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)

VIEWPOINTS_JSON = "/home/maniskill/phase_1_to_3/sequential_viewpoint_training/viewpoints.json"
PHASE15_CKPT = "/home/maniskill/phase_15_cross_embodiment/combined/combined_omniarm_best.pt"
DELIVERABLE_CKPT = os.path.join(CKPT_DIR, "universal_composite_omniarm.pt")


def collect_kinematic_dataset(num_transitions_per_robot=5000, batch_envs=64, seed=42):
    print("\n" + "=" * 80)
    print(f" COLLECTING KINEMATIC DATASET FOR EMBODIMENT ENGINE ({num_transitions_per_robot*2:,} PAIRS)")
    print("=" * 80)

    robots = [
        ("panda", 0, 7),
        ("xarm6_robotiq", 1, 6),
    ]

    total_samples = len(robots) * num_transitions_per_robot
    buf_cartesian = torch.empty((total_samples, 4), dtype=torch.float32)
    buf_qpos = torch.empty((total_samples, 7), dtype=torch.float32)
    buf_robot_id = torch.empty((total_samples,), dtype=torch.long)
    buf_delta_q = torch.empty((total_samples, 7), dtype=torch.float32)

    cur_idx = 0
    t0 = time.time()
    rng = np.random.RandomState(seed)

    for robot_uid, robot_id, n_dof in robots:
        print(f"\n--- Collecting kinematic pairs for {robot_uid} (DoF={n_dof}) ---")
        env = gym.make(
            "PickCube-v1",
            robot_uids=robot_uid,
            obs_mode="state",
            control_mode="pd_ee_delta_pos",
            num_envs=batch_envs,
        )
        vec_env = ManiSkillVectorEnv(env, batch_envs, ignore_terminations=True)

        steps_needed = int(np.ceil(num_transitions_per_robot / batch_envs))
        obs, _ = vec_env.reset(seed=int(seed + robot_id * 100))

        for step in range(steps_needed):
            cur_batch = min(batch_envs, num_transitions_per_robot - (step * batch_envs))
            agent = env.unwrapped.agent
            robot = agent.robot

            # Arm joints only (first n_dof joints)
            q_current = robot.get_qpos()[:, :n_dof].clone()

            # Random Cartesian displacement
            act = (torch.rand((batch_envs, 4), device=DEVICE) - 0.5) * 0.4
            act[:, 3] = torch.where(torch.rand(batch_envs, device=DEVICE) > 0.5, 1.0, -1.0)

            obs, rew, term, trunc, info = vec_env.step(act)

            q_next = robot.get_qpos()[:, :n_dof]
            dq = q_next - q_current

            # Pad to 7 DoF
            q_padded = torch.zeros((cur_batch, 7), device=DEVICE)
            dq_padded = torch.zeros((cur_batch, 7), device=DEVICE)
            q_padded[:, :n_dof] = q_current[:cur_batch]
            dq_padded[:, :n_dof] = dq[:cur_batch]

            buf_cartesian[cur_idx:cur_idx + cur_batch] = act[:cur_batch].cpu()
            buf_qpos[cur_idx:cur_idx + cur_batch] = q_padded.cpu()
            buf_robot_id[cur_idx:cur_idx + cur_batch] = robot_id
            buf_delta_q[cur_idx:cur_idx + cur_batch] = dq_padded.cpu()
            cur_idx += cur_batch

        vec_env.close()
        del vec_env, env
        torch.cuda.empty_cache()

    print(f"[OK] Kinematic dataset collected: {cur_idx:,} pairs in {time.time()-t0:.1f}s")
    return buf_cartesian[:cur_idx], buf_qpos[:cur_idx], buf_robot_id[:cur_idx], buf_delta_q[:cur_idx]


def train_kinematic_adapter(model, buf_cartesian, buf_qpos, buf_robot_id, buf_delta_q, epochs=20, batch_size=128):
    print("\n" + "=" * 80)
    print(" TRAINING SUB-MODULE 2: EMBODIMENT KINEMATIC ENGINE (NEURAL IK)")
    print("=" * 80)

    dataset = TensorDataset(buf_cartesian, buf_qpos, buf_robot_id, buf_delta_q)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    optimizer = optim.AdamW(model.kinematic_engine.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.kinematic_engine.train()
        total_loss = 0.0
        for cart, qpos, r_id, dq in loader:
            cart = cart.to(DEVICE)
            qpos = qpos.to(DEVICE)
            r_id = r_id.to(DEVICE)
            dq = dq.to(DEVICE)

            pred_dq = model.kinematic_engine.forward_ik(cart, qpos, r_id)
            loss = F.mse_loss(pred_dq, dq)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(loader)
        if ep % 5 == 0 or ep == epochs:
            print(f"  Kinematic Adapter Epoch {ep:2d}/{epochs:2d} | Loss (MSE): {avg_loss:.6f} | Time: {time.time()-t0:.1f}s")

    print("[OK] Kinematic Adapter training complete.")


def benchmark_composite_model_across_viewpoints(model, batch_envs=50):
    print("\n" + "=" * 80)
    print(" BENCHMARKING SINGLE COMPOSITE DELIVERABLE MODEL ACROSS 13 VIEWPOINTS")
    print(" Testing on BOTH Franka Panda (7 DoF) and xArm 6 Robotiq (6 DoF)")
    print("=" * 80)

    with open(VIEWPOINTS_JSON, "r") as f:
        vps = json.load(f)
    vp_dict = {vp["id"]: vp for vp in vps}

    eval_vps = [
        (17, "VP 17 (Front-Mid Anchor)", "Anchor"),
        (0, "VP 00 (Front-Low Anchor)", "Anchor"),
        (20, "VP 20 (Left-Mid Anchor)", "Anchor"),
        (19, "VP 19 (Right-Mid Anchor)", "Anchor"),
        (10, "In-Between VP 10 (Elevation Interp)", "In-Between"),
        (13, "In-Between VP 13 (Azimuth Interp)", "In-Between"),
        (11, "In-Between VP 11 (High-Elevation Interp)", "In-Between"),
        (12, "In-Between VP 12 (Flank Interp)", "In-Between"),
        (14, "In-Between VP 14 (Elevation Interp)", "In-Between"),
        (15, "In-Between VP 15 (Mid-Left Interp)", "In-Between"),
        (16, "In-Between VP 16 (Mid-Right Interp)", "In-Between"),
        (18, "In-Between VP 18 (Upper-Left Interp)", "In-Between"),
        (21, "In-Between VP 21 (Extreme Flank Interp)", "In-Between"),
    ]

    model.eval()
    benchmark_results = {"panda": [], "xarm6": []}

    for robot_uid, robot_id in [("panda", 0), ("xarm6_robotiq", 1)]:
        print(f"\n--- Testing Composite Deliverable on {robot_uid} (Embodiment ID={robot_id}) ---")
        base_shift = np.array([0.0, 0.0, 0.0]) if robot_uid == "panda" else np.array([-0.522 - (-0.615), 0.0, 0.0])

        for vid, name, vtype in eval_vps:
            meta = vp_dict[vid]
            cam_eye = (np.array(meta["world_eye"]) + base_shift).tolist()

            env = gym.make(
                "PickCube-v1",
                robot_uids=robot_uid,
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
                prop = get_universal_proprio(env, robot_uid).to(DEVICE)
                with torch.no_grad():
                    pts = extract_table_workspace_pc(obs, num_points=512)
                    # Use the single composite deliverable get_action method!
                    act_hardware = model.get_action(pts, prop, robot_id=robot_id)

                    # Saturation for gripper
                    act_hardware[:, 3] = torch.where(act_hardware[:, 3] > 0.0, torch.tensor(1.0, device=DEVICE), torch.tensor(-1.0, device=DEVICE))

                obs, rew, term, trunc, info = vec_env.step(act_hardware)
                if "is_grasped" in info:
                    grasped_any = grasped_any | info["is_grasped"]

            succ = info["final_info"]["episode"]["success_once"].float().mean().item() * 100.0
            grasp = grasped_any.float().mean().item() * 100.0

            benchmark_results["panda" if robot_uid == "panda" else "xarm6"].append({
                "id": vid,
                "name": name,
                "type": vtype,
                "azimuth": meta["azimuth_deg"],
                "elevation": meta["elevation_deg"],
                "success_rate": succ,
                "grasp_rate": grasp,
            })
            print(f"  [{vtype.upper():<10}] {name:<35}: Succ={succ:5.1f}% | Grasp={grasp:5.1f}%")
            vec_env.close()
            del vec_env, env
            torch.cuda.empty_cache()

    for r_key in ["panda", "xarm6"]:
        r_list = benchmark_results[r_key]
        anch = np.mean([r["success_rate"] for r in r_list if r["type"] == "Anchor"])
        interp = np.mean([r["success_rate"] for r in r_list if r["type"] == "In-Between"])
        tot = np.mean([r["success_rate"] for r in r_list])
        grp = np.mean([r["grasp_rate"] for r in r_list])
        print(f"\nSummary for {r_key.upper()}: Overall SR={tot:.1f}% | Anchor SR={anch:.1f}% | In-Between SR={interp:.1f}% | Grasp={grp:.1f}%")

    return benchmark_results


def main():
    print("=" * 80)
    print(" PHASE 16: COMPOSITE MODULAR ARCHITECTURE EXECUTION")
    print("=" * 80)

    # 1. Instantiate the Universal Composite Architecture
    composite_model = UniversalCompositeOmniPolicy().to(DEVICE)

    # 2. Weight Surgery: Copy pre-trained viewpoint generalist weights from Phase 15
    print("\n--- Step 1: Loading Pre-trained Viewpoint Engine Weights ---")
    composite_model.copy_viewpoint_weights_from_checkpoint(PHASE15_CKPT, device=DEVICE)

    # 3. Train Sub-Module 2: Embodiment Kinematic Engine
    print("\n--- Step 2: Training Kinematic Engine ---")
    buf_cart, buf_q, buf_id, buf_dq = collect_kinematic_dataset(num_transitions_per_robot=5000)
    train_kinematic_adapter(composite_model, buf_cart, buf_q, buf_id, buf_dq, epochs=20)

    # 4. Save the Single Unified Deliverable
    print("\n--- Step 3: Exporting Unified Deliverable Checkpoint ---")
    composite_model.save_deliverable(DELIVERABLE_CKPT)

    # 5. Benchmark the Single Deliverable across all 13 viewpoints
    print("\n--- Step 4: Validating Single Deliverable Across All Viewpoints & Robots ---")
    loaded_deliverable = UniversalCompositeOmniPolicy.load_deliverable(DELIVERABLE_CKPT, device=DEVICE)
    bench_results = benchmark_composite_model_across_viewpoints(loaded_deliverable, batch_envs=50)

    # 6. Save results to JSON
    out_json = os.path.join(EVAL_DIR, "composite_architecture_benchmark.json")
    with open(out_json, "w") as f:
        json.dump(bench_results, f, indent=2)
    print(f"\n[OK] Benchmark results written to {out_json}")


if __name__ == "__main__":
    main()
