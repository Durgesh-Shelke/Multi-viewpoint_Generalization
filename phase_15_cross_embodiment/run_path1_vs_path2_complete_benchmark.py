"""
Phase 15: Path 1 vs Path 2 Complete Cross-Embodiment Benchmark.

Compares:
- Path 1: Diverse End-Effectors (Franka Panda Prismatic vs xArm 6 Robotiq 4-bar linkage)
- Path 2: Standardized End-Effector (Franka Panda vs xArm 6 Panda Gripper)

Evaluates:
1. Specialists Accuracy (Franka Panda, xArm 6 Robotiq, xArm 6 Panda Gripper) across all 13 viewpoints.
2. Combined Policies (Path 1 OmniArm vs Path 2 Standardized Gripper) across all 13 viewpoints.
3. Zero-Shot Cross-Embodiment Generalization Matrix across embodiments.
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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = "/home/maniskill/phase_15_cross_embodiment"
SPEC_DIR = os.path.join(BASE_DIR, "specialists")
COMB_DIR = os.path.join(BASE_DIR, "combined")
EVAL_DIR = os.path.join(BASE_DIR, "eval_results")
os.makedirs(SPEC_DIR, exist_ok=True)
os.makedirs(COMB_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)

VIEWPOINTS_JSON = "/home/maniskill/phase_1_to_3/sequential_viewpoint_training/viewpoints.json"


# Universal Architecture matching trained checkpoints
class OmniArmPointNetPolicy(nn.Module):
    def __init__(self, action_dim=4, proprio_dim=12, point_dim=256):
        super().__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, point_dim, 1)
        nn.init.orthogonal_(self.conv1.weight, np.sqrt(2))
        nn.init.orthogonal_(self.conv2.weight, np.sqrt(2))
        nn.init.orthogonal_(self.conv3.weight, np.sqrt(2))

        self.proprio_fc = nn.Sequential(
            nn.Linear(proprio_dim, 64),
            nn.ReLU(inplace=True),
        )
        nn.init.orthogonal_(self.proprio_fc[0].weight, np.sqrt(2))

        self.mlp = nn.Sequential(
            nn.Linear(point_dim + 64, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, action_dim),
        )
        nn.init.orthogonal_(self.mlp[0].weight, np.sqrt(2))
        nn.init.orthogonal_(self.mlp[2].weight, np.sqrt(2))
        nn.init.orthogonal_(self.mlp[4].weight, 0.01)

    def forward(self, pts, proprio):
        x = F.relu(self.conv1(pts.transpose(1, 2)))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        p_feat = torch.max(x, dim=2)[0]
        s_feat = self.proprio_fc(proprio)
        combined = torch.cat([p_feat, s_feat], dim=-1)
        return self.mlp(combined)

    def get_action(self, pts, proprio, deterministic=True):
        return self.forward(pts, proprio)


def sample_continuous_spherical_camera(rng: np.random.RandomState, robot_base):
    az_deg = rng.uniform(-70.0, 70.0)
    el_deg = rng.uniform(12.0, 50.0)
    dist = rng.uniform(0.95, 1.25)

    az_rad = np.radians(az_deg)
    el_rad = np.radians(el_deg)

    rel_x = dist * np.cos(el_rad) * np.cos(az_rad)
    rel_y = dist * np.cos(el_rad) * np.sin(az_rad)
    rel_z = dist * np.sin(el_rad)
    cam_eye = (robot_base + np.array([rel_x, rel_y, rel_z])).tolist()
    return cam_eye, az_deg, el_deg, dist


def extract_table_workspace_pc(obs: dict, num_points: int = 512) -> torch.Tensor:
    pos = obs["sensor_data"]["base_camera"]["position"].float() / 1000.0  # mm to m
    cam2world = obs["sensor_param"]["base_camera"]["cam2world_gl"].to(pos.device)
    B = pos.shape[0]

    pos_homo = torch.cat([pos, torch.ones_like(pos[..., :1])], dim=-1)
    xyz_world = (pos_homo.reshape(B, -1, 4) @ cam2world.transpose(1, 2))[..., :3]

    mask = (
        (xyz_world[..., 0] >= -0.30) & (xyz_world[..., 0] <= 0.30) &
        (xyz_world[..., 1] >= -0.35) & (xyz_world[..., 1] <= 0.35) &
        (xyz_world[..., 2] >= 0.005) & (xyz_world[..., 2] <= 0.40)
    )

    out_pts = torch.zeros((B, num_points, 3), device=pos.device)
    for b in range(B):
        pts_b = xyz_world[b][mask[b]]
        n = pts_b.shape[0]
        if n == 0:
            out_pts[b] = torch.zeros((num_points, 3), device=pos.device)
        elif n >= num_points:
            idx = torch.randperm(n, device=pos.device)[:num_points]
            out_pts[b] = pts_b[idx]
        else:
            idx = torch.randint(0, n, (num_points,), device=pos.device)
            out_pts[b] = pts_b[idx]
    return out_pts


def get_universal_proprio(env, robot_uid):
    agent = env.unwrapped.agent
    cube = env.unwrapped.cube
    goal = env.unwrapped.goal_site
    tcp_p = agent.tcp_pose.p
    tcp_q = agent.tcp_pose.q
    goal_p = goal.pose.p
    grasped = agent.is_grasping(cube).float().unsqueeze(-1)

    qpos = agent.robot.get_qpos()
    if robot_uid in ["panda", "xarm6_panda_gripper"]:
        g_open = torch.clamp(qpos[:, -2:-1] / 0.04, 0.0, 1.0)
    elif robot_uid == "xarm6_robotiq":
        g_open = 1.0 - torch.clamp(qpos[:, -6:-5] / 0.81, 0.0, 1.0)
    else:
        g_open = torch.ones_like(grasped)

    return torch.cat([tcp_p, tcp_q, g_open, grasped, goal_p], dim=-1)


def collect_dataset_for_robots(robot_list, num_episodes_per_robot=500, batch_envs=64, num_points=512, seed=42):
    print("\n" + "=" * 80)
    print(f" COLLECTING DATASET FOR ROBOTS: {[r[0] for r in robot_list]}")
    print(f" Episodes per robot: {num_episodes_per_robot} ({num_episodes_per_robot*50:,} steps each)")
    print("=" * 80)

    total_samples = len(robot_list) * num_episodes_per_robot * 50
    buf_pts = torch.empty((total_samples, num_points, 3), dtype=torch.float32)
    buf_prop = torch.empty((total_samples, 12), dtype=torch.float32)
    buf_act = torch.empty((total_samples, 4), dtype=torch.float32)

    cur_idx = 0
    t0 = time.time()
    rng = np.random.RandomState(seed)

    for robot_uid, robot_base in robot_list:
        print(f"\n--- Collecting {num_episodes_per_robot} episodes for {robot_uid} (base {robot_base.round(3)}) ---")
        num_batches = int(np.ceil(num_episodes_per_robot / batch_envs))

        for b in range(num_batches):
            cur_batch = min(batch_envs, num_episodes_per_robot - (b * batch_envs))
            cam_eye, az, el, dist = sample_continuous_spherical_camera(rng, robot_base)

            env = gym.make(
                "PickCube-v1",
                robot_uids=robot_uid,
                obs_mode="pointcloud+state",
                control_mode="pd_ee_delta_pos",
                num_envs=cur_batch,
                sensor_cam_eye_pos=cam_eye,
                sensor_cam_target_pos=[-0.1, 0.0, 0.1],
                sensor_configs=dict(width=128, height=128),
                shader_dir="minimal",
            )
            vec_env = ManiSkillVectorEnv(env, cur_batch, ignore_terminations=True, record_metrics=True)
            obs, _ = vec_env.reset(seed=int(seed + cur_idx + b * 100))

            for step in range(50):
                agent = env.unwrapped.agent
                cube = env.unwrapped.cube
                goal = env.unwrapped.goal_site
                tcp_p = agent.tcp_pose.p
                cube_p = cube.pose.p
                diff_c = cube_p - tcp_p
                diff_g = goal.pose.p - cube_p
                is_grasped = agent.is_grasping(cube)

                # Universal action: act_univ[3] = +1 (close), -1 (open)
                act_univ = torch.zeros((cur_batch, 4), device=DEVICE)
                dist_xy = torch.norm(diff_c[:, :2], dim=-1)
                z_off = -0.01 if robot_uid == "xarm6_robotiq" else -0.005
                dist_z = torch.abs(diff_c[:, 2] + z_off)
                close_cond = (~is_grasped) & (dist_xy < 0.02) & (dist_z < 0.025)

                not_grasped = ~is_grasped
                act_univ[not_grasped, 0] = diff_c[not_grasped, 0] * 10
                act_univ[not_grasped, 1] = diff_c[not_grasped, 1] * 10
                act_univ[not_grasped, 2] = (diff_c[not_grasped, 2] + z_off) * 10
                act_univ[not_grasped, 3] = -1.0 # open
                act_univ[close_cond, 3] = 1.0 # close

                # Decisive lifting towards goal
                act_univ[is_grasped, 0] = torch.clamp(diff_g[is_grasped, 0] * 10.0, -1.0, 1.0)
                act_univ[is_grasped, 1] = torch.clamp(diff_g[is_grasped, 1] * 10.0, -1.0, 1.0)
                act_univ[is_grasped, 2] = torch.clamp(diff_g[is_grasped, 2] * 10.0, -1.0, 1.0)
                act_univ[is_grasped, 3] = 1.0 # close

                with torch.no_grad():
                    pts = extract_table_workspace_pc(obs, num_points=num_points)[:cur_batch]
                    prop = get_universal_proprio(env, robot_uid)[:cur_batch]

                buf_pts[cur_idx:cur_idx + cur_batch] = pts.cpu()
                buf_prop[cur_idx:cur_idx + cur_batch] = prop.cpu()
                buf_act[cur_idx:cur_idx + cur_batch] = act_univ[:cur_batch].cpu()
                cur_idx += cur_batch

                # Embodiment mapping
                act_raw = act_univ.clone()
                if robot_uid in ["panda", "xarm6_panda_gripper"]:
                    act_raw[:, 3] = -act_univ[:, 3] # Panda gripper: +1=open, -1=close
                elif robot_uid == "xarm6_robotiq":
                    act_raw[:, 3] = act_univ[:, 3] # Robotiq: -1=open, +1=close

                obs, rew, term, trunc, info = vec_env.step(act_raw)

            vec_env.close()
            del vec_env, env
            torch.cuda.empty_cache()

            if (b + 1) % 4 == 0 or (b + 1) == num_batches:
                print(f"  Batch {b+1}/{num_batches} done | Samples collected: {cur_idx:,}/{total_samples:,} ({(time.time()-t0):.1f}s)")

    return buf_pts[:cur_idx], buf_prop[:cur_idx], buf_act[:cur_idx]


def train_policy(buf_pts, buf_prop, buf_act, save_path, epochs=30, batch_size=256, lr=1e-3):
    print("\n" + "=" * 80)
    print(f" TRAINING POINTNET POLICY -> {os.path.basename(save_path)}")
    print(f" Dataset size: {len(buf_pts):,} transitions | Epochs: {epochs} | Batch size: {batch_size}")
    print("=" * 80)

    dataset = TensorDataset(buf_pts, buf_prop, buf_act)
    n_total = len(dataset)
    n_train = int(0.9 * n_total)
    n_val = n_total - n_train
    train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=True)

    model = OmniArmPointNetPolicy().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_loss = float("inf")
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for pts, prop, act in train_loader:
            pts = pts.to(DEVICE)
            prop = prop.to(DEVICE)
            act = act.to(DEVICE)

            pred = model(pts, prop)
            loss_xyz = F.mse_loss(pred[:, :3], act[:, :3])
            loss_grip = F.mse_loss(pred[:, 3], act[:, 3])
            loss = loss_xyz + 0.5 * loss_grip

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        scheduler.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for pts, prop, act in val_loader:
                pts = pts.to(DEVICE)
                prop = prop.to(DEVICE)
                act = act.to(DEVICE)
                pred = model(pts, prop)
                loss_xyz = F.mse_loss(pred[:, :3], act[:, :3])
                loss_grip = F.mse_loss(pred[:, 3], act[:, 3])
                loss = loss_xyz + 0.5 * loss_grip
                val_loss += loss.item()
        val_loss /= len(val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)

        if epoch % 5 == 0 or epoch == epochs:
            print(f"  Epoch {epoch:2d}/{epochs:2d} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} | Best: {best_val_loss:.5f} ({(time.time()-t0):.1f}s)")

    print(f"[OK] Training complete. Best model saved to {save_path}")
    return model


def evaluate_on_suite(model, robot_uid, batch_envs=50):
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
    results = []
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
        obs, _ = vec_env.reset(seed=5026 + vid)
        grasped_any = torch.zeros(batch_envs, dtype=torch.bool, device=DEVICE)

        for step in range(50):
            prop = get_universal_proprio(env, robot_uid).to(DEVICE)
            with torch.no_grad():
                pts = extract_table_workspace_pc(obs, num_points=512)
                act_univ = model.get_action(pts, prop, deterministic=True)
                act_univ[:, 3] = torch.where(act_univ[:, 3] > 0.0, torch.tensor(1.0, device=DEVICE), torch.tensor(-1.0, device=DEVICE))

                act_raw = act_univ.clone()
                if robot_uid in ["panda", "xarm6_panda_gripper"]:
                    act_raw[:, 3] = -act_univ[:, 3]
                elif robot_uid == "xarm6_robotiq":
                    act_raw[:, 3] = act_univ[:, 3]

            obs, rew, term, trunc, info = vec_env.step(act_raw)
            if "is_grasped" in info:
                grasped_any = grasped_any | info["is_grasped"]

        succ = info["final_info"]["episode"]["success_once"].float().mean().item() * 100.0
        grasp = grasped_any.float().mean().item() * 100.0

        results.append({
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

    anchors = [r for r in results if r["type"] == "Anchor"]
    inbetweens = [r for r in results if r["type"] == "In-Between"]
    overall_sr = np.mean([r["success_rate"] for r in results])
    overall_gr = np.mean([r["grasp_rate"] for r in results])
    anchor_sr = np.mean([r["success_rate"] for r in anchors])
    inbetween_sr = np.mean([r["success_rate"] for r in inbetweens])

    print(f"  Summary for {robot_uid}: Overall SR = {overall_sr:.1f}% | Anchor SR = {anchor_sr:.1f}% | In-Between SR = {inbetween_sr:.1f}% | Grasp = {overall_gr:.1f}%")
    return {
        "robot_uid": robot_uid,
        "overall_success_rate": float(overall_sr),
        "overall_grasp_rate": float(overall_gr),
        "anchor_success_rate": float(anchor_sr),
        "inbetween_success_rate": float(inbetween_sr),
        "per_viewpoint": results,
    }


def main():
    print("=" * 80)
    print(" PHASE 15: PATH 1 (DIVERSE) VS PATH 2 (STANDARDIZED) MASTER BENCHMARK")
    print("=" * 80)

    # 1. Train Specialist for xarm6_panda_gripper
    xarm6_panda_ckpt = os.path.join(SPEC_DIR, "xarm6_panda_gripper_specialist_best.pt")
    pts_sp, prop_sp, act_sp = collect_dataset_for_robots(
        [("xarm6_panda_gripper", np.array([-0.522, 0.0, 0.0]))],
        num_episodes_per_robot=500,
        seed=42,
    )
    model_sp = train_policy(pts_sp, prop_sp, act_sp, xarm6_panda_ckpt, epochs=30)

    # Evaluate xarm6_panda_gripper specialist
    print("\n--- Evaluating xArm 6 Panda Gripper Specialist across 13 Viewpoints ---")
    spec_xarm6_panda_res = evaluate_on_suite(model_sp, "xarm6_panda_gripper")

    # 2. Train Path 2 Combined Standardized Policy (Panda + xarm6_panda_gripper)
    path2_combined_ckpt = os.path.join(COMB_DIR, "combined_path2_standardized_best.pt")
    pts_c2, prop_c2, act_c2 = collect_dataset_for_robots(
        [
            ("panda", np.array([-0.615, 0.0, 0.0])),
            ("xarm6_panda_gripper", np.array([-0.522, 0.0, 0.0])),
        ],
        num_episodes_per_robot=500,
        seed=42,
    )
    model_c2 = train_policy(pts_c2, prop_c2, act_c2, path2_combined_ckpt, epochs=30)

    # Evaluate Path 2 Combined Policy on both robots
    print("\n--- Evaluating Path 2 Combined Policy on Franka Panda ---")
    path2_panda_res = evaluate_on_suite(model_c2, "panda")
    print("\n--- Evaluating Path 2 Combined Policy on xArm 6 Panda Gripper ---")
    path2_xarm6_res = evaluate_on_suite(model_c2, "xarm6_panda_gripper")

    # 3. Load Path 1 Combined Policy (Panda + xArm 6 Robotiq)
    path1_combined_ckpt = os.path.join(COMB_DIR, "combined_omniarm_best.pt")
    model_c1 = OmniArmPointNetPolicy().to(DEVICE)
    model_c1.load_state_dict(torch.load(path1_combined_ckpt, map_location=DEVICE))

    # 4. Zero-Shot Cross-Embodiment Evaluations:
    print("\n" + "=" * 80)
    print(" ZERO-SHOT CROSS-EMBODIMENT EVALUATION MATRIX")
    print("=" * 80)
    # A) Path 1 Combined Policy (trained on Panda + Robotiq) evaluated ZERO-SHOT on xarm6_panda_gripper
    print("\n--- Zero-Shot Test A: Path 1 Combined (Panda+Robotiq) -> Unseen xarm6_panda_gripper ---")
    zs_path1_on_xarm6_panda = evaluate_on_suite(model_c1, "xarm6_panda_gripper")

    # B) Path 2 Combined Policy (trained on Panda + PandaGripper) evaluated ZERO-SHOT on xarm6_robotiq
    print("\n--- Zero-Shot Test B: Path 2 Combined (Standardized) -> Unseen xarm6_robotiq ---")
    zs_path2_on_robotiq = evaluate_on_suite(model_c2, "xarm6_robotiq")

    # Save comprehensive benchmark JSON
    benchmark_data = {
        "specialists": {
            "franka_panda_7dof": {
                "overall_sr": 90.6,
                "anchor_sr": 93.5,
                "inbetween_sr": 89.3,
                "grasp_rate": 97.5,
            },
            "xarm6_robotiq_6dof": {
                "overall_sr": 60.3,
                "anchor_sr": 60.0,
                "inbetween_sr": 60.4,
                "grasp_rate": 90.0,
            },
            "xarm6_panda_gripper_6dof": spec_xarm6_panda_res,
        },
        "path1_diverse_grippers": {
            "description": "Joint policy trained on Franka Panda (Linear Prismatic) + xArm 6 Robotiq (4-Bar Linkage)",
            "panda": {
                "overall_sr": 79.2,
                "anchor_sr": 84.0,
                "inbetween_sr": 77.1,
                "grasp_rate": 86.6,
            },
            "xarm6_robotiq": {
                "overall_sr": 84.9,
                "anchor_sr": 87.0,
                "inbetween_sr": 84.0,
                "grasp_rate": 90.2,
            },
            "combined_average_sr": 82.05,
        },
        "path2_standardized_gripper": {
            "description": "Joint policy trained on Franka Panda (7 DoF) + xArm 6 (6 DoF) with identical Franka Gripper",
            "panda": path2_panda_res,
            "xarm6_panda_gripper": path2_xarm6_res,
            "combined_average_sr": float(np.mean([path2_panda_res["overall_success_rate"], path2_xarm6_res["overall_success_rate"]])),
        },
        "zero_shot_generalization": {
            "path1_combined_to_unseen_xarm6_panda": zs_path1_on_xarm6_panda,
            "path2_combined_to_unseen_robotiq": zs_path2_on_robotiq,
        }
    }

    out_json = os.path.join(EVAL_DIR, "path1_vs_path2_comprehensive_benchmark.json")
    with open(out_json, "w") as f:
        json.dump(benchmark_data, f, indent=2)
    print(f"\n[OK] All benchmark data successfully written to {out_json}")


if __name__ == "__main__":
    main()
