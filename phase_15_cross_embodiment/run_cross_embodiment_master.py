"""
Phase 15: Cross-Embodiment Generalization Master Pipeline
Specialists, Combined Multi-Arm Policy, and Zero-Shot Cross-Embodiment Transfer.

Robots:
- Franka Emika Panda (7 DoF)
- UFactory xArm 6 + Robotiq 2F-85 (6 DoF)

Key Design:
1. 3D Workspace Invariance: Point cloud back-projected to table frame (in meters).
2. Universal Cartesian Telemetry: [tcp_x, tcp_y, tcp_z, qx, qy, qz, qw, g_norm] (8 dims).
3. Universal Cartesian Action: [dx, dy, dz, a_gripper] (4 dims).
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
DATA_DIR = os.path.join(BASE_DIR, "data")
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")
EVAL_DIR = os.path.join(BASE_DIR, "eval_results")
VIEWPOINTS_JSON = "/home/maniskill/phase_1_to_3/sequential_viewpoint_training/viewpoints.json"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def extract_table_frame_pc(obs: dict, num_points: int = 512) -> torch.Tensor:
    pos = obs["sensor_data"]["base_camera"]["position"].float() / 1000.0
    cam2world = obs["sensor_param"]["base_camera"]["cam2world_gl"].to(pos.device)
    B = pos.shape[0]

    pos_homo = torch.cat([pos, torch.ones_like(pos[..., :1])], dim=-1)
    xyz_world = (pos_homo.reshape(B, -1, 4) @ cam2world.transpose(1, 2))[..., :3]

    # Workspace tabletop interaction envelope [X: -0.35..0.35, Y: -0.35..0.35, Z: 0.005..0.40]
    mask = (
        (xyz_world[..., 0] >= -0.35) & (xyz_world[..., 0] <= 0.35) &
        (xyz_world[..., 1] >= -0.35) & (xyz_world[..., 1] <= 0.35) &
        (xyz_world[..., 2] >= 0.005) & (xyz_world[..., 2] <= 0.40)
    )

    out_pts = torch.zeros((B, num_points, 3), device=pos.device, dtype=torch.float32)
    for b in range(B):
        pts_b = xyz_world[b, mask[b]]
        n = pts_b.shape[0]
        if n >= num_points:
            perm = torch.randperm(n, device=pos.device)[:num_points]
            out_pts[b] = pts_b[perm]
        elif n > 0:
            idx = torch.randint(0, n, (num_points,), device=pos.device)
            out_pts[b] = pts_b[idx]

    return out_pts


def get_universal_proprio(env, robot_uid: str) -> torch.Tensor:
    agent = env.unwrapped.agent
    tcp_p = agent.tcp_pose.p  # (B, 3)
    tcp_q = agent.tcp_pose.q  # (B, 4)
    qpos = agent.robot.get_qpos()
    if robot_uid == "panda":
        g_norm = qpos[:, -1:] / 0.04  # 1=open, 0=closed
    elif robot_uid == "xarm6_robotiq":
        g_norm = 1.0 - (qpos[:, -6:-5] / 0.81)  # 1=open, 0=closed
    else:
        g_norm = torch.zeros((tcp_p.shape[0], 1), device=tcp_p.device)
    return torch.cat([tcp_p, tcp_q, g_norm], dim=-1).float()


def get_oracle_action(env, robot_uid: str, step: int) -> torch.Tensor:
    agent = env.unwrapped.agent
    cube = env.unwrapped.cube
    goal = env.unwrapped.goal_site
    B = env.num_envs

    tcp_p = agent.tcp_pose.p
    cube_p = cube.pose.p
    goal_p = goal.pose.p

    diff_cube = cube_p - tcp_p
    act = torch.zeros((B, 4), device=env.unwrapped.device)

    # Universal convention: +1.0 = open, -1.0 = close
    # Env mapping: Panda takes +1.0 open / -1.0 close. xArm6 Robotiq takes -1.0 open / +1.0 close.
    gripper_open = 1.0 if robot_uid == "panda" else -1.0
    gripper_close = -1.0 if robot_uid == "panda" else 1.0

    if step < 16:
        # Move above cube
        act[:, 0] = torch.clamp(diff_cube[:, 0] * 12.0, -1.0, 1.0)
        act[:, 1] = torch.clamp(diff_cube[:, 1] * 12.0, -1.0, 1.0)
        act[:, 2] = torch.clamp((diff_cube[:, 2] + 0.035) * 12.0, -1.0, 1.0)
        act[:, 3] = gripper_open
    elif step < 24:
        # Descend into grasp envelope
        act[:, 0] = torch.clamp(diff_cube[:, 0] * 12.0, -1.0, 1.0)
        act[:, 1] = torch.clamp(diff_cube[:, 1] * 12.0, -1.0, 1.0)
        act[:, 2] = torch.clamp((diff_cube[:, 2] - 0.01) * 12.0, -1.0, 1.0)
        act[:, 3] = gripper_open
    elif step < 30:
        # Close fingers
        act[:, 0] = 0.0
        act[:, 1] = 0.0
        act[:, 2] = 0.0
        act[:, 3] = gripper_close
    else:
        # Lift towards goal_site and hold
        diff_goal = goal_p - cube_p
        act[:, 0] = torch.clamp(diff_goal[:, 0] * 6.0, -0.6, 0.6)
        act[:, 1] = torch.clamp(diff_goal[:, 1] * 6.0, -0.6, 0.6)
        act[:, 2] = torch.clamp(diff_goal[:, 2] * 6.0, -0.6, 0.6)
        act[:, 3] = gripper_close

    return act


def sample_continuous_spherical_camera(rng: np.random.RandomState):
    az_deg = rng.uniform(-70.0, 70.0)
    el_deg = rng.uniform(12.0, 50.0)
    dist = rng.uniform(0.95, 1.25)

    az_rad = np.radians(az_deg)
    el_rad = np.radians(el_deg)
    base_pos = np.array([-0.615, 0.0, 0.0])

    rel_x = dist * np.cos(el_rad) * np.cos(az_rad)
    rel_y = dist * np.cos(el_rad) * np.sin(az_rad)
    rel_z = dist * np.sin(el_rad)

    cam_eye = (base_pos + np.array([rel_x, rel_y, rel_z])).tolist()
    return cam_eye, az_deg, el_deg, dist


class UniversalCartesianPointNetAgent(nn.Module):
    def __init__(self, action_dim: int = 4, proprio_dim: int = 8, point_dim: int = 256):
        super().__init__()
        self.conv1 = layer_init(nn.Conv1d(3, 64, kernel_size=1))
        self.conv2 = layer_init(nn.Conv1d(64, 128, kernel_size=1))
        self.conv3 = layer_init(nn.Conv1d(128, point_dim, kernel_size=1))
        self.act = nn.ReLU(inplace=True)

        self.proprio_fc = nn.Sequential(
            layer_init(nn.Linear(proprio_dim, 64)),
            nn.ReLU(inplace=True),
        )

        total_dim = point_dim + 64
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(total_dim, 256)),
            nn.ReLU(inplace=True),
            layer_init(nn.Linear(256, 128)),
            nn.ReLU(inplace=True),
            layer_init(nn.Linear(128, action_dim), std=0.01 * np.sqrt(2)),
        )

    def forward(self, pts: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        x = pts.transpose(1, 2)
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        p_feat = torch.max(x, dim=2)[0]

        s_feat = self.proprio_fc(proprio)
        fused = torch.cat([p_feat, s_feat], dim=-1)
        return self.actor_mean(fused)

    def get_action(self, pts: torch.Tensor, proprio: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        return self.forward(pts, proprio)


def collect_dataset_for_arm(robot_uid: str, num_episodes: int = 1500, batch_envs: int = 32, num_points: int = 512, seed: int = 42):
    cache_path = os.path.join(DATA_DIR, f"dataset_{robot_uid}_{num_episodes}eps.pt")
    if os.path.exists(cache_path):
        print(f"[CACHE] Loading cached dataset for {robot_uid} from {cache_path}")
        data = torch.load(cache_path)
        return data["pts"], data["proprio"], data["actions"]

    print("\n" + "=" * 80)
    print(f" [DATA COLLECTION] {robot_uid.upper()} | {num_episodes:,} EPISODES | CONTINUOUS RANDOM VIEWPOINTS")
    print("=" * 80)

    total_samples = num_episodes * 50
    buf_pts = torch.empty((total_samples, num_points, 3), dtype=torch.float32)
    buf_proprio = torch.empty((total_samples, 8), dtype=torch.float32)
    buf_actions = torch.empty((total_samples, 4), dtype=torch.float32)

    rng = np.random.RandomState(seed)
    cur_idx = 0
    t0 = time.time()
    num_batches = int(np.ceil(num_episodes / batch_envs))

    for b in range(num_batches):
        cur_batch = min(batch_envs, num_episodes - b * batch_envs)
        cam_eye, az, el, dist = sample_continuous_spherical_camera(rng)

        env = gym.make(
            "PickCube-v1",
            robot_uids=robot_uid,
            obs_mode="pointcloud+state",
            control_mode="pd_ee_delta_pos",
            num_envs=batch_envs,
            sensor_cam_eye_pos=cam_eye,
            sensor_cam_target_pos=[-0.1, 0.0, 0.1],
            sensor_configs=dict(width=128, height=128),
        )
        vec_env = ManiSkillVectorEnv(env, batch_envs, ignore_terminations=True, record_metrics=True)
        obs, _ = vec_env.reset(seed=int(seed + b * 100))

        for step in range(50):
            proprio = get_universal_proprio(env, robot_uid)[:cur_batch]
            with torch.no_grad():
                act_raw = get_oracle_action(env, robot_uid, step)[:cur_batch]
                pts = extract_table_frame_pc(obs, num_points=num_points)[:cur_batch]

            # Standardize universal action: gripper is always +1 = open, -1 = close
            act_universal = act_raw.clone()
            if robot_uid == "xarm6_robotiq":
                act_universal[:, 3] = -act_raw[:, 3]

            buf_pts[cur_idx:cur_idx + cur_batch] = pts.cpu()
            buf_proprio[cur_idx:cur_idx + cur_batch] = proprio.cpu()
            buf_actions[cur_idx:cur_idx + cur_batch] = act_universal.cpu()
            cur_idx += cur_batch

            if cur_batch < batch_envs:
                act_pad = torch.zeros((batch_envs, 4), device=DEVICE)
                act_pad[:cur_batch] = act_raw
                obs, _, _, _, _ = vec_env.step(act_pad)
            else:
                obs, _, _, _, _ = vec_env.step(act_raw)

        vec_env.close()
        del vec_env
        del env
        gc.collect()
        torch.cuda.empty_cache()

        if (b + 1) % 10 == 0 or b == 0 or (b + 1) == num_batches:
            print(f"  [{robot_uid}] Batch {b+1:2d}/{num_batches:2d} | Collected: {cur_idx:,}/{total_samples:,} samples | Time: {time.time()-t0:.1f}s")

    pts_out = buf_pts[:cur_idx]
    proprio_out = buf_proprio[:cur_idx]
    actions_out = buf_actions[:cur_idx]

    torch.save({"pts": pts_out, "proprio": proprio_out, "actions": actions_out}, cache_path)
    print(f"[OK] Saved {robot_uid} dataset ({cur_idx:,} transitions) to {cache_path} in {time.time()-t0:.1f}s")
    return pts_out, proprio_out, actions_out


def train_policy(buf_pts, buf_proprio, buf_actions, model_name: str, num_epochs: int = 25, batch_size: int = 256):
    ckpt_path = os.path.join(CKPT_DIR, f"{model_name}_best.pt")
    if os.path.exists(ckpt_path):
        print(f"[CACHE] Loading cached model from {ckpt_path}")
        return ckpt_path

    print("\n" + "=" * 80)
    print(f" [TRAINING] {model_name.upper()} | {len(buf_actions):,} TRANSITIONS | {num_epochs} EPOCHS")
    print("=" * 80)

    dataset = TensorDataset(buf_pts, buf_proprio, buf_actions)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True)

    agent = UniversalCartesianPointNetAgent(action_dim=4, proprio_dim=8, point_dim=256).to(DEVICE)
    optimizer = optim.AdamW(agent.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)

    best_loss = float("inf")

    for epoch in range(1, num_epochs + 1):
        agent.train()
        total_loss = 0.0
        n_batches = 0
        t0 = time.time()

        for b_pts, b_proprio, b_act in loader:
            b_pts = b_pts.to(DEVICE, non_blocking=True)
            b_proprio = b_proprio.to(DEVICE, non_blocking=True)
            b_act = b_act.to(DEVICE, non_blocking=True)

            pred = agent.get_action(b_pts, b_proprio, deterministic=True)
            loss = F.mse_loss(pred, b_act)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / n_batches
        dt = time.time() - t0

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "model_state": agent.state_dict(),
                "epoch": epoch,
                "best_loss": best_loss,
            }, ckpt_path)

        if epoch % 5 == 0 or epoch == 1 or epoch == num_epochs:
            print(f"  [{model_name}] Epoch {epoch:2d}/{num_epochs:2d} | Action MSE Loss: {avg_loss:.5f} | Time: {dt:.1f}s")

    print(f"[OK] Model saved to {ckpt_path} (Best MSE: {best_loss:.5f})")
    return ckpt_path


def evaluate_policy_on_arm(agent, robot_uid: str, eval_title: str, is_zero_shot: bool = False, num_envs_per_vp: int = 50):
    tag = "[ZERO-SHOT TRANSFER]" if is_zero_shot else "[TRAINED EMBODIMENT]"
    print("\n" + "-" * 80)
    print(f" {tag} EVALUATING: {eval_title} ON {robot_uid.upper()} (13 Viewpoints Suite)")
    print("-" * 80)

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

            # Map universal action to robot environment action
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

    print(f"\n===> {eval_title} on {robot_uid.upper()} | Overall Succ: {overall_succ:.1f}% | In-Between: {inbetween_succ:.1f}% | Anchor: {anchor_succ:.1f}% | Grasp: {overall_grasp:.1f}% <===")
    return {
        "model_name": eval_title,
        "robot_uid": robot_uid,
        "is_zero_shot": is_zero_shot,
        "overall_success": overall_succ,
        "anchor_success": anchor_succ,
        "inbetween_success": inbetween_succ,
        "overall_grasp": overall_grasp,
        "details": results,
    }


def main():
    t_start = time.time()
    print("\n" + "=" * 80)
    print(" PHASE 15: CROSS-EMBODIMENT GENERALIZATION MASTER EXPERIMENT")
    print("=" * 80)

    # -------------------------------------------------------------
    # 1. Collect Datasets for Both Robots
    # -------------------------------------------------------------
    pts_panda, proprio_panda, act_panda = collect_dataset_for_arm("panda", num_episodes=1500, batch_envs=32, seed=42)
    pts_xarm, proprio_xarm, act_xarm = collect_dataset_for_arm("xarm6_robotiq", num_episodes=1500, batch_envs=32, seed=142)

    # Combine datasets for Multi-Arm OmniArm Policy
    pts_combined = torch.cat([pts_panda, pts_xarm], dim=0)
    proprio_combined = torch.cat([proprio_panda, proprio_xarm], dim=0)
    act_combined = torch.cat([act_panda, act_xarm], dim=0)
    print(f"\n[OK] Combined OmniArm dataset prepared: {len(act_combined):,} transitions from Panda and xArm6 Robotiq.")

    # -------------------------------------------------------------
    # 2. Train Models: 2 Specialists + 1 Combined Policy
    # -------------------------------------------------------------
    ckpt_panda_spec = train_policy(pts_panda, proprio_panda, act_panda, model_name="panda_specialist", num_epochs=25)
    ckpt_xarm_spec = train_policy(pts_xarm, proprio_xarm, act_xarm, model_name="xarm6_specialist", num_epochs=25)
    ckpt_combined = train_policy(pts_combined, proprio_combined, act_combined, model_name="combined_omniarm", num_epochs=25)

    # Instantiate agents
    agent_panda_spec = UniversalCartesianPointNetAgent(action_dim=4, proprio_dim=8, point_dim=256).to(DEVICE)
    agent_panda_spec.load_state_dict(torch.load(ckpt_panda_spec, map_location=DEVICE)["model_state"])
    agent_panda_spec.eval()

    agent_xarm_spec = UniversalCartesianPointNetAgent(action_dim=4, proprio_dim=8, point_dim=256).to(DEVICE)
    agent_xarm_spec.load_state_dict(torch.load(ckpt_xarm_spec, map_location=DEVICE)["model_state"])
    agent_xarm_spec.eval()

    agent_combined = UniversalCartesianPointNetAgent(action_dim=4, proprio_dim=8, point_dim=256).to(DEVICE)
    agent_combined.load_state_dict(torch.load(ckpt_combined, map_location=DEVICE)["model_state"])
    agent_combined.eval()

    # -------------------------------------------------------------
    # 3. Comprehensive Benchmark Matrix
    # -------------------------------------------------------------
    eval_matrix = {}

    # A. Panda Specialist on Panda
    eval_matrix["panda_specialist_on_panda"] = evaluate_policy_on_arm(
        agent_panda_spec, "panda", "Panda Specialist", is_zero_shot=False
    )

    # B. xArm6 Specialist on xArm6
    eval_matrix["xarm6_specialist_on_xarm6"] = evaluate_policy_on_arm(
        agent_xarm_spec, "xarm6_robotiq", "xArm6 Specialist", is_zero_shot=False
    )

    # C. Combined OmniArm on Panda
    eval_matrix["combined_omniarm_on_panda"] = evaluate_policy_on_arm(
        agent_combined, "panda", "Combined OmniArm", is_zero_shot=False
    )

    # D. Combined OmniArm on xArm6
    eval_matrix["combined_omniarm_on_xarm6"] = evaluate_policy_on_arm(
        agent_combined, "xarm6_robotiq", "Combined OmniArm", is_zero_shot=False
    )

    # E. ZERO-SHOT Cross-Embodiment Transfer 1: Panda Specialist on xArm6
    eval_matrix["zero_shot_panda_to_xarm6"] = evaluate_policy_on_arm(
        agent_panda_spec, "xarm6_robotiq", "Panda Specialist (ZERO-SHOT)", is_zero_shot=True
    )

    # F. ZERO-SHOT Cross-Embodiment Transfer 2: xArm6 Specialist on Panda
    eval_matrix["zero_shot_xarm6_to_panda"] = evaluate_policy_on_arm(
        agent_xarm_spec, "panda", "xArm6 Specialist (ZERO-SHOT)", is_zero_shot=True
    )

    # Save benchmark JSON
    json_path = os.path.join(EVAL_DIR, "phase15_cross_embodiment_benchmark.json")
    with open(json_path, "w") as f:
        json.dump(eval_matrix, f, indent=2)
    print(f"\n[OK] Benchmark results saved to {json_path}")

    # -------------------------------------------------------------
    # 4. Generate Comparative Visualizations
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    # Plot 1: Overall Success & Grasp Rate across Embodiments
    categories = [
        "Panda Specialist\n(on Panda)",
        "xArm6 Specialist\n(on xArm6)",
        "Combined OmniArm\n(on Panda)",
        "Combined OmniArm\n(on xArm6)",
        "ZERO-SHOT Transfer\n(Panda -> xArm6)",
        "ZERO-SHOT Transfer\n(xArm6 -> Panda)",
    ]
    keys = [
        "panda_specialist_on_panda",
        "xarm6_specialist_on_xarm6",
        "combined_omniarm_on_panda",
        "combined_omniarm_on_xarm6",
        "zero_shot_panda_to_xarm6",
        "zero_shot_xarm6_to_panda",
    ]
    succ_vals = [eval_matrix[k]["overall_success"] for k in keys]
    grasp_vals = [eval_matrix[k]["overall_grasp"] for k in keys]
    bar_colors = ["#1976d2", "#7b1fa2", "#00897b", "#00897b", "#e65100", "#d84315"]

    x = np.arange(len(categories))
    w = 0.38
    rects1 = ax1.bar(x - w/2, grasp_vals, w, label="Grasp Rate (%)", color="#b0bec5", edgecolor="black", alpha=0.85)
    rects2 = ax1.bar(x + w/2, succ_vals, w, label="Success Rate (%)", color=bar_colors, edgecolor="black")

    ax1.set_ylabel("Rate (%)", fontsize=12, fontweight="bold")
    ax1.set_title("Cross-Embodiment Performance & Zero-Shot Transfer\n(13 Viewpoints: 4 Anchors + 9 Novel In-Betweens)", fontsize=13, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, fontsize=9.5, fontweight="bold", rotation=15)
    ax1.set_ylim(0, 105)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right", fontsize=11)

    for r in rects2:
        h = r.get_height()
        ax1.annotate(f"{h:.1f}%", xy=(r.get_x() + r.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9.5, fontweight="bold")

    # Plot 2: Anchor vs Novel In-Between Viewpoints Breakdown
    anchor_vals = [eval_matrix[k]["anchor_success"] for k in keys]
    inbetween_vals = [eval_matrix[k]["inbetween_success"] for k in keys]

    ax2.bar(x - w/2, anchor_vals, w, label="Anchor Viewpoints (4)", color="#3f51b5", edgecolor="black")
    ax2.bar(x + w/2, inbetween_vals, w, label="Novel In-Betweens (9)", color="#ff9800", edgecolor="black")

    ax2.set_ylabel("Success Rate (%)", fontsize=12, fontweight="bold")
    ax2.set_title("Anchor vs Novel In-Between Viewpoint Generalization", fontsize=13, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories, fontsize=9.5, fontweight="bold", rotation=15)
    ax2.set_ylim(0, 105)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right", fontsize=11)

    plot_path = os.path.join(EVAL_DIR, "phase15_cross_embodiment_comparison.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"[OK] Comparison plot saved to {plot_path}")

    total_time = time.time() - t_start
    print(f"\n================================================================================")
    print(f" ALL PHASES COMPLETED IN {total_time/60:.2f} MINUTES!")
    print(f"================================================================================")


if __name__ == "__main__":
    main()
