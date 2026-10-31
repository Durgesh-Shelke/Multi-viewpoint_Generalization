"""
Phase 15: Unified Multi-Arm Combined Student Policy (combined_omniarm).

1. Collects 500 episodes of Franka Panda + 500 episodes of xArm 6 Robotiq (50,000 transitions total).
2. Uses unified 12-dim joint-agnostic proprioception:
   [tcp_pos (3), tcp_quat (4), gripper_open (1), is_grasped (1), goal_pos (3)]
3. Uses 512-point table-frame workspace 3D Point Clouds.
4. Uses universal action space: [dx, dy, dz, dgripper] (+1 = close, -1 = open).
5. Evaluates on BOTH Franka Panda and xArm 6 across the full 13-viewpoint suite.
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
from phase_13_pointcloud_generalization.pointnet_models import extract_robot_frame_pc

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = "/home/maniskill/phase_15_cross_embodiment"
COMBINED_DIR = os.path.join(BASE_DIR, "combined")
EVAL_DIR = os.path.join(BASE_DIR, "eval_results")
os.makedirs(COMBINED_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)

VIEWPOINTS_JSON = "/home/maniskill/phase_1_to_3/sequential_viewpoint_training/viewpoints.json"
COMBINED_CKPT = os.path.join(COMBINED_DIR, "combined_omniarm_best.pt")


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

    # Universal table-frame workspace filter (centered at table center [0, 0, 0])
    mask = (
        (xyz_world[..., 0] >= -0.30) & (xyz_world[..., 0] <= 0.30) &
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


def get_universal_proprio(env, robot_uid):
    agent = env.unwrapped.agent
    cube = env.unwrapped.cube
    goal = env.unwrapped.goal_site
    tcp_p = agent.tcp_pose.p
    tcp_q = agent.tcp_pose.q
    goal_p = goal.pose.p
    grasped = agent.is_grasping(cube).float().unsqueeze(-1)

    qpos = agent.robot.get_qpos()
    if robot_uid == "panda":
        # Panda finger qpos is in [0, 0.04]. 0.04=open, 0=closed
        g_open = torch.clamp(qpos[:, -2:-1] / 0.04, 0.0, 1.0)
    elif robot_uid == "xarm6_robotiq":
        # Robotiq mimic angle is in [0, 0.81]. 0=open, 0.81=closed
        g_open = 1.0 - torch.clamp(qpos[:, -6:-5] / 0.81, 0.0, 1.0)
    else:
        g_open = torch.ones_like(grasped)

    return torch.cat([tcp_p, tcp_q, g_open, grasped, goal_p], dim=-1)


def collect_multirobot_dataset(num_episodes_per_robot=500, batch_envs=64, num_points=512, seed=42):
    print("\n" + "=" * 80)
    print(f" [STAGE 1] COLLECTING {num_episodes_per_robot * 2:,} TOTAL EPISODES FOR COMBINED POLICY")
    print(" Robots: Franka Panda (7 DoF) & UFactory xArm 6 Robotiq (6 DoF)")
    print(" Representation: Table-Frame Workspace 3D Point Cloud + 12-dim Universal Proprioception")
    print("=" * 80)

    robots = [
        ("panda", np.array([-0.615, 0.0, 0.0])),
        ("xarm6_robotiq", np.array([-0.522, 0.0, 0.0])),
    ]

    total_samples = num_episodes_per_robot * 2 * 50
    buf_pts = torch.empty((total_samples, num_points, 3), dtype=torch.float32)
    buf_prop = torch.empty((total_samples, 12), dtype=torch.float32)
    buf_act = torch.empty((total_samples, 4), dtype=torch.float32)

    cur_idx = 0
    t0 = time.time()
    rng = np.random.RandomState(seed)

    for robot_uid, robot_base in robots:
        print(f"\n--- Collecting {num_episodes_per_robot} episodes for {robot_uid} ---")
        num_batches = int(np.ceil(num_episodes_per_robot / batch_envs))

        for b in range(num_batches):
            cur_batch = min(batch_envs, num_episodes_per_robot - (b * batch_envs))
            cam_eye, az, el, dist = sample_continuous_spherical_camera(rng, robot_base)

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
                act_univ = torch.zeros((batch_envs, 4), device=DEVICE)
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

                # Smooth lifting towards goal to prevent underactuated linkage slip
                act_univ[is_grasped, 0] = torch.clamp(diff_g[is_grasped, 0] * 5.0, -0.5, 0.5)
                act_univ[is_grasped, 1] = torch.clamp(diff_g[is_grasped, 1] * 5.0, -0.5, 0.5)
                act_univ[is_grasped, 2] = torch.clamp(diff_g[is_grasped, 2] * 5.0, -0.5, 0.5)
                act_univ[is_grasped, 3] = 1.0 # close

                with torch.no_grad():
                    pts = extract_table_workspace_pc(obs, num_points=num_points)[:cur_batch]
                    prop = get_universal_proprio(env, robot_uid)[:cur_batch]

                buf_pts[cur_idx:cur_idx + cur_batch] = pts.cpu()
                buf_prop[cur_idx:cur_idx + cur_batch] = prop.cpu()
                buf_act[cur_idx:cur_idx + cur_batch] = act_univ[:cur_batch].cpu()
                cur_idx += cur_batch

                # Hardware embodiment mapping:
                act_raw = act_univ.clone()
                if robot_uid == "panda":
                    act_raw[:, 3] = -act_univ[:, 3] # Panda: +1=open, -1=close
                elif robot_uid == "xarm6_robotiq":
                    act_raw[:, 3] = act_univ[:, 3] # Robotiq: -1=open, +1=close

                obs, _, _, _, _ = vec_env.step(act_raw)

            vec_env.close()
            del vec_env; del env; gc.collect()

            if (b + 1) % 2 == 0 or (b + 1) == num_batches:
                print(f"  [{robot_uid}] Batch {b+1:2d}/{num_batches} | Samples: {cur_idx:,}/{total_samples:,} | Time: {time.time()-t0:.1f}s")

    print(f"[OK] Multi-robot dataset collection complete! Total samples: {cur_idx:,}")
    return buf_pts[:cur_idx], buf_prop[:cur_idx], buf_act[:cur_idx]


class OmniArmPointNetPolicy(nn.Module):
    """
    Universal Cross-Embodiment PointNet Policy:
    Operates directly on workspace geometry and universal kinematics.
    """
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


def train_combined_omniarm(pts_data, prop_data, act_data, epochs=30, batch_size=256, lr=1e-3):
    print("\n" + "=" * 80)
    print(f" [STAGE 2] TRAINING COMBINED OMNIARM POLICY ({epochs} EPOCHS)")
    print("=" * 80)

    dataset = TensorDataset(pts_data, prop_data, act_data)
    val_size = int(len(dataset) * 0.1)
    train_size = len(dataset) - val_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=2)

    model = OmniArmPointNetPolicy(action_dim=4, proprio_dim=12, point_dim=256).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_loss = float("inf")
    t0 = time.time()

    for ep in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for bp, bs, ba in train_loader:
            bp = bp.to(DEVICE, non_blocking=True)
            bs = bs.to(DEVICE, non_blocking=True)
            ba = ba.to(DEVICE, non_blocking=True)

            pred = model(bp, bs)
            loss = F.mse_loss(pred, ba)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(bp)

        scheduler.step()
        train_loss /= train_size

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bp, bs, ba in val_loader:
                bp = bp.to(DEVICE, non_blocking=True)
                bs = bs.to(DEVICE, non_blocking=True)
                ba = ba.to(DEVICE, non_blocking=True)
                pred = model(bp, bs)
                val_loss += F.mse_loss(pred, ba).item() * len(bp)
        val_loss /= val_size

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), COMBINED_CKPT)
            mark = "(*) BEST"
        else:
            mark = ""

        if ep % 5 == 0 or ep == 1 or mark != "":
            print(f"  Epoch {ep:2d}/{epochs:2d} | Train MSE: {train_loss:.5f} | Val MSE: {val_loss:.5f} | LR: {scheduler.get_last_lr()[0]:.2e} {mark}")

    print(f"[OK] Training finished in {time.time() - t0:.1f}s. Saved to {COMBINED_CKPT}")
    return model


def evaluate_combined_suite():
    print("\n" + "=" * 80)
    print(" [STAGE 3] BENCHMARKING COMBINED OMNIARM POLICY ACROSS ALL 13 VIEWPOINTS")
    print(" Testing on BOTH Franka Panda and xArm 6 Robotiq (50 envs per viewpoint = 1,300 total tests)")
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

    model = OmniArmPointNetPolicy(action_dim=4, proprio_dim=12, point_dim=256).to(DEVICE)
    model.load_state_dict(torch.load(COMBINED_CKPT, map_location=DEVICE))
    model.eval()

    combined_results = {"panda": [], "xarm6": []}
    batch_envs = 50

    for robot_uid in ["panda", "xarm6_robotiq"]:
        print(f"\n--- Testing Combined Policy on {robot_uid} across 13 Viewpoints ---")
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

                    # Firm saturation
                    act_univ[:, 3] = torch.where(act_univ[:, 3] > 0.0, torch.tensor(1.0, device=DEVICE), torch.tensor(-1.0, device=DEVICE))

                # Route to robot hardware
                act_raw = act_univ.clone()
                if robot_uid == "panda":
                    act_raw[:, 3] = -act_univ[:, 3]
                elif robot_uid == "xarm6_robotiq":
                    act_raw[:, 3] = act_univ[:, 3]

                obs, _, _, _, info = vec_env.step(act_raw)
                if "is_grasped" in info:
                    grasped_any |= info["is_grasped"]

            succ = info["final_info"]["episode"]["success_once"].float().mean().item() * 100.0
            grasp = grasped_any.float().mean().item() * 100.0
            vec_env.close()
            del vec_env; del env; gc.collect()

            combined_results["panda" if robot_uid == "panda" else "xarm6"].append({
                "id": vid, "name": name, "type": vtype,
                "azimuth": meta["azimuth_deg"], "elevation": meta["elevation_deg"],
                "success_rate": succ, "grasp_rate": grasp
            })
            print(f"  [{vtype.upper():<10}] {name:<32}: Succ={succ:5.1f}% | Grasp={grasp:5.1f}%")

    out_json = os.path.join(EVAL_DIR, "combined_omniarm_benchmark.json")
    with open(out_json, "w") as f:
        json.dump(combined_results, f, indent=2)

    print("\n" + "=" * 80)
    print(" [BENCHMARK SUMMARY] COMBINED OMNIARM RESULTS")
    for r_key in ["panda", "xarm6"]:
        anch = np.mean([r["success_rate"] for r in combined_results[r_key] if r["type"] == "Anchor"])
        interp = np.mean([r["success_rate"] for r in combined_results[r_key] if r["type"] == "In-Between"])
        tot = np.mean([r["success_rate"] for r in combined_results[r_key]])
        grp = np.mean([r["grasp_rate"] for r in combined_results[r_key]])
        print(f" {r_key.upper():15s}: Anchor={anch:.1f}% | In-Between={interp:.1f}% | Overall={tot:.1f}% | Grasp={grp:.1f}%")
    print("=" * 80)


if __name__ == "__main__":
    pts, prop, act = collect_multirobot_dataset(num_episodes_per_robot=500, batch_envs=64, num_points=512, seed=42)
    train_combined_omniarm(pts, prop, act, epochs=30, batch_size=256, lr=1e-3)
    evaluate_combined_suite()
