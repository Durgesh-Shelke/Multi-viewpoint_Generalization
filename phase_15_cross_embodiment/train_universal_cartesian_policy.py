"""
Phase 15: Universal Cartesian PointNet Policy for Zero-Shot Cross-Embodiment Generalization.

Key Principle:
1. 3D Vision: Point cloud is back-projected into the metric table/workspace frame (in meters).
2. Proprioception: The policy receives ONLY the universal 7-dim Tool Center Point (TCP) telemetry
   [x_tcp, y_tcp, z_tcp, qx, qy, qz, qw] + gripper openness (total 8 dims).
   No joint angles or robot-specific kinematics are exposed to the policy!
3. Control Space: The policy outputs 4D delta Cartesian actions [dx, dy, dz, dgripper].
4. Evaluation:
   - Trained Arm: Franka Panda
   - ZERO-SHOT Unseen Arm 1: UFactory xArm 6 + Robotiq 2F-85
   - ZERO-SHOT Unseen Arm 2: Fetch Mobile Manipulator Arm
"""

import os
import sys
sys.path.append("/home/maniskill")
sys.path.append("/home/maniskill/examples/baselines/ppo")
import time
import json
import gc
from types import SimpleNamespace
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
from ppo import Agent as PrivilegedAgent

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = "/home/maniskill/phase_15_cross_embodiment"
OUTPUT_DIR = os.path.join(BASE_DIR, "eval_results")
CKPT_DIR = os.path.join(BASE_DIR, "omniarm")
VIEWPOINTS_JSON = "/home/maniskill/phase_1_to_3/sequential_viewpoint_training/viewpoints.json"
TEACHER_CKPT = "/home/maniskill/phase_12_privileged_state_distillation/teacher/privileged_state_teacher.pt"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)


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

    # Workspace tabletop envelope [X: -0.35..0.35, Y: -0.35..0.35, Z: 0.005..0.40]
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


def get_universal_tcp_proprio(env) -> torch.Tensor:
    """Extracts 8-dim universal hand proprioception: [tcp_x, tcp_y, tcp_z, qx, qy, qz, qw, gripper_openness]."""
    agent = env.unwrapped.agent
    tcp_p = agent.tcp_pose.p  # (B, 3)
    tcp_q = agent.tcp_pose.q  # (B, 4)
    qpos = agent.robot.get_qpos()
    g_open = qpos[:, -1:]     # (B, 1)
    return torch.cat([tcp_p, tcp_q, g_open], dim=-1).float()


class UniversalCartesianPointNetAgent(nn.Module):
    """
    Embodiment-Agnostic Cartesian Policy:
    Operates entirely in metric Cartesian space with 3D point cloud and 8-dim TCP proprioception.
    """
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

    def forward(self, pts: torch.Tensor, tcp_proprio: torch.Tensor) -> torch.Tensor:
        # PointNet max-pooling
        x = pts.transpose(1, 2)
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        p_feat = torch.max(x, dim=2)[0]

        s_feat = self.proprio_fc(tcp_proprio)
        fused = torch.cat([p_feat, s_feat], dim=-1)
        return self.actor_mean(fused)

    def get_action(self, pts: torch.Tensor, tcp_proprio: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        return self.forward(pts, tcp_proprio)


def load_panda_teacher():
    class DummySpace:
        def __init__(self, shape): self.shape = shape
    dummy_vec = SimpleNamespace(
        single_observation_space=DummySpace((42,)),
        single_action_space=DummySpace((4,))
    )
    teacher = PrivilegedAgent(dummy_vec).to(DEVICE)
    ckpt = torch.load(TEACHER_CKPT, map_location=DEVICE)
    teacher.load_state_dict(ckpt)
    teacher.eval()
    print(f"[OK] Privileged Teacher loaded from {TEACHER_CKPT}")
    return teacher


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


def collect_cartesian_dataset(teacher, num_episodes: int = 1500, batch_envs: int = 32, num_points: int = 512, seed: int = 42):
    print("\n" + "=" * 80)
    print(f" [STAGE 1] COLLECTING {num_episodes:,} EPISODES WITH UNIVERSAL TCP PROPRIOCEPTION")
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
            robot_uids="panda",
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
            state_42 = obs["state"][:cur_batch]
            tcp_proprio = get_universal_tcp_proprio(env)[:cur_batch]

            with torch.no_grad():
                act = teacher.get_action(state_42, deterministic=True)
                pts = extract_table_frame_pc(obs, num_points=num_points)[:cur_batch]

            buf_pts[cur_idx:cur_idx + cur_batch] = pts.cpu()
            buf_proprio[cur_idx:cur_idx + cur_batch] = tcp_proprio.cpu()
            buf_actions[cur_idx:cur_idx + cur_batch] = act.cpu()
            cur_idx += cur_batch

            if cur_batch < batch_envs:
                act_pad = torch.zeros((batch_envs, 4), device=DEVICE)
                act_pad[:cur_batch] = act
                obs, _, _, _, _ = vec_env.step(act_pad)
            else:
                obs, _, _, _, _ = vec_env.step(act)

        vec_env.close()
        del vec_env
        del env
        gc.collect()
        torch.cuda.empty_cache()

        if (b + 1) % 5 == 0 or b == 0 or (b + 1) == num_batches:
            print(f"  Batch {b+1:2d}/{num_batches:2d} | Collected: {cur_idx:,} / {total_samples:,} samples | Time: {time.time()-t0:.1f}s")

    print(f"\n[OK] Cartesian dataset collection completed: {cur_idx:,} transitions in {time.time()-t0:.1f}s")
    return buf_pts[:cur_idx], buf_proprio[:cur_idx], buf_actions[:cur_idx]


def train_universal_policy(buf_pts, buf_proprio, buf_actions, num_epochs: int = 25, batch_size: int = 256):
    print("\n" + "=" * 80)
    print(" [STAGE 2] TRAINING UNIVERSAL CARTESIAN POINTNET POLICY")
    print(f" Transitions: {len(buf_actions):,} | Epochs: {num_epochs} | Batch Size: {batch_size}")
    print("=" * 80)

    dataset = TensorDataset(buf_pts, buf_proprio, buf_actions)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True)

    agent = UniversalCartesianPointNetAgent(action_dim=4, proprio_dim=8, point_dim=256).to(DEVICE)
    optimizer = optim.AdamW(agent.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)

    best_loss = float("inf")
    ckpt_path = os.path.join(CKPT_DIR, "universal_cartesian_pointnet_best.pt")

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
            print(f"  Epoch {epoch:2d}/{num_epochs:2d} | Action MSE Loss: {avg_loss:.5f} | Time: {dt:.1f}s")

    print(f"\n[OK] Universal Cartesian Policy saved to {ckpt_path} (MSE: {best_loss:.5f})")
    return ckpt_path


def evaluate_on_robot(agent, robot_uid: str, is_zero_shot: bool = False):
    tag = "[ZERO-SHOT UNSEEN]" if is_zero_shot else "[TRAINED ARM]"
    print("\n" + "=" * 80)
    print(f" {tag} BENCHMARKING ON: {robot_uid.upper()}")
    print("=" * 80)

    with open(VIEWPOINTS_JSON) as f:
        vps = json.load(f)
    vp_dict = {v["id"]: v for v in vps}

    eval_vps = [
        (17, "VP 17 (Front-Mid Anchor)", "trained"),
        (0,  "VP 00 (Front-Low Anchor)", "trained"),
        (20, "VP 20 (Diag-Left Anchor)", "trained"),
        (19, "VP 19 (Diag-Right Anchor)", "trained"),
        (13, "VP 13 (In-Between Az: VP17-20)", "untrained"),
        (14, "VP 14 (In-Between Az: VP17-19)", "untrained"),
        (12, "VP 12 (Flank In-Between R)", "untrained"),
        (15, "VP 15 (Flank In-Between L)", "untrained"),
    ]

    results = []
    batch_envs = 50

    for vid, name, vtype in eval_vps:
        meta = vp_dict[vid]
        cam_eye = meta["world_eye"]

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
        eval_envs = ManiSkillVectorEnv(env, batch_envs, ignore_terminations=True, record_metrics=True)
        obs, _ = eval_envs.reset(seed=2026 + vid)
        grasped_any = torch.zeros(batch_envs, dtype=torch.bool, device=DEVICE)

        for step in range(50):
            tcp_proprio = get_universal_tcp_proprio(env)

            with torch.no_grad():
                pts = extract_table_frame_pc(obs, num_points=512)
                act_4d = agent.get_action(pts, tcp_proprio, deterministic=True)

            # Map 4D Cartesian action to robot action space
            if robot_uid == "fetch":
                # Fetch: actions 5..7 are delta arm pos [dx, dy, dz], action 8 is gripper
                act_full = torch.zeros((batch_envs, 9), device=DEVICE)
                act_full[:, 5:8] = act_4d[:, :3]
                act_full[:, 8] = act_4d[:, 3]
            else:
                act_full = act_4d

            obs, rew, term, trunc, infos = eval_envs.step(act_full)
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
        v_tag = "[ANCHOR]" if vtype == "trained" else "[IN-BETWEEN]"
        print(f"  {v_tag:<12} {name:<32}: Succ={succ_rate:5.1f}% | Grasp={grasp_rate:5.1f}%")

    avg_succ = np.mean([r["success_rate"] for r in results])
    avg_grasp = np.mean([r["grasp_rate"] for r in results])
    print(f"\n===> {robot_uid.upper()} Overall Success: {avg_succ:.1f}% | Grasp: {avg_grasp:.1f}% <===")
    return {
        "robot_uid": robot_uid,
        "is_zero_shot": is_zero_shot,
        "overall_success": avg_succ,
        "overall_grasp": avg_grasp,
        "details": results,
    }


def main():
    teacher = load_panda_teacher()
    buf_pts, buf_proprio, buf_actions = collect_cartesian_dataset(teacher, num_episodes=1500, batch_envs=32)
    best_ckpt = train_universal_policy(buf_pts, buf_proprio, buf_actions, num_epochs=25, batch_size=256)

    agent = UniversalCartesianPointNetAgent(action_dim=4, proprio_dim=8, point_dim=256).to(DEVICE)
    ckpt = torch.load(best_ckpt, map_location=DEVICE)
    agent.load_state_dict(ckpt.get("model_state", ckpt))
    agent.eval()

    all_results = {}

    # 1. Benchmark on Trained Arm: Panda
    all_results["panda"] = evaluate_on_robot(agent, "panda", is_zero_shot=False)

    # 2. ZERO-SHOT Benchmark on Unseen Arm 1: xArm 6 Robotiq
    all_results["xarm6_robotiq"] = evaluate_on_robot(agent, "xarm6_robotiq", is_zero_shot=True)

    # 3. ZERO-SHOT Benchmark on Unseen Arm 2: Fetch Arm
    all_results["fetch"] = evaluate_on_robot(agent, "fetch", is_zero_shot=True)

    # Save benchmark JSON
    json_path = os.path.join(OUTPUT_DIR, "phase15_zero_shot_cross_embodiment.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Plot Multi-Arm Comparison
    fig, ax = plt.subplots(figsize=(14, 6))
    robots = ["panda", "xarm6_robotiq", "fetch"]
    x = np.arange(len(robots))
    succs = [all_results[r]["overall_success"] for r in robots]
    grasps = [all_results[r]["overall_grasp"] for r in robots]
    labels = [
        f"Franka Panda\n[Trained Arm]\nSucc: {succs[0]:.1f}%",
        f"xArm 6 Robotiq\n[ZERO-SHOT UNSEEN]\nSucc: {succs[1]:.1f}%",
        f"Fetch Arm\n[ZERO-SHOT UNSEEN]\nSucc: {succs[2]:.1f}%",
    ]
    colors = ["#1565c0", "#7b1fa2", "#2e7d32"]

    width = 0.35
    ax.bar(x - width/2, grasps, width, label="Grasp Initiation Rate (%)", color="#b0bec5", edgecolor="black", alpha=0.85)
    ax.bar(x + width/2, succs, width, label="Pick-and-Lift Success Rate (%)", color=colors, edgecolor="black")

    ax.set_ylabel("Rate (%)", fontsize=12, fontweight="bold")
    ax.set_title("Phase 15: Zero-Shot Cross-Embodiment Generalization Across Robot Arms\n(Trained on Panda Only -> Evaluated Zero-Shot on xArm6 and Fetch)", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", fontsize=11)

    plot_path = os.path.join(OUTPUT_DIR, "phase15_zero_shot_cross_embodiment.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"\n[OK] Zero-shot cross-embodiment plot saved to {plot_path}")


if __name__ == "__main__":
    main()
