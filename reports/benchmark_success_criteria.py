"""
Benchmark Success Criteria Comparison:
Any-Time Success (ManiSkill default) vs. Terminal Success (at Step 50) vs. Sustained Success (>= 10 consecutive steps)
"""

import os
import sys
import torch
import numpy as np
import gymnasium as gym

import mani_skill.envs
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from mani_skill.utils import sapien_utils

REPO_DIR = "/home/maniskill/Multi-viewpoint_Generalization"
sys.path.append(REPO_DIR)
sys.path.append(os.path.join(REPO_DIR, "phase_13_pointcloud_generalization"))
from pointnet_models import PointNetAgent, extract_robot_frame_pc
from record_16view_grid_videos import compute_cam_eye, VIEWPOINTS_16

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tasks = [
    ("PickCube-v1", "pd_ee_delta_pos", 4, 29, f"{REPO_DIR}/phase_13_pointcloud_generalization/student/pointnet_student_best.pt", [-0.1, 0.0, 0.1]),
    ("PushCube-v1", "pd_ee_delta_pose", 7, 28, f"{REPO_DIR}/phase_17_multitask_replication/checkpoints/pointnet_pushcube_consistent_best.pt", [-0.1, 0.0, 0.1]),
    ("PullCube-v1", "pd_ee_delta_pose", 7, 28, f"{REPO_DIR}/phase_17_multitask_replication/checkpoints/pointnet_pullcube_consistent_best.pt", [-0.1, 0.0, 0.1]),
    ("LiftPegUpright-v1", "pd_ee_delta_pose", 7, 25, f"{REPO_DIR}/phase_18_multitask_extended/checkpoints/pointnet_liftpegupright-v1_best.pt", [0.0, 0.0, 0.05]),
    ("PokeCube-v1", "pd_joint_delta_pos", 8, 25, f"{REPO_DIR}/phase_18_multitask_extended/checkpoints/pointnet_pokecube-v1_best.pt", [0.0, 0.0, 0.05]),
]

print("=" * 90)
print(f" {'Task':<20} | {'Any-Time (ManiSkill)':<22} | {'Terminal (Step 50)':<20} | {'Sustained (>=10 steps)':<22}")
print("=" * 90)

for env_id, control_mode, action_dim, proprio_dim, ckpt_path, target_pos in tasks:
    policy = PointNetAgent(action_dim=action_dim, proprio_dim=proprio_dim, point_dim=256).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    st = ckpt["model_state"] if "model_state" in ckpt else ckpt
    policy.load_state_dict(st)
    policy.eval()

    vp = VIEWPOINTS_16[3] # Front-Mid Anchor
    cam_eye = compute_cam_eye(vp["az"], vp["el"], vp["dist"])
    cam_pose = sapien_utils.look_at(eye=cam_eye, target=target_pos)
    sensor_cfg = dict(base_camera=dict(pose=cam_pose, width=128, height=128))

    num_envs = 50
    env = gym.make(
        env_id,
        obs_mode="pointcloud+state",
        control_mode=control_mode,
        num_envs=num_envs,
        sensor_configs=sensor_cfg,
        shader_dir="minimal",
    )
    vec_env = ManiSkillVectorEnv(env, num_envs, ignore_terminations=True, record_metrics=True)
    obs, _ = vec_env.reset(seed=100)

    step_succ = []
    for step in range(50):
        proprio = obs["state"][:, :proprio_dim].to(DEVICE)
        with torch.no_grad():
            pts = extract_robot_frame_pc(obs, num_points=512)
            act = policy.get_action(pts, proprio, deterministic=True)
        obs, rew, term, trunc, info = vec_env.step(act)
        if trunc[0]:
            s = info["final_info"]["success"].cpu().numpy()
        else:
            s = info["success"].cpu().numpy()
        step_succ.append(s)

    step_succ = np.array(step_succ) # [50, num_envs]
    any_time = np.any(step_succ, axis=0).mean() * 100.0
    terminal = step_succ[-1].mean() * 100.0
    
    # Calculate consecutive sustained steps for each env
    sustained_10 = []
    for e in range(num_envs):
        history = step_succ[:, e]
        max_consec = 0
        cur_consec = 0
        for val in history:
            if val:
                cur_consec += 1
                if cur_consec > max_consec:
                    max_consec = cur_consec
            else:
                cur_consec = 0
        sustained_10.append(max_consec >= 10)
    sustained_10_sr = np.mean(sustained_10) * 100.0

    print(f" {env_id:<20} | {any_time:20.1f}% | {terminal:18.1f}% | {sustained_10_sr:20.1f}%")
    vec_env.close()

print("=" * 90)
