"""
Phase 14 Video Showcase Generator.

Records high-resolution MP4 rollout videos of PointNetAgent picking and lifting cubes
across novel in-between camera viewpoints with picture-in-picture (PiP) inset and telemetry overlays.
"""

import os
import sys
sys.path.append("/home/maniskill")
import cv2
import numpy as np
import torch
import gymnasium as gym

import mani_skill.envs
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from phase_13_pointcloud_generalization.pointnet_models import PointNetAgent, extract_robot_frame_pc

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
POINTNET_CKPT = "/home/maniskill/phase_13_pointcloud_generalization/student/pointnet_student_best.pt"
VIDEO_DIR = "/home/maniskill/phase_14_robustness_and_videos/videos"
os.makedirs(VIDEO_DIR, exist_ok=True)


def record_rollout_video(agent, cam_eye, video_name: str, title: str, seed: int = 42):
    env = gym.make(
        "PickCube-v1",
        obs_mode="pointcloud+state",
        control_mode="pd_ee_delta_pos",
        render_mode="rgb_array",
        num_envs=1,
        sensor_cam_eye_pos=cam_eye,
        sensor_cam_target_pos=[-0.1, 0.0, 0.1],
        sensor_configs=dict(width=128, height=128),
    )
    vec_env = ManiSkillVectorEnv(env, 1, ignore_terminations=True, record_metrics=True)
    obs, _ = vec_env.reset(seed=seed)

    video_path = os.path.join(VIDEO_DIR, video_name)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = 20
    writer = cv2.VideoWriter(video_path, fourcc, fps, (512, 512))

    for step in range(50):
        # Render third-person observer camera (512x512)
        frame_rgb = vec_env.render()[0].cpu().numpy()  # uint8 RGB
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        # Inset Robot Base Camera View (128x128)
        robot_cam_rgb = obs["sensor_data"]["base_camera"]["rgb"][0].cpu().numpy()
        robot_cam_bgr = cv2.cvtColor(robot_cam_rgb, cv2.COLOR_RGB2BGR)
        # Place in upper-right corner (offset: x in [364..492], y in [20..148])
        frame_bgr[20:148, 364:492] = robot_cam_bgr
        cv2.rectangle(frame_bgr, (363, 19), (493, 149), (0, 255, 255), 2)
        cv2.putText(frame_bgr, "Sensor View", (370, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        # Telemetry Text
        cv2.putText(frame_bgr, title, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.putText(frame_bgr, f"Step: {step+1:2d}/50", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Policy Action
        proprio_29 = obs["state"][:, :29].to(DEVICE)
        with torch.no_grad():
            pts = extract_robot_frame_pc(obs, num_points=512).to(DEVICE)
            act = agent.get_action(pts, proprio_29, deterministic=True)

        obs, rew, term, trunc, infos = vec_env.step(act)

        # Status text
        is_grasped = infos.get("is_grasped", torch.tensor([False]))[0].item()
        if is_grasped:
            cv2.putText(frame_bgr, "STATUS: OBJECT GRASPED", (20, 480), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        else:
            cv2.putText(frame_bgr, "STATUS: APPROACHING", (20, 480), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)

        writer.write(frame_bgr)

    writer.release()
    vec_env.close()
    del vec_env
    del env
    print(f"  [OK] Saved video: {video_path}")


def main():
    print("=" * 80)
    print(" [PHASE 14] GENERATING VIDEO SHOWCASE FOR NOVEL IN-BETWEEN VIEWPOINTS")
    print("=" * 80)

    agent = PointNetAgent(action_dim=4, proprio_dim=29, point_dim=256).to(DEVICE)
    ckpt = torch.load(POINTNET_CKPT, map_location=DEVICE)
    agent.load_state_dict(ckpt.get("model_state", ckpt))
    agent.eval()

    showcase_viewpoints = [
        ("vp13_inbetween_azimuth.mp4", [-0.0384, -0.3989, 0.4958], "VP 13 (In-Between Azimuth: -24.8 deg)"),
        ("vp15_flank_left.mp4", [-0.1066, -0.8000, 0.5459], "VP 15 (Flank In-Between Left: -57.3 deg)"),
        ("vp10_inbetween_elevation.mp4", [0.3752, 0.0505, 0.4450], "VP 10 (In-Between Elevation: 24.0 deg)"),
        ("vp12_flank_right.mp4", [-0.1264, 0.8523, 0.4757], "VP 12 (Flank In-Between Right: +60.2 deg)"),
        ("random_novel_diagonal.mp4", [0.1500, 0.5500, 0.7000], "Novel Random Camera (Az: +35 deg, El: 45 deg)"),
    ]

    for vname, eye, title in showcase_viewpoints:
        record_rollout_video(agent, eye, vname, title, seed=2026)

    print("\n[OK] All showcase videos generated successfully!")


if __name__ == "__main__":
    main()
