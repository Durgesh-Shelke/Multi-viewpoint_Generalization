"""
Generate Publication-Grade 16-Viewpoint Grid Videos (4x4 Split-Screen)
------------------------------------------------------------------------
Demonstrates simultaneous 16-angle continuous viewpoint invariance for
trained 3D PointNet models on tabletop manipulation tasks:
  1. PickCube-v1
  2. PushCube-v1
  3. PullCube-v1
  4. PokeCube-v1

CRUCIAL ARCHITECTURAL SPECIFICATION:
Each tile in the 4x4 grid is a completely independent, closed-loop rollout.
In Tile k, the PointNet policy actively perceives the scene in real time
from Viewpoint k's point cloud and drives the robot arm to solve the task.
"""

import os
import sys
import gc
import time
import subprocess
import cv2
import numpy as np
import torch
import gymnasium as gym
import mani_skill.envs
from mani_skill.utils import sapien_utils
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

REPO_DIR = "/home/maniskill/Multi-viewpoint_Generalization"
sys.path.append(REPO_DIR)
sys.path.append(os.path.join(REPO_DIR, "phase_13_pointcloud_generalization"))
from pointnet_models import PointNetAgent, extract_robot_frame_pc

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = os.path.join(REPO_DIR, "videos")
os.makedirs(OUT_DIR, exist_ok=True)

FFMPEG_BIN = "/opt/conda/lib/python3.9/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2"

BASE_POS = np.array([-0.615, 0.0, 0.0])

# 16 Diverse Continuous Viewpoints Covering the Entire Frontal Dome
VIEWPOINTS_16 = [
    # Row 1: Low Elevation Frontal & Flank Viewpoints
    {"id": 0, "name": "Front-Low", "type": "Anchor", "az": 0.0, "el": 12.3, "dist": 1.09},
    {"id": 1, "name": "Low-Left", "type": "Novel", "az": -18.5, "el": 15.0, "dist": 1.12},
    {"id": 2, "name": "Low-Right", "type": "Novel", "az": +18.5, "el": 15.0, "dist": 1.12},
    {"id": 3, "name": "Front-Mid", "type": "Anchor", "az": -4.7, "el": 31.8, "dist": 1.09},
    # Row 2: Mid Elevation & Diagonals
    {"id": 4, "name": "Diag-Left", "type": "Anchor", "az": -37.2, "el": 35.0, "dist": 1.09},
    {"id": 5, "name": "Diag-Right", "type": "Anchor", "az": +47.8, "el": 33.7, "dist": 1.09},
    {"id": 6, "name": "Mid-InBetw-L", "type": "Novel", "az": -22.0, "el": 28.0, "dist": 1.05},
    {"id": 7, "name": "Mid-InBetw-R", "type": "Novel", "az": +25.0, "el": 28.0, "dist": 1.05},
    # Row 3: High Elevation & Overhead
    {"id": 8, "name": "High-Mid", "type": "Novel", "az": -12.0, "el": 42.0, "dist": 1.15},
    {"id": 9, "name": "High-Right", "type": "Novel", "az": +32.0, "el": 42.0, "dist": 1.15},
    {"id": 10, "name": "High-Overhead", "type": "Novel", "az": 0.0, "el": 40.0, "dist": 1.10},
    {"id": 11, "name": "High-Diag-L", "type": "Novel", "az": -45.0, "el": 45.0, "dist": 1.18},
    # Row 4: Extreme Flanks & Zoom Variations
    {"id": 12, "name": "Flank-Left 60°", "type": "OOD", "az": -60.0, "el": 26.0, "dist": 1.10},
    {"id": 13, "name": "Flank-Right 62°", "type": "OOD", "az": +62.0, "el": 26.0, "dist": 1.10},
    {"id": 14, "name": "Close-Zoom", "type": "Zoom", "az": -15.0, "el": 22.0, "dist": 0.98},
    {"id": 15, "name": "Wide-Zoom", "type": "Zoom", "az": +15.0, "el": 30.0, "dist": 1.32},
]


def compute_cam_eye(az_deg, el_deg, dist):
    az_rad = np.radians(az_deg)
    el_rad = np.radians(el_deg)
    rel_x = dist * np.cos(el_rad) * np.cos(az_rad)
    rel_y = dist * np.cos(el_rad) * np.sin(az_rad)
    rel_z = dist * np.sin(el_rad)
    return (BASE_POS + np.array([rel_x, rel_y, rel_z])).tolist()


def overlay_tile_hud(img, vp_info, step, total_steps, is_success):
    """
    Overlays a clean, high-contrast HUD on an individual camera tile.
    """
    h, w = img.shape[:2]
    out = img.copy()

    # Semi-transparent top banner for Camera Info
    top_bar = np.zeros((32, w, 3), dtype=np.uint8)
    alpha = 0.65
    out[:32, :] = cv2.addWeighted(out[:32, :], 1.0 - alpha, top_bar, alpha, 0)

    # Label text & Type Tag
    vp_name = vp_info["name"]
    vp_id = vp_info["id"] + 1
    tag = vp_info["type"]
    tag_color = (0, 255, 255) if tag == "Anchor" else (100, 255, 100) if tag == "Novel" else (255, 180, 50) if tag == "OOD" else (220, 100, 255)

    label_str = f"VP {vp_id:02d}: {vp_name}"
    cv2.putText(out, label_str, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, label_str, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (20, 20, 20), 1, cv2.LINE_AA)

    # Tag badge in top banner
    cv2.putText(out, f"[{tag.upper()}]", (w - 78, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.44, tag_color, 1, cv2.LINE_AA)

    # Bottom bar for coordinates & active status
    bot_bar = np.zeros((24, w, 3), dtype=np.uint8)
    out[h - 24:, :] = cv2.addWeighted(out[h - 24:, :], 1.0 - alpha, bot_bar, alpha, 0)

    coord_str = f"Az: {vp_info['az']:+.0f}° | El: {vp_info['el']:.0f}° | Dist: {vp_info['dist']:.2f}m"
    cv2.putText(out, coord_str, (8, h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1, cv2.LINE_AA)

    # Success badge at bottom right
    if is_success:
        cv2.putText(out, "● SUCCESS", (w - 92, h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 0), 1, cv2.LINE_AA)
    else:
        cv2.putText(out, "● ACTIVE", (w - 80, h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1, cv2.LINE_AA)

    # Subtle 1px dark border around tile
    cv2.rectangle(out, (0, 0), (w - 1, h - 1), (40, 40, 40), 1)
    return out


def record_16view_grid_video(task_name: str, env_id: str, control_mode: str, action_dim: int,
                             ckpt_path: str, seed: int = 100, num_steps: int = 50,
                             target_pos=[-0.1, 0.0, 0.1], proprio_dim: int = 29):
    print("\n" + "=" * 80)
    print(f" RECORDING 16-VIEWPOINT CLOSED-LOOP GRID VIDEO FOR: {task_name.upper()}")
    print(f" Env: {env_id} | Control Mode: {control_mode} | Action Dim: {action_dim} | Proprio Dim: {proprio_dim} | Seed: {seed}")
    print(" 16 Independent Closed-Loop Rollouts (Real-time PointNet perception per tile)")
    print("=" * 80)

    # 1. Load Policy
    policy = PointNetAgent(action_dim=action_dim, proprio_dim=proprio_dim, point_dim=256).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    st = ckpt["model_state"] if "model_state" in ckpt else ckpt
    policy.load_state_dict(st)
    policy.eval()
    print(f"[OK] Policy loaded from {ckpt_path}")

    # 2. Execute 16 Independent Closed-Loop Rollouts
    print(f"\n[STAGE 1] Executing 16 independent closed-loop test rollouts (480x270 per tile)...")
    tile_w, tile_h = 480, 270 # 16:9 aspect ratio per tile -> 4x4 grid = exactly 1920x1080 (1080p Full HD)
    all_camera_frames = [] # Shape: [16, num_steps + 1, tile_h, tile_w, 3]
    view_successes = []

    t_start = time.time()

    for k, vp_info in enumerate(VIEWPOINTS_16):
        t0 = time.time()
        cam_eye = compute_cam_eye(vp_info["az"], vp_info["el"], vp_info["dist"])
        cam_pose = sapien_utils.look_at(eye=cam_eye, target=target_pos)
        sensor_cfg = dict(base_camera=dict(pose=cam_pose, width=tile_w, height=tile_h))

        env_cam = gym.make(
            env_id,
            obs_mode="pointcloud+state",
            control_mode=control_mode,
            num_envs=1,
            sensor_configs=sensor_cfg,
            shader_dir="minimal",
        )
        vec_env_cam = ManiSkillVectorEnv(env_cam, 1, ignore_terminations=True, record_metrics=True)
        obs, _ = vec_env_cam.reset(seed=seed)
        frames_k = []
        succ_latch = False

        # Step 0 initial observation & HUD
        rgb0 = obs["sensor_data"]["base_camera"]["rgb"][0].cpu().numpy()
        frames_k.append(overlay_tile_hud(rgb0, vp_info, 0, num_steps, False))

        for step in range(num_steps):
            proprio = obs["state"][:, :proprio_dim].to(DEVICE)
            with torch.no_grad():
                pts = extract_robot_frame_pc(obs, num_points=512)
                act = policy.get_action(pts, proprio, deterministic=True)
            obs, rew, term, trunc, info = vec_env_cam.step(act)
            if "success" in info and info["success"][0].item():
                succ_latch = True

            rgb = obs["sensor_data"]["base_camera"]["rgb"][0].cpu().numpy()
            hud_frame = overlay_tile_hud(rgb, vp_info, step + 1, num_steps, succ_latch)
            frames_k.append(hud_frame)

        vec_env_cam.close()
        del vec_env_cam
        del env_cam
        gc.collect()
        torch.cuda.empty_cache()

        view_successes.append(succ_latch)
        all_camera_frames.append(frames_k)
        status_str = "SUCCESS" if succ_latch else "FAILED"
        dt = time.time() - t0
        print(f"  Tile {k+1:02d}/16: {vp_info['name']:<18} (Az={vp_info['az']:+5.1f}°, El={vp_info['el']:4.1f}°, Dist={vp_info['dist']:.2f}m) -> {status_str} ({dt:.1f}s)", flush=True)

    print(f"\n[OK] All 16 independent test rollouts completed in {time.time()-t_start:.1f}s")
    print(f"     Score: {sum(view_successes)}/16 viewpoints succeeded ({sum(view_successes)/16*100:.1f}%)")

    # 3. Assemble 4x4 Grid Video into 1920x1080 Full HD Video
    print(f"\n[STAGE 2] Composing 4x4 Grid into 1920x1080 Full HD Video...")
    total_frames = len(all_camera_frames[0])
    raw_video_path = os.path.join(OUT_DIR, f"{task_name.lower()}_16view_grid_raw.mp4")
    final_video_path = os.path.join(OUT_DIR, f"{task_name.lower()}_16view_continuous_viewpoint_grid.mp4")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(raw_video_path, fourcc, 25.0, (1920, 1080))

    for f_idx in range(total_frames):
        # 4 rows of 4 columns
        rows = []
        for r in range(4):
            row_tiles = [all_camera_frames[r * 4 + c][f_idx] for c in range(4)]
            row_img = np.hstack(row_tiles)
            rows.append(row_img)
        grid_frame = np.vstack(rows) # Exactly (1080, 1920, 3)

        # Convert RGB to BGR for cv2
        grid_bgr = cv2.cvtColor(grid_frame, cv2.COLOR_RGB2BGR)
        out_writer.write(grid_bgr)

    # Hold the final frame for 1.5 seconds (38 frames) so viewer can admire task completion across all tiles
    final_bgr = cv2.cvtColor(grid_frame, cv2.COLOR_RGB2BGR)
    for _ in range(38):
        out_writer.write(final_bgr)

    out_writer.release()
    print(f"[OK] Raw video written: {raw_video_path}")

    # 4. Convert to H.264 using bundled ffmpeg for universal browser/slide compatibility (5x slowed down for human viewing)
    print(f"[STAGE 3] Encoding universal H.264 MP4 with ffmpeg (5x slowed down)...")
    cmd = [
        FFMPEG_BIN, "-y",
        "-i", raw_video_path,
        "-filter:v", "setpts=5.0*PTS",
        "-r", "25",
        "-c:v", "libx264",
        "-preset", "slow",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        final_video_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    os.remove(raw_video_path)

    # Copy to root repo and artifacts
    os.system(f"cp {final_video_path} {REPO_DIR}/{task_name.lower()}_16view_grid.mp4")
    curr_artifact_dir = "/root/.gemini/antigravity-cli/brain/4509f9ab-8307-466a-8346-a6d6f96fae85"
    if os.path.exists(curr_artifact_dir):
        os.makedirs(f"{curr_artifact_dir}/videos", exist_ok=True)
        os.system(f"cp {final_video_path} {curr_artifact_dir}/videos/{task_name.lower()}_16view_continuous_viewpoint_grid.mp4")
        os.system(f"cp {final_video_path} {curr_artifact_dir}/{task_name.lower()}_16view_grid.mp4")

    size_mb = os.path.getsize(final_video_path) / (1024 * 1024)
    print(f"[SUCCESS] 16-Viewpoint Grid Video Generated!")
    print(f"  Path: {final_video_path}")
    print(f"  Resolution: 1920x1080 (Full HD, 4x4 Grid)")
    print(f"  Size: {size_mb:.2f} MB")
    return final_video_path


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="all", choices=["all", "PickCube", "PushCube", "PullCube", "PokeCube", "LiftPegUpright"])
    args = parser.parse_args()

    tasks_dict = {
        "PushCube": {
            "name": "PushCube",
            "env_id": "PushCube-v1",
            "control_mode": "pd_ee_delta_pose",
            "action_dim": 7,
            "proprio_dim": 28,
            "ckpt": "/home/maniskill/Multi-viewpoint_Generalization/phase_17_multitask_replication/checkpoints/pointnet_pushcube_consistent_best.pt",
            "seed": 42,
            "target_pos": [-0.1, 0.0, 0.1]
        },
        "PullCube": {
            "name": "PullCube",
            "env_id": "PullCube-v1",
            "control_mode": "pd_ee_delta_pose",
            "action_dim": 7,
            "proprio_dim": 28,
            "ckpt": "/home/maniskill/Multi-viewpoint_Generalization/phase_17_multitask_replication/checkpoints/pointnet_pullcube_consistent_best.pt",
            "seed": 42,
            "target_pos": [-0.1, 0.0, 0.1]
        },
        "PickCube": {
            "name": "PickCube",
            "env_id": "PickCube-v1",
            "control_mode": "pd_ee_delta_pos",
            "action_dim": 4,
            "proprio_dim": 29,
            "ckpt": "/home/maniskill/Multi-viewpoint_Generalization/phase_13_pointcloud_generalization/student/pointnet_student_best.pt",
            "seed": 10,
            "target_pos": [-0.1, 0.0, 0.1]
        },
        "PokeCube": {
            "name": "PokeCube",
            "env_id": "PokeCube-v1",
            "control_mode": "pd_joint_delta_pos",
            "action_dim": 8,
            "proprio_dim": 29,
            "ckpt": "/home/maniskill/Multi-viewpoint_Generalization/phase_18_multitask_extended/checkpoints/pointnet_pokecube-v1_best.pt",
            "seed": 7,
            "target_pos": [0.0, 0.0, 0.05]
        },
        "LiftPegUpright": {
            "name": "LiftPegUpright",
            "env_id": "LiftPegUpright-v1",
            "control_mode": "pd_ee_delta_pose",
            "action_dim": 7,
            "proprio_dim": 25,
            "ckpt": "/home/maniskill/Multi-viewpoint_Generalization/phase_18_multitask_extended/checkpoints/pointnet_liftpegupright-v1_best.pt",
            "seed": 7,
            "target_pos": [0.0, 0.0, 0.05]
        }
    }

    if args.task == "all":
        tasks_to_run = list(tasks_dict.values())
    else:
        tasks_to_run = [tasks_dict[args.task]]

    for t in tasks_to_run:
        record_16view_grid_video(
            task_name=t["name"],
            env_id=t["env_id"],
            control_mode=t["control_mode"],
            action_dim=t["action_dim"],
            ckpt_path=t["ckpt"],
            seed=t["seed"],
            num_steps=50,
            target_pos=t["target_pos"],
            proprio_dim=t["proprio_dim"]
        )


if __name__ == "__main__":
    main()
