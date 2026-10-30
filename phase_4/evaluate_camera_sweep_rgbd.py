"""
13-Angle Semicircular Camera Sweep Benchmark for 4-Channel RGB-D Policy.
Evaluates viewpoint robustness along the horizontal arc on the sphere (R = 1.0942 m)
centered at the robot base [-0.615, 0.0, 0.0].

Angles: 0 deg (Side +Y) -> 90 deg (Front Training View) -> 180 deg (Side -Y) in 15-degree increments.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import gymnasium as gym
import imageio
import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, "/home/maniskill")
sys.path.insert(0, "/home/maniskill/examples/baselines/ppo")

import mani_skill.envs
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper, FlattenRGBDObservationWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from ppo_rgbd import Agent

BASE_POS = np.array([-0.615, 0.0, 0.0])
TARGET_POS = np.array([-0.1, 0.0, 0.1])
R_HORIZ = 0.915
Z_CAM = 0.600
SPHERE_RADIUS = np.sqrt(R_HORIZ**2 + Z_CAM**2)  # 1.0942 m


def get_camera_eye(deg: float) -> list:
    """Computes camera eye coordinates for a given angle in degrees along the semicircular arc."""
    rad = np.deg2rad(deg)
    x_rel = R_HORIZ * np.sin(rad)
    y_rel = R_HORIZ * np.cos(rad)
    eye = [float(BASE_POS[0] + x_rel), float(BASE_POS[1] + y_rel), float(Z_CAM)]
    return eye


def evaluate_single_angle(
    agent: Agent,
    angle_deg: float,
    episodes: int = 1000,
    num_envs: int = 100,
    device: str = "cuda",
) -> dict:
    eye = get_camera_eye(angle_deg)
    target = TARGET_POS.tolist()

    print(f"\n{'='*72}")
    print(f" EVALUATING ANGLE: {angle_deg:5.1f} deg | Eye: [{eye[0]:6.3f}, {eye[1]:6.3f}, {eye[2]:6.3f}]")
    print(f" Target Runs: {episodes:,} | Parallel Instances: {num_envs}")
    print(f"{'='*72}")

    env = gym.make(
        "PickCube-v1",
        num_envs=num_envs,
        obs_mode="rgbd",
        control_mode="pd_ee_delta_pos",
        render_mode=None,
        sim_backend="physx_cuda",
        sensor_cam_eye_pos=eye,
        sensor_cam_target_pos=target,
    )
    env = FlattenRGBDObservationWrapper(env, rgb=True, depth=True, state=True, sep_depth=True)
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    env = ManiSkillVectorEnv(env, num_envs, ignore_terminations=False, record_metrics=True)

    obs, _ = env.reset(seed=int(angle_deg * 1000 + 42))

    completed = 0
    successes = 0
    returns = []
    lens = []
    t0 = time.time()

    with torch.no_grad():
        while completed < episodes:
            action = agent.get_action(obs, deterministic=True)
            obs, rew, term, trunc, infos = env.step(action)

            if "final_info" in infos:
                mask = infos["_final_info"]
                for i in range(len(mask)):
                    if mask[i]:
                        ep_info = infos["final_info"]["episode"]
                        succ_key = "success_once" if "success_once" in ep_info else ("success" if "success" in ep_info else "success_at_end")
                        succ = bool(ep_info[succ_key][i].item() if torch.is_tensor(ep_info[succ_key]) else ep_info[succ_key])
                        if succ:
                            successes += 1
                        completed += 1
                        returns.append(float(ep_info["return"][i].item() if torch.is_tensor(ep_info["return"]) else ep_info["return"]))
                        lens.append(float(ep_info["episode_len"][i].item() if torch.is_tensor(ep_info["episode_len"]) else ep_info["episode_len"]))
                        if completed >= episodes:
                            break

    env.close()
    elapsed = time.time() - t0
    rate = (successes / completed) * 100.0 if completed > 0 else 0.0
    mean_ret = float(np.mean(returns)) if returns else 0.0
    mean_len = float(np.mean(lens)) if lens else 0.0

    print(f" >>> Angle {angle_deg:5.1f} deg Result: {successes}/{completed} Successes ({rate:.2f}%) | Len: {mean_len:.1f} steps | Return: {mean_ret:.2f} | Time: {elapsed:.1f}s")

    return {
        "angle_deg": angle_deg,
        "camera_eye": eye,
        "target_pos": target,
        "episodes": completed,
        "success_count": successes,
        "success_rate": round(rate, 2),
        "mean_return": round(mean_ret, 4),
        "mean_length": round(mean_len, 2),
        "elapsed_seconds": round(elapsed, 2),
    }


def record_angle_video(
    agent: Agent,
    angle_deg: float,
    output_path: str,
    seed: int = 42,
    max_steps: int = 50,
):
    eye = get_camera_eye(angle_deg)
    target = TARGET_POS.tolist()

    env = gym.make(
        "PickCube-v1",
        num_envs=1,
        obs_mode="rgbd",
        control_mode="pd_ee_delta_pos",
        render_mode="rgb_array",
        sim_backend="physx_cuda",
        sensor_cam_eye_pos=eye,
        sensor_cam_target_pos=target,
    )
    env = FlattenRGBDObservationWrapper(env, rgb=True, depth=True, state=True, sep_depth=True)
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    env = ManiSkillVectorEnv(env, 1, ignore_terminations=False, record_metrics=True)

    obs, _ = env.reset(seed=seed)
    frames = []

    with torch.no_grad():
        for step in range(max_steps):
            render_frame = env.render()
            if isinstance(render_frame, torch.Tensor):
                render_frame = render_frame.cpu().numpy()
            if render_frame.ndim == 4:
                render_frame = render_frame[0]

            sensor_frame = obs["rgb"][0].cpu().numpy()
            sensor_large = cv2.resize(sensor_frame, (512, 512), interpolation=cv2.INTER_NEAREST)

            cv2.putText(
                render_frame,
                "Workspace View (Third-Person)",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                sensor_large,
                f"Sensor Cam: {angle_deg:.0f} deg (RGB-D Policy)",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )

            composite = np.concatenate([sensor_large, render_frame], axis=1)
            frames.append(composite)

            action = agent.get_action(obs, deterministic=True)
            obs, rew, term, trunc, info = env.step(action)
            if term.any() or trunc.any():
                for _ in range(8):
                    frames.append(composite)
                break

    env.close()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    imageio.mimsave(output_path, frames, fps=20)
    print(f"[Video] Saved {len(frames)} frames to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="13-Angle Camera Sweep for RGB-D Policy")
    parser.add_argument("--checkpoint", type=str, default="/home/maniskill/phase_4/final_models/clean_rgbd_baseline_front_view_91percent.pt")
    parser.add_argument("--episodes", type=int, default=1000, help="Episodes per angle")
    parser.add_argument("--num_envs", type=int, default=100, help="Parallel environments")
    parser.add_argument("--output_json", type=str, default="/home/maniskill/phase_4/runs/sim2real_45mm_block_rgbd/sweep_results_front_rgbd.json")
    args = parser.parse_args()

    print("\n" + "=" * 78)
    print(" 13-ANGLE HORIZONTAL CAMERA SWEEP BENCHMARK (RGB-D BASELINE)")
    print(f" Checkpoint:  {args.checkpoint}")
    print(f" Episodes/VP: {args.episodes:,} ({args.episodes * 13:,} Total Episodes)")
    print(f" Instances:   {args.num_envs} Parallel GPU Envs")
    print(f" Output JSON: {args.output_json}")
    print("=" * 78 + "\n")

    # Load agent
    dummy_env = gym.make("PickCube-v1", num_envs=2, obs_mode="rgbd", control_mode="pd_ee_delta_pos", sim_backend="physx_cuda")
    dummy_env = FlattenRGBDObservationWrapper(dummy_env, rgb=True, depth=True, state=True, sep_depth=True)
    dummy_env = ManiSkillVectorEnv(dummy_env, 2, ignore_terminations=False, record_metrics=True)
    sample_obs, _ = dummy_env.reset()
    agent = Agent(dummy_env, sample_obs=sample_obs).cuda()
    agent.load_state_dict(torch.load(args.checkpoint, map_location="cuda"))
    agent.eval()
    dummy_env.close()

    angles = [0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0, 105.0, 120.0, 135.0, 150.0, 165.0, 180.0]
    all_results = []
    total_start = time.time()

    for angle in angles:
        res = evaluate_single_angle(agent, angle, episodes=args.episodes, num_envs=args.num_envs)
        all_results.append(res)
        if angle in [0.0, 45.0, 90.0, 135.0, 180.0]:
            out_vid = os.path.join(os.path.dirname(args.output_json), f"angle_{int(angle):03d}deg.mp4")
            record_angle_video(agent, angle, output_path=out_vid)

    total_time = time.time() - total_start

    summary = {
        "modality": "RGB-D (4 Channels: RGB + Metric Depth)",
        "training_view": "90.0 deg (Front View: [0.3, 0.0, 0.6])",
        "checkpoint": args.checkpoint,
        "total_angles": len(angles),
        "episodes_per_angle": args.episodes,
        "total_episodes": len(angles) * args.episodes,
        "total_elapsed_seconds": round(total_time, 2),
        "mean_success_rate": round(float(np.mean([r["success_rate"] for r in all_results])), 2),
        "results": all_results,
    }

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(summary, f, indent=2)

    # Also mirror to research_reports
    os.makedirs("/home/maniskill/phase_4/reports", exist_ok=True)
    with open("/home/maniskill/phase_4/reports/sweep_results_front_rgbd.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 78)
    print(" 13-ANGLE HORIZONTAL SWEEP COMPLETED SUCCESSFULLY")
    print(f" Total Episodes:    {summary['total_episodes']:,}")
    print(f" Mean Success Rate: {summary['mean_success_rate']:.2f}%")
    print(f" Total Runtime:     {total_time:.1f}s ({total_time / 60:.1f} min)")
    print(f" Saved to:          {args.output_json}")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()
