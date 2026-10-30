"""
Camera Sweep Evaluation Script for Sim-to-Real Visual Manipulation Policy.

Evaluates viewpoint generalization across a semicircular arc on the sphere centered at the robot base.
Angles: 0 deg (side view, +Y) -> 90 deg (front view, +X) -> 180 deg (opposite side view, -Y) in 15-degree steps.
Sphere Radius: R = 1.0942 m centered at robot base [-0.615, 0.0, 0.0].
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
from ppo_rgb import Agent

BASE_POS = np.array([-0.615, 0.0, 0.0])
TARGET_POS = np.array([-0.1, 0.0, 0.1])
R_HORIZ = 0.915
Z_CAM = 0.600
SPHERE_RADIUS = np.sqrt(R_HORIZ**2 + Z_CAM**2)  # 1.0942 m


def get_camera_eye(deg: float) -> list:
    """
    Computes camera eye coordinates for a given angle in degrees along the semicircular arc.
    0 deg:   Side view (+Y) -> [-0.615, +0.915, 0.600]
    90 deg:  Front view (+X) -> [ 0.300,  0.000, 0.600] (Training view)
    180 deg: Side view (-Y) -> [-0.615, -0.915, 0.600]
    """
    rad = np.deg2rad(deg)
    x_rel = R_HORIZ * np.sin(rad)
    y_rel = R_HORIZ * np.cos(rad)
    eye = [float(BASE_POS[0] + x_rel), float(BASE_POS[1] + y_rel), float(Z_CAM)]
    return eye


def evaluate_single_angle(
    agent: Agent,
    angle_deg: float,
    episodes: int = 1000,
    num_envs: int = 48,
    device: str = "cuda",
) -> dict:
    eye = get_camera_eye(angle_deg)
    target = TARGET_POS.tolist()

    print(f"\n{'='*70}")
    print(f" EVALUATING ANGLE: {angle_deg:5.1f} deg | eye: [{eye[0]:6.3f}, {eye[1]:6.3f}, {eye[2]:6.3f}]")
    print(f" Target Runs: {episodes:,} | Parallel Instances: {num_envs}")
    print(f"{'='*70}")

    env = gym.make(
        "PickCube-v1",
        num_envs=num_envs,
        obs_mode="rgb",
        control_mode="pd_ee_delta_pos",
        render_mode=None,
        sim_backend="physx_cuda",
        sensor_cam_eye_pos=eye,
        sensor_cam_target_pos=target,
    )
    env = FlattenRGBDObservationWrapper(env, rgb=True, depth=False, state=True)
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    env = ManiSkillVectorEnv(env, num_envs, ignore_terminations=False, record_metrics=True)

    obs, _ = env.reset(seed=int(angle_deg * 1000 + 42))

    episodes_completed = 0
    success_count = 0
    returns = []
    episode_lens = []

    start_time = time.time()

    with torch.no_grad():
        while episodes_completed < episodes:
            action = agent.get_action(obs, deterministic=True)
            obs, rew, term, trunc, infos = env.step(action)

            if "final_info" in infos:
                mask = infos["_final_info"]
                for i in range(len(mask)):
                    if mask[i]:
                        ep_info = infos["final_info"]["episode"]
                        succ_key = (
                            "success_once"
                            if "success_once" in ep_info
                            else ("success" if "success" in ep_info else "success_at_end")
                        )
                        succ = bool(ep_info[succ_key][i].item() if torch.is_tensor(ep_info[succ_key]) else ep_info[succ_key])
                        ret = float(ep_info["return"][i].item() if torch.is_tensor(ep_info["return"]) else ep_info["return"])
                        ep_len = float(ep_info["episode_len"][i].item() if torch.is_tensor(ep_info["episode_len"]) else ep_info["episode_len"])

                        if succ:
                            success_count += 1
                        returns.append(ret)
                        episode_lens.append(ep_len)
                        episodes_completed += 1

                        if episodes_completed >= episodes:
                            break

    elapsed = time.time() - start_time
    env.close()

    success_rate = (success_count / episodes) * 100
    mean_return = float(np.mean(returns)) if returns else 0.0
    mean_len = float(np.mean(episode_lens)) if episode_lens else 0.0

    print(f" -> Completed: {success_count}/{episodes} ({success_rate:5.2f}%) in {elapsed:.1f}s | Mean Return: {mean_return:.2f}")

    return {
        "angle_deg": float(angle_deg),
        "eye_position": eye,
        "episodes": episodes,
        "success_count": success_count,
        "success_rate": round(success_rate, 2),
        "mean_return": round(mean_return, 2),
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
        obs_mode="rgb",
        control_mode="pd_ee_delta_pos",
        render_mode="rgb_array",
        sim_backend="physx_cuda",
        sensor_cam_eye_pos=eye,
        sensor_cam_target_pos=target,
    )
    env = FlattenRGBDObservationWrapper(env, rgb=True, depth=False, state=True)
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
                f"Workspace View (Third-Person)",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                sensor_large,
                f"Sensor Cam: {angle_deg:.0f} deg (Policy Input)",
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


def run_sweep(
    checkpoint_path: str = "/home/maniskill/phase_1_to_3/visual_manipulation/best_sim2real_policy.pt",
    start_deg: float = 0.0,
    end_deg: float = 180.0,
    step_deg: float = 15.0,
    episodes_per_angle: int = 1000,
    num_envs: int = 48,
    output_dir: str = "/home/maniskill/phase_1_to_3/visual_manipulation/camera_sweep_results",
    device: str = "cuda",
):
    os.makedirs(output_dir, exist_ok=True)
    angles = np.arange(start_deg, end_deg + 1e-5, step_deg).tolist()

    print("\n" + "=" * 75)
    print(" CAMERA VIEWPOINT SWEEP EVALUATION")
    print(f" Checkpoint:         {checkpoint_path}")
    print(f" Angles:             {angles} (Total: {len(angles)} angles)")
    print(f" Episodes per Angle: {episodes_per_angle:,}")
    print(f" Parallel Instances: {num_envs}")
    print(f" Output Directory:   {output_dir}")
    print("=" * 75)

    # Initialize a dummy env to instantiate agent
    dummy_env = gym.make(
        "PickCube-v1",
        num_envs=1,
        obs_mode="rgb",
        control_mode="pd_ee_delta_pos",
        sim_backend="physx_cuda",
    )
    dummy_env = FlattenRGBDObservationWrapper(dummy_env, rgb=True, depth=False, state=True)
    if isinstance(dummy_env.action_space, gym.spaces.Dict):
        dummy_env = FlattenActionSpaceWrapper(dummy_env)
    dummy_env = ManiSkillVectorEnv(dummy_env, 1, ignore_terminations=False, record_metrics=True)
    sample_obs, _ = dummy_env.reset()

    agent = Agent(dummy_env, sample_obs=sample_obs).to(device)
    agent.load_state_dict(torch.load(checkpoint_path, map_location=device))
    agent.eval()
    dummy_env.close()

    results = []
    total_start = time.time()

    for idx, angle in enumerate(angles):
        res = evaluate_single_angle(
            agent=agent,
            angle_deg=angle,
            episodes=episodes_per_angle,
            num_envs=num_envs,
            device=device,
        )
        results.append(res)

        # Record demo video at key angles (start 0, quarter 45, center 90, 135, end 180)
        if int(angle) in [0, 45, 90, 135, 180]:
            video_file = os.path.join(output_dir, f"angle_{int(angle):03d}deg.mp4")
            record_angle_video(agent=agent, angle_deg=angle, output_path=video_file)

    total_time = time.time() - total_start

    # Save results json
    json_path = os.path.join(output_dir, "sweep_results.json")
    with open(json_path, "w") as f:
        json.dump(
            {
                "checkpoint": checkpoint_path,
                "sphere_radius": float(SPHERE_RADIUS),
                "episodes_per_angle": episodes_per_angle,
                "total_walltime_sec": round(total_time, 2),
                "results": results,
            },
            f,
            indent=2,
        )

    # Print summary table
    print("\n" + "=" * 75)
    print(" CAMERA VIEWPOINT SWEEP RESULTS SUMMARY")
    print("=" * 75)
    print(f"{'Angle':>7} | {'Eye X':>7} {'Eye Y':>7} {'Eye Z':>7} | {'Success Rate':>12} | {'Mean Ret':>8} | {'Mean Len':>8}")
    print("-" * 75)
    for r in results:
        eye = r["eye_position"]
        print(
            f"{r['angle_deg']:5.0f}°  | {eye[0]:7.3f} {eye[1]:7.3f} {eye[2]:7.3f} | {r['success_rate']:11.2f}% | {r['mean_return']:8.2f} | {r['mean_length']:8.1f}"
        )
    print("=" * 75)
    print(f"Total Sweep Time: {total_time/60:.1f} minutes")
    print(f"Results saved to: {json_path}\n")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate visual policy across camera angles")
    parser.add_argument("--checkpoint", type=str, default="/home/maniskill/phase_1_to_3/visual_manipulation/best_sim2real_policy.pt")
    parser.add_argument("--start_deg", type=float, default=0.0)
    parser.add_argument("--end_deg", type=float, default=180.0)
    parser.add_argument("--step_deg", type=float, default=15.0)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--num_envs", type=int, default=48)
    parser.add_argument("--output_dir", type=str, default="/home/maniskill/phase_1_to_3/visual_manipulation/camera_sweep_results")
    args = parser.parse_args()

    run_sweep(
        checkpoint_path=args.checkpoint,
        start_deg=args.start_deg,
        end_deg=args.end_deg,
        step_deg=args.step_deg,
        episodes_per_angle=args.episodes,
        num_envs=args.num_envs,
        output_dir=args.output_dir,
    )
