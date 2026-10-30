"""
Evaluation and Video Recording tool for Visual Manipulation Policies.
"""

import argparse
import os
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, "/home/maniskill")

import mani_skill.envs
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from model import VisualAgent


def evaluate_policy(
    agent,
    env_id: str = "PickCube-v1",
    control_mode: str = "pd_ee_delta_pos",
    num_episodes: int = 10,
    max_steps: int = 100,
    video_dir: str = None,
    device: str = "cuda",
):
    agent.eval()
    render_mode = "rgb_array" if video_dir else None
    env = gym.make(
        env_id,
        obs_mode="rgb",
        control_mode=control_mode,
        num_envs=1,
        render_mode=render_mode,
    )
    env = FlattenRGBDObservationWrapper(env, rgb=True, depth=False, state=True)

    if video_dir:
        os.makedirs(video_dir, exist_ok=True)
        env = RecordEpisode(env, output_dir=video_dir, max_steps_per_video=max_steps, save_trajectory=False)

    successes = []
    returns = []

    for ep in range(num_episodes):
        obs, _ = env.reset(seed=ep + 2000)
        ep_return = 0.0
        success = False

        for step in range(max_steps):
            obs_gpu = {k: v.to(device) for k, v in obs.items()}
            with torch.no_grad():
                action = agent.get_action(obs_gpu, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += reward.item() if torch.is_tensor(reward) else reward

            if "success" in info:
                success = success or bool(
                    info["success"].any().item() if torch.is_tensor(info["success"]) else info["success"]
                )
            if terminated or truncated:
                break

        successes.append(float(success))
        returns.append(ep_return)

    env.close()
    mean_success = np.mean(successes)
    mean_return = np.mean(returns)
    return mean_success, mean_return


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained visual manipulation policy")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to policy checkpoint (.pt)")
    parser.add_argument("--env_id", type=str, default="PickCube-v1")
    parser.add_argument("--control_mode", type=str, default="pd_ee_delta_pos")
    parser.add_argument("--episodes", type=int, default=5, help="Number of evaluation episodes")
    parser.add_argument("--max_steps", type=int, default=100, help="Max steps per episode")
    parser.add_argument("--video_dir", type=str, default="eval_videos", help="Directory to save MP4 videos")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using {device}")
    print(f"[Loading] Checkpoint: {args.checkpoint}")

    agent = VisualAgent(action_dim=4, state_dim=29, in_channels=3).to(device)
    agent.load_state_dict(torch.load(args.checkpoint, map_location=device))

    print(f"\n=== Evaluating on {args.env_id} ({args.episodes} episodes) ===")
    succ, ret = evaluate_policy(
        agent,
        env_id=args.env_id,
        control_mode=args.control_mode,
        num_episodes=args.episodes,
        max_steps=args.max_steps,
        video_dir=args.video_dir,
        device=device,
    )

    print("\n---------------- Results ----------------")
    print(f"Success Rate: {succ * 100:.1f}%")
    print(f"Mean Return:  {ret:.2f}")
    if args.video_dir:
        print(f"Videos saved to: {Path(args.video_dir).resolve()}")
    print("----------------------------------------\n")


if __name__ == "__main__":
    main()
