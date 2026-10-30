"""
50 Million Step Visual Reinforcement Learning for Sim-to-Real Object Manipulation (Option A: 4.5cm Block).
Pure Vision-to-Action PPO: Zero object labels, dynamic robot DoF encoding, GPU parallelized.
"""

import argparse
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, "/home/maniskill")

import mani_skill.envs
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from model import PureVisuomotorAgent


def evaluate_policy(agent, env_id="PickCube-v1", episodes=5, max_steps=50, video_dir=None, device="cuda"):
    agent.eval()
    render_mode = "rgb_array" if video_dir else None
    env = gym.make(
        env_id,
        obs_mode="rgb",
        control_mode="pd_ee_delta_pos",
        num_envs=1,
        render_mode=render_mode,
    )
    env = FlattenRGBDObservationWrapper(env, rgb=True, depth=False, state=True)

    if video_dir:
        os.makedirs(video_dir, exist_ok=True)
        env = RecordEpisode(env, output_dir=video_dir, max_steps_per_video=max_steps, save_trajectory=False)

    successes = []
    returns = []

    for ep in range(episodes):
        obs, _ = env.reset(seed=ep + 7000)
        ep_return = 0.0
        success = False

        for step in range(max_steps):
            obs_dict = {
                "rgb": obs["rgb"].to(device),
                "tcp_state": obs["state"][:, 18:29].to(device),
                "qpos": obs["state"][:, :9].to(device),
                "qvel": obs["state"][:, 9:18].to(device),
            }

            with torch.no_grad():
                action = agent.get_action(obs_dict, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += reward.item() if torch.is_tensor(reward) else reward

            if "success" in info:
                s = info["success"].any().item() if torch.is_tensor(info["success"]) else info["success"]
                success = success or bool(s)

            if terminated or truncated:
                break

        successes.append(float(success))
        returns.append(ep_return)

    env.close()
    return float(np.mean(successes) * 100), float(np.mean(returns))


def main():
    parser = argparse.ArgumentParser(description="50M Step Visual RL for Sim-to-Real Object Manipulation")
    parser.add_argument("--num_envs", type=int, default=512, help="Number of parallel GPU environments (max for 8GB VRAM)")
    parser.add_argument("--total_timesteps", type=int, default=50_000_000, help="Total environment steps")
    parser.add_argument("--save_dir", type=str, default="/home/maniskill/visual_manipulation/checkpoints/sim2real_block", help="Checkpoint directory")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--eval_freq", type=int, default=25, help="Iteration frequency for evaluations")
    parser.add_argument("--save_freq", type=int, default=50, help="Iteration frequency for saving checkpoints")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    video_dir = os.path.join(args.save_dir, "eval_videos")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*75}")
    print(f" 50M-STEP VISUAL RL: OPTION A (4.5cm MATTE BLOCK)")
    print(f" Hardware: {torch.cuda.get_device_name(0)}")
    print(f" Parallel Instances: {args.num_envs} GPU simulation environments")
    print(f" Target Timesteps:   {args.total_timesteps:,} steps")
    print(f" Checkpoint Output:  {args.save_dir}")
    print(f"{'='*75}\n")

    # Environment setup
    env = gym.make(
        "PickCube-v1",
        obs_mode="rgb",
        control_mode="pd_ee_delta_pos",
        num_envs=args.num_envs,
    )
    env = FlattenRGBDObservationWrapper(env, rgb=True, depth=False, state=True)

    # Agent setup
    agent = PureVisuomotorAgent(action_dim=4, in_channels=3, tcp_dim=11, max_dof=16).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.lr, eps=1e-5)

    num_steps = 50
    batch_size = args.num_envs * num_steps
    num_iterations = args.total_timesteps // batch_size

    print(f"[PPO Config] Batch Size: {batch_size} | Iterations: {num_iterations:,} | Rollout Steps: {num_steps}")

    obs, _ = env.reset(seed=42)
    start_time = time.time()
    best_success = -1.0

    for iteration in range(1, num_iterations + 1):
        obs_buf_rgb = torch.zeros((num_steps, args.num_envs, 128, 128, 3), dtype=torch.uint8, device=device)
        obs_buf_tcp = torch.zeros((num_steps, args.num_envs, 11), dtype=torch.float32, device=device)
        obs_buf_qpos = torch.zeros((num_steps, args.num_envs, 9), dtype=torch.float32, device=device)
        obs_buf_qvel = torch.zeros((num_steps, args.num_envs, 9), dtype=torch.float32, device=device)
        actions_buf = torch.zeros((num_steps, args.num_envs, 4), device=device)
        logprobs_buf = torch.zeros((num_steps, args.num_envs), device=device)
        rewards_buf = torch.zeros((num_steps, args.num_envs), device=device)
        dones_buf = torch.zeros((num_steps, args.num_envs), device=device)
        values_buf = torch.zeros((num_steps, args.num_envs), device=device)

        agent.eval()
        for step in range(num_steps):
            rgb = obs["rgb"]
            tcp_state = obs["state"][:, 18:29]
            qpos = obs["state"][:, :9]
            qvel = obs["state"][:, 9:18]

            obs_buf_rgb[step] = rgb
            obs_buf_tcp[step] = tcp_state
            obs_buf_qpos[step] = qpos
            obs_buf_qvel[step] = qvel

            obs_dict = {
                "rgb": rgb,
                "tcp_state": tcp_state,
                "qpos": qpos,
                "qvel": qvel,
            }

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(obs_dict)

            actions_buf[step] = action
            logprobs_buf[step] = logprob
            values_buf[step] = value.flatten()

            obs, reward, terminated, truncated, info = env.step(action)
            rewards_buf[step] = reward.view(-1)
            dones_buf[step] = (terminated | truncated).view(-1)

        # GAE Advantage Estimation
        with torch.no_grad():
            last_obs_dict = {
                "rgb": obs["rgb"],
                "tcp_state": obs["state"][:, 18:29],
                "qpos": obs["state"][:, :9],
                "qvel": obs["state"][:, 9:18],
            }
            next_value = agent.critic(agent.extract_features(last_obs_dict)).flatten()
            advantages = torch.zeros_like(rewards_buf)
            lastgaelam = 0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - dones_buf[t].float()
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones_buf[t + 1].float()
                    nextvalues = values_buf[t + 1]
                delta = rewards_buf[t] + 0.95 * nextvalues * nextnonterminal - values_buf[t]
                advantages[t] = lastgaelam = delta + 0.95 * 0.95 * nextnonterminal * lastgaelam
            returns = advantages + values_buf

        # Flatten buffers
        b_rgb = obs_buf_rgb.view(-1, 128, 128, 3)
        b_tcp = obs_buf_tcp.view(-1, 11)
        b_qpos = obs_buf_qpos.view(-1, 9)
        b_qvel = obs_buf_qvel.view(-1, 9)
        b_actions = actions_buf.view(-1, 4)
        b_logprobs = logprobs_buf.view(-1)
        b_advantages = advantages.view(-1)
        b_returns = returns.view(-1)

        # PPO Update
        agent.train()
        minibatch_size = batch_size // 4
        indices = np.arange(batch_size)

        for epoch in range(4):
            np.random.shuffle(indices)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_idx = indices[start:end]

                mb_obs = {
                    "rgb": b_rgb[mb_idx],
                    "tcp_state": b_tcp[mb_idx],
                    "qpos": b_qpos[mb_idx],
                    "qvel": b_qvel[mb_idx],
                }

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(mb_obs, b_actions[mb_idx])
                ratio = (newlogprob - b_logprobs[mb_idx]).exp()

                mb_adv = b_advantages[mb_idx]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                pg_loss = torch.max(-mb_adv * ratio, -mb_adv * torch.clamp(ratio, 0.8, 1.2)).mean()
                v_loss = 0.5 * ((newvalue.flatten() - b_returns[mb_idx]) ** 2).mean()

                loss = pg_loss - 0.01 * entropy.mean() + 0.5 * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                optimizer.step()

        # Logging & Progress
        elapsed = time.time() - start_time
        global_steps = iteration * batch_size
        sps = int(global_steps / elapsed)
        mean_rew = rewards_buf.mean().item()

        if iteration % 5 == 0 or iteration == 1:
            print(f"Iter {iteration:04d}/{num_iterations:04d} | Steps: {global_steps:9,d}/{args.total_timesteps:,} | Mean Reward: {mean_rew:.3f} | SPS: {sps:,} | Elapsed: {elapsed/60:.1f}m")

        # Periodic Evaluation
        if iteration % args.eval_freq == 0:
            print(f"\n--> Running Evaluation (Iteration {iteration})...")
            succ, ret = evaluate_policy(agent, env_id="PickCube-v1", episodes=5, video_dir=video_dir, device=device)
            print(f"--> Eval Success Rate: {succ:.1f}% | Eval Mean Return: {ret:.2f}\n")

            if succ >= best_success:
                best_success = succ
                torch.save(agent.state_dict(), os.path.join(args.save_dir, "best_model.pt"))
                print(f"--> [New Best Model Saved] Success: {succ:.1f}%\n")

        # Periodic Checkpoint
        if iteration % args.save_freq == 0:
            ckpt_path = os.path.join(args.save_dir, f"ckpt_{iteration}.pt")
            torch.save(agent.state_dict(), ckpt_path)
            print(f"[Checkpoint] Saved to {ckpt_path}\n")

    env.close()

    # Final Model Save
    final_path = os.path.join(args.save_dir, "final_model.pt")
    torch.save(agent.state_dict(), final_path)
    print(f"\n{'='*75}")
    print(f" TRAINING COMPLETE! 50 Million Steps finished in {elapsed/3600:.2f} hours.")
    print(f" Final Model Saved to: {final_path}")
    print(f" Best Model Saved to:  {os.path.join(args.save_dir, 'best_model.pt')}")
    print(f"{'='*75}\n")


if __name__ == "__main__":
    main()
