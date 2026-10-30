"""
Federated Reinforcement Learning (FedRL) Client Training Runner.
Trains 5 independent local policies across 5 tasks (Apple, Pear, Mustard Bottle, Soup Can, Sphere).
Each client runs with 150 parallel simulation instances and saves isolated weights for downstream FedAvg aggregation.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, "/home/maniskill")

import fragile_pick_env
from fragile_pick_env import OBJECT_NAMES, OBJECT_PROFILES
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from model import EmbodimentAgnosticVisualAgent


def evaluate_client(agent, object_name: str, episodes: int = 5, max_steps: int = 50, video_dir: str = None, device: str = "cuda"):
    agent.eval()
    render_mode = "rgb_array" if video_dir else None
    env = gym.make(
        "FragilePick-v1",
        obs_mode="rgb",
        control_mode="pd_ee_delta_pos",
        object_name=object_name,
        num_envs=1,
        render_mode=render_mode,
    )
    env = FlattenRGBDObservationWrapper(env, rgb=True, depth=False, state=True)

    if video_dir:
        os.makedirs(video_dir, exist_ok=True)
        env = RecordEpisode(env, output_dir=video_dir, max_steps_per_video=max_steps, save_trajectory=False)

    successes = []
    pristine_successes = []
    peak_forces = []
    damage_pcts = []
    returns = []

    profile = OBJECT_PROFILES[object_name]
    obj_id = profile["id"]

    for ep in range(episodes):
        obs, _ = env.reset(seed=ep + 5000)
        ep_return = 0.0
        success = False
        ep_peak_force = 0.0
        ep_damage = 0.0

        for step in range(max_steps):
            obs_dict = {
                "rgb": obs["rgb"].to(device),
                "tcp_state": obs["state"][:, 18:29].to(device),  # 11-dim universal TCP
                "qpos": obs["state"][:, :9].to(device),          # 9-dim joint pos
                "qvel": obs["state"][:, 9:18].to(device),        # 9-dim joint vel
                "object_id": torch.full((1, 1), obj_id, dtype=torch.long, device=device),
            }

            with torch.no_grad():
                action = agent.get_action(obs_dict, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += reward.item() if torch.is_tensor(reward) else reward

            force = info["peak_force"].item() if torch.is_tensor(info["peak_force"]) else info["peak_force"]
            dmg = info["damage_pct"].item() if torch.is_tensor(info["damage_pct"]) else info["damage_pct"]
            ep_peak_force = max(ep_peak_force, force)
            ep_damage = max(ep_damage, dmg)

            if "success" in info:
                s = info["success"].any().item() if torch.is_tensor(info["success"]) else info["success"]
                success = success or bool(s)

            if terminated or truncated:
                break

        successes.append(float(success))
        pristine_successes.append(float(success and (ep_damage == 0.0)))
        peak_forces.append(ep_peak_force)
        damage_pcts.append(ep_damage)
        returns.append(ep_return)

    env.close()

    return {
        "success_rate": float(np.mean(successes) * 100),
        "pristine_success_rate": float(np.mean(pristine_successes) * 100),
        "mean_peak_force": float(np.mean(peak_forces)),
        "mean_damage_pct": float(np.mean(damage_pcts)),
        "mean_return": float(np.mean(returns)),
        "f_safe": profile["f_safe"],
        "f_max": profile["f_max"],
    }


def train_federated_client(
    object_name: str,
    base_save_dir: str,
    num_envs: int = 150,
    total_timesteps: int = 150_000,
    device: str = "cuda",
):
    profile = OBJECT_PROFILES[object_name]
    obj_id = profile["id"]
    client_dir = os.path.join(base_save_dir, object_name)
    os.makedirs(client_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f" [FedRL Client] Training: {object_name.upper()} (Client ID: {obj_id})")
    print(f" Instances: {num_envs} parallel simulation envs")
    print(f" Force Bounds: F_safe = {profile['f_safe']} N | F_max = {profile['f_max']} N")
    print(f" Target Directory: {client_dir}")
    print(f"{'='*70}\n")

    # Initialize environment
    env = gym.make(
        "FragilePick-v1",
        obs_mode="rgb",
        control_mode="pd_ee_delta_pos",
        object_name=object_name,
        num_envs=num_envs,
    )
    env = FlattenRGBDObservationWrapper(env, rgb=True, depth=False, state=True)

    # Initialize Embodiment-Agnostic Agent
    agent = EmbodimentAgnosticVisualAgent(action_dim=4, in_channels=3, tcp_dim=11, max_dof=16, num_known_objects=10).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=3e-4, eps=1e-5)

    num_steps = 50
    batch_size = num_envs * num_steps
    num_iterations = max(1, total_timesteps // batch_size)

    obs, _ = env.reset(seed=100 + obj_id)
    start_time = time.time()

    for iteration in range(1, num_iterations + 1):
        obs_buf_rgb = torch.zeros((num_steps, num_envs, 128, 128, 3), dtype=torch.uint8, device=device)
        obs_buf_tcp = torch.zeros((num_steps, num_envs, 11), dtype=torch.float32, device=device)
        obs_buf_qpos = torch.zeros((num_steps, num_envs, 9), dtype=torch.float32, device=device)
        obs_buf_qvel = torch.zeros((num_steps, num_envs, 9), dtype=torch.float32, device=device)
        actions_buf = torch.zeros((num_steps, num_envs, 4), device=device)
        logprobs_buf = torch.zeros((num_steps, num_envs), device=device)
        rewards_buf = torch.zeros((num_steps, num_envs), device=device)
        dones_buf = torch.zeros((num_steps, num_envs), device=device)
        values_buf = torch.zeros((num_steps, num_envs), device=device)

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
                "object_id": torch.full((num_envs, 1), obj_id, dtype=torch.long, device=device),
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
                "object_id": torch.full((num_envs, 1), obj_id, dtype=torch.long, device=device),
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

        # Flatten buffers for PPO updates
        b_rgb = obs_buf_rgb.view(-1, 128, 128, 3)
        b_tcp = obs_buf_tcp.view(-1, 11)
        b_qpos = obs_buf_qpos.view(-1, 9)
        b_qvel = obs_buf_qvel.view(-1, 9)
        b_actions = actions_buf.view(-1, 4)
        b_logprobs = logprobs_buf.view(-1)
        b_advantages = advantages.view(-1)
        b_returns = returns.view(-1)

        agent.train()
        minibatch_size = batch_size // 5
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
                    "object_id": torch.full((len(mb_idx), 1), obj_id, dtype=torch.long, device=device),
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

        if iteration % 5 == 0 or iteration == num_iterations:
            elapsed = time.time() - start_time
            sps = int((iteration * batch_size) / elapsed)
            mean_rew = rewards_buf.mean().item()
            dmg = info["damage_pct"].float().mean().item()
            force = info["peak_force"].float().mean().item()
            print(f"[{object_name:14s}] Iter {iteration:03d}/{num_iterations:03d} | Mean Reward: {mean_rew:.3f} | Force: {force:.1f}N | Damage: {dmg:.1f}% | SPS: {sps}")

    env.close()

    # Save local client model
    client_ckpt = os.path.join(client_dir, "policy.pt")
    torch.save(agent.state_dict(), client_ckpt)

    # Evaluate client and save evaluation video
    video_dir = os.path.join(client_dir, "eval_video")
    metrics = evaluate_client(agent, object_name=object_name, episodes=3, video_dir=video_dir, device=device)

    # Save metrics JSON
    with open(os.path.join(client_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n[Completed] Client {object_name} saved to: {client_ckpt}")
    print(f" -> Success: {metrics['success_rate']:.1f}% | Pristine (0% Damage): {metrics['pristine_success_rate']:.1f}% | Peak Force: {metrics['mean_peak_force']:.1f}N")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Federated RL Multi-Client Training")
    parser.add_argument("--num_envs", type=int, default=150, help="Parallel simulation instances per client")
    parser.add_argument("--timesteps", type=int, default=150_000, help="Timesteps per client round")
    parser.add_argument("--save_dir", type=str, default="/home/maniskill/visual_manipulation/checkpoints/federated_runs", help="Base directory for client models")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Federated Training on {device} with {args.num_envs} instances per client...")

    summary = {}
    for obj in OBJECT_NAMES:
        metrics = train_federated_client(
            object_name=obj,
            base_save_dir=args.save_dir,
            num_envs=args.num_envs,
            total_timesteps=args.timesteps,
            device=device,
        )
        summary[obj] = metrics

    # Save aggregated overview
    summary_path = os.path.join(args.save_dir, "federated_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print(" FEDERATED CLIENT TRAINING ROUND COMPLETED")
    print("=" * 80)
    print(f"{'Client / Object':<18} | {'Safe F':<8} | {'Peak F':<8} | {'Damage %':<10} | {'Pristine %':<12} | {'Success %':<10}")
    print("-" * 80)
    for obj, m in summary.items():
        print(f"{obj:<18} | {m['f_safe']:<6.1f}N | {m['mean_peak_force']:<6.1f}N | {m['mean_damage_pct']:<8.1f}% | {m['pristine_success_rate']:<10.1f}% | {m['success_rate']:<8.1f}%")
    print("=" * 80)
    print(f"\nAll models saved in separate directories under: {args.save_dir}\n")


if __name__ == "__main__":
    main()
