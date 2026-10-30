"""
Training and Evaluation runner for Force-Sensitive & Object-Conditioned Manipulation (FragilePick-v1).
Supports all 5 objects: Apple, Pear, Mustard Bottle, Soup Can, Sphere.
Tracks contact forces, % damage, and rewards pristine zero-damage manipulation.
"""

import argparse
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
from model import VisualAgent


def evaluate_fragile_policy(
    agent: VisualAgent,
    object_name: str,
    episodes: int = 5,
    max_steps: int = 50,
    video_dir: str = None,
    device: str = "cuda",
):
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

    for ep in range(episodes):
        obs, _ = env.reset(seed=ep + 3000)
        ep_return = 0.0
        success = False
        is_pristine = True
        ep_peak_force = 0.0
        ep_damage = 0.0

        for step in range(max_steps):
            obs_gpu = {k: v.to(device) for k, v in obs.items()}
            with torch.no_grad():
                action = agent.get_action(obs_gpu, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += reward.item() if torch.is_tensor(reward) else reward

            force = info["peak_force"].item() if torch.is_tensor(info["peak_force"]) else info["peak_force"]
            dmg = info["damage_pct"].item() if torch.is_tensor(info["damage_pct"]) else info["damage_pct"]
            ep_peak_force = max(ep_peak_force, force)
            ep_damage = max(ep_damage, dmg)

            if "success" in info:
                s = info["success"].any().item() if torch.is_tensor(info["success"]) else info["success"]
                success = success or bool(s)
            if "is_pristine" in info:
                p = info["is_pristine"].all().item() if torch.is_tensor(info["is_pristine"]) else info["is_pristine"]
                is_pristine = is_pristine and bool(p)

            if terminated or truncated:
                break

        successes.append(float(success))
        pristine_successes.append(float(success and (ep_damage == 0.0)))
        peak_forces.append(ep_peak_force)
        damage_pcts.append(ep_damage)
        returns.append(ep_return)

    env.close()

    metrics = {
        "success_rate": np.mean(successes) * 100,
        "pristine_success_rate": np.mean(pristine_successes) * 100,
        "mean_peak_force": np.mean(peak_forces),
        "mean_damage_pct": np.mean(damage_pcts),
        "mean_return": np.mean(returns),
        "f_safe": profile["f_safe"],
        "f_max": profile["f_max"],
    }
    return metrics


def train_object_session(
    object_name: str,
    total_timesteps: int = 150_000,
    num_envs: int = 64,
    save_dir: str = "checkpoints/fragile_pick",
    device: str = "cuda",
):
    print(f"\n=======================================================")
    print(f" Starting Training Session: {object_name.upper()}")
    print(f" Force Specs: F_safe = {OBJECT_PROFILES[object_name]['f_safe']} N | F_max = {OBJECT_PROFILES[object_name]['f_max']} N")
    print(f" Profile: {OBJECT_PROFILES[object_name]['desc']}")
    print(f"=======================================================\n")

    os.makedirs(save_dir, exist_ok=True)
    out_ckpt = os.path.join(save_dir, f"{object_name}_policy.pt")

    # Create parallel training environments on GPU
    env = gym.make(
        "FragilePick-v1",
        obs_mode="rgb",
        control_mode="pd_ee_delta_pos",
        object_name=object_name,
        num_envs=num_envs,
    )
    env = FlattenRGBDObservationWrapper(env, rgb=True, depth=False, state=True)

    # Agent setup
    agent = VisualAgent(action_dim=4, state_dim=35, in_channels=3).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=3e-4, eps=1e-5)

    # Load pre-trained vision weights if available
    pretrained_bc = Path(__file__).resolve().parent / "checkpoints" / "bc_pretrained.pt"
    if pretrained_bc.exists():
        try:
            sd = torch.load(pretrained_bc, map_location=device)
            # Load CNN feature weights
            cnn_sd = {k: v for k, v in sd.items() if "feature_net.extractors.rgb" in k}
            agent.load_state_dict(cnn_sd, strict=False)
            print(f"[Init] Loaded pre-trained vision weights from {pretrained_bc.name}")
        except Exception as e:
            print(f"[Init] Vision weight warm-start skipped: {e}")

    # PPO Rollout Storage
    num_steps = 50
    batch_size = num_envs * num_steps
    num_iterations = max(1, total_timesteps // batch_size)

    obs, _ = env.reset(seed=42)
    start_time = time.time()

    for iteration in range(1, num_iterations + 1):
        obs_buf_rgb = torch.zeros((num_steps, num_envs, 128, 128, 3), dtype=torch.uint8, device=device)
        obs_buf_state = torch.zeros((num_steps, num_envs, 35), dtype=torch.float32, device=device)
        actions_buf = torch.zeros((num_steps, num_envs, 4), device=device)
        logprobs_buf = torch.zeros((num_steps, num_envs), device=device)
        rewards_buf = torch.zeros((num_steps, num_envs), device=device)
        dones_buf = torch.zeros((num_steps, num_envs), device=device)
        values_buf = torch.zeros((num_steps, num_envs), device=device)

        agent.eval()
        for step in range(num_steps):
            obs_buf_rgb[step] = obs["rgb"]
            obs_buf_state[step] = obs["state"]

            obs_gpu = {"rgb": obs["rgb"], "state": obs["state"]}
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(obs_gpu)

            actions_buf[step] = action
            logprobs_buf[step] = logprob
            values_buf[step] = value.flatten()

            obs, reward, terminated, truncated, info = env.step(action)
            rewards_buf[step] = reward.view(-1)
            dones_buf[step] = (terminated | truncated).view(-1)

        # Compute Generalized Advantage Estimation (GAE)
        with torch.no_grad():
            next_value = agent.get_value({"rgb": obs["rgb"], "state": obs["state"]}).flatten()
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

        # Flatten rollout buffers
        b_rgb = obs_buf_rgb.view(-1, 128, 128, 3)
        b_state = obs_buf_state.view(-1, 35)
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

                mb_obs = {"rgb": b_rgb[mb_idx], "state": b_state[mb_idx]}
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(mb_obs, b_actions[mb_idx])
                logratio = newlogprob - b_logprobs[mb_idx]
                ratio = logratio.exp()

                mb_adv = b_advantages[mb_idx]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - 0.2, 1 + 0.2)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                v_loss = 0.5 * ((newvalue.flatten() - b_returns[mb_idx]) ** 2).mean()
                entropy_loss = entropy.mean()

                loss = pg_loss - 0.01 * entropy_loss + 0.5 * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                optimizer.step()

        if iteration % 10 == 0 or iteration == num_iterations:
            elapsed = time.time() - start_time
            sps = int((iteration * batch_size) / elapsed)
            mean_rew = rewards_buf.mean().item()
            dmg = info["damage_pct"].float().mean().item()
            force = info["peak_force"].float().mean().item()
            print(f"[{object_name:14s}] Iter {iteration:03d}/{num_iterations:03d} | Mean Reward: {mean_rew:.3f} | Peak Force: {force:.1f}N | Damage: {dmg:.1f}% | SPS: {sps}")

    env.close()

    # Save checkpoint
    torch.save(agent.state_dict(), out_ckpt)
    print(f"\n[Saved] Checkpoint for {object_name} saved to: {out_ckpt}")

    # Run quick evaluation & video recording
    eval_video_dir = os.path.join(save_dir, "videos", object_name)
    eval_metrics = evaluate_fragile_policy(agent, object_name=object_name, episodes=3, video_dir=eval_video_dir, device=device)
    print(f"\n--- {object_name.upper()} Evaluation Results ---")
    print(f"Success Rate:          {eval_metrics['success_rate']:.1f}%")
    print(f"Pristine (0% Damage):  {eval_metrics['pristine_success_rate']:.1f}%")
    print(f"Mean Peak Force:       {eval_metrics['mean_peak_force']:.2f} N (Safe limit: {eval_metrics['f_safe']} N)")
    print(f"Mean Damage:           {eval_metrics['mean_damage_pct']:.1f}%")
    print(f"Mean Episode Return:   {eval_metrics['mean_return']:.2f}")
    print(f"Evaluation Video:      {eval_video_dir}/0.mp4\n")

    return eval_metrics


def main():
    parser = argparse.ArgumentParser(description="Train or Evaluate Force-Sensitive Policies")
    parser.add_argument("--object", type=str, default="apple", choices=OBJECT_NAMES + ["all"], help="Target object to train")
    parser.add_argument("--timesteps", type=int, default=100_000, help="Timesteps per object")
    parser.add_argument("--num_envs", type=int, default=64, help="Parallel environments on GPU")
    parser.add_argument("--save_dir", type=str, default="checkpoints/fragile_pick", help="Directory for checkpoints and videos")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    targets = OBJECT_NAMES if args.object == "all" else [args.object]

    print(f"\nTraining pipeline scheduled for {len(targets)} object(s): {', '.join(targets)}\n")
    results = {}
    for obj_name in targets:
        metrics = train_object_session(
            object_name=obj_name,
            total_timesteps=args.timesteps,
            num_envs=args.num_envs,
            save_dir=args.save_dir,
            device=device,
        )
        results[obj_name] = metrics

    print("\n==================== FINAL BENCHMARK SUMMARY ====================")
    print(f"{'Object':<16} | {'Safe F':<8} | {'Peak F':<8} | {'Damage %':<10} | {'Pristine %':<12} | {'Success %':<10}")
    print("-" * 75)
    for obj, m in results.items():
        print(f"{obj:<16} | {m['f_safe']:<6.1f}N | {m['mean_peak_force']:<6.1f}N | {m['mean_damage_pct']:<8.1f}% | {m['pristine_success_rate']:<10.1f}% | {m['success_rate']:<8.1f}%")
    print("=================================================================\n")


if __name__ == "__main__":
    main()
