import argparse
import os
import random
import sys
import time
from collections import defaultdict

import gymnasium as gym
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Add paths
sys.path.insert(0, "/home/maniskill")
sys.path.insert(0, "/home/maniskill/examples/baselines/ppo")

import mani_skill.envs
from mani_skill.utils import common
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from ppo_rgb import Agent


class VisualPickDataset(Dataset):
    def __init__(self, demo_path, device="cpu"):
        super().__init__()
        print(f"Loading demonstrations from {demo_path}...")
        f = h5py.File(demo_path, "r")
        rgbs, states, actions = [], [], []

        for k in sorted(f.keys()):
            traj = f[k]
            act = torch.from_numpy(traj["actions"][:]).float()
            N = len(act)
            rgb = torch.from_numpy(traj["obs"]["sensor_data"]["base_camera"]["rgb"][:N])  # uint8 (N, 128, 128, 3)

            # Reconstruct exact state in dictionary insertion order
            agent_dict = {
                "qpos": torch.from_numpy(traj["obs"]["agent"]["qpos"][:N]),
                "qvel": torch.from_numpy(traj["obs"]["agent"]["qvel"][:N]),
            }
            extra_dict = {
                "is_grasped": torch.from_numpy(traj["obs"]["extra"]["is_grasped"][:N]).unsqueeze(-1),
                "tcp_pose": torch.from_numpy(traj["obs"]["extra"]["tcp_pose"][:N]),
                "goal_pos": torch.from_numpy(traj["obs"]["extra"]["goal_pos"][:N]),
            }
            state = common.flatten_state_dict({"agent": agent_dict, "extra": extra_dict}, use_torch=True).float()

            rgbs.append(rgb)
            states.append(state)
            actions.append(act)

        self.rgbs = torch.cat(rgbs, dim=0)
        self.states = torch.cat(states, dim=0)
        self.actions = torch.cat(actions, dim=0)
        self.device = device
        print(f"Dataset loaded: {len(self.actions)} transitions. RGB: {self.rgbs.shape}, State: {self.states.shape}, Actions: {self.actions.shape}")

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        return {
            "rgb": self.rgbs[idx],
            "state": self.states[idx],
            "action": self.actions[idx],
        }


def evaluate(agent, env_id, control_mode, num_episodes=10, max_steps=100, record_dir=None, device="cuda"):
    agent.eval()
    render_mode = "rgb_array" if record_dir else None
    env = gym.make(env_id, obs_mode="rgb", control_mode=control_mode, num_envs=1, render_mode=render_mode)
    env = FlattenRGBDObservationWrapper(env, rgb=True, depth=False, state=True)
    if record_dir:
        os.makedirs(record_dir, exist_ok=True)
        env = RecordEpisode(env, output_dir=record_dir, max_steps_per_video=max_steps, save_trajectory=False)

    successes = []
    returns = []

    for ep in range(num_episodes):
        obs, _ = env.reset(seed=ep + 1000)
        ep_return = 0.0
        success = False
        for step in range(max_steps):
            obs_gpu = {k: v.to(device) for k, v in obs.items()}
            with torch.no_grad():
                action = agent.get_action(obs_gpu, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += reward.item() if torch.is_tensor(reward) else reward
            if "success" in info:
                success = success or bool(info["success"].any().item() if torch.is_tensor(info["success"]) else info["success"])
            if terminated or truncated:
                break
        successes.append(float(success))
        returns.append(ep_return)

    env.close()
    mean_success = np.mean(successes)
    mean_return = np.mean(returns)
    return mean_success, mean_return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo_path", type=str, default="/root/.maniskill/demos/PickCube-v1/motionplanning/trajectory.rgb.pd_ee_delta_pos.physx_cpu.h5")
    parser.add_argument("--env_id", type=str, default="PickCube-v1")
    parser.add_argument("--control_mode", type=str, default="pd_ee_delta_pos")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--eval_freq", type=int, default=10)
    parser.add_argument("--num_eval_episodes", type=int, default=10)
    parser.add_argument("--save_dir", type=str, default="runs/PickCube-v1__visual_bc")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    # Mock env to instantiate Agent with exact dimensions
    class MockEnv:
        class unwrapped:
            class single_action_space:
                shape = (4,)

    sample_obs = {
        "rgb": torch.zeros(1, 128, 128, 3, dtype=torch.uint8),
        "state": torch.zeros(1, 29, dtype=torch.float32),
    }
    agent = Agent(MockEnv(), sample_obs).to(device)

    if args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint}")
        agent.load_state_dict(torch.load(args.checkpoint, map_location=device))

    if args.evaluate:
        print("Running evaluation...")
        eval_video_dir = os.path.join(args.save_dir, "eval_videos")
        success, ret = evaluate(agent, args.env_id, args.control_mode, num_episodes=args.num_eval_episodes, record_dir=eval_video_dir, device=device)
        print(f"Evaluation Success Rate: {success * 100:.1f}%, Mean Return: {ret:.2f}")
        print(f"Videos saved to: {eval_video_dir}")
        return

    # Dataset & DataLoader
    dataset = VisualPickDataset(args.demo_path)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(agent.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_loss = float("inf")
    best_eval_success = 0.0

    print("--- Starting Behavioral Cloning Pre-training on Visual Observations ---")
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        agent.train()
        train_loss = 0.0
        for batch in train_loader:
            obs = {
                "rgb": batch["rgb"].to(device),
                "state": batch["state"].to(device),
            }
            targets = batch["action"].to(device)

            features = agent.feature_net(obs)
            preds = agent.actor_mean(features)
            loss = F.mse_loss(preds, targets)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item() * len(targets)

        scheduler.step()
        train_loss /= len(train_set)

        # Validation
        agent.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                obs = {
                    "rgb": batch["rgb"].to(device),
                    "state": batch["state"].to(device),
                }
                targets = batch["action"].to(device)
                features = agent.feature_net(obs)
                preds = agent.actor_mean(features)
                val_loss += F.mse_loss(preds, targets).item() * len(targets)
        val_loss /= len(val_set)

        print(f"Epoch {epoch:02d}/{args.epochs:02d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | LR: {scheduler.get_last_lr()[0]:.6f}")

        # Save best validation model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(agent.state_dict(), os.path.join(args.save_dir, "best_val_ckpt.pt"))

        # Evaluate policy periodically
        if epoch % args.eval_freq == 0:
            print(f"--> Evaluating policy on environment (Epoch {epoch})...")
            succ, ret = evaluate(agent, args.env_id, args.control_mode, num_episodes=args.num_eval_episodes, device=device)
            print(f"--> Evaluation Success Rate: {succ * 100:.1f}%, Mean Return: {ret:.2f}")
            if succ >= best_eval_success:
                best_eval_success = succ
                torch.save(agent.state_dict(), os.path.join(args.save_dir, "best_eval_ckpt.pt"))

    # Save final model
    torch.save(agent.state_dict(), os.path.join(args.save_dir, "final_ckpt.pt"))
    print(f"Training completed in {time.time() - start_time:.2f}s!")

    # Final evaluation with video recording
    print("Recording final evaluation videos...")
    eval_video_dir = os.path.join(args.save_dir, "eval_videos")
    succ, ret = evaluate(agent, args.env_id, args.control_mode, num_episodes=5, record_dir=eval_video_dir, device=device)
    print(f"Final Evaluation Success Rate: {succ * 100:.1f}%, Mean Return: {ret:.2f}")
    print(f"Videos saved to: {eval_video_dir}")


if __name__ == "__main__":
    main()
