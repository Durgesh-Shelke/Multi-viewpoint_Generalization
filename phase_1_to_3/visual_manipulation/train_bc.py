"""
Behavioral Cloning Pre-training for Visual Robot Manipulation.
Trains the visual policy (NatureCNN + Action Head) directly on camera frames and demonstrations.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

# Ensure visual_manipulation and mani_skill are on sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, "/home/maniskill")

from dataset import VisualPickDataset
from model import VisualAgent
from evaluate import evaluate_policy


def main():
    parser = argparse.ArgumentParser(description="Pre-train Visual Policy using Behavioral Cloning")
    parser.add_argument("--demo_path", type=str, default=str(Path.home() / ".maniskill" / "demos" / "PickCube-v1" / "motionplanning" / "trajectory.rgb.pd_ee_delta_pos.physx_cpu.h5"))
    parser.add_argument("--env_id", type=str, default="PickCube-v1")
    parser.add_argument("--control_mode", type=str, default="pd_ee_delta_pos")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--eval_freq", type=int, default=10)
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--save_dir", type=str, default="checkpoints/bc")
    parser.add_argument("--max_demos", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using {device}")

    # Load dataset
    dataset = VisualPickDataset(args.demo_path, max_demos=args.max_demos)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    # Initialize agent
    agent = VisualAgent(action_dim=4, state_dim=29, in_channels=3).to(device)
    optimizer = torch.optim.AdamW(agent.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_loss = float("inf")
    start_time = time.time()
    print("\n=== Starting Behavioral Cloning Training ===")

    for epoch in range(1, args.epochs + 1):
        agent.train()
        train_loss = 0.0
        for batch in train_loader:
            obs = {"rgb": batch["rgb"].to(device), "state": batch["state"].to(device)}
            targets = batch["action"].to(device)

            preds = agent.actor_mean(agent.feature_net(obs))
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
                obs = {"rgb": batch["rgb"].to(device), "state": batch["state"].to(device)}
                targets = batch["action"].to(device)
                preds = agent.actor_mean(agent.feature_net(obs))
                val_loss += F.mse_loss(preds, targets).item() * len(targets)
        val_loss /= len(val_set)

        print(f"Epoch {epoch:02d}/{args.epochs:02d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | LR: {scheduler.get_last_lr()[0]:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(agent.state_dict(), os.path.join(args.save_dir, "best_val_model.pt"))

        if epoch % args.eval_freq == 0:
            print(f"--> Periodic Evaluation (Epoch {epoch})...")
            succ, ret = evaluate_policy(agent, env_id=args.env_id, control_mode=args.control_mode, num_episodes=args.eval_episodes, device=device)
            print(f"--> Success Rate: {succ * 100:.1f}%, Mean Return: {ret:.2f}")

    final_ckpt = os.path.join(args.save_dir, "final_bc_model.pt")
    torch.save(agent.state_dict(), final_ckpt)
    print(f"\n[Finished] Training completed in {time.time() - start_time:.1f}s.")
    print(f"[Saved] Model weights saved to: {final_ckpt}")


if __name__ == "__main__":
    main()
