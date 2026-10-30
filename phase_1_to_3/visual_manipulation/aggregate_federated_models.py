"""
Federated Model Aggregation (FedAvg & Task Arithmetic).
Combines models trained independently across separate tasks into a single unified global policy.
"""

import argparse
import copy
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, "/home/maniskill")

import fragile_pick_env
from fragile_pick_env import OBJECT_NAMES
from model import EmbodimentAgnosticVisualAgent
from train_federated_clients import evaluate_client


def federated_averaging(client_state_dicts, weights=None):
    """
    Standard FedAvg: Computes weighted average of parameters across all client models.
    """
    num_clients = len(client_state_dicts)
    if weights is None:
        weights = [1.0 / num_clients] * num_clients

    global_dict = copy.deepcopy(client_state_dicts[0])

    for key in global_dict.keys():
        if "num_batches_tracked" in key:
            continue
        # Weighted parameter sum
        global_dict[key] = sum(w * client[key].float() for w, client in zip(weights, client_state_dicts))

    return global_dict


def main():
    parser = argparse.ArgumentParser(description="Aggregate Federated RL Models")
    parser.add_argument("--runs_dir", type=str, default="/home/maniskill/visual_manipulation/checkpoints/federated_runs", help="Directory containing client subdirectories")
    parser.add_argument("--eval_episodes", type=int, default=5, help="Evaluation episodes per object")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*70}")
    print(f" FEDERATED REINFORCEMENT LEARNING AGGREGATION")
    print(f" Loading client models from: {args.runs_dir}")
    print(f"{'='*70}\n")

    client_dicts = []
    loaded_objects = []

    for obj in OBJECT_NAMES:
        ckpt_path = os.path.join(args.runs_dir, obj, "policy.pt")
        if os.path.exists(ckpt_path):
            sd = torch.load(ckpt_path, map_location="cpu")
            client_dicts.append(sd)
            loaded_objects.append(obj)
            print(f" [Loaded] Client: {obj:<16} from {ckpt_path}")
        else:
            print(f" [Warning] Missing checkpoint for {obj} at {ckpt_path}")

    if not client_dicts:
        print("Error: No client checkpoints found to aggregate.")
        sys.exit(1)

    print(f"\nAggregating {len(client_dicts)} models via FedAvg (Federated Averaging)...")
    global_sd = federated_averaging(client_dicts)

    global_ckpt_path = os.path.join(args.runs_dir, "global_aggregated_model.pt")
    torch.save(global_sd, global_ckpt_path)
    print(f"[Saved] Global Aggregated Model saved to: {global_ckpt_path}")

    # Evaluate Global Aggregated Model across ALL objects
    print("\n" + "=" * 70)
    print(" EVALUATING GLOBAL AGGREGATED POLICY ACROSS ALL OBJECTS")
    print("=" * 70)

    global_agent = EmbodimentAgnosticVisualAgent(action_dim=4, in_channels=3, tcp_dim=11, max_dof=16, num_known_objects=10).to(device)
    global_agent.load_state_dict(global_sd)

    eval_summary = {}
    video_base = os.path.join(args.runs_dir, "global_eval_videos")

    for obj in loaded_objects:
        video_dir = os.path.join(video_base, obj)
        m = evaluate_client(global_agent, object_name=obj, episodes=args.eval_episodes, video_dir=video_dir, device=device)
        eval_summary[obj] = m
        print(f"[{obj:<14s}] Success: {m['success_rate']:<5.1f}% | Pristine (0% dmg): {m['pristine_success_rate']:<5.1f}% | Peak Force: {m['mean_peak_force']:<5.1f}N | Dmg: {m['mean_damage_pct']:<5.1f}%")

    with open(os.path.join(args.runs_dir, "global_evaluation_summary.json"), "w") as f:
        json.dump(eval_summary, f, indent=2)

    print("\n" + "=" * 70)
    print(f"Aggregation and multi-object evaluation complete. Videos in: {video_base}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
