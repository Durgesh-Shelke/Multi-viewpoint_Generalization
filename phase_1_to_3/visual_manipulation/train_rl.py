"""
Visual Reinforcement Learning (PPO) Fine-tuner for Robotic Manipulation.
Loads pre-trained visual policy weights (from BC) and optimizes on reward signals via GPU parallel simulation.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Fine-tune or train Visual Policy using PPO")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to pre-trained checkpoint to start from")
    parser.add_argument("--env_id", type=str, default="PickCube-v1", help="ManiSkill environment ID")
    parser.add_argument("--control_mode", type=str, default="pd_ee_delta_pos", help="Target controller mode")
    parser.add_argument("--num_envs", type=int, default=128, help="Number of parallel environments (tuned for 8GB VRAM)")
    parser.add_argument("--num_eval_envs", type=int, default=8, help="Number of parallel evaluation environments")
    parser.add_argument("--total_timesteps", type=int, default=1_000_000, help="Total environment steps")
    parser.add_argument("--update_epochs", type=int, default=8, help="PPO update epochs per rollout")
    parser.add_argument("--num_minibatches", type=int, default=8, help="Minibatches per update")
    parser.add_argument("--eval_freq", type=int, default=5, help="Epoch frequency for evaluations")
    parser.add_argument("--exp_name", type=str, default="visual_ppo_finetune", help="Experiment name")
    args = parser.parse_args()

    ppo_script = Path("/home/maniskill/examples/baselines/ppo/ppo_rgb.py")
    if not ppo_script.exists():
        print(f"Error: {ppo_script} not found.")
        sys.exit(1)

    cmd = [
        "python3",
        str(ppo_script),
        f"--env_id={args.env_id}",
        f"--control_mode={args.control_mode}",
        f"--num_envs={args.num_envs}",
        f"--num_eval_envs={args.num_eval_envs}",
        f"--total_timesteps={args.total_timesteps}",
        f"--update_epochs={args.update_epochs}",
        f"--num_minibatches={args.num_minibatches}",
        f"--eval_freq={args.eval_freq}",
        f"--exp_name={args.exp_name}",
        "--save_model",
    ]

    if args.checkpoint:
        cmd.append(f"--checkpoint={args.checkpoint}")

    cmd_str = " ".join(cmd)
    print(f"\n[Running Visual PPO] {cmd_str}\n")
    subprocess.run(cmd_str, shell=True, check=True)


if __name__ == "__main__":
    main()
