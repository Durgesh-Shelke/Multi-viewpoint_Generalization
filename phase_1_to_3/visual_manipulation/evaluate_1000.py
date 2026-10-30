"""
Rigorous 1,000-Episode Evaluation Benchmark for Sim-to-Real Visual Manipulation Policy.
Uses GPU-parallel simulation (100 parallel environments) to evaluate 1,000 test runs in seconds.
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

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, "/home/maniskill")
sys.path.insert(0, "/home/maniskill/examples/baselines/ppo")

import mani_skill.envs
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper, FlattenRGBDObservationWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from ppo_rgb import Agent


def run_benchmark_1000(checkpoint_path: str, num_eval_envs: int = 100, target_episodes: int = 1000, device: str = "cuda", output_path: str = None):
    print(f"\n{'='*75}")
    print(f" BENCHMARK EVALUATION: {target_episodes} TEST RUNS")
    print(f" Checkpoint: {checkpoint_path}")
    print(f" Parallel Eval Instances: {num_eval_envs}")
    print(f"{'='*75}\n")

    env_kwargs = dict(
        obs_mode="rgb",
        control_mode="pd_ee_delta_pos",
        render_mode=None,
        sim_backend="physx_cuda",
    )
    env = gym.make("PickCube-v1", num_envs=num_eval_envs, **env_kwargs)
    env = FlattenRGBDObservationWrapper(env, rgb=True, depth=False, state=True)

    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    env = ManiSkillVectorEnv(env, num_eval_envs, ignore_terminations=False, record_metrics=True)

    obs, _ = env.reset(seed=12345)
    agent = Agent(env, sample_obs=obs).to(device)
    print(f"[Loading] Weights from {checkpoint_path}...")
    agent.load_state_dict(torch.load(checkpoint_path, map_location=device))
    agent.eval()

    episodes_completed = 0
    success_count = 0
    returns = []
    episode_lens = []

    start_time = time.time()
    step_count = 0

    with torch.no_grad():
        while episodes_completed < target_episodes:
            action = agent.get_action(obs, deterministic=True)
            obs, rew, term, trunc, infos = env.step(action)
            step_count += 1

            if "final_info" in infos:
                mask = infos["_final_info"]
                for i in range(len(mask)):
                    if mask[i]:
                        ep_info = infos["final_info"]["episode"]
                        succ_key = "success_once" if "success_once" in ep_info else ("success" if "success" in ep_info else "success_at_end")
                        succ = bool(ep_info[succ_key][i].item() if torch.is_tensor(ep_info[succ_key]) else ep_info[succ_key])
                        ret = float(ep_info["return"][i].item() if torch.is_tensor(ep_info["return"]) else ep_info["return"])
                        ep_len = float(ep_info["episode_len"][i].item() if torch.is_tensor(ep_info["episode_len"]) else ep_info["episode_len"])

                        if succ:
                            success_count += 1
                        returns.append(ret)
                        episode_lens.append(ep_len)
                        episodes_completed += 1

                        if episodes_completed % 100 == 0 or episodes_completed == target_episodes:
                            curr_rate = (success_count / episodes_completed) * 100
                            print(f" [Progress] Evaluated {episodes_completed:4d}/{target_episodes} | Current Success Rate: {curr_rate:5.1f}%")

                        if episodes_completed >= target_episodes:
                            break

    elapsed = time.time() - start_time
    env.close()

    final_success_rate = (success_count / target_episodes) * 100
    mean_return = float(np.mean(returns))
    mean_length = float(np.mean(episode_lens))

    print("\n" + "=" * 75)
    print(f" FINAL 1,000-RUN BENCHMARK REPORT")
    print("=" * 75)
    print(f" Total Test Episodes:   {target_episodes:,}")
    print(f" Successful Picks:      {success_count:,} / {target_episodes:,}")
    print(f" Success Rate:          {final_success_rate:.2f}%")
    print(f" Target Goal:           95.00%")
    print(f" Goal Met:              {'YES (>= 95%)' if final_success_rate >= 95.0 else 'NO (< 95%)'}")
    print(f" Mean Return:           {mean_return:.2f}")
    print(f" Mean Episode Length:   {mean_length:.1f} steps")
    print(f" Benchmark Walltime:    {elapsed:.1f}s")
    print("=" * 75 + "\n")

    report = {
        "checkpoint": checkpoint_path,
        "target_episodes": target_episodes,
        "success_count": success_count,
        "success_rate": final_success_rate,
        "goal_met": bool(final_success_rate >= 95.0),
        "mean_return": mean_return,
        "mean_length": mean_length,
        "elapsed_seconds": elapsed,
    }

    report_path = output_path or os.path.join(os.path.dirname(checkpoint_path), "benchmark_1000_results.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint file (.pt)")
    parser.add_argument("--num_eval_envs", type=int, default=100)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    run_benchmark_1000(args.checkpoint, num_eval_envs=args.num_eval_envs, target_episodes=args.episodes, output_path=args.output)
