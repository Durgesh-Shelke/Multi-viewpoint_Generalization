"""
Single-Viewpoint Specialist Training Harness for 128x128 RGB-D (Candidate A Architecture).
Trains an independent policy for a designated viewpoint with orthogonal random initialization,
early stopping upon reaching >=95.0% grasp success, and zero-loss pause/resume support.
"""

import argparse
import gc
import json
import os
import random
import signal
import sys
import time
from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, "/home/maniskill")

import mani_skill.envs
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper, FlattenRGBDObservationWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from models import UnifiedAblationAgent

STOP_REQUESTED = False


def sig_handler(signum, frame):
    global STOP_REQUESTED
    print(f"\n[Interrupt] Signal {signum} received! Stopping cleanly after iteration...")
    STOP_REQUESTED = True


signal.signal(signal.SIGINT, sig_handler)
signal.signal(signal.SIGTERM, sig_handler)


def save_checkpoint(
    ckpt_path: str,
    agent: nn.Module,
    optimizer: optim.Optimizer,
    iteration: int,
    global_step: int,
    best_eval_success: float,
    history_data: list,
    elapsed_seconds: float,
    completed: bool = False,
):
    tmp_path = ckpt_path + ".tmp"
    torch.save(
        {
            "iteration": iteration,
            "global_step": global_step,
            "best_eval_success": best_eval_success,
            "model_state": agent.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "history_data": history_data,
            "elapsed_seconds": elapsed_seconds,
            "completed": completed,
        },
        tmp_path,
    )
    os.replace(tmp_path, ckpt_path)


def save_best_weights_only(best_path: str, agent: nn.Module, best_eval_success: float, global_step: int):
    tmp_path = best_path + ".tmp"
    torch.save(
        {
            "global_step": global_step,
            "best_eval_success": best_eval_success,
            "model_state": agent.state_dict(),
        },
        tmp_path,
    )
    os.replace(tmp_path, best_path)


def evaluate_policy(agent: nn.Module, eval_envs, num_episodes: int = 100, max_steps: int = 50, device: str = "cuda"):
    agent.eval()
    eval_obs, _ = eval_envs.reset()
    eval_successes = 0
    eval_eps = 0
    returns = []
    lengths = []

    with torch.no_grad():
        for _ in range(max_steps):
            act = agent.get_action(eval_obs, deterministic=True)
            eval_obs, _, _, _, eval_infos = eval_envs.step(act)
            if "final_info" in eval_infos:
                mask = eval_infos.get("_final_info", None)
                if mask is not None:
                    for i in range(len(mask)):
                        if mask[i]:
                            ep_info = eval_infos["final_info"].get("episode", {})
                            succ_key = "success_once" if "success_once" in ep_info else "success"
                            succ = False
                            if succ_key in ep_info:
                                val = ep_info[succ_key][i]
                                succ = bool(val.item() if torch.is_tensor(val) else val)
                            if succ:
                                eval_successes += 1
                            if "r" in ep_info:
                                r_val = ep_info["r"][i]
                                returns.append(float(r_val.item() if torch.is_tensor(r_val) else r_val))
                            if "l" in ep_info:
                                l_val = ep_info["l"][i]
                                lengths.append(float(l_val.item() if torch.is_tensor(l_val) else l_val))
                            eval_eps += 1

    agent.train()
    success_rate = (eval_successes / max(eval_eps, 1)) * 100.0 if eval_eps > 0 else 0.0
    mean_ret = float(np.mean(returns)) if returns else 0.0
    mean_len = float(np.mean(lengths)) if lengths else 0.0
    return success_rate, mean_ret, mean_len


@dataclass
class Args:
    viewpoint_id: int = 17
    total_timesteps: int = 5000000
    num_envs: int = 128
    num_eval_envs: int = 50
    eval_episodes: int = 100
    eval_freq: int = 25  # Eval every 25 iterations = 160,000 steps
    early_stop_threshold: float = 95.0
    num_steps: int = 50
    num_minibatches: int = 8
    update_epochs: int = 4
    lr: float = 3e-4
    gamma: float = 0.8
    gae_lambda: float = 0.9
    clip_coef: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = 0.2
    seed: int = 42
    output_dir: str = "/home/maniskill/phase_6_viewpoint_specialists"
    viewpoints_json: str = "/home/maniskill/phase_1_to_3/sequential_viewpoint_training/viewpoints.json"


def train_specialist(args: Args):
    global STOP_REQUESTED
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load viewpoint coordinates
    with open(args.viewpoints_json) as f:
        all_vps = json.load(f)
    vp_dict = {vp["id"]: vp for vp in all_vps}
    if args.viewpoint_id not in vp_dict:
        raise ValueError(f"Viewpoint ID {args.viewpoint_id} not found in {args.viewpoints_json}")

    vp = vp_dict[args.viewpoint_id]
    cam_eye = vp["world_eye"]
    cam_target = [-0.1, 0.0, 0.1]
    az = vp["azimuth_deg"]
    el = vp["elevation_deg"]

    run_seed = args.seed + args.viewpoint_id
    random.seed(run_seed)
    np.random.seed(run_seed)
    torch.manual_seed(run_seed)
    torch.backends.cudnn.deterministic = True

    ckpt_dir = os.path.join(args.output_dir, "checkpoints")
    eval_dir = os.path.join(args.output_dir, "eval_results")
    tb_dir = os.path.join(args.output_dir, "tb_logs", f"specialist_vp_{args.viewpoint_id:02d}")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)
    os.makedirs(tb_dir, exist_ok=True)

    latest_ckpt_path = os.path.join(ckpt_dir, f"specialist_vp_{args.viewpoint_id:02d}_latest.pt")
    best_ckpt_path = os.path.join(ckpt_dir, f"specialist_vp_{args.viewpoint_id:02d}_best.pt")
    metrics_json_path = os.path.join(ckpt_dir, f"specialist_vp_{args.viewpoint_id:02d}_train_metrics.json")
    history_json_path = os.path.join(eval_dir, f"specialist_vp_{args.viewpoint_id:02d}_history.json")
    history_csv_path = os.path.join(eval_dir, f"specialist_vp_{args.viewpoint_id:02d}_history.csv")
    pause_file = os.path.join(args.output_dir, ".pause")

    # Check if already completed (must hit target ceiling or max steps)
    if os.path.exists(metrics_json_path) and os.path.exists(best_ckpt_path):
        with open(metrics_json_path) as f:
            m = json.load(f)
        succ = m.get("best_eval_success", 0.0)
        tot_steps = m.get("total_timesteps", 0)
        if succ >= args.early_stop_threshold or tot_steps >= args.total_timesteps:
            print(f"[Specialist VP {args.viewpoint_id:02d}] Already completed! Target success: {succ:.1f}% (Steps: {tot_steps:,}). Skipping.")
            return

    batch_size = args.num_envs * args.num_steps
    minibatch_size = batch_size // args.num_minibatches
    total_iterations = args.total_timesteps // batch_size

    print(f"\n{'='*78}")
    print(f" TRAINING SPECIALIST: VIEWPOINT {args.viewpoint_id:02d}")
    print(f" Azimuth: {az:+6.1f}° | Elevation: {el:4.1f}° | Eye: {cam_eye}")
    print(f" Input: 128x128 4-Channel RGB-D | Orthogonal Random Init")
    print(f" Envs: {args.num_envs} | Batch: {batch_size} | Total Steps: {args.total_timesteps:,}")
    print(f" Target Early Stopping: >={args.early_stop_threshold:.1f}% Eval Success")
    print(f"{'='*78}\n")

    # Environment setup
    env_kwargs = dict(
        obs_mode="rgbd",
        render_mode="all",
        sim_backend="physx_cuda",
        sensor_cam_eye_pos=cam_eye,
        sensor_cam_target_pos=cam_target,
        control_mode="pd_ee_delta_pos",
        sensor_configs=dict(width=128, height=128),
    )

    eval_envs = gym.make("PickCube-v1", num_envs=args.num_eval_envs, reconfiguration_freq=None, **env_kwargs)
    envs = gym.make("PickCube-v1", num_envs=args.num_envs, reconfiguration_freq=None, **env_kwargs)

    envs = FlattenRGBDObservationWrapper(envs, rgb=True, depth=True, state=True, sep_depth=True)
    eval_envs = FlattenRGBDObservationWrapper(eval_envs, rgb=True, depth=True, state=True, sep_depth=True)

    if isinstance(envs.action_space, gym.spaces.Dict):
        envs = FlattenActionSpaceWrapper(envs)
        eval_envs = FlattenActionSpaceWrapper(eval_envs)

    envs = ManiSkillVectorEnv(envs, args.num_envs, ignore_terminations=False, record_metrics=True)
    eval_envs = ManiSkillVectorEnv(eval_envs, args.num_eval_envs, ignore_terminations=True, record_metrics=True)

    # Initialize Agent (Candidate A: 128x128 RGB-D)
    obs, _ = envs.reset(seed=run_seed)
    agent = UnifiedAblationAgent(envs, obs, candidate="A").to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.lr, eps=1e-5)
    writer = SummaryWriter(tb_dir)

    start_iteration = 1
    global_step = 0
    best_eval_success = 0.0
    history_data = []
    elapsed_seconds = 0.0

    # Resumption check
    if os.path.exists(latest_ckpt_path):
        try:
            print(f"[Resume] Found checkpoint at {latest_ckpt_path}. Restoring...")
            ckpt = torch.load(latest_ckpt_path, map_location=device)
            agent.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["optimizer_state"])
            start_iteration = ckpt.get("iteration", 0) + 1
            global_step = ckpt.get("global_step", 0)
            best_eval_success = ckpt.get("best_eval_success", 0.0)
            history_data = ckpt.get("history_data", [])
            elapsed_seconds = ckpt.get("elapsed_seconds", 0.0)
            print(f"[Resume] Restored state from Iteration {start_iteration - 1}, Step {global_step:,}, Best Eval: {best_eval_success:.1f}%")
        except Exception as e:
            print(f"[Resume Error] Could not load checkpoint: {e}. Starting fresh.")

    # Storage buffers
    obs_rgb = torch.zeros((args.num_steps, args.num_envs, 128, 128, 3), dtype=torch.uint8, device=device)
    obs_depth = torch.zeros((args.num_steps, args.num_envs, 128, 128, 1), dtype=torch.int16, device=device)
    obs_state = torch.zeros((args.num_steps, args.num_envs, 29), dtype=torch.float32, device=device)

    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape, device=device)
    logprobs = torch.zeros((args.num_steps, args.num_envs), device=device)
    rewards = torch.zeros((args.num_steps, args.num_envs), device=device)
    dones = torch.zeros((args.num_steps, args.num_envs), device=device)
    values = torch.zeros((args.num_steps, args.num_envs), device=device)

    next_obs = obs
    next_done = torch.zeros(args.num_envs, device=device)

    start_walltime = time.time() - elapsed_seconds
    peak_vram_mb = 0.0

    print(f"[Training] Launching PPO loop from iteration {start_iteration} to {total_iterations}...")

    early_stopped = False
    iteration = start_iteration - 1
    for iteration in range(start_iteration, total_iterations + 1):
        iter_start_time = time.time()

        # Learning rate annealing
        frac = 1.0 - (iteration - 1.0) / total_iterations
        lrnow = frac * args.lr
        optimizer.param_groups[0]["lr"] = lrnow

        # Rollout phase
        final_values = torch.zeros((args.num_steps, args.num_envs), device=device)
        for step in range(args.num_steps):
            global_step += args.num_envs
            obs_rgb[step] = next_obs["rgb"]
            obs_depth[step] = next_obs["depth"]
            obs_state[step] = next_obs["state"]
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            next_obs, reward, terminations, truncations, infos = envs.step(action)
            next_done = torch.logical_or(terminations, truncations).to(torch.float32)
            rewards[step] = reward.view(-1)

            # Proper Gymnasium truncation bootstrapping
            if "final_info" in infos and "_final_info" in infos:
                done_mask = infos["_final_info"]
                for k in infos["final_observation"]:
                    infos["final_observation"][k] = infos["final_observation"][k][done_mask]
                with torch.no_grad():
                    final_values[step, torch.arange(args.num_envs, device=device)[done_mask]] = agent.get_value(infos["final_observation"]).view(-1)

        # GAE bootstrap with truncation handling
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    next_not_done = 1.0 - next_done
                    nextvalues = next_value
                else:
                    next_not_done = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                real_next_values = next_not_done * nextvalues + final_values[t]
                delta = rewards[t] + args.gamma * real_next_values - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * next_not_done * lastgaelam
            returns = advantages + values

        # Flatten batch
        b_obs = {
            "rgb": obs_rgb.reshape((-1, 128, 128, 3)),
            "depth": obs_depth.reshape((-1, 128, 128, 1)),
            "state": obs_state.reshape((-1, 29)),
        }
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # PPO Update
        b_inds = np.arange(batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                mb_obs = {k: v[mb_inds] for k, v in b_obs.items()}
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(mb_obs, b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                entropy_loss = entropy.mean()

                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        iter_elapsed = time.time() - iter_start_time
        sps = int(batch_size / iter_elapsed)
        cur_vram = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0
        peak_vram_mb = max(peak_vram_mb, cur_vram)

        # Periodic Evaluation & Checkpointing
        if iteration % args.eval_freq == 0 or iteration == total_iterations or STOP_REQUESTED:
            eval_succ, eval_ret, eval_len = evaluate_policy(agent, eval_envs, num_episodes=args.eval_episodes, device=device)
            current_elapsed = time.time() - start_walltime

            print(
                f"[Iter {iteration:04d}/{total_iterations} | Step {global_step:,}] "
                f"Eval Success: {eval_succ:5.1f}% | Return: {eval_ret:5.2f} | Length: {eval_len:4.1f} | "
                f"SPS: {sps} | Best: {best_eval_success:5.1f}% | VRAM: {cur_vram:.0f}MB"
            )

            # Record history
            history_data.append({
                "iteration": iteration,
                "global_step": global_step,
                "eval_success": eval_succ,
                "eval_return": eval_ret,
                "eval_length": eval_len,
                "sps": sps,
                "elapsed_seconds": round(current_elapsed, 1),
            })

            # Save latest checkpoint for resume
            save_checkpoint(
                latest_ckpt_path,
                agent,
                optimizer,
                iteration,
                global_step,
                max(best_eval_success, eval_succ),
                history_data,
                current_elapsed,
                completed=False,
            )

            # Check for new best model
            if eval_succ >= best_eval_success:
                best_eval_success = eval_succ
                save_best_weights_only(best_ckpt_path, agent, best_eval_success, global_step)
                print(f"  >>> [NEW BEST] VP {args.viewpoint_id:02d} Model Saved! Success: {best_eval_success:.1f}% -> {best_ckpt_path}")

            # Early stopping check
            if eval_succ >= args.early_stop_threshold:
                print(f"\n{'*'*78}")
                print(f" [EARLY STOPPING TRIGGERED] VP {args.viewpoint_id:02d} reached {eval_succ:.1f}% >= {args.early_stop_threshold:.1f}%!")
                print(f" Plateau mastered at Step {global_step:,} (Iter {iteration}). Cleanly completing training.")
                print(f"{'*'*78}\n")
                early_stopped = True
                break

        # Check for pause flag
        if os.path.exists(pause_file):
            print(f"\n[PAUSE] Detected {pause_file}. Saving state and entering sleep loop...")
            current_elapsed = time.time() - start_walltime
            save_checkpoint(
                latest_ckpt_path,
                agent,
                optimizer,
                iteration,
                global_step,
                best_eval_success,
                history_data,
                current_elapsed,
                completed=False,
            )
            torch.cuda.empty_cache()
            gc.collect()

            while os.path.exists(pause_file) and not STOP_REQUESTED:
                time.sleep(2)
            print("[RESUME] Pause cleared. Resuming active training...")
            start_walltime = time.time() - current_elapsed

        if STOP_REQUESTED:
            print("[STOP] Graceful stop requested. Checkpoint saved. Exiting.")
            break

    total_walltime = time.time() - start_walltime

    # Save final evaluation history JSON & CSV
    with open(history_json_path, "w") as f:
        json.dump(history_data, f, indent=2)

    import csv
    if history_data:
        with open(history_csv_path, "w", newline="") as f:
            dict_writer = csv.DictWriter(f, fieldnames=list(history_data[0].keys()))
            dict_writer.writeheader()
            dict_writer.writerows(history_data)

    # Save final completion metrics
    metrics = {
        "viewpoint_id": args.viewpoint_id,
        "azimuth_deg": az,
        "elevation_deg": el,
        "world_eye": cam_eye,
        "completed": True if (best_eval_success >= args.early_stop_threshold or global_step >= args.total_timesteps) else False,
        "early_stopped": early_stopped,
        "total_timesteps": global_step,
        "total_iterations": iteration,
        "best_eval_success": best_eval_success,
        "total_walltime_seconds": round(total_walltime, 2),
        "avg_sps": int(global_step / max(total_walltime, 1)),
        "peak_vram_mb": round(peak_vram_mb, 1),
        "best_checkpoint": best_ckpt_path,
    }
    with open(metrics_json_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n[COMPLETED] Viewpoint {args.viewpoint_id:02d} Training Finished!")
    print(f" Steps: {global_step:,} | Best Eval: {best_eval_success:.1f}% | Walltime: {total_walltime / 60:.1f} mins | Avg SPS: {metrics['avg_sps']}")
    print(f" Best Weights: {best_ckpt_path}\n")

    envs.close()
    eval_envs.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--viewpoint-id", type=int, default=17)
    parser.add_argument("--total-timesteps", type=int, default=5000000)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--num-eval-envs", type=int, default=50)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--eval-freq", type=int, default=25)
    parser.add_argument("--early-stop-threshold", type=float, default=95.0)
    parser.add_argument("--output-dir", type=str, default="/home/maniskill/phase_6_viewpoint_specialists")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    args_dataclass = Args(
        viewpoint_id=args.viewpoint_id,
        total_timesteps=args.total_timesteps,
        num_envs=args.num_envs,
        num_eval_envs=args.num_eval_envs,
        eval_episodes=args.eval_episodes,
        eval_freq=args.eval_freq,
        early_stop_threshold=args.early_stop_threshold,
        output_dir=args.output_dir,
        seed=args.seed,
    )
    train_specialist(args_dataclass)
