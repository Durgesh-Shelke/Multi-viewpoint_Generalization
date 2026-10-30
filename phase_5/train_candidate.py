"""
Training Pipeline for Resolution and Asymmetric Sensor Fusion Ablation Study.
Supports Candidate A, B, C, D with full hyperparameter parity.
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
    print(f"\n[Interrupt] Signal {signum} received! Stopping cleanly...")
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
    torch.save({
        "iteration": iteration,
        "global_step": global_step,
        "best_eval_success": best_eval_success,
        "agent_state": agent.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "history_data": history_data,
        "elapsed_seconds": elapsed_seconds,
        "completed": completed,
    }, tmp_path)
    os.replace(tmp_path, ckpt_path)


def save_history_csv(output_dir: str, candidate: str, history_data: list):
    if not history_data:
        return
    import csv
    eval_hist_dir = os.path.join(output_dir, "eval_results")
    os.makedirs(eval_hist_dir, exist_ok=True)
    csv_file = os.path.join(eval_hist_dir, f"candidate_{candidate}_history.csv")
    tmp_file = csv_file + ".tmp"
    keys = history_data[0].keys()
    with open(tmp_file, "w", newline="") as f:
        dict_writer = csv.DictWriter(f, keys)
        dict_writer.writeheader()
        dict_writer.writerows(history_data)
    os.replace(tmp_file, csv_file)


class DictArray(object):
    def __init__(self, buffer_shape, element_space, data_dict=None, device=None):
        self.buffer_shape = buffer_shape
        if data_dict:
            self.data = data_dict
        else:
            assert isinstance(element_space, gym.spaces.dict.Dict)
            self.data = {}
            for k, v in element_space.items():
                if isinstance(v, gym.spaces.dict.Dict):
                    self.data[k] = DictArray(buffer_shape, v, device=device)
                else:
                    dtype = (
                        torch.float32
                        if v.dtype in (np.float32, np.float64)
                        else (
                            torch.uint8
                            if v.dtype == np.uint8
                            else (
                                torch.int16
                                if v.dtype == np.int16
                                else (torch.int32 if v.dtype == np.int32 else torch.int64)
                            )
                        )
                    )
                    self.data[k] = torch.zeros(buffer_shape + v.shape, dtype=dtype, device=device)

    def __getitem__(self, index):
        if isinstance(index, str):
            return self.data[index]
        return {k: v[index] for k, v in self.data.items()}

    def __setitem__(self, index, value):
        if isinstance(index, str):
            self.data[index] = value
        else:
            for k, v in value.items():
                self.data[k][index] = v

    def reshape(self, shape):
        t = len(self.buffer_shape)
        new_dict = {}
        for k, v in self.data.items():
            if isinstance(v, DictArray):
                new_dict[k] = v.reshape(shape)
            else:
                new_dict[k] = v.reshape(shape + v.shape[t:])
        new_buffer_shape = next(iter(new_dict.values())).shape[: len(shape)]
        return DictArray(new_buffer_shape, None, data_dict=new_dict)


def train_single_candidate(
    candidate: str = "A",
    total_timesteps: int = 3000000,
    num_envs: int = 256,
    num_eval_envs: int = 8,
    num_minibatches: Optional[int] = None,
    learning_rate: float = 3e-4,
    target_kl: float = 0.2,
    eval_freq: int = 25,
    output_dir: str = "/home/maniskill/phase_5",
    seed: int = 42,
    device_str: str = "cuda",
    resume: bool = True,
) -> dict:
    global STOP_REQUESTED
    candidate = candidate.upper()
    res = 128 if candidate == "A" else 256
    if num_minibatches is None:
        num_minibatches = 8 if candidate == "A" else 16

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(device_str)

    num_steps = 50
    batch_size = num_envs * num_steps
    minibatch_size = batch_size // num_minibatches
    num_iterations = int(np.ceil(total_timesteps / batch_size))

    ckpt_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    tb_dir = os.path.join(output_dir, "tb_logs", f"candidate_{candidate}")
    os.makedirs(tb_dir, exist_ok=True)
    writer = SummaryWriter(tb_dir)

    print("\n" + "=" * 78)
    print(f" TRAINING CANDIDATE {candidate} ({res}x{res})")
    print(f" Modality / Architecture: Candidate {candidate}")
    print(f" Resolution:              {res} x {res}")
    print(f" Total Timesteps:         {total_timesteps:,} ({num_iterations} iterations)")
    print(f" Parallel Instances:      {num_envs} train | {num_eval_envs} eval")
    print(f" Minibatch Size:          {minibatch_size} (16 minibatches)")
    print("=" * 78 + "\n")

    cam_eye = [0.3, 0.0, 0.6]
    cam_target = [-0.1, 0.0, 0.1]
    env_kwargs = dict(
        obs_mode="rgbd",
        render_mode="all",
        sim_backend="physx_cuda",
        sensor_cam_eye_pos=cam_eye,
        sensor_cam_target_pos=cam_target,
        control_mode="pd_ee_delta_pos",
        sensor_configs=dict(width=res, height=res),
    )

    eval_envs = gym.make("PickCube-v1", num_envs=num_eval_envs, reconfiguration_freq=None, **env_kwargs)
    envs = gym.make("PickCube-v1", num_envs=num_envs, reconfiguration_freq=None, **env_kwargs)

    envs = FlattenRGBDObservationWrapper(envs, rgb=True, depth=True, state=True, sep_depth=True)
    eval_envs = FlattenRGBDObservationWrapper(eval_envs, rgb=True, depth=True, state=True, sep_depth=True)

    if isinstance(envs.action_space, gym.spaces.Dict):
        envs = FlattenActionSpaceWrapper(envs)
        eval_envs = FlattenActionSpaceWrapper(eval_envs)

    envs = ManiSkillVectorEnv(envs, num_envs, ignore_terminations=False, record_metrics=True)
    eval_envs = ManiSkillVectorEnv(eval_envs, num_eval_envs, ignore_terminations=True, record_metrics=True)

    sample_obs, _ = envs.reset(seed=seed)
    eval_obs, _ = eval_envs.reset(seed=seed)
    agent = UnifiedAblationAgent(envs, sample_obs=sample_obs, candidate=candidate).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=learning_rate, eps=1e-5)

    # Rollout storage
    obs_storage = DictArray((num_steps, num_envs), envs.single_observation_space, device=device)
    actions = torch.zeros((num_steps, num_envs) + envs.single_action_space.shape, device=device)
    logprobs = torch.zeros((num_steps, num_envs), device=device)
    rewards = torch.zeros((num_steps, num_envs), device=device)
    dones = torch.zeros((num_steps, num_envs), device=device)
    values = torch.zeros((num_steps, num_envs), device=device)

    global_step = 0
    best_eval_success = 0.0
    sps_list = []
    history_data = []
    start_iteration = 1
    loaded_elapsed = 0.0

    latest_ckpt_path = os.path.join(ckpt_dir, f"candidate_{candidate}_latest.pt")
    final_ckpt_path = os.path.join(ckpt_dir, f"candidate_{candidate}.pt")
    pause_file = os.path.join(output_dir, ".pause")

    if resume and os.path.exists(latest_ckpt_path):
        print(f"[RESUME] Found existing checkpoint: {latest_ckpt_path}")
        try:
            ckpt = torch.load(latest_ckpt_path, map_location=device)
            agent.load_state_dict(ckpt["agent_state"])
            optimizer.load_state_dict(ckpt["optimizer_state"])
            start_iteration = ckpt["iteration"] + 1
            global_step = ckpt["global_step"]
            best_eval_success = ckpt.get("best_eval_success", 0.0)
            history_data = ckpt.get("history_data", [])
            loaded_elapsed = ckpt.get("elapsed_seconds", 0.0)
            print(f"[RESUME] Successfully restored state! Resuming from iteration {start_iteration}/{num_iterations} (step {global_step:,}), best eval: {best_eval_success:.1f}%")
        except Exception as e:
            print(f"[RESUME WARNING] Failed to load checkpoint {latest_ckpt_path}: {e}. Starting fresh.")

    start_time = time.time() - loaded_elapsed

    next_obs, _ = envs.reset(seed=seed)
    next_done = torch.zeros(num_envs, device=device)

    for iteration in range(start_iteration, num_iterations + 1):
        if STOP_REQUESTED:
            print(f"\n[Interrupt] Stop requested. Checkpointing Candidate {candidate} at iteration {iteration - 1}...")
            save_checkpoint(latest_ckpt_path, agent, optimizer, iteration - 1, global_step, best_eval_success, history_data, time.time() - start_time)
            save_history_csv(output_dir, candidate, history_data)
            break

        # Check for pause flag
        if os.path.exists(pause_file):
            print(f"\n[PAUSED] Detected pause flag: {pause_file}")
            save_checkpoint(latest_ckpt_path, agent, optimizer, iteration - 1, global_step, best_eval_success, history_data, time.time() - start_time)
            save_history_csv(output_dir, candidate, history_data)
            print(f"[PAUSED] Saved state at iteration {iteration - 1} (step {global_step:,}). Pausing execution (GPU idle)...")
            torch.cuda.empty_cache()
            while os.path.exists(pause_file) and not STOP_REQUESTED:
                time.sleep(1)
            if STOP_REQUESTED:
                print(f"\n[Interrupt] Stop requested while paused.")
                break
            print(f"[RESUMED] Pause flag removed. Resuming Candidate {candidate} at iteration {iteration} (step {global_step:,})!\n")

        # Periodic Evaluation
        if iteration % eval_freq == 0 or iteration == start_iteration:
            eval_obs, _ = eval_envs.reset()
            eval_successes = 0
            eval_eps = 0
            t_eval = time.time()
            with torch.no_grad():
                for _ in range(50):
                    act = agent.get_action(eval_obs, deterministic=True)
                    eval_obs, _, _, _, eval_infos = eval_envs.step(act)
                    if "final_info" in eval_infos:
                        mask = eval_infos["_final_info"]
                        for i in range(len(mask)):
                            if mask[i]:
                                ep_info = eval_infos["final_info"]["episode"]
                                succ_key = "success_once" if "success_once" in ep_info else "success"
                                succ = bool(ep_info[succ_key][i].item() if torch.is_tensor(ep_info[succ_key]) else ep_info[succ_key])
                                if succ:
                                    eval_successes += 1
                                eval_eps += 1
            eval_rate = (eval_successes / eval_eps) * 100.0 if eval_eps > 0 else 0.0
            if eval_rate >= best_eval_success and eval_rate > 0.0:
                best_eval_success = eval_rate
                best_ckpt_path = os.path.join(ckpt_dir, f"candidate_{candidate}_best.pt")
                torch.save(agent.state_dict(), best_ckpt_path)
                torch.save(agent.state_dict(), final_ckpt_path)
            else:
                best_eval_success = max(best_eval_success, eval_rate)
            writer.add_scalar("eval/success_rate", eval_rate, global_step)
            print(f" >>> [Candidate {candidate} | Step {global_step:7,d}] Eval Success: {eval_rate:5.1f}% (best: {best_eval_success:.1f}%) in {time.time()-t_eval:.1f}s")

            cur_sps = int(global_step / (time.time() - start_time)) if (time.time() - start_time) > 0 else 0
            cur_rew = round(float(rewards.mean().item()), 3) if global_step > 0 else 0.0
            history_data.append({
                "iteration": iteration,
                "global_step": global_step,
                "eval_success_rate": round(eval_rate, 2),
                "best_eval_success": round(best_eval_success, 2),
                "mean_reward": cur_rew,
                "sps": cur_sps,
                "elapsed_seconds": round(time.time() - start_time, 2),
            })
            eval_hist_dir = os.path.join(output_dir, "eval_results")
            os.makedirs(eval_hist_dir, exist_ok=True)
            with open(os.path.join(eval_hist_dir, f"candidate_{candidate}_history.json"), "w") as f:
                json.dump(history_data, f, indent=2)

            save_checkpoint(latest_ckpt_path, agent, optimizer, iteration, global_step, best_eval_success, history_data, time.time() - start_time)
            save_history_csv(output_dir, candidate, history_data)

        # Rollout collection
        final_values = torch.zeros((num_steps, num_envs), device=device)
        for step in range(num_steps):
            global_step += num_envs
            obs_storage[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            next_obs, reward, terminations, truncations, infos = envs.step(action)
            next_done = torch.logical_or(terminations, truncations).to(torch.float32)
            rewards[step] = reward.view(-1)

            if "final_info" in infos:
                done_mask = infos["_final_info"]
                for k in infos["final_observation"]:
                    infos["final_observation"][k] = infos["final_observation"][k][done_mask]
                with torch.no_grad():
                    final_values[step, torch.arange(num_envs, device=device)[done_mask]] = agent.get_value(infos["final_observation"]).view(-1)

        # Advantage computation (GAE with proper truncation bootstrapping)
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    next_not_done = 1.0 - next_done
                    nextvalues = next_value
                else:
                    next_not_done = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                real_next_values = next_not_done * nextvalues + final_values[t]
                delta = rewards[t] + 0.8 * real_next_values - values[t]
                advantages[t] = lastgaelam = delta + 0.8 * 0.9 * next_not_done * lastgaelam
            returns = advantages + values

        # Flatten batch
        b_obs = obs_storage.reshape((-1,))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing Policy & Value Network with Target KL safeguard
        b_inds = np.arange(batch_size)
        for epoch in range(4):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()

                if target_kl is not None and approx_kl > target_kl:
                    break

                mb_advantages = b_advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - 0.2, 1 + 0.2)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                loss = pg_loss + 0.5 * v_loss - 0.0 * entropy.mean()

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                optimizer.step()

            if target_kl is not None and approx_kl > target_kl:
                break

        elapsed = time.time() - start_time
        sps = int(global_step / elapsed) if elapsed > 0 else 0
        sps_list.append(sps)
        mean_rew = float(rewards.mean().item())

        if iteration % 10 == 0 or iteration == 1:
            print(f" [Candidate {candidate}] Iter {iteration:3d}/{num_iterations} | Step: {global_step:7,d} | Rew: {mean_rew:5.2f} | SPS: {sps} | Best Eval: {best_eval_success:.1f}%")

    total_time = time.time() - start_time
    peak_vram_mb = round(torch.cuda.max_memory_allocated() / 1e6, 2)
    avg_sps = int(np.mean(sps_list[5:])) if len(sps_list) > 5 else sps
    is_completed = (global_step >= total_timesteps or iteration >= num_iterations)

    # Save resume checkpoint and full history
    save_checkpoint(
        latest_ckpt_path,
        agent,
        optimizer,
        num_iterations if is_completed else iteration - 1,
        global_step,
        best_eval_success,
        history_data,
        total_time,
        completed=is_completed,
    )
    save_history_csv(output_dir, candidate, history_data)

    if is_completed:
        # Save final model weights (preferring peak best_model if available)
        best_ckpt_path = os.path.join(ckpt_dir, f"candidate_{candidate}_best.pt")
        if os.path.exists(best_ckpt_path) and best_eval_success > 0.0:
            import shutil
            shutil.copy(best_ckpt_path, final_ckpt_path)
            print(f"[Best Model Loaded] Copied peak eval checkpoint ({best_eval_success:.1f}%) to {final_ckpt_path}")
        else:
            torch.save(agent.state_dict(), final_ckpt_path)
        metrics_path = os.path.join(ckpt_dir, f"candidate_{candidate}_train_metrics.json")
        metrics_data = {
            "candidate": candidate,
            "resolution": res,
            "checkpoint": final_ckpt_path,
            "total_timesteps": global_step,
            "completed": True,
            "total_walltime_seconds": round(total_time, 2),
            "avg_sps": avg_sps,
            "peak_vram_mb": peak_vram_mb,
            "best_eval_success": round(best_eval_success, 2),
        }
        with open(metrics_path, "w") as f:
            json.dump(metrics_data, f, indent=2)
        print(f"\n[Completed] Candidate {candidate} finished all {total_timesteps:,} steps!")
        print(f"[Saved] Final model weights: {final_ckpt_path}")
        print(f"[Saved] Training metrics: {metrics_path}")
    else:
        print(f"\n[Interrupted/Paused] Candidate {candidate} stopped at step {global_step:,} / {total_timesteps:,}.")
        print(f"[Saved] Resume state preserved at: {latest_ckpt_path}")

    print(f"  Total Time: {total_time/60:.1f} mins | Avg SPS: {avg_sps} | Peak VRAM: {peak_vram_mb} MB | Best Eval: {best_eval_success:.1f}%")

    envs.close()
    eval_envs.close()
    writer.close()

    del agent, optimizer, envs, eval_envs, obs_storage
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "candidate": candidate,
        "resolution": res,
        "checkpoint": final_ckpt_path,
        "total_timesteps": global_step,
        "completed": is_completed,
        "total_walltime_seconds": round(total_time, 2),
        "avg_sps": avg_sps,
        "peak_vram_mb": peak_vram_mb,
        "best_eval_success": round(best_eval_success, 2),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=str, default="A", choices=["A", "B", "C", "D", "E"])
    parser.add_argument("--total-timesteps", type=int, default=10000000)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--num-minibatches", type=int, default=None, help="Number of minibatches (default: 8 for 128x128, 16 for 256x256)")
    parser.add_argument("--target-kl", type=float, default=0.2, help="PPO target KL divergence early stopping threshold")
    parser.add_argument("--eval-freq", type=int, default=25)
    parser.add_argument("--output-dir", type=str, default="/home/maniskill/phase_5")
    parser.add_argument("--resume", action="store_true", default=True, help="Resume from existing checkpoint if available (default: True)")
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="Start training from scratch, ignoring checkpoints")
    args = parser.parse_args()

    result = train_single_candidate(
        candidate=args.candidate,
        total_timesteps=args.total_timesteps,
        num_envs=args.num_envs,
        num_minibatches=args.num_minibatches,
        target_kl=args.target_kl,
        eval_freq=args.eval_freq,
        output_dir=args.output_dir,
        resume=args.resume,
    )
    if not result.get("completed", False):
        sys.exit(130)
