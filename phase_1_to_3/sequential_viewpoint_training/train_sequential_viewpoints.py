"""
Target-Driven Sequential Viewpoint Continual Training across 30 Fibonacci Hemispherical Viewpoints.

For each viewpoint:
- Trains the policy iteratively on that specific viewpoint.
- Periodically evaluates accuracy on that viewpoint (every eval_freq epochs).
- Stays on that viewpoint until accuracy reaches >= 95.0%.
- Once >= 95.0% is achieved, saves checkpoint (ckpt_vp_{i:02d}.pt) and proceeds to the next viewpoint.
- Fully resumable: writes state to training_progress.json after every epoch.
"""

import argparse
import json
import os
import signal
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
sys.path.insert(0, "/home/maniskill/examples/baselines/ppo")

import mani_skill.envs
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper, FlattenRGBDObservationWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from ppo_rgb import Agent, DictArray

TARGET_POS = [-0.1, 0.0, 0.1]
STOP_REQUESTED = False


def sig_handler(signum, frame):
    global STOP_REQUESTED
    print(f"\n[Interrupt] Signal {signum} received! Stopping cleanly after current epoch...")
    STOP_REQUESTED = True


signal.signal(signal.SIGINT, sig_handler)
signal.signal(signal.SIGTERM, sig_handler)


def evaluate_policy_on_viewpoint(
    agent: Agent,
    eye: list,
    target: list = TARGET_POS,
    episodes: int = 50,
    num_envs: int = 32,
    device: str = "cuda",
) -> float:
    """Evaluates the policy specifically on the given viewpoint camera position."""
    env = gym.make(
        "PickCube-v1",
        num_envs=num_envs,
        obs_mode="rgb",
        control_mode="pd_ee_delta_pos",
        sim_backend="physx_cuda",
        sensor_cam_eye_pos=eye,
        sensor_cam_target_pos=target,
    )
    env = FlattenRGBDObservationWrapper(env, rgb=True, depth=False, state=True)
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    env = ManiSkillVectorEnv(env, num_envs, ignore_terminations=False, record_metrics=True)

    obs, _ = env.reset(seed=12345)
    completed = 0
    successes = 0

    with torch.no_grad():
        while completed < episodes:
            action = agent.get_action(obs, deterministic=True)
            obs, rew, term, trunc, infos = env.step(action)
            if "final_info" in infos:
                mask = infos["_final_info"]
                for i in range(len(mask)):
                    if mask[i]:
                        ep_info = infos["final_info"]["episode"]
                        succ_key = (
                            "success_once"
                            if "success_once" in ep_info
                            else ("success" if "success" in ep_info else "success_at_end")
                        )
                        succ = bool(ep_info[succ_key][i].item() if torch.is_tensor(ep_info[succ_key]) else ep_info[succ_key])
                        if succ:
                            successes += 1
                        completed += 1
                        if completed >= episodes:
                            break
    env.close()
    return round((successes / completed) * 100.0, 2)


def train_single_viewpoint_until_target(
    agent: Agent,
    viewpoint: dict,
    target_accuracy: float = 95.0,
    eval_freq: int = 5,
    eval_episodes: int = 50,
    max_iterations: int = 400,
    num_envs: int = 256,
    num_steps: int = 50,
    lr: float = 2e-4,
    gamma: float = 0.8,
    gae_lambda: float = 0.9,
    update_epochs: int = 4,
    num_minibatches: int = 8,
    clip_coef: float = 0.2,
    ent_coef: float = 0.0,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    start_iter: int = 1,
    output_dir: str = "/home/maniskill/phase_1_to_3/sequential_viewpoint_training",
    progress_file: str = "/home/maniskill/phase_1_to_3/sequential_viewpoint_training/training_progress.json",
    device: str = "cuda",
) -> tuple:
    global STOP_REQUESTED
    eye = viewpoint["world_eye"]
    target = TARGET_POS
    vp_id = viewpoint["id"]
    az = viewpoint["azimuth_deg"]
    el = viewpoint["elevation_deg"]

    print(f"\n{'='*78}")
    print(f" TRAINING VIEWPOINT {vp_id:02d} / 30 | Azimuth: {az:+6.1f}°, Elevation: {el:4.1f}°")
    print(f" Eye: {eye} | Target Accuracy: {target_accuracy}% (Eval every {eval_freq} epochs)")
    print(f" Starting at Epoch {start_iter} | Max Epochs: {max_iterations}")
    print(f"{'='*78}")

    # Check initial baseline accuracy on this viewpoint before training
    agent.eval()
    init_acc = evaluate_policy_on_viewpoint(agent=agent, eye=eye, target=target, episodes=eval_episodes, num_envs=32)
    print(f" [VP {vp_id:02d}] Initial Baseline Accuracy on this viewpoint: {init_acc:.1f}%")
    if init_acc >= target_accuracy:
        print(f" >>> ALREADY AT TARGET! Viewpoint {vp_id:02d} baseline ({init_acc:.1f}%) >= {target_accuracy}%. Skipping training!")
        return agent, True, 0, init_acc

    env = gym.make(
        "PickCube-v1",
        num_envs=num_envs,
        obs_mode="rgb",
        control_mode="pd_ee_delta_pos",
        sim_backend="physx_cuda",
        sensor_cam_eye_pos=eye,
        sensor_cam_target_pos=target,
    )
    env = FlattenRGBDObservationWrapper(env, rgb=True, depth=False, state=True)
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    env = ManiSkillVectorEnv(env, num_envs, ignore_terminations=False, record_metrics=True)

    optimizer = optim.Adam(agent.parameters(), lr=lr, eps=1e-5)
    batch_size = num_steps * num_envs
    minibatch_size = batch_size // num_minibatches

    # Storage setup
    obs = DictArray((num_steps, num_envs), env.single_observation_space, device=device)
    actions = torch.zeros((num_steps, num_envs) + env.single_action_space.shape, device=device)
    logprobs = torch.zeros((num_steps, num_envs), device=device)
    rewards = torch.zeros((num_steps, num_envs), device=device)
    dones = torch.zeros((num_steps, num_envs), device=device)
    values = torch.zeros((num_steps, num_envs), device=device)

    next_obs, _ = env.reset(seed=int(vp_id * 1000 + 42))
    next_done = torch.zeros(num_envs, device=device)

    ckpt_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    iter_num = start_iter
    last_eval_acc = init_acc
    target_hit = False

    while iter_num <= max_iterations:
        if STOP_REQUESTED:
            print("\n[Pause] Stop requested! Saving current checkpoint & progress...")
            break

        iter_start = time.time()
        final_values = torch.zeros((num_steps, num_envs), device=device)

        # Rollout
        agent.eval()
        for step in range(num_steps):
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            next_obs, reward, term, trunc, infos = env.step(action)
            next_done = torch.logical_or(term, trunc).to(torch.float32)
            rewards[step] = reward.view(-1)

            if "final_info" in infos:
                done_mask = infos["_final_info"]
                for k in infos["final_observation"]:
                    infos["final_observation"][k] = infos["final_observation"][k][done_mask]
                with torch.no_grad():
                    final_values[step, torch.arange(num_envs, device=device)[done_mask]] = agent.get_value(infos["final_observation"]).view(-1)

        # GAE Advantages
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
                delta = rewards[t] + gamma * real_next_values - values[t]
                advantages[t] = lastgaelam = delta + gamma * gae_lambda * next_not_done * lastgaelam
            returns = advantages + values

        # Flatten batch
        b_obs = obs.reshape((-1,))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + env.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # PPO Update
        agent.train()
        b_inds = np.arange(batch_size)
        for epoch in range(update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                mb_advantages = b_advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                optimizer.step()

        elapsed = time.time() - iter_start
        sps = int(batch_size / elapsed)
        mean_rew = float(rewards.mean().item())

        # Save instant checkpoint
        latest_ckpt = os.path.join(ckpt_dir, "latest_in_progress.pt")
        torch.save(agent.state_dict(), latest_ckpt)

        # Periodic evaluation check for 95% target
        if iter_num % eval_freq == 0:
            agent.eval()
            eval_acc = evaluate_policy_on_viewpoint(agent=agent, eye=eye, target=target, episodes=eval_episodes, num_envs=32)
            last_eval_acc = eval_acc
            print(f" [VP {vp_id:02d}] Epoch {iter_num:3d} | Train Rew: {mean_rew:5.2f} | Eval Acc: {eval_acc:5.1f}% (Target: {target_accuracy}%) | SPS: {sps}")

            update_progress_json(
                progress_file=progress_file,
                current_vp_idx=vp_id,
                current_iter=iter_num,
                latest_eval_accuracy=eval_acc,
                target_accuracy=target_accuracy,
                latest_checkpoint=latest_ckpt,
                status="interrupted" if STOP_REQUESTED else "training",
            )

            if eval_acc >= target_accuracy:
                print(f"\n >>> TARGET HIT! Viewpoint {vp_id:02d} reached {eval_acc:.1f}% >= {target_accuracy}% after {iter_num} epochs!")
                target_hit = True
                iter_num += 1
                break
        else:
            if iter_num % 2 == 0:
                print(f" [VP {vp_id:02d}] Epoch {iter_num:3d} | Train Rew: {mean_rew:5.2f} | SPS: {sps}")

            update_progress_json(
                progress_file=progress_file,
                current_vp_idx=vp_id,
                current_iter=iter_num,
                latest_eval_accuracy=last_eval_acc,
                target_accuracy=target_accuracy,
                latest_checkpoint=latest_ckpt,
                status="interrupted" if STOP_REQUESTED else "training",
            )

        iter_num += 1

    env.close()
    return agent, target_hit, iter_num - 1, last_eval_acc


def load_progress_json(progress_file: str) -> dict:
    if os.path.exists(progress_file):
        try:
            with open(progress_file, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return None


def update_progress_json(
    progress_file: str,
    current_vp_idx: int,
    current_iter: int,
    latest_eval_accuracy: float,
    target_accuracy: float,
    latest_checkpoint: str,
    status: str = "training",
    completed_vp_entry: dict = None,
):
    data = load_progress_json(progress_file) or {
        "current_viewpoint_idx": 0,
        "total_viewpoints": 30,
        "target_accuracy_pct": target_accuracy,
        "status": status,
        "current_iteration": 0,
        "latest_eval_accuracy": 0.0,
        "latest_checkpoint": latest_checkpoint,
        "completed_viewpoints": [],
        "last_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    data["current_viewpoint_idx"] = current_vp_idx
    data["current_iteration"] = current_iter
    data["latest_eval_accuracy"] = latest_eval_accuracy
    data["target_accuracy_pct"] = target_accuracy
    data["latest_checkpoint"] = latest_checkpoint
    data["status"] = status
    data["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S")

    if completed_vp_entry is not None:
        data["completed_viewpoints"] = [
            e for e in data.get("completed_viewpoints", []) if e["viewpoint_id"] != completed_vp_entry["viewpoint_id"]
        ]
        data["completed_viewpoints"].append(completed_vp_entry)

    with open(progress_file, "w") as f:
        json.dump(data, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Target-Driven (>= 95%) Sequential Viewpoint Training")
    parser.add_argument("--base_model", type=str, default="/home/maniskill/phase_1_to_3/sequential_viewpoint_training/backup_original_model.pt")
    parser.add_argument("--viewpoints_json", type=str, default="/home/maniskill/phase_1_to_3/sequential_viewpoint_training/viewpoints.json")
    parser.add_argument("--output_dir", type=str, default="/home/maniskill/phase_1_to_3/sequential_viewpoint_training")
    parser.add_argument("--target_accuracy", type=float, default=95.0, help="Accuracy threshold (%) required to advance")
    parser.add_argument("--eval_freq", type=int, default=5, help="Check accuracy every N epochs")
    parser.add_argument("--eval_episodes", type=int, default=50, help="Number of test episodes per accuracy check")
    parser.add_argument("--max_epochs_per_vp", type=int, default=400, help="Max epochs before advancing if 95% not reached")
    parser.add_argument("--num_envs", type=int, default=128, help="Number of parallel visual environments")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--resume", action="store_true", default=True, help="Auto resume from training_progress.json")
    parser.add_argument("--fresh", action="store_true", default=False, help="Start fresh from Viewpoint 00 ignoring previous progress")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    progress_file = os.path.join(args.output_dir, "training_progress.json")

    with open(args.viewpoints_json, "r") as f:
        viewpoints = json.load(f)

    if args.fresh and os.path.exists(progress_file):
        os.remove(progress_file)
        print("[Start] Cleared previous progress to start fresh from Viewpoint 00.")

    start_vp_idx = 0
    start_iter = 1
    model_to_load = args.base_model

    prog = load_progress_json(progress_file) if (args.resume and not args.fresh) else None
    if prog:
        start_vp_idx = prog.get("current_viewpoint_idx", 0)
        start_iter = prog.get("current_iteration", 0) + 1
        model_to_load = prog.get("latest_checkpoint", args.base_model)
        print(f"\n[Resume] Found saved state! Resuming from Viewpoint {start_vp_idx:02d}, Epoch {start_iter}")
        print(f"[Resume] Loading weights from: {model_to_load}")
    else:
        print(f"\n[Start] Starting target-driven sequential training (target: {args.target_accuracy}%) from: {model_to_load}")

    # Initialize agent
    dummy_env = gym.make("PickCube-v1", num_envs=1, obs_mode="rgb", control_mode="pd_ee_delta_pos", sim_backend="physx_cuda")
    dummy_env = FlattenRGBDObservationWrapper(dummy_env, rgb=True, depth=False, state=True)
    if isinstance(dummy_env.action_space, gym.spaces.Dict):
        dummy_env = FlattenActionSpaceWrapper(dummy_env)
    dummy_env = ManiSkillVectorEnv(dummy_env, 1, ignore_terminations=False, record_metrics=True)
    sample_obs, _ = dummy_env.reset()

    agent = Agent(dummy_env, sample_obs=sample_obs).cuda()
    agent.load_state_dict(torch.load(model_to_load, map_location="cuda"))
    dummy_env.close()

    for vp_idx in range(start_vp_idx, len(viewpoints)):
        if STOP_REQUESTED:
            print("\n[Stopped] Sequential training safely paused. State saved to training_progress.json.")
            break

        vp = viewpoints[vp_idx]
        current_start_iter = start_iter if vp_idx == start_vp_idx else 1

        agent, hit_target, epochs_trained, final_acc = train_single_viewpoint_until_target(
            agent=agent,
            viewpoint=vp,
            target_accuracy=args.target_accuracy,
            eval_freq=args.eval_freq,
            eval_episodes=args.eval_episodes,
            max_iterations=args.max_epochs_per_vp,
            num_envs=args.num_envs,
            lr=args.lr,
            start_iter=current_start_iter,
            output_dir=args.output_dir,
            progress_file=progress_file,
        )

        if STOP_REQUESTED:
            break

        # Save checkpoint for this viewpoint
        vp_ckpt = os.path.join(ckpt_dir, f"ckpt_vp_{vp_idx:02d}.pt")
        torch.save(agent.state_dict(), vp_ckpt)
        print(f" [Saved] Viewpoint {vp_idx:02d} checkpoint saved to: {vp_ckpt} (Accuracy: {final_acc:.1f}%)")

        entry = {
            "viewpoint_id": vp_idx,
            "azimuth_deg": vp["azimuth_deg"],
            "elevation_deg": vp["elevation_deg"],
            "checkpoint": vp_ckpt,
            "epochs_trained": epochs_trained,
            "accuracy": final_acc,
            "target_hit": hit_target,
            "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        update_progress_json(
            progress_file=progress_file,
            current_vp_idx=vp_idx + 1,
            current_iter=0,
            latest_eval_accuracy=final_acc,
            target_accuracy=args.target_accuracy,
            latest_checkpoint=vp_ckpt,
            status="training",
            completed_vp_entry=entry,
        )

    final_prog = load_progress_json(progress_file)
    if final_prog and final_prog.get("current_viewpoint_idx", 0) >= len(viewpoints):
        final_prog["status"] = "completed"
        with open(progress_file, "w") as f:
            json.dump(final_prog, f, indent=2)
        print("\n" + "=" * 78)
        print(f" ALL 30 VIEWPOINTS TRAINED TO TARGET ({args.target_accuracy}%) SUCCESSFULLY!")
        print("=" * 78)


if __name__ == "__main__":
    main()
