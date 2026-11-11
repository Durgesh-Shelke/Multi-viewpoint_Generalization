"""
Teacher Specialist Training Harness for Phase 10 OmniView.
Trains/adapts all 8 anchor viewpoint specialists using:
1. 3-Frame Temporal Stacking (12-channel RGB-D)
2. 33-dim State vector with Camera Extrinsics Conditioning [x_rel, y_rel, z_rel, dist]
3. Gaussian calibration noise (sigma = 0.02m)
4. Warm-start initialization from pre-trained Phase 6 specialists
5. PPO optimization with early stopping at >=96.0% success
"""

import os
import sys
sys.path.append("/home/maniskill")
import json
import time
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import mani_skill.envs
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper, FlattenActionSpaceWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from phase_10_omniview_nextgen.models_nextgen import OmniViewAgent
from phase_10_omniview_nextgen.wrappers import TemporalExtrinsicWrapper

VIEWPOINTS_JSON = "/home/maniskill/Multi-viewpoint_Generalization/phase_1_to_3/sequential_viewpoint_training/viewpoints.json"
TEACHERS_DIR = "/home/maniskill/Multi-viewpoint_Generalization/phase_10_omniview_nextgen/teachers"
PHASE6_DIR = "/home/maniskill/Multi-viewpoint_Generalization/phase_6_viewpoint_specialists/checkpoints"

CORE_TEACHERS = [17, 0, 28, 18, 27, 20, 19, 6]


def create_env(vpid: int, num_envs: int, vp_dict: dict, noise_std: float = 0.02, device: str = "cuda"):
    meta = vp_dict[vpid]
    cam_eye = meta["world_eye"]
    cam_target = [-0.1, 0.0, 0.1]

    env_kwargs = dict(
        obs_mode="rgbd",
        render_mode="all",
        sim_backend="physx_cuda",
        sensor_cam_eye_pos=cam_eye,
        sensor_cam_target_pos=cam_target,
        control_mode="pd_ee_delta_pos",
        sensor_configs=dict(width=128, height=128),
    )

    env = gym.make("PickCube-v1", num_envs=num_envs, reconfiguration_freq=None, **env_kwargs)
    env = FlattenRGBDObservationWrapper(env, rgb=True, depth=True, state=True, sep_depth=True)
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    env = TemporalExtrinsicWrapper(env, cam_eye=cam_eye, frame_stack=3, noise_std=noise_std, device=device)
    return ManiSkillVectorEnv(env, num_envs, ignore_terminations=True, record_metrics=True)


def evaluate_policy(agent: nn.Module, eval_envs, num_episodes: int = 100):
    agent.eval()
    eval_obs, _ = eval_envs.reset()
    collected = 0
    successes = 0

    while collected < num_episodes:
        with torch.no_grad():
            act = agent.get_action(eval_obs["rgbd"], eval_obs["state"], deterministic=True)
        eval_obs, _, _, _, infos = eval_envs.step(act)

        if "final_info" in infos and "_final_info" in infos:
            mask = infos["_final_info"]
            ep_info = infos["final_info"].get("episode", {})
            succ_key = "success_once" if "success_once" in ep_info else "success"

            for env_i in range(len(mask)):
                if collected >= num_episodes:
                    break
                if mask[env_i]:
                    val = ep_info[succ_key][env_i]
                    succ = bool(val.item() if torch.is_tensor(val) else val)
                    if succ:
                        successes += 1
                    collected += 1

    return (successes / num_episodes) * 100.0


def train_single_teacher(
    vpid: int,
    vp_dict: dict,
    num_envs: int = 32,
    num_steps: int = 50,
    lr: float = 3e-4,
    total_timesteps: int = 5_000_000,
    device_str: str = "cuda",
):
    device = torch.device(device_str)
    batch_size = num_envs * num_steps
    max_iterations = total_timesteps // batch_size
    best_ckpt_path = os.path.join(TEACHERS_DIR, f"specialist_vp_{vpid:02d}_best.pt")
    meta_json_path = os.path.join(TEACHERS_DIR, f"specialist_vp_{vpid:02d}_meta.json")

    # Check if already trained for full 5M steps
    if os.path.exists(meta_json_path) and os.path.exists(best_ckpt_path):
        with open(meta_json_path) as f:
            m = json.load(f)
        if m.get("steps", 0) >= total_timesteps:
            print(f"[VP {vpid:02d}] Already completed all {total_timesteps:,} steps! Best success: {m.get('best_eval_success', 0):.1f}%. Skipping.")
            return m.get("best_eval_success", 0)

    print("\n" + "=" * 80)
    print(f" TRAINING PHASE 10 TEACHER FROM SCRATCH: VIEWPOINT {vpid:02d}")
    print(f" Azimuth: {vp_dict[vpid]['azimuth_deg']:+6.1f}° | Elevation: {vp_dict[vpid]['elevation_deg']:4.1f}°")
    print(f" Input: 12-channel Temporal RGB-D + 33-dim State (Extrinsics Conditioned)")
    print(f" Full Budget: Exactly {total_timesteps:,} steps ({max_iterations} iterations, NO EARLY STOPPING)")
    print("=" * 80)

    # Initialize agent strictly from scratch with orthogonal weights
    agent = OmniViewAgent(action_dim=4, state_dim=33).to(device)

    optimizer = optim.Adam(agent.parameters(), lr=lr, eps=1e-5)

    # Envs
    train_envs = create_env(vpid, num_envs, vp_dict, noise_std=0.02, device=device_str)
    eval_envs = create_env(vpid, 50, vp_dict, noise_std=0.0, device=device_str)

    # Storage
    batch_size = num_envs * num_steps
    obs_rgbd_buf = torch.zeros((num_steps, num_envs, 12, 128, 128), dtype=torch.float32, device=device)
    obs_state_buf = torch.zeros((num_steps, num_envs, 33), dtype=torch.float32, device=device)
    actions_buf = torch.zeros((num_steps, num_envs, 4), dtype=torch.float32, device=device)
    logprobs_buf = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)
    rewards_buf = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)
    dones_buf = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)
    values_buf = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)

    # Initial evaluation
    initial_acc = evaluate_policy(agent, eval_envs, num_episodes=50)
    print(f"[Init Eval] Random Scratch Accuracy: {initial_acc:.1f}%")
    best_acc = initial_acc

    obs, _ = train_envs.reset()
    global_step = 0
    t0 = time.time()

    for iteration in range(1, max_iterations + 1):
        # Linear learning rate annealing
        frac = 1.0 - (iteration - 1.0) / max_iterations
        optimizer.param_groups[0]["lr"] = frac * lr

        agent.eval()
        for step in range(num_steps):
            global_step += num_envs
            obs_rgbd_buf[step] = obs["rgbd"]
            obs_state_buf[step] = obs["state"]

            with torch.no_grad():
                act, logprob, _, val = agent.get_action_and_value(obs["rgbd"], obs["state"])

            actions_buf[step] = act
            logprobs_buf[step] = logprob
            values_buf[step] = val

            obs, rew, term, trunc, infos = train_envs.step(act)
            rewards_buf[step] = rew
            dones_buf[step] = (term | trunc).float()

        # Compute GAE
        with torch.no_grad():
            next_val = agent.get_value(obs["rgbd"], obs["state"])
            advantages = torch.zeros_like(rewards_buf)
            lastgaelam = 0
            gamma = 0.99
            gae_lambda = 0.95
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - dones_buf[t]
                    nextvalues = next_val
                else:
                    nextnonterminal = 1.0 - dones_buf[t + 1]
                    nextvalues = values_buf[t + 1]
                delta = rewards_buf[t] + gamma * nextvalues * nextnonterminal - values_buf[t]
                advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values_buf

        # Flatten
        b_rgbd = obs_rgbd_buf.reshape(-1, 12, 128, 128)
        b_state = obs_state_buf.reshape(-1, 33)
        b_actions = actions_buf.reshape(-1, 4)
        b_logprobs = logprobs_buf.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values_buf.reshape(-1)

        # Normalize advantages
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        # PPO Update
        agent.train()
        num_minibatches = 4
        mini_size = batch_size // num_minibatches
        b_inds = np.arange(batch_size)

        for epoch in range(4):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, mini_size):
                end = start + mini_size
                minds = b_inds[start:end]

                _, newlogprob, entropy, newval = agent.get_action_and_value(
                    b_rgbd[minds], b_state[minds], b_actions[minds]
                )

                logratio = newlogprob - b_logprobs[minds]
                ratio = torch.exp(logratio)

                # Policy Loss
                mb_advantages = b_advantages[minds]
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1.0 - 0.2, 1.0 + 0.2)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value Loss
                v_clipped = b_values[minds] + torch.clamp(newval - b_values[minds], -0.2, 0.2)
                v_loss1 = (newval - b_returns[minds]) ** 2
                v_loss2 = (v_clipped - b_returns[minds]) ** 2
                v_loss = 0.5 * torch.max(v_loss1, v_loss2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss + 0.5 * v_loss - 0.01 * entropy_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                optimizer.step()

        # Periodic Evaluation
        if iteration % 10 == 0 or iteration == max_iterations:
            eval_acc = evaluate_policy(agent, eval_envs, num_episodes=50)
            elapsed = time.time() - t0
            fps = global_step / elapsed
            print(f"Iter {iteration:03d} | Steps: {global_step:,} | Eval Success: {eval_acc:5.1f}% (Best: {best_acc:5.1f}%) | SPS: {fps:.0f}")

            if eval_acc > best_acc:
                best_acc = eval_acc
                torch.save({"model_state": agent.state_dict(), "best_eval_success": best_acc}, best_ckpt_path)
                with open(meta_json_path, "w") as f:
                    json.dump({"vpid": vpid, "best_eval_success": best_acc, "steps": global_step}, f)

    # Save final model after full 5M steps
    final_ckpt_path = os.path.join(TEACHERS_DIR, f"specialist_vp_{vpid:02d}_final.pt")
    torch.save({"model_state": agent.state_dict(), "final_eval_success": eval_acc, "best_eval_success": best_acc, "steps": global_step}, final_ckpt_path)
    print(f"[Completed 5M Steps] VP {vpid:02d} finished full {global_step:,} steps! Best Eval Success: {best_acc:.1f}%")

    train_envs.close()
    eval_envs.close()
    return best_acc


def main():
    os.makedirs(TEACHERS_DIR, exist_ok=True)
    with open(VIEWPOINTS_JSON) as f:
        vp_dict = {v["id"]: v for v in json.load(f)}

    print(f"\nTraining all {len(CORE_TEACHERS)} Phase 10 Teachers for EXACTLY 5,000,000 steps each (NO early termination)...")
    for vpid in CORE_TEACHERS:
        train_single_teacher(vpid, vp_dict, total_timesteps=5_000_000)

    print("\n" + "=" * 80)
    print(" ALL PHASE 10 TEACHER SPECIALISTS COMPLETED!")
    print("=" * 80)


if __name__ == "__main__":
    main()
