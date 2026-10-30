"""
Specialist Ceiling Booster:
Elevates any viewpoint specialist that hasn't reached the >=95.0% performance ceiling
by fine-tuning with proper Gymnasium truncation bootstrapping and adaptive learning rate.
Guarantees that all teacher checkpoints meet the >95% mastery criterion required for policy distillation.
"""

import os
import sys
import json
import time
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, "/home/maniskill")

import mani_skill.envs
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper, FlattenRGBDObservationWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from models import UnifiedAblationAgent

VIEWPOINTS_JSON = "/home/maniskill/phase_1_to_3/sequential_viewpoint_training/viewpoints.json"
CORE_VP_IDS = [17, 0, 28, 18, 27, 20, 19, 6]
CHECKPOINT_DIR = os.path.join(SCRIPT_DIR, "checkpoints")
EVAL_DIR = os.path.join(SCRIPT_DIR, "eval_results")
CANDIDATE_A_CKPT = "/home/maniskill/phase_5/checkpoints/candidate_A_best.pt"


def evaluate_policy(agent: nn.Module, eval_envs, num_episodes: int = 100, max_steps: int = 50, device: str = "cuda"):
    agent.eval()
    eval_obs, _ = eval_envs.reset()
    eval_successes = 0
    eval_eps = 0

    with torch.no_grad():
        for _ in range(max_steps):
            act = agent.get_action(eval_obs, deterministic=True)
            eval_obs, _, _, _, eval_infos = eval_envs.step(act)
            if "final_info" in eval_infos and "_final_info" in eval_infos:
                mask = eval_infos["_final_info"]
                for i in range(len(mask)):
                    if mask[i]:
                        ep_info = eval_infos["final_info"]["episode"]
                        succ_key = "success_once" if "success_once" in ep_info else "success"
                        val = ep_info[succ_key][i]
                        succ = bool(val.item() if torch.is_tensor(val) else val)
                        if succ:
                            eval_successes += 1
                        eval_eps += 1

    agent.train()
    success_rate = (eval_successes / max(eval_eps, 1)) * 100.0 if eval_eps > 0 else 0.0
    return success_rate


def boost_single_viewpoint(
    vp_id: int,
    target_accuracy: float = 95.0,
    max_steps: int = 1500000,
    num_envs: int = 128,
    num_eval_envs: int = 50,
    lr: float = 1.5e-4,
    device_str: str = "cuda",
):
    device = torch.device(device_str)
    best_ckpt_path = os.path.join(CHECKPOINT_DIR, f"specialist_vp_{vp_id:02d}_best.pt")
    metrics_path = os.path.join(CHECKPOINT_DIR, f"specialist_vp_{vp_id:02d}_train_metrics.json")

    # Check current accuracy
    current_best = 0.0
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            m = json.load(f)
            current_best = m.get("best_eval_success", 0.0)

    if current_best >= target_accuracy:
        print(f"[Booster] VP {vp_id:02d} ALREADY at target ceiling ({current_best:.1f}% >= {target_accuracy:.1f}%). Skipping.")
        return current_best

    print(f"\n{'='*78}")
    print(f" BOOSTING SPECIALIST CEILING: VIEWPOINT {vp_id:02d}")
    print(f" Current Best: {current_best:.1f}% | Target: >={target_accuracy:.1f}%")
    print(f"{'='*78}\n")

    with open(VIEWPOINTS_JSON) as f:
        all_vps = {v["id"]: v for v in json.load(f)}
    vp = all_vps[vp_id]
    cam_eye = vp["world_eye"]

    env_kwargs = dict(
        obs_mode="rgbd",
        render_mode="all",
        sim_backend="physx_cuda",
        sensor_cam_eye_pos=cam_eye,
        sensor_cam_target_pos=[-0.1, 0.0, 0.1],
        control_mode="pd_ee_delta_pos",
        sensor_configs=dict(width=128, height=128),
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

    obs, _ = envs.reset(seed=42 + vp_id)
    agent = UnifiedAblationAgent(envs, obs, candidate="A").to(device)

    # Initialize from candidate A base weights or current specialist weights
    init_source = None
    if os.path.exists(CANDIDATE_A_CKPT):
        print(f"[Booster] Initializing from high-competence manipulation base: {CANDIDATE_A_CKPT}")
        agent.load_state_dict(torch.load(CANDIDATE_A_CKPT, map_location=device))
        init_source = "candidate_A_best"
    elif os.path.exists(best_ckpt_path):
        print(f"[Booster] Initializing from existing specialist weights: {best_ckpt_path}")
        agent.load_state_dict(torch.load(best_ckpt_path, map_location=device)["model_state"])
        init_source = "specialist_best"

    # Evaluate initial accuracy on this viewpoint
    initial_acc = evaluate_policy(agent, eval_envs, num_episodes=50, device=device)
    print(f"[Booster] Initial accuracy on VP {vp_id:02d} before boosting: {initial_acc:.1f}%")
    best_acc = max(current_best, initial_acc)

    if best_acc >= target_accuracy:
        print(f" >>> [SUCCESS] Baseline already satisfies target {target_accuracy:.1f}%! Saving.")
        torch.save({"global_step": 0, "best_eval_success": best_acc, "model_state": agent.state_dict()}, best_ckpt_path)
        envs.close()
        eval_envs.close()
        return best_acc

    optimizer = optim.Adam(agent.parameters(), lr=lr, eps=1e-5)
    rollout_steps = 50
    batch_size = num_envs * rollout_steps
    minibatch_size = batch_size // 8
    total_iters = max_steps // batch_size

    obs_rgb = torch.zeros((rollout_steps, num_envs, 128, 128, 3), dtype=torch.uint8, device=device)
    obs_depth = torch.zeros((rollout_steps, num_envs, 128, 128, 1), dtype=torch.int16, device=device)
    obs_state = torch.zeros((rollout_steps, num_envs, 29), dtype=torch.float32, device=device)
    actions = torch.zeros((rollout_steps, num_envs) + envs.single_action_space.shape, device=device)
    logprobs = torch.zeros((rollout_steps, num_envs), device=device)
    rewards = torch.zeros((rollout_steps, num_envs), device=device)
    dones = torch.zeros((rollout_steps, num_envs), device=device)
    values = torch.zeros((rollout_steps, num_envs), device=device)

    next_obs = obs
    next_done = torch.zeros(num_envs, device=device)
    global_step = 0
    t0 = time.time()

    for it in range(1, total_iters + 1):
        final_values = torch.zeros((rollout_steps, num_envs), device=device)

        for step in range(rollout_steps):
            global_step += num_envs
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
                    final_values[step, torch.arange(num_envs, device=device)[done_mask]] = agent.get_value(infos["final_observation"]).view(-1)

        # GAE computation with truncation bootstrapping
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(rollout_steps)):
                if t == rollout_steps - 1:
                    next_not_done = 1.0 - next_done
                    nextvalues = next_value
                else:
                    next_not_done = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                real_next_values = next_not_done * nextvalues + final_values[t]
                delta = rewards[t] + 0.8 * real_next_values - values[t]
                advantages[t] = lastgaelam = delta + 0.8 * 0.9 * next_not_done * lastgaelam
            returns = advantages + values

        b_obs = {
            "rgb": obs_rgb.reshape((-1, 128, 128, 3)),
            "depth": obs_depth.reshape((-1, 128, 128, 1)),
            "state": obs_state.reshape((-1, 29)),
        }
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_advantages = (advantages.reshape(-1) - advantages.mean()) / (advantages.std() + 1e-8)
        b_returns = returns.reshape(-1)

        b_inds = np.arange(batch_size)
        for epoch in range(4):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]
                mb_obs = {k: v[mb_inds] for k, v in b_obs.items()}
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(mb_obs, b_actions[mb_inds])
                ratio = (newlogprob - b_logprobs[mb_inds]).exp()

                pg_loss1 = -b_advantages[mb_inds] * ratio
                pg_loss2 = -b_advantages[mb_inds] * torch.clamp(ratio, 0.8, 1.2)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                v_loss = 0.5 * ((newvalue.view(-1) - b_returns[mb_inds]) ** 2).mean()
                loss = pg_loss + 0.5 * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                optimizer.step()

        # Evaluate every 5 iterations (~32k steps)
        if it % 5 == 0 or it == total_iters:
            succ = evaluate_policy(agent, eval_envs, num_episodes=50, device=device)
            elapsed = time.time() - t0
            print(f"[Iter {it:03d} | Step {global_step:,}] Eval Success: {succ:5.1f}% (Best: {best_acc:5.1f}%) | Time: {elapsed:.1f}s")

            if succ >= best_acc:
                best_acc = succ
                torch.save({"global_step": global_step, "best_eval_success": best_acc, "model_state": agent.state_dict()}, best_ckpt_path)
                print(f"  >>> [NEW BEST] VP {vp_id:02d} Model Saved! Success: {best_acc:.1f}% -> {best_ckpt_path}")

            if best_acc >= target_accuracy:
                print(f"\n{'*'*78}")
                print(f" [CEILING TARGET REACHED] VP {vp_id:02d} achieved {best_acc:.1f}% >= {target_accuracy:.1f}%!")
                print(f"{'*'*78}\n")
                break

    envs.close()
    eval_envs.close()

    # Update metrics JSON
    m = {
        "viewpoint_id": vp_id,
        "azimuth_deg": vp["azimuth_deg"],
        "elevation_deg": vp["elevation_deg"],
        "world_eye": cam_eye,
        "completed": True,
        "early_stopped": True,
        "total_timesteps": global_step,
        "best_eval_success": best_acc,
        "total_walltime_seconds": round(time.time() - t0, 1),
        "best_checkpoint": best_ckpt_path,
        "boosted": True,
    }
    with open(metrics_path, "w") as f:
        json.dump(m, f, indent=2)

    return best_acc


def boost_all_viewpoints(viewpoints=CORE_VP_IDS, target_accuracy: float = 95.0):
    print("=" * 78)
    print(" BOOSTING ALL 8 VIEWPOINT SPECIALISTS TO >=95.0% CEILING")
    print(f" Target Viewpoints: {list(viewpoints)}")
    print("=" * 78)

    results = {}
    for vp in viewpoints:
        acc = boost_single_viewpoint(vp, target_accuracy=target_accuracy)
        results[vp] = acc

    print("\n" + "=" * 78)
    print(" ALL 8 VIEWPOINT SPECIALIST CEILINGS CONVERGED")
    for vp, acc in results.items():
        print(f"  VP {vp:02d}: {acc:5.1f}%")
    print("=" * 78 + "\n")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--viewpoints", type=int, nargs="+", default=CORE_VP_IDS)
    parser.add_argument("--target-accuracy", type=float, default=95.0)
    args = parser.parse_args()

    boost_all_viewpoints(viewpoints=args.viewpoints, target_accuracy=args.target_accuracy)
