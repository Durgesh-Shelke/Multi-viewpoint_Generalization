"""
Maniwhere Training Pipeline: Visual Generalizable Framework for Reinforcement Learning (CoRL 2024).
End-to-end PPO training with:
1. Dual-camera synchronous simulation (Canonical Fixed VP 17 + Dynamic Moving Viewpoints).
2. Spatial Transformer Network (Perspective STN) for feature map rectification.
3. Multi-View Representation Objective: Symmetrized InfoNCE (tau=0.1) + Feature Map Alignment (lambda=200).
4. Proper Gymnasium truncation bootstrapping for stable Bellman value estimates.
"""

import os
import sys
import json
import time
import argparse
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
from models_maniwhere import ManiwhereAgent
from pick_cube_dual_view import PickCubeDualViewEnv, DualViewWrapper

VIEWPOINTS_JSON = "/home/maniskill/phase_1_to_3/sequential_viewpoint_training/viewpoints.json"
CORE_VP_IDS = [17, 0, 28, 18, 27, 20, 19, 6]
MOVING_VP_IDS = [0, 28, 18, 27, 20, 19, 6]


def evaluate_policy_single_view(
    agent: nn.Module,
    vp_id: int,
    num_envs: int = 50,
    max_steps: int = 50,
    device: str = "cuda",
):
    with open(VIEWPOINTS_JSON) as f:
        all_vps = json.load(f)
    vp_dict = {vp["id"]: vp for vp in all_vps}
    vp = vp_dict[vp_id]
    cam_eye = vp["world_eye"]
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
    eval_envs = gym.make("PickCube-v1", num_envs=num_envs, reconfiguration_freq=None, **env_kwargs)
    eval_envs = FlattenRGBDObservationWrapper(eval_envs, rgb=True, depth=True, state=True, sep_depth=True)
    if isinstance(eval_envs.action_space, gym.spaces.Dict):
        eval_envs = FlattenActionSpaceWrapper(eval_envs)
    eval_envs = ManiSkillVectorEnv(eval_envs, num_envs, ignore_terminations=True, record_metrics=True)

    agent.eval()
    obs, _ = eval_envs.reset(seed=1234 + vp_id)
    successes = 0
    eps = 0

    with torch.no_grad():
        for _ in range(max_steps):
            act = agent.get_action(obs, deterministic=True)
            obs, _, _, _, infos = eval_envs.step(act)
            if "final_info" in infos and "_final_info" in infos:
                mask = infos["_final_info"]
                for i in range(len(mask)):
                    if mask[i]:
                        ep = infos["final_info"].get("episode", {})
                        k = "success_once" if "success_once" in ep else "success"
                        if k in ep:
                            val = ep[k][i]
                            if bool(val.item() if torch.is_tensor(val) else val):
                                successes += 1
                        eps += 1

    eval_envs.close()
    agent.train()
    return (successes / max(eps, 1)) * 100.0 if eps > 0 else 0.0


def evaluate_all_viewpoints(agent: nn.Module, viewpoints=CORE_VP_IDS, num_envs: int = 50, device: str = "cuda"):
    results = {}
    for vp in viewpoints:
        results[vp] = evaluate_policy_single_view(agent, vp, num_envs=num_envs, device=device)
    mean_succ = float(np.mean(list(results.values())))
    min_succ = float(np.min(list(results.values())))
    return results, mean_succ, min_succ


def train_maniwhere(
    total_timesteps: int = 3000000,
    num_envs: int = 64,
    rollout_steps: int = 50,
    lr: float = 3e-4,
    stn_lr: float = 1e-4,
    gamma: float = 0.8,
    gae_lambda: float = 0.9,
    clip_coef: float = 0.2,
    ent_coef: float = 0.005,
    vf_coef: float = 0.5,
    maniwhere_coef: float = 0.5,
    lambda_feat: float = 200.0,
    temperature: float = 0.1,
    eval_freq_iters: int = 10,
    output_dir: str = "/home/maniskill/phase_9_maniwhere_replication",
    warm_start_ckpt: str = None,
    device_str: str = "cuda",
):
    device = torch.device(device_str)
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    eval_dir = os.path.join(output_dir, "eval_results")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)

    best_ckpt_path = os.path.join(ckpt_dir, "maniwhere_best.pt")
    history_csv = os.path.join(eval_dir, "maniwhere_training_history.csv")
    metrics_json = os.path.join(eval_dir, "maniwhere_metrics.json")

    print("\n" + "=" * 78)
    print(" STARTING MANIWHERE TRAINING: MULTI-VIEW REPRESENTATION + PERSPECTIVE STN")
    print(f" Target Steps: {total_timesteps:,} | Envs: {num_envs} | Rollout Steps: {rollout_steps}")
    print(f" Multi-View Loss: InfoNCE (tau={temperature}) + Feature Alignment (lambda={lambda_feat})")
    print(f" STN Architecture: Perspective 3x3 Homography (lr={stn_lr})")
    if warm_start_ckpt:
        print(f" Warm-Start Initialization: {warm_start_ckpt}")
    print("=" * 78 + "\n")

    # Initial Moving Viewpoint
    moving_vpid = 18

    def make_dual_env(mv_id):
        env = gym.make(
            "PickCubeDualView-v1",
            obs_mode="rgbd",
            num_envs=num_envs,
            sim_backend="physx_cuda",
            control_mode="pd_ee_delta_pos",
            moving_vpid=mv_id,
        )
        env = DualViewWrapper(env)
        if isinstance(env.action_space, gym.spaces.Dict):
            env = FlattenActionSpaceWrapper(env)
        env = ManiSkillVectorEnv(env, num_envs, ignore_terminations=False, record_metrics=True)
        return env

    envs = make_dual_env(moving_vpid)
    obs, _ = envs.reset(seed=42)

    # Initialize Maniwhere Agent
    agent = ManiwhereAgent(action_dim=4, state_dim=29, feature_dim=256).to(device)

    # Optional Warm-Start from Candidate A manipulation base
    if warm_start_ckpt and os.path.exists(warm_start_ckpt):
        print(f"[Warm-Start] Loading pretrained manipulation base: {warm_start_ckpt}")
        base_state = torch.load(warm_start_ckpt, map_location=device)
        if "model_state" in base_state:
            base_state = base_state["model_state"]

        key_mapping = {
            "feature_net.cnn.0.weight": "encoder.conv1.0.weight",
            "feature_net.cnn.0.bias": "encoder.conv1.0.bias",
            "feature_net.cnn.2.weight": "encoder.conv2.0.weight",
            "feature_net.cnn.2.bias": "encoder.conv2.0.bias",
            "feature_net.cnn.4.weight": "encoder.conv2.2.weight",
            "feature_net.cnn.4.bias": "encoder.conv2.2.bias",
            "feature_net.cnn.7.weight": "encoder.fc.0.weight",
            "feature_net.cnn.7.bias": "encoder.fc.0.bias",
            "feature_net.state_fc.0.weight": "state_net.0.weight",
            "feature_net.state_fc.0.bias": "state_net.0.bias",
            "actor_mean.0.weight": "actor_mean.0.weight",
            "actor_mean.0.bias": "actor_mean.0.bias",
            "actor_mean.2.weight": "actor_mean.2.weight",
            "actor_mean.2.bias": "actor_mean.2.bias",
            "actor_logstd": "actor_logstd",
            "critic.0.weight": "critic.0.weight",
            "critic.0.bias": "critic.0.bias",
            "critic.2.weight": "critic.2.weight",
            "critic.2.bias": "critic.2.bias",
        }

        agent_state = agent.state_dict()
        transferred = 0
        for src_k, dst_k in key_mapping.items():
            if src_k in base_state and dst_k in agent_state:
                if base_state[src_k].shape == agent_state[dst_k].shape:
                    agent_state[dst_k] = base_state[src_k]
                    transferred += 1
        agent.load_state_dict(agent_state)
        print(f"[Warm-Start] Successfully transferred {transferred} weight tensors! (STN preserved at Identity H=I)")

    # Optimizers: separate parameter groups for base and STN
    stn_params = list(agent.encoder.stn.parameters())
    base_params = [p for n, p in agent.named_parameters() if "stn" not in n]

    optimizer = optim.Adam([
        {"params": base_params, "lr": lr, "eps": 1e-5},
        {"params": stn_params, "lr": stn_lr, "eps": 1e-5},
    ])

    batch_size = num_envs * rollout_steps
    minibatch_size = batch_size // 4
    total_iters = total_timesteps // batch_size

    # Storage buffers
    obs_fixed_rgb = torch.zeros((rollout_steps, num_envs, 128, 128, 3), dtype=torch.uint8, device=device)
    obs_fixed_depth = torch.zeros((rollout_steps, num_envs, 128, 128, 1), dtype=torch.int16, device=device)
    obs_fixed_state = torch.zeros((rollout_steps, num_envs, 29), dtype=torch.float32, device=device)

    obs_move_rgb = torch.zeros((rollout_steps, num_envs, 128, 128, 3), dtype=torch.uint8, device=device)
    obs_move_depth = torch.zeros((rollout_steps, num_envs, 128, 128, 1), dtype=torch.int16, device=device)
    obs_move_state = torch.zeros((rollout_steps, num_envs, 29), dtype=torch.float32, device=device)

    actions = torch.zeros((rollout_steps, num_envs, 4), device=device)
    logprobs = torch.zeros((rollout_steps, num_envs), device=device)
    rewards = torch.zeros((rollout_steps, num_envs), device=device)
    dones = torch.zeros((rollout_steps, num_envs), device=device)
    values = torch.zeros((rollout_steps, num_envs), device=device)

    next_obs = obs
    next_done = torch.zeros(num_envs, device=device)
    global_step = 0
    best_mean_succ = 0.0
    history = []
    t0 = time.time()

    with open(history_csv, "w") as f:
        f.write("iteration,global_step,mean_success,min_success,rl_loss,maniwhere_loss,j_con,j_feat,walltime_seconds\n")

    for it in range(1, total_iters + 1):
        # Cycle moving viewpoint across rounds to expose policy to diverse angles
        curr_moving_vp = MOVING_VP_IDS[(it - 1) % len(MOVING_VP_IDS)]
        if curr_moving_vp != moving_vpid and it % 5 == 1:
            envs.close()
            moving_vpid = curr_moving_vp
            envs = make_dual_env(moving_vpid)
            next_obs, _ = envs.reset(seed=42 + it)
            next_done = torch.zeros(num_envs, device=device)

        final_values = torch.zeros((rollout_steps, num_envs), device=device)

        # 1. Environment Rollout
        for step in range(rollout_steps):
            global_step += num_envs

            obs_fixed_rgb[step] = next_obs["fixed"]["rgb"]
            obs_fixed_depth[step] = next_obs["fixed"]["depth"]
            obs_fixed_state[step] = next_obs["fixed"]["state"]

            obs_move_rgb[step] = next_obs["move"]["rgb"]
            obs_move_depth[step] = next_obs["move"]["depth"]
            obs_move_state[step] = next_obs["move"]["state"]

            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs["fixed"])
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            next_obs, reward, terminations, truncations, infos = envs.step(action)
            next_done = torch.logical_or(terminations, truncations).to(torch.float32)
            rewards[step] = reward.view(-1)

            # Proper Gymnasium truncation bootstrapping
            if "final_info" in infos and "_final_info" in infos:
                done_mask = infos["_final_info"]
                if "final_observation" in infos:
                    final_obs = infos["final_observation"]
                    if "cam_canonical" in final_obs.get("sensor_data", {}):
                        # Construct fixed observation for bootstrapped state
                        can_rgb = final_obs["sensor_data"]["cam_canonical"]["rgb"][done_mask]
                        can_depth = final_obs["sensor_data"]["cam_canonical"]["depth"][done_mask]
                        with torch.no_grad():
                            fb_obs = {
                                "rgb": can_rgb,
                                "depth": can_depth,
                                "state": next_obs["fixed"]["state"][done_mask],
                            }
                            final_values[step, torch.arange(num_envs, device=device)[done_mask]] = agent.get_value(fb_obs).view(-1)

        # 2. GAE Computation
        with torch.no_grad():
            next_value = agent.get_value(next_obs["fixed"]).reshape(1, -1)
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
                delta = rewards[t] + gamma * real_next_values - values[t]
                advantages[t] = lastgaelam = delta + gamma * gae_lambda * next_not_done * lastgaelam
            returns = advantages + values

        # 3. Flatten Rollout Batches
        b_fixed_obs = {
            "rgb": obs_fixed_rgb.reshape((-1, 128, 128, 3)),
            "depth": obs_fixed_depth.reshape((-1, 128, 128, 1)),
            "state": obs_fixed_state.reshape((-1, 29)),
        }
        b_move_obs = {
            "rgb": obs_move_rgb.reshape((-1, 128, 128, 3)),
            "depth": obs_move_depth.reshape((-1, 128, 128, 1)),
            "state": obs_move_state.reshape((-1, 29)),
        }
        b_actions = actions.reshape((-1, 4))
        b_logprobs = logprobs.reshape(-1)
        b_advantages = (advantages.reshape(-1) - advantages.mean()) / (advantages.std() + 1e-8)
        b_returns = returns.reshape(-1)

        # 4. PPO + Maniwhere Multi-View Optimization
        b_inds = np.arange(batch_size)
        epoch_rl_loss = 0.0
        epoch_mw_loss = 0.0
        epoch_j_con = 0.0
        epoch_j_feat = 0.0
        num_updates = 0

        for epoch in range(4):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                mb_fixed = {k: v[mb_inds] for k, v in b_fixed_obs.items()}
                mb_move = {k: v[mb_inds] for k, v in b_move_obs.items()}

                # Policy & Value predictions on Fixed observation
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(mb_fixed, b_actions[mb_inds])
                ratio = (newlogprob - b_logprobs[mb_inds]).exp()

                pg_loss1 = -b_advantages[mb_inds] * ratio
                pg_loss2 = -b_advantages[mb_inds] * torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                v_loss = 0.5 * ((newvalue.view(-1) - b_returns[mb_inds]) ** 2).mean()
                ent_loss = entropy.mean()

                rl_loss = pg_loss + vf_coef * v_loss - ent_coef * ent_loss

                # Maniwhere Multi-View Representation Loss (Eq. 3)
                l_maniwhere, j_con, j_feat = agent.compute_maniwhere_representation_loss(
                    mb_fixed, mb_move, temperature=temperature, lambda_feat=lambda_feat
                )

                total_loss = rl_loss + maniwhere_coef * l_maniwhere

                optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                optimizer.step()

                epoch_rl_loss += rl_loss.item()
                epoch_mw_loss += l_maniwhere.item()
                epoch_j_con += j_con.item()
                epoch_j_feat += j_feat.item()
                num_updates += 1

        avg_rl = epoch_rl_loss / num_updates
        avg_mw = epoch_mw_loss / num_updates
        avg_con = epoch_j_con / num_updates
        avg_feat = epoch_j_feat / num_updates

        # 5. Periodic Evaluation across All 8 Viewpoints
        if it % eval_freq_iters == 0 or it == total_iters:
            vp_results, mean_succ, min_succ = evaluate_all_viewpoints(agent, viewpoints=CORE_VP_IDS, num_envs=50, device=device)
            elapsed = time.time() - t0

            print(f"[Iter {it:03d}/{total_iters} | Step {global_step:,}] Mean Success: {mean_succ:5.1f}% | Worst: {min_succ:5.1f}% | RL Loss: {avg_rl:.3f} | MW Loss: {avg_mw:.3f} (InfoNCE: {avg_con:.3f}, Feat: {avg_feat:.6f}) | Time: {elapsed/60:.1f}m")
            print(f"  Viewpoints: VP17: {vp_results[17]:.0f}% | VP00: {vp_results[0]:.0f}% | VP28: {vp_results[28]:.0f}% | VP18: {vp_results[18]:.0f}% | VP27: {vp_results[27]:.0f}% | VP20: {vp_results[20]:.0f}% | VP19: {vp_results[19]:.0f}% | VP06: {vp_results[6]:.0f}%")

            with open(history_csv, "a") as f:
                f.write(f"{it},{global_step},{mean_succ},{min_succ},{avg_rl:.4f},{avg_mw:.4f},{avg_con:.4f},{avg_feat:.6f},{elapsed:.1f}\n")

            if mean_succ >= best_mean_succ:
                best_mean_succ = mean_succ
                torch.save({
                    "iteration": it,
                    "global_step": global_step,
                    "best_mean_success": best_mean_succ,
                    "per_viewpoint": vp_results,
                    "model_state": agent.state_dict(),
                }, best_ckpt_path)
                print(f"  >>> [NEW BEST MODEL] Mean Success: {best_mean_succ:.1f}% -> {best_ckpt_path}")

    envs.close()

    metrics = {
        "framework": "Maniwhere (CoRL 2024 Replication)",
        "total_timesteps": global_step,
        "best_mean_success": best_mean_succ,
        "total_walltime_seconds": round(time.time() - t0, 1),
        "checkpoint": best_ckpt_path,
    }
    with open(metrics_json, "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n[Done] Maniwhere Training Finished!")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-timesteps", type=int, default=3000000)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--eval-freq", type=int, default=10)
    parser.add_argument("--warm-start", type=str, default="/home/maniskill/phase_5/checkpoints/candidate_A_best.pt")
    args = parser.parse_args()

    train_maniwhere(
        total_timesteps=args.total_timesteps,
        num_envs=args.num_envs,
        eval_freq_iters=args.eval_freq,
        warm_start_ckpt=args.warm_start if args.warm_start and args.warm_start.lower() != "none" else None,
    )
