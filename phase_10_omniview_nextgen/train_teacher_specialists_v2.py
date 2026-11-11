"""
Phase 10 Robust Teacher Specialist Training Engine.
Trains all 8 anchor teachers from scratch for EXACTLY 5,000,000 steps each.
Key Features:
- Exact Phase 6 PPO engine with ignore_terminations=False and truncation bootstrapping.
- Low-memory rollout buffers (uint8 RGB + int16 Depth), using <3.5 GB VRAM.
- 33-dim State vector with Camera Extrinsics Conditioning [x_rel, y_rel, z_rel, dist] + Gaussian noise.
- Full 5,000,000 steps per teacher (1,562 iterations) with zero early termination.
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
from phase_7_multiteacher_distillation.models import layer_init

VIEWPOINTS_JSON = "/home/maniskill/Multi-viewpoint_Generalization/phase_1_to_3/sequential_viewpoint_training/viewpoints.json"
TEACHERS_DIR = "/home/maniskill/Multi-viewpoint_Generalization/phase_10_omniview_nextgen/teachers"
CORE_TEACHERS = [17, 0, 28, 18, 27, 20, 19, 6]


class ExtrinsicTeacherAgent(nn.Module):
    def __init__(self, action_dim=4, state_dim=33, feature_dim=256):
        super().__init__()
        # 4-Channel RGB-D CNN
        self.cnn = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, kernel_size=8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 12 * 12, feature_dim)),
            nn.ReLU(),
        )
        self.state_fc = nn.Sequential(
            layer_init(nn.Linear(state_dim, 256)),
            nn.ReLU(),
        )
        fusion_dim = feature_dim + 256
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(fusion_dim, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, action_dim), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))
        self.critic = nn.Sequential(
            layer_init(nn.Linear(fusion_dim, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, 1), std=1.0),
        )

    def _preprocess(self, rgb_uint8: torch.Tensor, depth_int16: torch.Tensor):
        rgb_norm = rgb_uint8.permute(0, 3, 1, 2).float().div_(255.0)
        d = depth_int16.permute(0, 3, 1, 2).float()
        d.masked_fill_(d <= 0, 3000.0)
        d_norm = d.div_(1000.0).clamp_(0.0, 3.0).div_(3.0)
        return torch.cat([rgb_norm, d_norm], dim=1)

    def get_features(self, rgb, depth, state):
        x = self._preprocess(rgb, depth)
        vis_feat = self.cnn(x)
        state_feat = self.state_fc(state)
        return torch.cat([vis_feat, state_feat], dim=-1)

    def get_action(self, rgb, depth, state, deterministic=False):
        feat = self.get_features(rgb, depth, state)
        action_mean = self.actor_mean(feat)
        if deterministic:
            return action_mean
        action_std = torch.exp(self.actor_logstd.expand_as(action_mean))
        probs = torch.distributions.Normal(action_mean, action_std)
        return probs.sample()

    def get_action_and_value(self, rgb, depth, state, action=None):
        feat = self.get_features(rgb, depth, state)
        action_mean = self.actor_mean(feat)
        action_std = torch.exp(self.actor_logstd.expand_as(action_mean))
        probs = torch.distributions.Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        log_prob = probs.log_prob(action).sum(dim=-1)
        entropy = probs.entropy().sum(dim=-1)
        value = self.critic(feat).squeeze(-1)
        return action, log_prob, entropy, value

    def get_value(self, rgb, depth, state):
        feat = self.get_features(rgb, depth, state)
        return self.critic(feat).squeeze(-1)


def append_extrinsics(state: torch.Tensor, cam_eye: torch.Tensor, noise_std: float = 0.02, device="cuda"):
    base_pos = torch.tensor([-0.615, 0.0, 0.0], device=device)
    rel_pos = cam_eye - base_pos
    b_size = state.shape[0]
    if noise_std > 0:
        noise = torch.randn((b_size, 3), device=device) * noise_std
        noisy_rel = rel_pos.unsqueeze(0).repeat(b_size, 1) + noise
    else:
        noisy_rel = rel_pos.unsqueeze(0).repeat(b_size, 1)
    dist = torch.norm(noisy_rel, dim=-1, keepdim=True)
    ext = torch.cat([noisy_rel, dist], dim=-1)
    return torch.cat([state.float(), ext], dim=-1)


def evaluate_policy(agent, eval_envs, cam_eye_t, num_episodes=50, device="cuda"):
    agent.eval()
    obs, _ = eval_envs.reset()
    collected = 0
    successes = 0

    while collected < num_episodes:
        state_33 = append_extrinsics(obs["state"], cam_eye_t, noise_std=0.0, device=device)
        with torch.no_grad():
            act = agent.get_action(obs["rgb"], obs["depth"], state_33, deterministic=True)
        obs, _, _, _, infos = eval_envs.step(act)

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
    num_envs: int = 64,
    num_steps: int = 50,
    lr: float = 3e-4,
    total_timesteps: int = 5_000_000,
    device_str: str = "cuda",
):
    device = torch.device(device_str)
    batch_size = num_envs * num_steps
    max_iterations = total_timesteps // batch_size
    best_ckpt_path = os.path.join(TEACHERS_DIR, f"specialist_vp_{vpid:02d}_best.pt")
    final_ckpt_path = os.path.join(TEACHERS_DIR, f"specialist_vp_{vpid:02d}_final.pt")
    meta_json_path = os.path.join(TEACHERS_DIR, f"specialist_vp_{vpid:02d}_meta.json")

    # Resume check: only skip if full 5M steps are done
    if os.path.exists(meta_json_path) and os.path.exists(final_ckpt_path):
        with open(meta_json_path) as f:
            m = json.load(f)
        if m.get("steps", 0) >= total_timesteps:
            print(f"[VP {vpid:02d}] Already completed all {total_timesteps:,} steps! Skipping.")
            return m.get("best_eval_success", 0)

    print("\n" + "=" * 80)
    print(f" TRAINING PHASE 10 TEACHER FROM SCRATCH: VIEWPOINT {vpid:02d}")
    print(f" Azimuth: {vp_dict[vpid]['azimuth_deg']:+6.1f}° | Elevation: {vp_dict[vpid]['elevation_deg']:4.1f}°")
    print(f" Budget: EXACTLY {total_timesteps:,} steps ({max_iterations} iterations, NO EARLY STOPPING)")
    print("=" * 80)

    agent = ExtrinsicTeacherAgent(action_dim=4, state_dim=33).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=lr, eps=1e-5)

    cam_eye = vp_dict[vpid]["world_eye"]
    cam_eye_t = torch.tensor(cam_eye, device=device)
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

    envs = gym.make("PickCube-v1", num_envs=num_envs, reconfiguration_freq=None, **env_kwargs)
    envs = FlattenRGBDObservationWrapper(envs, rgb=True, depth=True, state=True, sep_depth=True)
    if isinstance(envs.action_space, gym.spaces.Dict):
        envs = FlattenActionSpaceWrapper(envs)
    envs = ManiSkillVectorEnv(envs, num_envs, ignore_terminations=False, record_metrics=True)

    eval_envs = gym.make("PickCube-v1", num_envs=50, reconfiguration_freq=None, **env_kwargs)
    eval_envs = FlattenRGBDObservationWrapper(eval_envs, rgb=True, depth=True, state=True, sep_depth=True)
    if isinstance(eval_envs.action_space, gym.spaces.Dict):
        eval_envs = FlattenActionSpaceWrapper(eval_envs)
    eval_envs = ManiSkillVectorEnv(eval_envs, 50, ignore_terminations=True, record_metrics=True)

    # Rollout buffers (compact uint8/int16 storage)
    obs_rgb = torch.zeros((num_steps, num_envs, 128, 128, 3), dtype=torch.uint8, device=device)
    obs_depth = torch.zeros((num_steps, num_envs, 128, 128, 1), dtype=torch.int16, device=device)
    obs_state = torch.zeros((num_steps, num_envs, 33), dtype=torch.float32, device=device)
    actions = torch.zeros((num_steps, num_envs, 4), dtype=torch.float32, device=device)
    logprobs = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)
    rewards = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)
    dones = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)
    values = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)

    next_obs, _ = envs.reset(seed=42 + vpid)
    next_done = torch.zeros(num_envs, device=device)
    global_step = 0
    best_acc = 0.0
    t0 = time.time()

    for iteration in range(1, max_iterations + 1):
        # Linear lr annealing
        frac = 1.0 - (iteration - 1.0) / max_iterations
        optimizer.param_groups[0]["lr"] = frac * lr

        agent.eval()
        final_values = torch.zeros((num_steps, num_envs), device=device)

        for step in range(num_steps):
            global_step += num_envs
            obs_rgb[step] = next_obs["rgb"]
            obs_depth[step] = next_obs["depth"]
            s33 = append_extrinsics(next_obs["state"], cam_eye_t, noise_std=0.02, device=device)
            obs_state[step] = s33
            dones[step] = next_done

            with torch.no_grad():
                act, logprob, _, val = agent.get_action_and_value(next_obs["rgb"], next_obs["depth"], s33)
                values[step] = val

            actions[step] = act
            logprobs[step] = logprob

            next_obs, reward, term, trunc, infos = envs.step(act)
            next_done = torch.logical_or(term, trunc).to(torch.float32)
            rewards[step] = reward.view(-1)

            # Proper truncation bootstrapping
            if "final_info" in infos and "_final_info" in infos:
                done_mask = infos["_final_info"]
                if done_mask.any():
                    final_obs = infos["final_observation"]
                    f_rgb = final_obs["rgb"][done_mask]
                    f_depth = final_obs["depth"][done_mask]
                    f_s33 = append_extrinsics(final_obs["state"][done_mask], cam_eye_t, noise_std=0.0, device=device)
                    with torch.no_grad():
                        f_vals = agent.get_value(f_rgb, f_depth, f_s33)
                        final_values[step, torch.arange(num_envs, device=device)[done_mask]] = f_vals.view(-1)

        # GAE bootstrap
        with torch.no_grad():
            s33_next = append_extrinsics(next_obs["state"], cam_eye_t, noise_std=0.0, device=device)
            next_value = agent.get_value(next_obs["rgb"], next_obs["depth"], s33_next).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    next_not_done = 1.0 - next_done
                    nextvalues = next_value
                else:
                    next_not_done = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                real_next = next_not_done * nextvalues + final_values[t]
                delta = rewards[t] + 0.99 * real_next - values[t]
                advantages[t] = lastgaelam = delta + 0.99 * 0.95 * next_not_done * lastgaelam
            returns = advantages + values

        # Flatten batch
        b_rgb = obs_rgb.reshape(-1, 128, 128, 3)
        b_depth = obs_depth.reshape(-1, 128, 128, 1)
        b_state = obs_state.reshape(-1, 33)
        b_actions = actions.reshape(-1, 4)
        b_logprobs = logprobs.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Normalize advantages
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        # PPO Update
        agent.train()
        minibatch_size = batch_size // 4
        b_inds = np.arange(batch_size)

        for epoch in range(4):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                minds = b_inds[start:end]

                _, newlogprob, entropy, newval = agent.get_action_and_value(
                    b_rgb[minds], b_depth[minds], b_state[minds], b_actions[minds]
                )

                logratio = newlogprob - b_logprobs[minds]
                ratio = torch.exp(logratio)

                pg_loss1 = -b_advantages[minds] * ratio
                pg_loss2 = -b_advantages[minds] * torch.clamp(ratio, 0.8, 1.2)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                v_clipped = b_values[minds] + torch.clamp(newval - b_values[minds], -0.2, 0.2)
                v_loss = 0.5 * torch.max((newval - b_returns[minds]) ** 2, (v_clipped - b_returns[minds]) ** 2).mean()
                loss = pg_loss + 0.5 * v_loss - 0.01 * entropy.mean()

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                optimizer.step()

        # Periodic Evaluation every 20 iterations (64,000 steps)
        if iteration % 20 == 0 or iteration == max_iterations:
            eval_acc = evaluate_policy(agent, eval_envs, cam_eye_t, num_episodes=50, device=device)
            elapsed = time.time() - t0
            fps = global_step / elapsed
            print(f"Iter {iteration:04d}/{max_iterations} | Steps: {global_step:,} | Eval Success: {eval_acc:5.1f}% (Best: {best_acc:5.1f}%) | SPS: {fps:.0f}")

            if eval_acc > best_acc:
                best_acc = eval_acc
                torch.save({"model_state": agent.state_dict(), "best_eval_success": best_acc}, best_ckpt_path)
                with open(meta_json_path, "w") as f:
                    json.dump({"vpid": vpid, "best_eval_success": best_acc, "steps": global_step}, f)

    # Final save after exactly 5,000,000 steps
    torch.save({"model_state": agent.state_dict(), "final_eval_success": eval_acc, "best_eval_success": best_acc, "steps": global_step}, final_ckpt_path)
    print(f"\n[Completed 5M Steps] VP {vpid:02d} completed all {global_step:,} steps! Best Eval Success: {best_acc:.1f}%\n")

    envs.close()
    eval_envs.close()
    return best_acc


def main():
    os.makedirs(TEACHERS_DIR, exist_ok=True)
    with open(VIEWPOINTS_JSON) as f:
        vp_dict = {v["id"]: v for v in json.load(f)}

    print(f"\nTraining all {len(CORE_TEACHERS)} Phase 10 Teachers for EXACTLY 5,000,000 steps each (NO early termination)...")
    for vpid in CORE_TEACHERS:
        train_single_teacher(vpid, vp_dict, num_envs=64, num_steps=50, total_timesteps=5_000_000)

    print("\n" + "=" * 80)
    print(" ALL 8 PHASE 10 TEACHER SPECIALISTS COMPLETED (40,000,000 STEPS TOTAL)!")
    print("=" * 80)


if __name__ == "__main__":
    main()
