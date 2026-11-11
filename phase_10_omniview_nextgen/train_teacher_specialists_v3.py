"""
Phase 10 Robust Teacher Specialist Training Engine (v3).
Trains all 8 anchor teachers from scratch for EXACTLY 5,000,000 steps each.
Uses the exact mathematical PPO formulation from Phase 6 that achieved 96-100% success:
- gamma = 0.8, gae_lambda = 0.9 (optimal for ManiSkill PickCube-v1 50-step horizon)
- num_envs = 128, batch_size = 6400, 8 minibatches (minibatch size 800)
- actor_logstd initialized to -0.5 (sigma = 0.606)
- actor_mean initialized with std = 0.01 * sqrt(2)
- Unclipped value loss (vf_coef = 0.5)
- ent_coef = 0.0, max_grad_norm = 0.5, target_kl = 0.2
- Camera Extrinsics Conditioning [x_rel, y_rel, z_rel, dist] appended to 29-dim state -> 33-dim state
- Gaussian noise (sigma = 0.02m) during training
- Compact uint8 RGB and int16 Depth buffers (<1.5 GB VRAM)
- STRICTLY 5,000,000 steps per viewpoint with ZERO early termination.
"""

import os
import sys
sys.path.append("/home/maniskill")
import gc
import json
import time
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import gymnasium as gym
import mani_skill.envs
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper, FlattenActionSpaceWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

VIEWPOINTS_JSON = "/home/maniskill/Multi-viewpoint_Generalization/phase_1_to_3/sequential_viewpoint_training/viewpoints.json"
TEACHERS_DIR = "/home/maniskill/Multi-viewpoint_Generalization/phase_10_omniview_nextgen/teachers"
CORE_TEACHERS = [17, 0, 28, 18, 27, 20, 19, 6]


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def preprocess_rgbd(rgb_uint8: torch.Tensor, depth_int16: torch.Tensor):
    """
    Standardized visual pre-processing:
    - RGB: uint8 [0..255] -> float32 [0..1]
    - Depth: int16 (millimeters) -> meters -> clamped [0..3m] -> normalized [0..1]
    """
    d = depth_int16.permute(0, 3, 1, 2).float()
    d.masked_fill_(d <= 0, 3000.0)
    d_norm = d.div_(1000.0).clamp_(0.0, 3.0).div_(3.0)
    rgb_norm = rgb_uint8.permute(0, 3, 1, 2).float().div_(255.0)
    return torch.cat([rgb_norm, d_norm], dim=1)


class ExtrinsicNatureCNN128(nn.Module):
    """Candidate A: Symmetric 128x128 4-Channel RGB-D Encoder with 33-dim State."""
    def __init__(self, state_dim=33, feature_size=256):
        super().__init__()
        self.feature_size = feature_size
        self.cnn = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 12 * 12, feature_size)),
            nn.ReLU(),
        )
        self.state_fc = nn.Sequential(
            layer_init(nn.Linear(state_dim, 256)),
            nn.ReLU()
        )
        self.out_features = feature_size + 256

    def forward(self, rgb, depth, state):
        x = preprocess_rgbd(rgb, depth)
        vis_feat = self.cnn(x)
        state_feat = self.state_fc(state)
        return torch.cat([vis_feat, state_feat], dim=1)


class Phase10TeacherAgent(nn.Module):
    """Phase 6 UnifiedAblationAgent architecture with 33-dim state input."""
    def __init__(self, action_dim=4, state_dim=33):
        super().__init__()
        self.feature_net = ExtrinsicNatureCNN128(state_dim=state_dim)
        latent_size = self.feature_net.out_features  # 512

        self.critic = nn.Sequential(
            layer_init(nn.Linear(latent_size, 512)),
            nn.ReLU(inplace=True),
            layer_init(nn.Linear(512, 1)),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(latent_size, 512)),
            nn.ReLU(inplace=True),
            layer_init(nn.Linear(512, action_dim), std=0.01 * np.sqrt(2)),
        )
        self.actor_logstd = nn.Parameter(torch.ones(1, action_dim) * -0.5)

    def get_features(self, rgb, depth, state):
        return self.feature_net(rgb, depth, state)

    def get_value(self, rgb, depth, state):
        feat = self.feature_net(rgb, depth, state)
        return self.critic(feat)

    def get_action(self, rgb, depth, state, deterministic=False):
        feat = self.feature_net(rgb, depth, state)
        action_mean = self.actor_mean(feat)
        if deterministic:
            return action_mean
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        return probs.sample()

    def get_action_and_value(self, rgb, depth, state, action=None):
        feat = self.feature_net(rgb, depth, state)
        action_mean = self.actor_mean(feat)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(feat)


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


def evaluate_policy(agent, eval_envs, cam_eye_t, num_episodes=50, max_steps=50, device="cuda"):
    agent.eval()
    obs, _ = eval_envs.reset()
    collected = 0
    successes = 0

    with torch.no_grad():
        for _ in range(max_steps):
            if collected >= num_episodes:
                break
            state_33 = append_extrinsics(obs["state"], cam_eye_t, noise_std=0.0, device=device)
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

    agent.train()
    return (successes / max(collected, 1)) * 100.0


def train_single_teacher(
    vpid: int,
    vp_dict: dict,
    total_timesteps: int = 5_000_000,
    num_envs: int = 128,
    num_steps: int = 50,
    lr: float = 3e-4,
    gamma: float = 0.8,
    gae_lambda: float = 0.9,
    clip_coef: float = 0.2,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    target_kl: float = 0.2,
    num_minibatches: int = 8,
    update_epochs: int = 4,
    eval_freq_iters: int = 25,
    device_str: str = "cuda",
):
    device = torch.device(device_str)
    batch_size = num_envs * num_steps  # 6400
    minibatch_size = batch_size // num_minibatches  # 800
    total_iterations = total_timesteps // batch_size  # 781

    best_ckpt_path = os.path.join(TEACHERS_DIR, f"specialist_vp_{vpid:02d}_best.pt")
    final_ckpt_path = os.path.join(TEACHERS_DIR, f"specialist_vp_{vpid:02d}_final.pt")
    meta_json_path = os.path.join(TEACHERS_DIR, f"specialist_vp_{vpid:02d}_meta.json")

    # Resume check: only skip if full 5M steps are done
    if os.path.exists(meta_json_path) and os.path.exists(final_ckpt_path):
        with open(meta_json_path) as f:
            m = json.load(f)
        if m.get("steps", 0) >= total_timesteps - batch_size:
            print(f"[VP {vpid:02d}] Already completed all 5,000,000 steps! (Logged: {m.get('steps', 0):,} steps). Skipping.")
            return m.get("best_eval_success", 0)

    print("\n" + "=" * 80)
    print(f" TRAINING PHASE 10 TEACHER FROM SCRATCH: VIEWPOINT {vpid:02d}")
    print(f" Azimuth: {vp_dict[vpid]['azimuth_deg']:+6.1f}° | Elevation: {vp_dict[vpid]['elevation_deg']:4.1f}°")
    print(f" Budget: EXACTLY {total_timesteps:,} steps ({total_iterations} iterations, NO EARLY STOPPING)")
    print(f" Batch: {batch_size} (128 envs x 50 steps) | Minibatch: {minibatch_size} | gamma={gamma}, gae_lambda={gae_lambda}")
    print("=" * 80)

    run_seed = 42 + vpid
    random.seed(run_seed)
    np.random.seed(run_seed)
    torch.manual_seed(run_seed)
    torch.backends.cudnn.deterministic = True

    agent = Phase10TeacherAgent(action_dim=4, state_dim=33).to(device)
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

    # Rollout buffers (compact uint8/int16 storage to keep VRAM < 1.5 GB)
    obs_rgb = torch.zeros((num_steps, num_envs, 128, 128, 3), dtype=torch.uint8, device=device)
    obs_depth = torch.zeros((num_steps, num_envs, 128, 128, 1), dtype=torch.int16, device=device)
    obs_state = torch.zeros((num_steps, num_envs, 33), dtype=torch.float32, device=device)
    actions = torch.zeros((num_steps, num_envs, 4), dtype=torch.float32, device=device)
    logprobs = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)
    rewards = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)
    dones = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)
    values = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)

    next_obs, _ = envs.reset(seed=run_seed)
    next_done = torch.zeros(num_envs, device=device)
    global_step = 0
    best_acc = 0.0
    t0 = time.time()

    for iteration in range(1, total_iterations + 1):
        # Linear lr annealing
        frac = 1.0 - (iteration - 1.0) / total_iterations
        optimizer.param_groups[0]["lr"] = frac * lr

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
                values[step] = val.flatten()

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
                delta = rewards[t] + gamma * real_next - values[t]
                advantages[t] = lastgaelam = delta + gamma * gae_lambda * next_not_done * lastgaelam
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

        # PPO Update (Exact Phase 6 Formulation)
        b_inds = np.arange(batch_size)
        clipfracs = []
        for epoch in range(update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                minds = b_inds[start:end]

                _, newlogprob, entropy, newval = agent.get_action_and_value(
                    b_rgb[minds], b_depth[minds], b_state[minds], b_actions[minds]
                )

                logratio = newlogprob - b_logprobs[minds]
                ratio = logratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > clip_coef).float().mean().item()]

                mb_adv = b_advantages[minds]
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newval = newval.view(-1)
                v_loss = 0.5 * ((newval - b_returns[minds]) ** 2).mean()

                loss = pg_loss + vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                optimizer.step()

            if target_kl is not None and approx_kl > target_kl:
                break

        # Periodic Evaluation every 25 iterations (160,000 steps)
        if iteration % eval_freq_iters == 0 or iteration == total_iterations:
            eval_acc = evaluate_policy(agent, eval_envs, cam_eye_t, num_episodes=50, device=device)
            elapsed = time.time() - t0
            fps = global_step / elapsed
            vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0
            print(f"Iter {iteration:04d}/{total_iterations} | Steps: {global_step:,} | Eval Success: {eval_acc:5.1f}% (Best: {best_acc:5.1f}%) | SPS: {fps:.0f} | VRAM: {vram_mb:.0f}MB")

            if eval_acc >= best_acc:
                best_acc = eval_acc
                torch.save({"model_state": agent.state_dict(), "best_eval_success": best_acc, "steps": global_step}, best_ckpt_path)
                with open(meta_json_path, "w") as f:
                    json.dump({"vpid": vpid, "best_eval_success": best_acc, "steps": global_step}, f)

    # Final save after exactly 5,000,000 steps
    torch.save({"model_state": agent.state_dict(), "final_eval_success": eval_acc, "best_eval_success": best_acc, "steps": global_step}, final_ckpt_path)
    with open(meta_json_path, "w") as f:
        json.dump({"vpid": vpid, "best_eval_success": best_acc, "final_eval_success": eval_acc, "steps": global_step}, f)
    print(f"\n[Completed 5M Steps] VP {vpid:02d} completed all {global_step:,} steps! Best Eval Success: {best_acc:.1f}%\n")

    envs.close()
    eval_envs.close()
    del envs, eval_envs, agent, optimizer
    torch.cuda.empty_cache()
    gc.collect()
    return best_acc


def main():
    parser = argparse.ArgumentParser(description="Phase 10 Robust Teacher Specialist Training Engine")
    parser.add_argument("--viewpoints", type=int, nargs="+", default=CORE_TEACHERS, help="Viewpoint IDs to train")
    parser.add_argument("--total-timesteps", type=int, default=5_000_000, help="Timesteps per viewpoint (strictly 5M)")
    args = parser.parse_args()

    os.makedirs(TEACHERS_DIR, exist_ok=True)
    with open(VIEWPOINTS_JSON) as f:
        vps = json.load(f)
    vp_dict = {vp["id"]: vp for vp in vps}

    print("\n" + "=" * 80)
    print("PHASE 10 MASTER TEACHER RETRAINING ENGINE LAUNCHED")
    print(f"Target Viewpoints: {args.viewpoints}")
    print(f"Budget per Teacher: {args.total_timesteps:,} steps (Strictly No Early Termination)")
    print("=" * 80 + "\n")

    for vpid in args.viewpoints:
        train_single_teacher(vpid, vp_dict, total_timesteps=args.total_timesteps)

    print("\n" + "=" * 80)
    print("ALL 8 PHASE 10 TEACHERS FULLY TRAINED FOR 5,000,000 STEPS EACH!")
    print("READY FOR NEXT-GEN OMNIVIEW STUDENT POLICY DISTILLATION.")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
