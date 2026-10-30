"""
Joint Multi-View Domain Randomization Training with 4-Channel RGB-D Observations.
Center: Robot Base [-0.615, 0.0, 0.0], R = 1.0942 m, 30 Fibonacci Hemispherical Viewpoints.

Features:
- Observation Mode: RGB-D (RGB 128x128x3 uint8 + Metric Depth 128x128x1 int16)
- Vision Backbone: 4-Channel NatureCNN [R, G, B, Depth_Normalized]
- Metric Depth Preprocessing: int16 millimeters -> meters [0..3m] -> normalized [0..1]
- Warm-Start Adapter: Transfers 3-channel RGB front-view weights into 4-channel RGB-D,
  initializing the depth weights with 0.0 so initial policy retains 100% front-view capability.
- Multi-Camera Vectorization: 128 parallel environments concurrently rendering 30 viewpoints.
- Vectorized Periodic Hemispherical Evaluation: Benchmarks all 30 viewpoints simultaneously.
"""

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

import cv2
import gymnasium as gym
import numpy as np
import sapien.core as sapien
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, "/home/maniskill")
sys.path.insert(0, "/home/maniskill/examples/baselines/ppo")
sys.path.insert(0, "/home/maniskill/phase_1_to_3/visual_manipulation")

import mani_skill.envs
from mani_skill.utils import sapien_utils
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper, FlattenRGBDObservationWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from ppo_rgb import DictArray

TARGET_POS = [-0.1, 0.0, 0.1]
STOP_REQUESTED = False


def sig_handler(signum, frame):
    global STOP_REQUESTED
    print(f"\n[Interrupt] Signal {signum} received! Stopping cleanly after current epoch...")
    STOP_REQUESTED = True


signal.signal(signal.SIGINT, sig_handler)
signal.signal(signal.SIGTERM, sig_handler)


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class RGBDNatureCNN(nn.Module):
    """
    4-Channel NatureCNN Backbone for Joint RGB-D Processing.
    Input: RGB (3 channels in [0, 1]) + Depth (1 channel metric distance in [0, 1]).
    """

    def __init__(self, sample_obs, in_channels: int = 4, feature_size: int = 256):
        super().__init__()
        self.out_features = 0
        self.feature_size = feature_size
        self.in_channels = in_channels

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 128, 128)
            n_flatten = self.cnn(dummy).shape[1]
            self.fc = nn.Sequential(nn.Linear(n_flatten, feature_size), nn.ReLU())

        self.out_features += feature_size

        if "state" in sample_obs:
            state_size = sample_obs["state"].shape[-1]
            self.state_fc = nn.Sequential(nn.Linear(state_size, 256), nn.ReLU())
            self.out_features += 256

    def forward(self, observations) -> torch.Tensor:
        # 1. Normalize RGB to [0, 1]
        rgb = observations["rgb"].float() / 255.0  # (B, H, W, 3)

        # 2. Process Depth: int16 (millimeters) -> meters -> normalized [0, 1]
        depth = observations["depth"].float()
        # Invalid/far pixels (<= 0) replaced with 3.0 m (3000 mm)
        depth = torch.where(depth <= 0, torch.full_like(depth, 3000.0), depth)
        depth_m = torch.clamp(depth / 1000.0, 0.0, 3.0)
        depth_norm = depth_m / 3.0  # (B, H, W, 1) in [0, 1]

        # 3. Concatenate into 4-channel tensor (B, 4, 128, 128)
        rgbd = torch.cat([rgb, depth_norm], dim=-1).permute(0, 3, 1, 2)
        vis_feat = self.fc(self.cnn(rgbd))

        if hasattr(self, "state_fc") and "state" in observations:
            state_feat = self.state_fc(observations["state"])
            return torch.cat([vis_feat, state_feat], dim=1)

        return vis_feat


class RGBDAgent(nn.Module):
    """Actor-Critic Policy with 4-Channel RGB-D Perception."""

    def __init__(self, env, sample_obs):
        super().__init__()
        action_dim = int(np.prod(env.single_action_space.shape))
        self.feature_net = RGBDNatureCNN(sample_obs)
        latent_size = self.feature_net.out_features

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

    def get_features(self, x):
        return self.feature_net(x)

    def get_value(self, x):
        return self.critic(self.feature_net(x))

    def get_action(self, x, deterministic=False):
        feat = self.feature_net(x)
        action_mean = self.actor_mean(feat)
        if deterministic:
            return action_mean
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        return probs.sample()

    def get_action_and_value(self, x, action=None):
        feat = self.feature_net(x)
        action_mean = self.actor_mean(feat)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(feat)


def load_model_weights(agent: RGBDAgent, checkpoint_path: str, device: str = "cuda"):
    """
    Intelligently loads weights from either:
    1) A 3-channel RGB checkpoint (e.g. backup_original_model.pt), warm-starting
       channels 0..2 with RGB weights and channel 3 with 0.0.
    2) A 4-channel RGB-D checkpoint, directly restoring all weights.
    """
    ckpt = torch.load(checkpoint_path, map_location=device)
    if "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]

    target_sd = agent.state_dict()

    # Check if checkpoint is from pure RGB (3-channel) or RGB-D (4-channel)
    is_3channel_rgb = "feature_net.extractors.rgb.0.0.weight" in ckpt

    if is_3channel_rgb:
        print(f"[Warm-Start] Detected 3-channel RGB checkpoint. Converting conv1 to 4-channel RGB-D...")
        for k, v in ckpt.items():
            if k == "feature_net.extractors.rgb.0.0.weight":
                target_sd["feature_net.cnn.0.weight"][:, :3, :, :] = v
                # Zero-init depth weights so initial policy behavior is 100% preserved
                target_sd["feature_net.cnn.0.weight"][:, 3:4, :, :] = 0.0
            elif k == "feature_net.extractors.rgb.0.0.bias":
                target_sd["feature_net.cnn.0.bias"] = v
            elif k == "feature_net.extractors.rgb.0.2.weight":
                target_sd["feature_net.cnn.2.weight"] = v
            elif k == "feature_net.extractors.rgb.0.2.bias":
                target_sd["feature_net.cnn.2.bias"] = v
            elif k == "feature_net.extractors.rgb.0.4.weight":
                target_sd["feature_net.cnn.4.weight"] = v
            elif k == "feature_net.extractors.rgb.0.4.bias":
                target_sd["feature_net.cnn.4.bias"] = v
            elif k == "feature_net.extractors.rgb.1.0.weight":
                target_sd["feature_net.fc.0.weight"] = v
            elif k == "feature_net.extractors.rgb.1.0.bias":
                target_sd["feature_net.fc.0.bias"] = v
            elif k == "feature_net.extractors.state.weight":
                target_sd["feature_net.state_fc.0.weight"] = v
            elif k == "feature_net.extractors.state.bias":
                target_sd["feature_net.state_fc.0.bias"] = v
            elif k in target_sd:
                target_sd[k] = v
        agent.load_state_dict(target_sd)
        print("[Warm-Start] Weights successfully transferred from RGB checkpoint into 4-channel RGB-D agent!")
    else:
        # Direct RGB-D checkpoint load
        agent.load_state_dict(ckpt)
        print(f"[Load] Successfully loaded 4-channel RGB-D checkpoint from: {checkpoint_path}")


def assign_viewpoints_to_envs(env, viewpoints: list, target: list = TARGET_POS, shuffle: bool = False):
    """Assigns the 30 Fibonacci viewpoints across the parallel sub-cameras."""
    cam = env.unwrapped._sensors["base_camera"]
    num_subcams = len(cam.camera._render_cameras)
    indices = np.arange(num_subcams)
    if shuffle:
        np.random.shuffle(indices)

    for i, rcam in enumerate(cam.camera._render_cameras):
        vp_idx = int(indices[i] % len(viewpoints))
        vp = viewpoints[vp_idx]
        pose = sapien_utils.look_at(eye=vp["world_eye"], target=target)
        sp = sapien.Pose(pose.p[0].cpu().numpy(), pose.q[0].cpu().numpy())
        rcam.set_local_pose(sp)

    cam.camera._cached_local_pose = None


def evaluate_hemisphere(agent: RGBDAgent, viewpoints: list, episodes_per_vp: int = 12, device: str = "cuda") -> dict:
    """
    Vectorized benchmark evaluating all 30 Fibonacci viewpoints concurrently in ~4 seconds.
    Uses 120 parallel environments (4 environments allocated to each viewpoint).
    """
    num_vps = len(viewpoints)
    envs_per_vp = 4
    num_envs = num_vps * envs_per_vp  # 120 envs
    agent.eval()

    eval_env = gym.make("PickCube-v1", num_envs=num_envs, obs_mode="rgbd", control_mode="pd_ee_delta_pos", sim_backend="physx_cuda")
    cam = eval_env.unwrapped._sensors["base_camera"]
    for i, rcam in enumerate(cam.camera._render_cameras):
        vp = viewpoints[i % num_vps]
        pose = sapien_utils.look_at(eye=vp["world_eye"], target=TARGET_POS)
        sp = sapien.Pose(pose.p[0].cpu().numpy(), pose.q[0].cpu().numpy())
        rcam.set_local_pose(sp)
    cam.camera._cached_local_pose = None

    eval_env = FlattenRGBDObservationWrapper(eval_env, rgb=True, depth=True, state=True, sep_depth=True)
    if isinstance(eval_env.action_space, gym.spaces.Dict):
        eval_env = FlattenActionSpaceWrapper(eval_env)
    eval_env = ManiSkillVectorEnv(eval_env, num_envs, ignore_terminations=False, record_metrics=True)

    obs, _ = eval_env.reset(seed=999)
    vp_successes = np.zeros(num_vps)
    vp_completed = np.zeros(num_vps)

    # Need roughly (episodes_per_vp / envs_per_vp) * 50 steps = 150 steps for 12 episodes/vp
    max_steps = int((episodes_per_vp / envs_per_vp) * 50) + 10

    with torch.no_grad():
        for step in range(max_steps):
            action = agent.get_action(obs, deterministic=True)
            obs, rew, term, trunc, infos = eval_env.step(action)
            if "final_info" in infos:
                mask = infos["_final_info"]
                for i in range(len(mask)):
                    if mask[i]:
                        vp_idx = i % num_vps
                        ep_info = infos["final_info"]["episode"]
                        succ_key = "success_once" if "success_once" in ep_info else ("success" if "success" in ep_info else "success_at_end")
                        succ = bool(ep_info[succ_key][i].item() if torch.is_tensor(ep_info[succ_key]) else ep_info[succ_key])
                        if succ:
                            vp_successes[vp_idx] += 1
                        vp_completed[vp_idx] += 1

    eval_env.close()

    rates = []
    for vp_idx in range(num_vps):
        c = vp_completed[vp_idx]
        s = vp_successes[vp_idx]
        rate = (s / c) * 100.0 if c > 0 else 0.0
        rates.append(round(float(rate), 1))

    mean_score = float(np.mean(rates))
    return {
        "mean_hemispherical_success": round(mean_score, 2),
        "min_viewpoint_success": round(float(np.min(rates)), 2),
        "max_viewpoint_success": round(float(np.max(rates)), 2),
        "success_rates": rates,
        "episodes_per_vp": int(np.min(vp_completed)),
    }


def train_joint_multiview(
    base_model: str,
    viewpoints_file: str,
    output_dir: str = "/home/maniskill/phase_1_to_3/joint_multiview_training",
    total_timesteps: int = 10000000,
    num_envs: int = 128,
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
    eval_freq_iters: int = 25,
    save_freq_iters: int = 25,
    resume: bool = True,
    device: str = "cuda",
):
    global STOP_REQUESTED
    os.makedirs(output_dir, exist_ok=True)
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    progress_file = os.path.join(output_dir, "joint_training_progress.json")

    with open(viewpoints_file, "r") as f:
        viewpoints = json.load(f)

    batch_size = num_steps * num_envs
    minibatch_size = batch_size // num_minibatches
    total_iterations = total_timesteps // batch_size

    # Check for resume
    start_iter = 1
    model_to_load = base_model
    global_step = 0

    if resume and os.path.exists(progress_file):
        try:
            with open(progress_file, "r") as f:
                prog = json.load(f)
            start_iter = prog.get("current_iteration", 0) + 1
            global_step = prog.get("global_step", 0)
            model_to_load = prog.get("latest_checkpoint", base_model)
            print(f"\n[Resume] Found saved progress! Resuming from Iteration {start_iter}/{total_iterations} (Global Step {global_step:,})")
            print(f"[Resume] Loading weights from: {model_to_load}")
        except Exception as e:
            print(f"[Start] Could not read resume progress: {e}. Starting fresh from: {model_to_load}")
    else:
        print(f"\n[Start] Starting fresh joint RGB-D training from: {model_to_load}")

    print("\n" + "=" * 78)
    print(" JOINT MULTI-VIEW DOMAIN RANDOMIZATION TRAINING (RGB-D 4-CHANNEL)")
    print(f" Observation Mode:    RGB-D (RGB [0..1] + Metric Depth [0..1] Normalized)")
    print(f" Input Channels:      4 Channels [R, G, B, Depth]")
    print(f" Base Model:          {model_to_load}")
    print(f" Viewpoints:          {len(viewpoints)} Fibonacci Equal-Area Cameras (Concurrent)")
    print(f" Parallel Instances:  {num_envs} envs (~{num_envs/len(viewpoints):.1f} envs/viewpoint)")
    print(f" Iterations:          {start_iter} -> {total_iterations} ({batch_size} steps/iter)")
    print(f" Output Directory:    {output_dir}")
    print("=" * 78 + "\n")

    # Create joint multi-view environment with RGB-D
    env = gym.make("PickCube-v1", num_envs=num_envs, obs_mode="rgbd", control_mode="pd_ee_delta_pos", sim_backend="physx_cuda")
    env = FlattenRGBDObservationWrapper(env, rgb=True, depth=True, state=True, sep_depth=True)
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    env = ManiSkillVectorEnv(env, num_envs, ignore_terminations=False, record_metrics=True)

    # Distribute the 30 viewpoints across the 128 parallel environments
    assign_viewpoints_to_envs(env, viewpoints, TARGET_POS, shuffle=False)

    obs_buf = DictArray((num_steps, num_envs), env.single_observation_space, device=device)
    actions_buf = torch.zeros((num_steps, num_envs) + env.single_action_space.shape, device=device)
    logprobs_buf = torch.zeros((num_steps, num_envs), device=device)
    rewards_buf = torch.zeros((num_steps, num_envs), device=device)
    dones_buf = torch.zeros((num_steps, num_envs), device=device)
    values_buf = torch.zeros((num_steps, num_envs), device=device)

    next_obs, _ = env.reset(seed=42)
    next_done = torch.zeros(num_envs, device=device)

    agent = RGBDAgent(env, sample_obs=next_obs).to(device)
    if model_to_load and os.path.exists(model_to_load):
        load_model_weights(agent, model_to_load, device=device)

    optimizer = optim.Adam(agent.parameters(), lr=lr, eps=1e-5)

    latest_eval_metrics = None

    for iteration in range(start_iter, total_iterations + 1):
        if STOP_REQUESTED:
            print("\n[Pause] Stop requested! Safely pausing joint training...")
            break

        iter_start = time.time()
        final_values = torch.zeros((num_steps, num_envs), device=device)

        # Rollout across all 30 viewpoints simultaneously
        agent.eval()
        for step in range(num_steps):
            global_step += num_envs
            obs_buf[step] = next_obs
            dones_buf[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values_buf[step] = value.flatten()
            actions_buf[step] = action
            logprobs_buf[step] = logprob

            next_obs, reward, term, trunc, infos = env.step(action)
            next_done = torch.logical_or(term, trunc).to(torch.float32)
            rewards_buf[step] = reward.view(-1)

            if "final_info" in infos:
                done_mask = infos["_final_info"]
                for k in infos["final_observation"]:
                    infos["final_observation"][k] = infos["final_observation"][k][done_mask]
                with torch.no_grad():
                    final_values[step, torch.arange(num_envs, device=device)[done_mask]] = agent.get_value(infos["final_observation"]).view(-1)

        # GAE Advantages
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards_buf).to(device)
            lastgaelam = 0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    next_not_done = 1.0 - next_done
                    nextvalues = next_value
                else:
                    next_not_done = 1.0 - dones_buf[t + 1]
                    nextvalues = values_buf[t + 1]
                real_next_values = next_not_done * nextvalues + final_values[t]
                delta = rewards_buf[t] + gamma * real_next_values - values_buf[t]
                advantages[t] = lastgaelam = delta + gamma * gae_lambda * next_not_done * lastgaelam
            returns = advantages + values_buf

        # Flatten batch
        b_obs = obs_buf.reshape((-1,))
        b_logprobs = logprobs_buf.reshape(-1)
        b_actions = actions_buf.reshape((-1,) + env.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values_buf.reshape(-1)

        # PPO Optimization
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

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

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
        mean_rew = float(rewards_buf.mean().item())

        # Save instant latest checkpoint
        latest_ckpt = os.path.join(ckpt_dir, "latest_joint_rgbd_model.pt")
        torch.save(agent.state_dict(), latest_ckpt)

        # Periodic checkpoint
        if iteration % save_freq_iters == 0:
            saved_ckpt = os.path.join(ckpt_dir, f"ckpt_joint_rgbd_{iteration:04d}.pt")
            torch.save(agent.state_dict(), saved_ckpt)
            print(f" [Checkpoint] Saved RGB-D model at Iteration {iteration} to: {saved_ckpt}")

        # Periodic full hemispherical benchmark evaluation
        if iteration % eval_freq_iters == 0:
            print(f"\n[Eval] Running Vectorized 30-Viewpoint Hemispherical Benchmark at Iteration {iteration}...")
            latest_eval_metrics = evaluate_hemisphere(agent, viewpoints, episodes_per_vp=12, device=device)
            print(f" >>> [Iteration {iteration:4d}] Overall Mean Hemispherical Generalization: {latest_eval_metrics['mean_hemispherical_success']:.2f}% (Min: {latest_eval_metrics['min_viewpoint_success']:.1f}%, Max: {latest_eval_metrics['max_viewpoint_success']:.1f}%)\n")

        if iteration % 5 == 0 or iteration == start_iter:
            eval_str = f" | Hemisphere Mean: {latest_eval_metrics['mean_hemispherical_success']:.1f}%" if latest_eval_metrics else ""
            print(f" Iteration {iteration:4d}/{total_iterations} | Global Step: {global_step:8,d} | Rew: {mean_rew:5.2f} | SPS: {sps}{eval_str}")

        # Update progress JSON
        prog_data = {
            "current_iteration": iteration,
            "total_iterations": total_iterations,
            "global_step": global_step,
            "mean_reward": round(mean_rew, 3),
            "latest_hemispherical_metrics": latest_eval_metrics,
            "latest_checkpoint": latest_ckpt,
            "status": "interrupted" if STOP_REQUESTED else "training",
            "last_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(progress_file, "w") as f:
            json.dump(prog_data, f, indent=2)

    env.close()
    print(f"\n[Done] Training stopped at Iteration {iteration}. Checkpoint: {latest_ckpt}")


def main():
    parser = argparse.ArgumentParser(description="Joint Multi-View Domain Randomization Training (RGB-D 4-Channel)")
    parser.add_argument("--base_model", type=str, default="/home/maniskill/phase_1_to_3/sequential_viewpoint_training/backup_original_model.pt")
    parser.add_argument("--viewpoints_file", type=str, default="/home/maniskill/phase_1_to_3/sequential_viewpoint_training/viewpoints.json")
    parser.add_argument("--output_dir", type=str, default="/home/maniskill/phase_1_to_3/joint_multiview_training")
    parser.add_argument("--total_timesteps", type=int, default=10000000)
    parser.add_argument("--num_envs", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--eval_freq", type=int, default=25, help="Benchmark all 30 viewpoints every N iterations")
    parser.add_argument("--save_freq", type=int, default=50)
    parser.add_argument("--resume", action="store_true", default=False)
    args = parser.parse_args()

    train_joint_multiview(
        base_model=args.base_model,
        viewpoints_file=args.viewpoints_file,
        output_dir=args.output_dir,
        total_timesteps=args.total_timesteps,
        num_envs=args.num_envs,
        lr=args.lr,
        eval_freq_iters=args.eval_freq,
        save_freq_iters=args.save_freq,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
