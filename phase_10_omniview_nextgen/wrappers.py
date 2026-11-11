"""
Temporal Frame Stacking and Camera Extrinsics Conditioning Wrappers for Phase 10.
"""

import gymnasium as gym
import torch


class TemporalExtrinsicWrapper(gym.ObservationWrapper):
    """
    Maintains a 3-frame history of standardized RGB-D frames (12 channels total)
    and augments the 29-dim robot proprioceptive state with 4-dim relative camera extrinsics
    [x_rel, y_rel, z_rel, dist] injected with Gaussian calibration noise.
    """
    def __init__(self, env, cam_eye, frame_stack: int = 3, noise_std: float = 0.02, device: str = "cuda"):
        super().__init__(env)
        self.frame_stack = frame_stack
        self.noise_std = noise_std
        self.device = torch.device(device)
        self.cam_eye = torch.as_tensor(cam_eye, dtype=torch.float32, device=self.device)
        # Franka robot base position in ManiSkill PickCube-v1 world frame
        self.base_pos = torch.tensor([-0.615, 0.0, 0.0], dtype=torch.float32, device=self.device)
        self.history = []

    def _prepare_single_frame(self, obs: dict) -> torch.Tensor:
        # RGB: [B, H, W, 3] -> [B, 3, H, W] in [0, 1]
        rgb = obs["rgb"].permute(0, 3, 1, 2).float() / 255.0

        # Depth: [B, H, W, 1] -> [B, 1, H, W] in [0, 1] (standardized mm -> meters -> clamped to 3m -> [0, 1])
        d = obs["depth"].permute(0, 3, 1, 2).float()
        d.masked_fill_(d <= 0, 3000.0)
        depth = d.div_(1000.0).clamp_(0.0, 3.0).div_(3.0)

        # 4-channel RGB-D
        return torch.cat([rgb, depth], dim=1)

    def _get_noisy_extrinsics(self, batch_size: int) -> torch.Tensor:
        rel_pos = self.cam_eye - self.base_pos
        if self.noise_std > 0:
            noise = torch.randn((batch_size, 3), device=self.device) * self.noise_std
            noisy_rel = rel_pos.unsqueeze(0).repeat(batch_size, 1) + noise
        else:
            noisy_rel = rel_pos.unsqueeze(0).repeat(batch_size, 1)

        dist = torch.norm(noisy_rel, dim=-1, keepdim=True)
        return torch.cat([noisy_rel, dist], dim=-1)  # [B, 4]

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        rgbd = self._prepare_single_frame(obs)
        # Initialize buffer with 3 copies of the initial frame
        self.history = [rgbd] * self.frame_stack

        stacked_rgbd = torch.cat(self.history, dim=1)  # [B, 12, H, W]
        cam_ext = self._get_noisy_extrinsics(obs["state"].shape[0])
        new_state = torch.cat([obs["state"].float(), cam_ext], dim=-1)  # [B, 33]

        return {"rgbd": stacked_rgbd, "state": new_state}, info

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        rgbd = self._prepare_single_frame(obs)

        # FIFO push
        self.history.pop(0)
        self.history.append(rgbd)

        stacked_rgbd = torch.cat(self.history, dim=1)  # [B, 12, H, W]
        cam_ext = self._get_noisy_extrinsics(obs["state"].shape[0])
        new_state = torch.cat([obs["state"].float(), cam_ext], dim=-1)  # [B, 33]

        return {"rgbd": stacked_rgbd, "state": new_state}, reward, term, trunc, info
