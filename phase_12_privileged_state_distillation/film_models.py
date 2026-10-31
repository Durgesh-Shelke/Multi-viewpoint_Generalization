"""
Phase 12 Architecture: FiLM-Conditioned Visual Policy for Viewpoint Generalization.
Applies Feature-wise Linear Modulation (FiLM) inside the convolutional backbone,
allowing camera extrinsics [x_rel, y_rel, z_rel, dist] to spatially modulate
visual representations at multiple convolutional scales.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def preprocess_rgbd(rgb_uint8: torch.Tensor, depth_int16: torch.Tensor):
    """
    Standardized visual pre-processing:
    - RGB: uint8 [0..255] -> float32 [0..1]
    - Depth: int16 (mm) -> meters -> clamped [0..3m] -> normalized [0..1]
    """
    d = depth_int16.permute(0, 3, 1, 2).float()
    d.masked_fill_(d <= 0, 3000.0)
    d_norm = d.div_(1000.0).clamp_(0.0, 3.0).div_(3.0)
    rgb_norm = rgb_uint8.permute(0, 3, 1, 2).float().div_(255.0)
    return torch.cat([rgb_norm, d_norm], dim=1)


class FiLMBlock(nn.Module):
    """
    Feature-wise Linear Modulation:
    h_out = gamma(c_ext) * Conv(h_in) + beta(c_ext)
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0, ext_dim=4):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.film_gen = nn.Sequential(
            nn.Linear(ext_dim, out_channels * 2),
        )
        # Initialize gamma around 1.0 and beta around 0.0
        nn.init.constant_(self.film_gen[0].weight, 0.0)
        nn.init.constant_(self.film_gen[0].bias[:out_channels], 1.0)
        nn.init.constant_(self.film_gen[0].bias[out_channels:], 0.0)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x, ext):
        h = self.conv(x)
        film_params = self.film_gen(ext)
        gamma, beta = torch.chunk(film_params, 2, dim=-1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return self.act(gamma * h + beta)


class FiLMNatureCNN128(nn.Module):
    """
    128x128 4-Channel RGB-D Encoder with Early FiLM Extrinsics Modulation.
    State: 33 dimensions (29 proprioceptive + 4 camera extrinsics).
    """
    def __init__(self, state_dim=33, ext_dim=4, out_features=512):
        super().__init__()
        self.state_dim = state_dim
        self.ext_dim = ext_dim
        self.out_features = out_features

        # 3 FiLM-modulated Conv stages for 128x128 RGB-D
        self.block1 = FiLMBlock(in_channels=4, out_channels=32, kernel_size=8, stride=4, padding=0, ext_dim=ext_dim)
        self.block2 = FiLMBlock(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=0, ext_dim=ext_dim)
        self.block3 = FiLMBlock(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0, ext_dim=ext_dim)

        self.fc = nn.Sequential(
            layer_init(nn.Linear(64 * 12 * 12, out_features)),
            nn.ReLU(inplace=True),
        )
        self.state_fc = nn.Sequential(
            layer_init(nn.Linear(state_dim, 64)),
            nn.ReLU(inplace=True),
        )

    def forward(self, rgb, depth, state):
        # Extract extrinsics (last 4 elements of state: [x_rel, y_rel, z_rel, dist])
        ext = state[:, -self.ext_dim:]
        x = preprocess_rgbd(rgb, depth)

        h1 = self.block1(x, ext)
        h2 = self.block2(h1, ext)
        h3 = self.block3(h2, ext)

        flat = torch.flatten(h3, start_dim=1)
        vis_feat = self.fc(flat)
        state_feat = self.state_fc(state)

        return torch.cat([vis_feat, state_feat], dim=1)


class FiLMPhase12Agent(nn.Module):
    def __init__(self, action_dim=4, state_dim=33):
        super().__init__()
        self.feature_net = FiLMNatureCNN128(state_dim=state_dim, ext_dim=4)
        latent_size = self.feature_net.out_features + 64  # 512 + 64 = 576

        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(latent_size, 256)),
            nn.ReLU(inplace=True),
            layer_init(nn.Linear(256, 128)),
            nn.ReLU(inplace=True),
            layer_init(nn.Linear(128, action_dim), std=0.01 * np.sqrt(2)),
        )
        self.actor_logstd = nn.Parameter(torch.ones(1, action_dim) * -0.5)

    def get_action(self, rgb, depth, state, deterministic=False):
        feat = self.feature_net(rgb, depth, state)
        action_mean = self.actor_mean(feat)
        if deterministic:
            return action_mean
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        return probs.sample()
