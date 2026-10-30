"""
Modular Visual Policy Architectures for Resolution & Asymmetric Sensor Fusion Study.
Candidates:
  - Candidate A: Symmetric 128x128 RGB-D (Baseline)
  - Candidate B: Symmetric 256x256 RGB-D
  - Candidate C: Asymmetric Decoupled (Low-Res RGB 64x64 + High-Res Depth 256x256)
  - Candidate D: High-Res Pure Depth (256x256 Depth Only)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def preprocess_rgbd(observations, candidate="A"):
    """
    Standardized visual pre-processing:
    - RGB: uint8 [0..255] -> float32 [0..1]
    - Depth: int16 (millimeters) -> meters -> clamped [0..3m] -> normalized [0..1]
    """
    rgb = observations.get("rgb", None)
    depth = observations.get("depth", None)

    d_norm = None
    if depth is not None:
        d = depth.permute(0, 3, 1, 2).float()
        d.masked_fill_(d <= 0, 3000.0)
        d_norm = d.div_(1000.0).clamp_(0.0, 3.0).div_(3.0)

    rgb_norm = None
    if rgb is not None:
        rgb_norm = rgb.permute(0, 3, 1, 2).float().div_(255.0)

    return rgb_norm, d_norm


class NatureCNN128(nn.Module):
    """Candidate A: Symmetric 128x128 4-Channel RGB-D Encoder."""

    def __init__(self, sample_obs, feature_size=256):
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
        self.out_features = feature_size
        if "state" in sample_obs:
            state_dim = sample_obs["state"].shape[-1]
            self.state_fc = nn.Sequential(layer_init(nn.Linear(state_dim, 256)), nn.ReLU())
            self.out_features += 256

    def forward(self, observations):
        rgb_norm, d_norm = preprocess_rgbd(observations, "A")
        x = torch.cat([rgb_norm, d_norm], dim=1)
        vis_feat = self.cnn(x)
        if hasattr(self, "state_fc") and "state" in observations:
            return torch.cat([vis_feat, self.state_fc(observations["state"])], dim=1)
        return vis_feat


class NatureCNN256(nn.Module):
    """Candidate B: Symmetric 256x256 4-Channel RGB-D Encoder."""

    def __init__(self, sample_obs, feature_size=256):
        super().__init__()
        self.feature_size = feature_size
        self.cnn = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 128, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(128 * 12 * 12, feature_size)),
            nn.ReLU(),
        )
        self.out_features = feature_size
        if "state" in sample_obs:
            state_dim = sample_obs["state"].shape[-1]
            self.state_fc = nn.Sequential(layer_init(nn.Linear(state_dim, 256)), nn.ReLU())
            self.out_features += 256

    def forward(self, observations):
        rgb_norm, d_norm = preprocess_rgbd(observations, "B")
        x = torch.cat([rgb_norm, d_norm], dim=1)
        vis_feat = self.cnn(x)
        if hasattr(self, "state_fc") and "state" in observations:
            return torch.cat([vis_feat, self.state_fc(observations["state"])], dim=1)
        return vis_feat


class AsymmetricDecoupledCNN(nn.Module):
    """Candidate C: Asymmetric Dual-Stream (Low-Res RGB 64x64 + High-Res Depth 256x256)."""

    def __init__(self, sample_obs, rgb_features=128, depth_features=256):
        super().__init__()
        self.rgb_cnn = nn.Sequential(
            layer_init(nn.Conv2d(3, 16, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(16, 32, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 32, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(32 * 12 * 12, rgb_features)),
            nn.ReLU(),
        )
        self.depth_cnn = nn.Sequential(
            layer_init(nn.Conv2d(1, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 12 * 12, depth_features)),
            nn.ReLU(),
        )
        self.out_features = rgb_features + depth_features
        if "state" in sample_obs:
            state_dim = sample_obs["state"].shape[-1]
            self.state_fc = nn.Sequential(layer_init(nn.Linear(state_dim, 256)), nn.ReLU())
            self.out_features += 256

    def forward(self, observations):
        rgb_norm, d_norm = preprocess_rgbd(observations, "C")
        # Downscale RGB to 64x64 for semantic encoding
        rgb_64 = F.interpolate(rgb_norm, size=(64, 64), mode="bilinear", align_corners=False)
        rgb_feat = self.rgb_cnn(rgb_64)
        depth_feat = self.depth_cnn(d_norm)
        vis_feat = torch.cat([rgb_feat, depth_feat], dim=1)
        if hasattr(self, "state_fc") and "state" in observations:
            return torch.cat([vis_feat, self.state_fc(observations["state"])], dim=1)
        return vis_feat


class PureDepthCNN256(nn.Module):
    """Candidate D: Pure Depth 256x256 (RGB completely ignored)."""

    def __init__(self, sample_obs, feature_size=256):
        super().__init__()
        self.feature_size = feature_size
        self.cnn = nn.Sequential(
            layer_init(nn.Conv2d(1, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 128, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(128 * 12 * 12, feature_size)),
            nn.ReLU(),
        )
        self.out_features = feature_size
        if "state" in sample_obs:
            state_dim = sample_obs["state"].shape[-1]
            self.state_fc = nn.Sequential(layer_init(nn.Linear(state_dim, 256)), nn.ReLU())
            self.out_features += 256

    def forward(self, observations):
        _, d_norm = preprocess_rgbd(observations, "D")
        vis_feat = self.cnn(d_norm)
        if hasattr(self, "state_fc") and "state" in observations:
            return torch.cat([vis_feat, self.state_fc(observations["state"])], dim=1)
        return vis_feat


class PureRGB256CNN(nn.Module):
    """Candidate E: Pure RGB 256x256 (Depth completely ignored)."""

    def __init__(self, sample_obs, feature_size=256):
        super().__init__()
        self.feature_size = feature_size
        self.cnn = nn.Sequential(
            layer_init(nn.Conv2d(3, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 128, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(128 * 12 * 12, feature_size)),
            nn.ReLU(),
        )
        self.out_features = feature_size
        if "state" in sample_obs:
            state_dim = sample_obs["state"].shape[-1]
            self.state_fc = nn.Sequential(layer_init(nn.Linear(state_dim, 256)), nn.ReLU())
            self.out_features += 256

    def forward(self, observations):
        rgb_norm, _ = preprocess_rgbd(observations, "E")
        vis_feat = self.cnn(rgb_norm)
        if hasattr(self, "state_fc") and "state" in observations:
            return torch.cat([vis_feat, self.state_fc(observations["state"])], dim=1)
        return vis_feat


class UnifiedAblationAgent(nn.Module):
    """Unified Actor-Critic Policy Supporting All 5 Candidates."""

    def __init__(self, envs, sample_obs, candidate="A"):
        super().__init__()
        self.candidate = candidate.upper()
        action_dim = int(np.prod(envs.unwrapped.single_action_space.shape))

        if self.candidate == "A":
            self.feature_net = NatureCNN128(sample_obs)
        elif self.candidate == "B":
            self.feature_net = NatureCNN256(sample_obs)
        elif self.candidate == "C":
            self.feature_net = AsymmetricDecoupledCNN(sample_obs)
        elif self.candidate == "D":
            self.feature_net = PureDepthCNN256(sample_obs)
        elif self.candidate == "E":
            self.feature_net = PureRGB256CNN(sample_obs)
        else:
            raise ValueError(f"Unknown candidate: {candidate}")

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
