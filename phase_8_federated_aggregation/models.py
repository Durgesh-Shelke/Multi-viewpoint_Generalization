"""
Modular Visual Policy Architectures for Federated Policy Aggregation.
Candidate A: Symmetric 128x128 4-Channel RGB-D Encoder.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def preprocess_rgbd(observations):
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
        rgb_norm, d_norm = preprocess_rgbd(observations)
        x = torch.cat([rgb_norm, d_norm], dim=1)
        vis_feat = self.cnn(x)
        if hasattr(self, "state_fc") and "state" in observations:
            return torch.cat([vis_feat, self.state_fc(observations["state"])], dim=1)
        return vis_feat


class UnifiedAblationAgent(nn.Module):
    def __init__(self, envs, sample_obs, candidate="A"):
        super().__init__()
        action_dim = int(np.prod(envs.unwrapped.single_action_space.shape))
        self.feature_net = NatureCNN128(sample_obs)
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
