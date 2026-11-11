"""
Phase 10 Next-Gen Architecture: OmniViewAgent with Temporal Frame Stacking and Extrinsic Conditioning.
- Visual: 3-Frame Stacked RGB-D (12 channels: [RGBD_{t-2}, RGBD_{t-1}, RGBD_t])
- State: 33-dimensional vector (29 proprioception + 4 relative camera pose [x_rel, y_rel, z_rel, dist])
- Action: 4-dimensional continuous control (pd_ee_delta_pos)
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class OmniViewAgent(nn.Module):
    def __init__(self, action_dim: int = 4, state_dim: int = 33, feature_dim: int = 256):
        super().__init__()
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.feature_dim = feature_dim

        # 12-Channel Temporal Visual Encoder (3 frames x 4 channels RGB-D)
        self.cnn = nn.Sequential(
            layer_init(nn.Conv2d(12, 32, kernel_size=8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 12 * 12, feature_dim)),
            nn.ReLU(),
        )

        # 33-Dimensional State & Camera Extrinsics MLP
        self.state_fc = nn.Sequential(
            layer_init(nn.Linear(state_dim, 256)),
            nn.ReLU(),
        )

        fusion_dim = feature_dim + 256  # 256 (visual) + 256 (state + extrinsics) = 512

        # Actor Network (Gaussian Policy)
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(fusion_dim, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, action_dim), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

        # Critic Network (Value Function)
        self.critic = nn.Sequential(
            layer_init(nn.Linear(fusion_dim, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, 1), std=1.0),
        )

    def get_features(self, rgbd: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        vis_feat = self.cnn(rgbd)
        state_feat = self.state_fc(state)
        return torch.cat([vis_feat, state_feat], dim=-1)

    def get_action(self, rgbd: torch.Tensor, state: torch.Tensor, deterministic: bool = False):
        feat = self.get_features(rgbd, state)
        action_mean = self.actor_mean(feat)
        if deterministic:
            return action_mean
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        return probs.sample()

    def get_action_and_value(self, rgbd: torch.Tensor, state: torch.Tensor, action: torch.Tensor = None):
        feat = self.get_features(rgbd, state)
        action_mean = self.actor_mean(feat)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)

        if action is None:
            action = probs.sample()

        log_prob = probs.log_prob(action).sum(dim=-1)
        entropy = probs.entropy().sum(dim=-1)
        value = self.critic(feat).squeeze(-1)
        return action, log_prob, entropy, value

    def get_value(self, rgbd: torch.Tensor, state: torch.Tensor):
        feat = self.get_features(rgbd, state)
        return self.critic(feat).squeeze(-1)

    def warm_start_from_single_frame(self, old_state_dict: dict):
        """
        Transfers pre-trained 4-channel single-frame weights into the 12-channel temporal network.
        - Conv1: 4 channels replicated across 3 frames (scaled by 1/3)
        - State MLP: First 29 columns copied directly; new 4 columns initialized to zero.
        - Higher conv layers, actor, and critic copied verbatim.
        """
        new_dict = self.state_dict()
        for k, v in old_state_dict.items():
            if "feature_net.cnn.0.weight" in k:
                # [32, 4, 8, 8] -> [32, 12, 8, 8]
                w_expanded = v.repeat(1, 3, 1, 1) / 3.0
                new_dict["cnn.0.weight"].copy_(w_expanded)
            elif "feature_net.cnn.0.bias" in k:
                new_dict["cnn.0.bias"].copy_(v)
            elif "feature_net.cnn.2." in k:
                new_dict[k.replace("feature_net.", "")].copy_(v)
            elif "feature_net.cnn.4." in k:
                new_dict[k.replace("feature_net.", "")].copy_(v)
            elif "feature_net.cnn.7." in k:
                new_dict[k.replace("feature_net.", "")].copy_(v)
            elif "feature_net.state_fc.0.weight" in k:
                if v.shape[-1] == self.state_dim:
                    new_dict["state_fc.0.weight"].copy_(v)
                else:
                    n_cols = min(v.shape[-1], self.state_dim)
                    new_dict["state_fc.0.weight"][:, :n_cols].copy_(v[:, :n_cols])
                    new_dict["state_fc.0.weight"][:, n_cols:].zero_()
            elif "feature_net.state_fc.0.bias" in k:
                new_dict["state_fc.0.bias"].copy_(v)
            elif "actor_mean." in k or "critic." in k or "actor_logstd" in k:
                new_dict[k].copy_(v)

        self.load_state_dict(new_dict)
        return self
