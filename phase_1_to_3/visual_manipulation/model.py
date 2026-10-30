"""
Embodiment-Agnostic Visuomotor Policy for Robotic Manipulation (Option A: 4.5cm Block).
Pure visual-motor control: Zero object dependencies or artificial labels.
Features:
- Camera RGB stream (128x128x3) -> NatureCNN
- Universal Task-Space TCP pose & gripper state
- Dynamic DoF Joint Encoder (works for any robot arm)
- End-Effector Delta Control: [dx, dy, dz, gripper]
"""

from typing import Dict, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class DynamicJointEncoder(nn.Module):
    """
    Encodes joint states dynamically regardless of robot degrees of freedom (DoF).
    Treats each joint as a token [pos, vel, joint_index] and aggregates via attention pooling.
    """

    def __init__(self, max_dof: int = 16, token_dim: int = 64, out_dim: int = 128):
        super().__init__()
        self.joint_id_embed = nn.Embedding(max_dof, token_dim // 2)
        self.joint_mlp = nn.Sequential(
            nn.Linear(2 + token_dim // 2, token_dim),
            nn.ReLU(),
            nn.Linear(token_dim, out_dim),
        )
        self.pool_attention = nn.Sequential(
            nn.Linear(out_dim, 1),
            nn.Softmax(dim=1),
        )

    def forward(self, qpos: torch.Tensor, qvel: torch.Tensor) -> torch.Tensor:
        b, dof = qpos.shape
        device = qpos.device

        joint_ids = torch.arange(dof, device=device).unsqueeze(0).expand(b, -1)
        id_embeds = self.joint_id_embed(joint_ids)

        joint_features = torch.cat([qpos.unsqueeze(-1), qvel.unsqueeze(-1), id_embeds], dim=-1)
        tokens = self.joint_mlp(joint_features)

        attn_weights = self.pool_attention(tokens)
        pooled = (tokens * attn_weights).sum(dim=1)
        return pooled


class PureVisuomotorAgent(nn.Module):
    """
    End-to-End Pure Visual Policy.
    Maps directly from Camera Pixels + Arm State to Control Actions.
    Zero object conditioning or class assumptions.
    """

    def __init__(
        self,
        action_dim: int = 4,
        in_channels: int = 3,
        tcp_dim: int = 11,
        max_dof: int = 16,
    ):
        super().__init__()
        self.action_dim = action_dim

        # 1. Vision Backbone (NatureCNN)
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(9216, 256),
            nn.ReLU(),
        )

        # 2. Universal TCP / Task-Space projection
        self.tcp_mlp = nn.Sequential(
            nn.Linear(tcp_dim, 64),
            nn.ReLU(),
        )

        # 3. Dynamic DoF Joint Encoder
        self.joint_encoder = DynamicJointEncoder(max_dof=max_dof, token_dim=64, out_dim=128)

        # Latent Fusion: Vision (256) + TCP (64) + Dynamic Joints (128) = 448 dims
        latent_dim = 256 + 64 + 128

        # Actor head
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(latent_dim, 512)),
            nn.ReLU(inplace=True),
            layer_init(nn.Linear(512, action_dim), std=0.01 * np.sqrt(2)),
        )
        self.actor_logstd = nn.Parameter(torch.ones(1, action_dim) * -0.5)

        # Critic head
        self.critic = nn.Sequential(
            layer_init(nn.Linear(latent_dim, 512)),
            nn.ReLU(inplace=True),
            layer_init(nn.Linear(512, 1)),
        )

    def extract_features(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Vision
        rgb = obs["rgb"].float().permute(0, 3, 1, 2) / 255.0
        vis_feat = self.cnn(rgb)

        # Universal TCP pose
        tcp_feat = self.tcp_mlp(obs["tcp_state"])

        # Dynamic Joint State
        joint_feat = self.joint_encoder(obs["qpos"], obs["qvel"])

        # Multi-modal fusion
        latent = torch.cat([vis_feat, tcp_feat, joint_feat], dim=-1)
        return latent

    def get_action(self, obs: Dict[str, torch.Tensor], deterministic: bool = False) -> torch.Tensor:
        features = self.extract_features(obs)
        mean = self.actor_mean(features)
        if deterministic:
            return mean
        std = torch.exp(self.actor_logstd.expand_as(mean))
        return Normal(mean, std).sample()

    def get_action_and_value(self, obs: Dict[str, torch.Tensor], action: torch.Tensor = None):
        features = self.extract_features(obs)
        mean = self.actor_mean(features)
        std = torch.exp(self.actor_logstd.expand_as(mean))
        probs = Normal(mean, std)
        if action is None:
            action = probs.sample()
        value = self.critic(features)
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), value
