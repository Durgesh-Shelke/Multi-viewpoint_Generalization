"""
Maniwhere Model Architecture: Visual Generalizable Framework for Reinforcement Learning (CoRL 2024).
Integrates:
1. Perspective Spatial Transformer Network (STN) with learnable 3x3 homography.
2. Dual-Stream Visual Encoder with intermediate feature map extraction.
3. Multi-View Representation Objective: Symmetrized InfoNCE Contrastive Loss + Feature Map Alignment Loss.
4. Actor-Critic Policy Network for Continuous Control.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal


class PerspectiveSTN(nn.Module):
    """
    Spatial Transformer Network with 3x3 Perspective Homography Transformation.
    Warp arbitrary camera view feature maps to align with the canonical camera frame.
    """
    def __init__(self, in_channels: int = 32):
        super().__init__()
        self.localization = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
        )
        # Initialize to identity transformation (H = eye(3))
        self.localization[-1].weight.data.zero_()
        self.localization[-1].bias.data.copy_(
            torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=torch.float32)
        )

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.size()
        theta = self.localization(x)

        # Unpack into 3x3 Homography Matrix with H[2, 2] = 1.0
        h11, h12, h13, h21, h22, h23, h31, h32 = theta.unbind(dim=1)
        one = torch.ones_like(h11)
        H_mat = torch.stack([
            h11, h12, h13,
            h21, h22, h23,
            h31, h32, one
        ], dim=1).view(B, 3, 3)

        # Generate normalized coordinate grid in [-1, 1]
        y, x_grid = torch.meshgrid(
            torch.linspace(-1.0, 1.0, H, device=x.device),
            torch.linspace(-1.0, 1.0, W, device=x.device),
            indexing="ij",
        )
        ones = torch.ones_like(x_grid)
        target_coords = torch.stack([x_grid, y, ones], dim=0).view(3, -1)  # (3, H*W)
        target_coords = target_coords.unsqueeze(0).repeat(B, 1, 1)  # (B, 3, H*W)

        # Perspective projection: (x^s, y^s, w^s)^T = H * (x^t, y^t, 1)^T
        src_coords = torch.bmm(H_mat, target_coords)
        eps = 1e-6
        xs = src_coords[:, 0, :] / (src_coords[:, 2, :].abs() + eps)
        ys = src_coords[:, 1, :] / (src_coords[:, 2, :].abs() + eps)

        # Numerical clamping to avoid grid sampling invalidity
        xs = torch.clamp(xs, -2.0, 2.0)
        ys = torch.clamp(ys, -2.0, 2.0)

        grid = torch.stack([xs, ys], dim=2).view(B, H, W, 2)
        warped = F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=True)
        return warped, H_mat


class ManiwhereVisualEncoder(nn.Module):
    """
    4-Channel RGB-D Visual Encoder with Embedded Perspective STN and Contrastive Projection Head.
    """
    def __init__(self, in_channels: int = 4, feature_dim: int = 256, proj_dim: int = 128):
        super().__init__()
        # Conv Stage 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
        )
        # Perspective Spatial Transformer Network (STN)
        self.stn = PerspectiveSTN(in_channels=32)

        # Conv Stage 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute flat size: 128x128 -> (128-8)/4 + 1 = 31 -> (31-4)/2 + 1 = 14 -> (14-3)/1 + 1 = 12 -> 64*12*12 = 9216
        flat_dim = 64 * 12 * 12
        self.fc = nn.Sequential(
            nn.Linear(flat_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
        )

        # Contrastive Projector (for Multi-View InfoNCE Loss)
        self.projector = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, proj_dim),
        )

    def forward_features(self, rgbd: torch.Tensor):
        """Returns visual representation, intermediate STN feature map, and contrastive projection."""
        # Convert NHWC -> NCHW
        if rgbd.dim() == 4 and rgbd.shape[1] not in [3, 4] and rgbd.shape[-1] in [3, 4]:
            rgbd = rgbd.permute(0, 3, 1, 2)

        x1 = self.conv1(rgbd)
        x_stn, H_mat = self.stn(x1)
        x2 = self.conv2(x_stn)
        z_vis = self.fc(x2)
        proj = F.normalize(self.projector(z_vis), dim=-1)
        return z_vis, x_stn, proj, H_mat

    def forward(self, rgbd: torch.Tensor):
        z_vis, _, _, _ = self.forward_features(rgbd)
        return z_vis


class ManiwhereAgent(nn.Module):
    """
    Actor-Critic Agent implementing the Maniwhere multi-view framework.
    """
    def __init__(self, action_dim: int = 4, state_dim: int = 29, feature_dim: int = 256):
        super().__init__()
        self.encoder = ManiwhereVisualEncoder(in_channels=4, feature_dim=feature_dim)

        self.state_net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
        )

        fusion_dim = feature_dim + 64

        # Actor Network (End-Effector Delta Position Control)
        self.actor_mean = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

        # Critic Network (Value Function)
        self.critic = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def _prepare_rgbd(self, obs: dict) -> torch.Tensor:
        rgb = obs["rgb"].float() / 255.0
        depth = obs["depth"].float() / 1000.0  # normalize mm to meters
        if rgb.dim() == 3:
            rgb = rgb.unsqueeze(0)
            depth = depth.unsqueeze(0)
        # Concatenate along channel dimension
        if rgb.shape[-1] == 3:
            # NHWC -> concat to NHW4 -> N4HW
            rgbd = torch.cat([rgb, depth], dim=-1).permute(0, 3, 1, 2)
        else:
            # NCHW
            rgbd = torch.cat([rgb, depth], dim=1)
        return rgbd

    def get_features(self, obs: dict):
        rgbd = self._prepare_rgbd(obs)
        z_vis, feat_stn, proj, H_mat = self.encoder.forward_features(rgbd)
        state = obs["state"]
        if state.dim() == 1:
            state = state.unsqueeze(0)
        z_state = self.state_net(state)
        z_fusion = torch.cat([z_vis, z_state], dim=-1)
        return z_fusion, feat_stn, proj, H_mat

    def get_value(self, obs: dict) -> torch.Tensor:
        z_fusion, _, _, _ = self.get_features(obs)
        return self.critic(z_fusion)

    def get_action_and_value(self, obs: dict, action: torch.Tensor = None):
        z_fusion, _, _, _ = self.get_features(obs)
        action_mean = self.actor_mean(z_fusion)
        action_std = self.actor_logstd.exp().expand_as(action_mean)
        dist = Normal(action_mean, action_std)

        if action is None:
            action = dist.rsample()

        logprob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.critic(z_fusion)
        return action, logprob, entropy, value

    def get_action(self, obs: dict, deterministic: bool = True):
        z_fusion, _, _, _ = self.get_features(obs)
        action_mean = self.actor_mean(z_fusion)
        if deterministic:
            return action_mean
        action_std = self.actor_logstd.exp().expand_as(action_mean)
        dist = Normal(action_mean, action_std)
        return dist.sample()

    def compute_maniwhere_representation_loss(
        self,
        obs_fixed: dict,
        obs_move: dict,
        temperature: float = 0.1,
        lambda_feat: float = 200.0,
    ):
        """
        Computes Maniwhere Multi-View Representation Objective (Eq. 1, 2, 3 in CoRL 2024 paper):
        L_Maniwhere = J_con(theta) + lambda * J_feat(theta)
        """
        rgbd_fixed = self._prepare_rgbd(obs_fixed)
        rgbd_move = self._prepare_rgbd(obs_move)

        _, feat_fixed, proj_fixed, _ = self.encoder.forward_features(rgbd_fixed)
        _, feat_move, proj_move, _ = self.encoder.forward_features(rgbd_move)

        # 1. Feature Alignment Loss J_feat (Eq. 2): L2 distance between normalized feature maps
        flat_feat_fixed = F.normalize(feat_fixed.flatten(start_dim=1), dim=-1)
        flat_feat_move = F.normalize(feat_move.flatten(start_dim=1), dim=-1)
        j_feat = F.mse_loss(flat_feat_fixed, flat_feat_move)

        # 2. Symmetrized InfoNCE Contrastive Loss J_con (Eq. 1):
        # proj_fixed: (B, D), proj_move: (B, D)
        B = proj_fixed.size(0)
        sim_matrix = torch.matmul(proj_fixed, proj_move.T) / temperature  # (B, B)
        labels = torch.arange(B, device=proj_fixed.device)

        loss_fixed_to_move = F.cross_entropy(sim_matrix, labels)
        loss_move_to_fixed = F.cross_entropy(sim_matrix.T, labels)
        j_con = 0.5 * (loss_fixed_to_move + loss_move_to_fixed)

        # Total Maniwhere Representation Loss
        l_maniwhere = j_con + lambda_feat * j_feat
        return l_maniwhere, j_con, j_feat
