"""
Phase 13: 3D Point-Cloud / Base-Frame Back-Projection Architecture.

Solves the 2D projective camera dilemma by:
1. Back-projecting depth observations into 3D Cartesian space in the robot base frame.
2. Filtering the manipulation workspace volume (X in [0.35, 0.85], Y in [-0.35, 0.35], Z in [0.005, 0.40]).
3. Encoding the resulting 3D point cloud with a permutation-invariant PointNet encoder.
4. Conditioning the policy on the robot base frame geometry, which is mathematically
   invariant to camera position, azimuth, elevation, and distance.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def extract_robot_frame_pc(obs: dict, num_points: int = 512) -> torch.Tensor:
    """
    Direct GPU back-projection and robot-frame transformation:
    1. Reads camera-space coordinates from 'position' (in mm).
    2. Converts to homogeneous coordinates and multiplies by 'cam2world_gl'.
    3. Transforms from world frame to robot base frame (base_pos = [-0.615, 0.0, 0.0]).
    4. Filters workspace bounding box [X: 0.35..0.85, Y: -0.35..0.35, Z: 0.005..0.40].
    5. Samples exactly num_points points per environment.
    """
    pos = obs["sensor_data"]["base_camera"]["position"].float() / 1000.0  # mm to m
    cam2world = obs["sensor_param"]["base_camera"]["cam2world_gl"]
    B = pos.shape[0]

    pos_homo = torch.cat([pos, torch.ones_like(pos[..., :1])], dim=-1)
    xyz_world = (pos_homo.reshape(B, -1, 4) @ cam2world.transpose(1, 2))[..., :3]

    robot_base = torch.tensor([-0.615, 0.0, 0.0], device=pos.device)
    xyz_robot = xyz_world - robot_base

    # Filter workspace above table plane
    mask = (
        (xyz_robot[..., 0] >= 0.35) & (xyz_robot[..., 0] <= 0.85) &
        (xyz_robot[..., 1] >= -0.35) & (xyz_robot[..., 1] <= 0.35) &
        (xyz_robot[..., 2] >= 0.005) & (xyz_robot[..., 2] <= 0.40)
    )

    out_pts = torch.zeros((B, num_points, 3), device=pos.device, dtype=torch.float32)
    for b in range(B):
        pts_b = xyz_robot[b, mask[b]]
        n = pts_b.shape[0]
        if n >= num_points:
            perm = torch.randperm(n, device=pos.device)[:num_points]
            out_pts[b] = pts_b[perm]
        elif n > 0:
            idx = torch.randint(0, n, (num_points,), device=pos.device)
            out_pts[b] = pts_b[idx]

    return out_pts


class PointNetEncoder(nn.Module):
    """
    Permutation-invariant PointNet feature extractor:
    Input: (B, N, 3) 3D Cartesian coordinates in robot frame.
    Output: (B, out_dim) global geometric feature vector.
    """
    def __init__(self, in_channels: int = 3, out_dim: int = 256):
        super().__init__()
        self.conv1 = layer_init(nn.Conv1d(in_channels, 64, kernel_size=1))
        self.conv2 = layer_init(nn.Conv1d(64, 128, kernel_size=1))
        self.conv3 = layer_init(nn.Conv1d(128, out_dim, kernel_size=1))
        self.act = nn.ReLU(inplace=True)
        self.out_dim = out_dim

    def forward(self, pts: torch.Tensor) -> torch.Tensor:
        # pts: (B, N, 3) -> transpose to (B, 3, N)
        x = pts.transpose(1, 2)
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        # Max-pooling across points ensures permutation invariance
        global_feat = torch.max(x, dim=2)[0]
        return global_feat


class PointNetAgent(nn.Module):
    """
    3D PointNet Policy Agent:
    Fuses 256-dim 3D geometric embedding with 64-dim robot proprioception.
    Outputs 4-dim delta end-effector position action [dx, dy, dz, dgripper].
    """
    def __init__(self, action_dim: int = 4, proprio_dim: int = 29, point_dim: int = 256):
        super().__init__()
        self.point_encoder = PointNetEncoder(in_channels=3, out_dim=point_dim)
        self.proprio_fc = nn.Sequential(
            layer_init(nn.Linear(proprio_dim, 64)),
            nn.ReLU(inplace=True),
        )

        total_dim = point_dim + 64  # 256 + 64 = 320
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(total_dim, 256)),
            nn.ReLU(inplace=True),
            layer_init(nn.Linear(256, 128)),
            nn.ReLU(inplace=True),
            layer_init(nn.Linear(128, action_dim), std=0.01 * np.sqrt(2)),
        )
        self.actor_logstd = nn.Parameter(torch.ones(1, action_dim) * -0.5)

    def forward(self, pts: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        p_feat = self.point_encoder(pts)
        s_feat = self.proprio_fc(proprio)
        combined = torch.cat([p_feat, s_feat], dim=-1)
        return self.actor_mean(combined)

    def get_action(self, pts: torch.Tensor, proprio: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        action_mean = self.forward(pts, proprio)
        if deterministic:
            return action_mean
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        return probs.sample()
