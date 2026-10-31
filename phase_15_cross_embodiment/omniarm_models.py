"""
Phase 15: Cross-Embodiment Multi-Arm Architecture (OmniArm PointNet).

Supports cross-embodiment generalization across diverse robotic arms:
1. Franka Emika Panda (7 DoF + parallel gripper, 4-dim EE action)
2. UFactory xArm 6 + Robotiq 2F-85 (6 DoF + parallel gripper, 4-dim EE action)
3. Low-Cost SO-100 (5 DoF + jaw gripper, 6-dim joint action)
4. Trossen WidowX 250 AI (6 DoF + parallel gripper, 8-dim joint action)

Key Design:
- Shared 3D Point-Cloud Visual Backbone: Encodes workspace Cartesian geometry into
  a universal 256-dim feature representation that is invariant to camera perspective
  and identical across all robot embodiments.
- Embodiment-Conditioned Proprioceptive Adapters: Projects robot-specific joint
  and gripper telemetry into a standardized 64-dim representation.
- Modular Embodiment Action Heads: Dispatches fused representation to the appropriate
  kinematic control head for the active robot arm.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

EMBODIMENT_NAMES = ["panda", "xarm6_robotiq", "so100", "widowxai"]
ROBOT_PROPRIO_DIMS = {
    "panda": 29,
    "xarm6_robotiq": 35,
    "so100": 23,
    "widowxai": 27,
}
ROBOT_ACTION_DIMS = {
    "panda": 4,           # pd_ee_delta_pos
    "xarm6_robotiq": 4,   # pd_ee_delta_pos
    "so100": 6,           # pd_joint_delta_pos
    "widowxai": 8,        # pd_joint_delta_pos
}
ROBOT_CONTROL_MODES = {
    "panda": "pd_ee_delta_pos",
    "xarm6_robotiq": "pd_ee_delta_pos",
    "so100": "pd_joint_delta_pos",
    "widowxai": "pd_joint_delta_pos",
}


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def extract_robot_frame_pc(obs: dict, num_points: int = 512, robot_base_xyz=(-0.615, 0.0, 0.0)) -> torch.Tensor:
    """
    Back-projects depth observations into 3D Cartesian space in the robot base frame.
    """
    pos = obs["sensor_data"]["base_camera"]["position"].float() / 1000.0
    cam2world = obs["sensor_param"]["base_camera"]["cam2world_gl"].to(pos.device)
    B = pos.shape[0]

    pos_homo = torch.cat([pos, torch.ones_like(pos[..., :1])], dim=-1)
    xyz_world = (pos_homo.reshape(B, -1, 4) @ cam2world.transpose(1, 2))[..., :3]

    robot_base = torch.tensor(robot_base_xyz, device=pos.device, dtype=torch.float32)
    xyz_robot = xyz_world - robot_base

    # Workspace filter (tabletop interaction envelope)
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
    """Universal 3D point cloud encoder."""
    def __init__(self, in_channels: int = 3, out_dim: int = 256):
        super().__init__()
        self.conv1 = layer_init(nn.Conv1d(in_channels, 64, kernel_size=1))
        self.conv2 = layer_init(nn.Conv1d(64, 128, kernel_size=1))
        self.conv3 = layer_init(nn.Conv1d(128, out_dim, kernel_size=1))
        self.act = nn.ReLU(inplace=True)
        self.out_dim = out_dim

    def forward(self, pts: torch.Tensor) -> torch.Tensor:
        x = pts.transpose(1, 2)
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        return torch.max(x, dim=2)[0]


class SingleArmPointNetAgent(nn.Module):
    """Specialist policy for a single robot arm embodiment."""
    def __init__(self, robot_uid: str = "panda"):
        super().__init__()
        self.robot_uid = robot_uid
        proprio_dim = ROBOT_PROPRIO_DIMS[robot_uid]
        action_dim = ROBOT_ACTION_DIMS[robot_uid]

        self.point_encoder = PointNetEncoder(in_channels=3, out_dim=256)
        self.proprio_fc = nn.Sequential(
            layer_init(nn.Linear(proprio_dim, 64)),
            nn.ReLU(inplace=True),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(256 + 64, 256)),
            nn.ReLU(inplace=True),
            layer_init(nn.Linear(256, 128)),
            nn.ReLU(inplace=True),
            layer_init(nn.Linear(128, action_dim), std=0.01 * np.sqrt(2)),
        )
        self.actor_logstd = nn.Parameter(torch.ones(1, action_dim) * -0.5)

    def forward(self, pts: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        p_feat = self.point_encoder(pts)
        s_feat = self.proprio_fc(proprio)
        return self.actor_mean(torch.cat([p_feat, s_feat], dim=-1))

    def get_action(self, pts: torch.Tensor, proprio: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        action_mean = self.forward(pts, proprio)
        if deterministic:
            return action_mean
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        return probs.sample()


class UniversalOmniArmAgent(nn.Module):
    """
    Universal Cross-Embodiment Policy:
    A single model that controls any of the 4 robot arms across continuous camera viewpoints.
    """
    def __init__(self):
        super().__init__()
        # Shared 3D geometric representation
        self.point_encoder = PointNetEncoder(in_channels=3, out_dim=256)

        # Embodiment-specific proprioceptive adapters
        self.proprio_adapters = nn.ModuleDict({
            r: nn.Sequential(
                layer_init(nn.Linear(ROBOT_PROPRIO_DIMS[r], 64)),
                nn.ReLU(inplace=True),
            )
            for r in EMBODIMENT_NAMES
        })

        # Core shared policy trunk
        self.shared_trunk = nn.Sequential(
            layer_init(nn.Linear(256 + 64, 256)),
            nn.ReLU(inplace=True),
            layer_init(nn.Linear(256, 128)),
            nn.ReLU(inplace=True),
        )

        # Modular action heads per embodiment
        self.action_heads = nn.ModuleDict({
            r: layer_init(nn.Linear(128, ROBOT_ACTION_DIMS[r]), std=0.01 * np.sqrt(2))
            for r in EMBODIMENT_NAMES
        })

    def forward(self, pts: torch.Tensor, proprio: torch.Tensor, robot_uid: str) -> torch.Tensor:
        p_feat = self.point_encoder(pts)
        s_feat = self.proprio_adapters[robot_uid](proprio)
        fused = torch.cat([p_feat, s_feat], dim=-1)
        latent = self.shared_trunk(fused)
        return self.action_heads[robot_uid](latent)

    def get_action(self, pts: torch.Tensor, proprio: torch.Tensor, robot_uid: str, deterministic: bool = True) -> torch.Tensor:
        return self.forward(pts, proprio, robot_uid)
