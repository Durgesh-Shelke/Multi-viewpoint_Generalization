"""
Phase 16: Composite Modular Universal Architecture.

Combines:
1. ViewpointPerceptionEngine: 3D PointNet spatial reasoning engine (solves Viewpoint Generalization).
2. EmbodimentKinematicEngine: Kinematic adaptation engine (solves DoF/Embodiment Generalization).

Provides a single unified torch.nn.Module deliverable ('UniversalCompositeOmniPolicy').
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ViewpointPerceptionEngine(nn.Module):
    """
    Sub-Module 1: Viewpoint Generalist Engine.
    Operates on metric 3D table workspace geometry and universal 12-dim proprioception.
    Outputs: Universal Task-Space Cartesian Action: [dx, dy, dz, a_gripper] (+1 = close, -1 = open).
    """
    def __init__(self, action_dim=4, proprio_dim=12, point_dim=256):
        super().__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, point_dim, 1)
        nn.init.orthogonal_(self.conv1.weight, np.sqrt(2))
        nn.init.orthogonal_(self.conv2.weight, np.sqrt(2))
        nn.init.orthogonal_(self.conv3.weight, np.sqrt(2))

        self.proprio_fc = nn.Sequential(
            nn.Linear(proprio_dim, 64),
            nn.ReLU(inplace=True),
        )
        nn.init.orthogonal_(self.proprio_fc[0].weight, np.sqrt(2))

        self.mlp = nn.Sequential(
            nn.Linear(point_dim + 64, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, action_dim),
        )
        nn.init.orthogonal_(self.mlp[0].weight, np.sqrt(2))
        nn.init.orthogonal_(self.mlp[2].weight, np.sqrt(2))
        nn.init.orthogonal_(self.mlp[4].weight, 0.01)

    def forward(self, pts, proprio):
        # pts: [B, N, 3] -> [B, 3, N]
        x = F.relu(self.conv1(pts.transpose(1, 2)))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        p_feat = torch.max(x, dim=2)[0]
        s_feat = self.proprio_fc(proprio)
        combined = torch.cat([p_feat, s_feat], dim=-1)
        return self.mlp(combined)


class EmbodimentKinematicEngine(nn.Module):
    """
    Sub-Module 2: Embodiment & DoF Kinematic Engine.
    Adapts universal Cartesian intent to specific robot embodiment constraints,
    handling gripper sign conventions, actuator limits, and neural differential IK.
    """
    ROBOT_MAP = {
        "panda": 0,
        "xarm6_robotiq": 1,
        "xarm6_panda_gripper": 2,
    }

    def __init__(self, max_dof=7, embed_dim=16):
        super().__init__()
        self.max_dof = max_dof
        self.embed_dim = embed_dim
        self.robot_embed = nn.Embedding(10, embed_dim)

        # 1. Cartesian Actuation Adapter (maps universal -> hardware controller action)
        self.cartesian_adapter = nn.Sequential(
            nn.Linear(4 + embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
        )

        # 2. Neural Differential Inverse Kinematics (maps Cartesian delta + q -> delta q)
        self.neural_ik = nn.Sequential(
            nn.Linear(4 + max_dof + embed_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, max_dof),
        )

    def forward_cartesian(self, cartesian_univ, robot_id):
        """
        Adapts universal [dx, dy, dz, a_gripper] (+1=close) to hardware Cartesian control.
        """
        B = cartesian_univ.shape[0]
        if isinstance(robot_id, int):
            robot_id_t = torch.full((B,), robot_id, dtype=torch.long, device=cartesian_univ.device)
        else:
            robot_id_t = robot_id
        emb = self.robot_embed(robot_id_t)
        inp = torch.cat([cartesian_univ, emb], dim=-1)
        # Residual adaptation
        delta = self.cartesian_adapter(inp)
        out = cartesian_univ + 0.1 * delta

        # Explicit sign routing for gripper (Panda vs Robotiq)
        is_robotiq = (robot_id_t == 1).unsqueeze(-1)
        # For Franka parallel: -1 is close, +1 is open (inverse of universal)
        # For Robotiq linkage: +1 is close, -1 is open (same as universal)
        grip_act = torch.where(is_robotiq, cartesian_univ[:, 3:4], -cartesian_univ[:, 3:4])
        out = torch.cat([out[:, :3], grip_act], dim=-1)
        return out

    def forward_ik(self, cartesian_univ, qpos_arm, robot_id):
        """
        Computes joint velocity delta dq for the arm using Neural Differential IK.
        """
        B = cartesian_univ.shape[0]
        if isinstance(robot_id, int):
            robot_id_t = torch.full((B,), robot_id, dtype=torch.long, device=cartesian_univ.device)
        else:
            robot_id_t = robot_id
        emb = self.robot_embed(robot_id_t)

        # Pad qpos to max_dof if needed
        if qpos_arm.shape[-1] < self.max_dof:
            pad = torch.zeros((B, self.max_dof - qpos_arm.shape[-1]), device=qpos_arm.device)
            qpos_padded = torch.cat([qpos_arm, pad], dim=-1)
        else:
            qpos_padded = qpos_arm[..., :self.max_dof]

        inp = torch.cat([cartesian_univ, qpos_padded, emb], dim=-1)
        delta_q = self.neural_ik(inp)
        return delta_q


class UniversalCompositeOmniPolicy(nn.Module):
    """
    The Single Deliverable Composite Model:
    Assembled from two decoupled, independently-developed submodules:
    1. viewpoint_engine (3D PointNet Spatial Perception)
    2. kinematic_engine (DoF Kinematic Adaptation & Neural IK)
    """
    def __init__(self, action_dim=4, proprio_dim=12, point_dim=256, max_dof=7):
        super().__init__()
        self.viewpoint_engine = ViewpointPerceptionEngine(action_dim, proprio_dim, point_dim)
        self.kinematic_engine = EmbodimentKinematicEngine(max_dof)

    def forward(self, pts, proprio, qpos_arm=None, robot_id=0):
        # Step 1: Solve Viewpoint Invariance (Spatial Perception in 3D Metric Coordinates)
        cartesian_univ = self.viewpoint_engine(pts, proprio)

        # Step 2: Solve DoF Invariance (Kinematic Embodiment Adaptation)
        cartesian_hardware = self.kinematic_engine.forward_cartesian(cartesian_univ, robot_id)

        joint_delta = None
        if qpos_arm is not None:
            joint_delta = self.kinematic_engine.forward_ik(cartesian_univ, qpos_arm, robot_id)

        return {
            "cartesian_universal": cartesian_univ,
            "cartesian_hardware": cartesian_hardware,
            "joint_delta": joint_delta,
        }

    def get_action(self, pts, proprio, robot_id=0):
        """Standard Cartesian interface matching ManiSkill controllers."""
        self.eval()
        with torch.no_grad():
            res = self.forward(pts, proprio, robot_id=robot_id)
            return res["cartesian_hardware"]

    def copy_viewpoint_weights_from_checkpoint(self, checkpoint_path, device="cuda"):
        """Performs weight surgery copying pre-trained spatial perception weights."""
        ckpt = torch.load(checkpoint_path, map_location=device)
        self.viewpoint_engine.load_state_dict(ckpt, strict=True)
        print(f"[OK] Viewpoint perception weights successfully loaded from {checkpoint_path}")

    def save_deliverable(self, save_path):
        """Exports the assembled composite policy as a single standalone checkpoint."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(self.state_dict(), save_path)
        print(f"[OK] Universal Composite Model exported to {save_path}")

    @classmethod
    def load_deliverable(cls, checkpoint_path, device="cuda"):
        """Loads the single deliverable model."""
        model = cls().to(device)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()
        print(f"[OK] Universal Composite Model loaded from {checkpoint_path}")
        return model
