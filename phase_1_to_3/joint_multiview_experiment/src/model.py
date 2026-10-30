"""
Vision-based Actor-Critic Policy with NatureCNN Backbone per SPEC Section 1 & 3.1.

Supports:
- Modality: RGB (3 input channels) or RGB-D (4 input channels: RGB + normalized metric depth).
- Fully orthogonal initialization of all convolutional and linear layers.
- Visual branch: NatureCNN -> Linear(9216 -> 256) -> ReLU.
- State branch: Linear(29 -> 256) -> ReLU.
- Fused representation: 512-dimensional latent vector (256 visual + 256 state).
- Actor head: Linear(512 -> 512) -> ReLU -> Linear(512 -> action_dim, std=0.01 * sqrt(2)).
- Critic head: Linear(512 -> 512) -> ReLU -> Linear(512 -> 1, std=1.0).

Python 3.9 compatible only.
"""

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal


def layer_init(
    layer: nn.Module,
    std: float = np.sqrt(2),
    bias_const: float = 0.0,
) -> nn.Module:
    """Initializes layer weights orthogonally and biases with a constant."""
    if hasattr(layer, "weight") and layer.weight is not None:
        torch.nn.init.orthogonal_(layer.weight, std)
    if hasattr(layer, "bias") and layer.bias is not None:
        torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def preprocess_depth(depth_tensor: torch.Tensor) -> torch.Tensor:
    """
    Standard depth preprocessing per SPEC Section 1:
    - depth in millimeters (int16 or float)
    - replace non-positive/far (<= 0) with 3000.0 mm
    - clamp to [0.0, 3.0] meters
    - normalize to [0.0, 1.0]
    """
    d = depth_tensor.float()
    d = torch.where(d <= 0, torch.full_like(d, 3000.0), d)
    d_m = torch.clamp(d / 1000.0, 0.0, 3.0)
    return d_m / 3.0


class NatureCNN(nn.Module):
    """
    NatureCNN Visual Feature Extractor with fully orthogonal initialization.
    Accepts 3 channels (RGB) or 4 channels (RGB-D) at 128x128 resolution.
    """

    def __init__(
        self,
        in_channels: int = 4,
        feature_size: int = 256,
        state_dim: int = 29,
        include_state: bool = True,
        state_activation: str = "none",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.feature_size = feature_size
        self.include_state = include_state
        self.state_activation = str(state_activation).lower().strip()
        self.out_features = feature_size

        # Convolutional layers (NatureCNN)
        self.cnn = nn.Sequential(
            layer_init(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=32,
                    kernel_size=8,
                    stride=4,
                    padding=0,
                )
            ),
            nn.ReLU(),
            layer_init(
                nn.Conv2d(
                    in_channels=32,
                    out_channels=64,
                    kernel_size=4,
                    stride=2,
                    padding=0,
                )
            ),
            nn.ReLU(),
            layer_init(
                nn.Conv2d(
                    in_channels=64,
                    out_channels=64,
                    kernel_size=3,
                    stride=1,
                    padding=0,
                )
            ),
            nn.ReLU(),
            nn.Flatten(),
        )

        # 128x128 through NatureCNN produces 64 * 12 * 12 = 9216 flattened features
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 128, 128)
            n_flatten = self.cnn(dummy).shape[1]
            self.fc = nn.Sequential(
                layer_init(nn.Linear(n_flatten, feature_size)),
                nn.ReLU(),
            )

        if include_state and state_dim > 0:
            if self.state_activation == "relu":
                self.state_fc = nn.Sequential(
                    layer_init(nn.Linear(state_dim, 256)),
                    nn.ReLU(),
                )
            elif self.state_activation == "none":
                self.state_fc = layer_init(nn.Linear(state_dim, 256))
            else:
                raise ValueError(
                    f"Unsupported state_activation: {state_activation!r}. Must be 'none' or 'relu'."
                )
            self.out_features += 256
        else:
            self.state_fc = None

    def forward(self, observations: Union[Dict[str, torch.Tensor], torch.Tensor]) -> torch.Tensor:
        """
        Extracts fused visual and state features.
        
        Args:
            observations: Dict containing:
                - "rgb": (B, H, W, 3) or (B, 3, H, W)
                - "depth": (B, H, W, 1) or (B, 1, H, W) [optional/for RGB-D]
                - "rgbd": (B, H, W, 4) or (B, 4, H, W) [optional]
                - "state": (B, state_dim)
                Or a raw image tensor (B, C, H, W) or (B, H, W, C).
                
        Returns:
            Fused feature tensor of shape (B, 512) or (B, feature_size).
        """
        state_tensor = None

        if isinstance(observations, dict):
            state_tensor = observations.get("state")

            if self.in_channels == 3:
                # RGB observation
                if "rgb" in observations:
                    vis_input = observations["rgb"].float()
                    if vis_input.shape[-1] == 3:
                        vis_input = vis_input.permute(0, 3, 1, 2)
                    if vis_input.max() > 1.0:
                        vis_input = vis_input / 255.0
                elif "rgbd" in observations:
                    vis_input = observations["rgbd"].float()
                    if vis_input.shape[-1] >= 3:
                        vis_input = vis_input[:, :, :, :3].permute(0, 3, 1, 2)
                    else:
                        vis_input = vis_input[:, :3, :, :]
                    if vis_input.max() > 1.0:
                        vis_input = vis_input / 255.0
                else:
                    raise KeyError(f"Expected 'rgb' in observations dict, got keys: {list(observations.keys())}")

            elif self.in_channels == 4:
                # RGB-D observation
                if "rgbd" in observations:
                    vis_input = observations["rgbd"].float()
                    if vis_input.shape[-1] == 4:
                        vis_input = vis_input.permute(0, 3, 1, 2)
                    # Normalize RGB channels if uint8 range
                    if vis_input[:, :3].max() > 1.0:
                        vis_input[:, :3] = vis_input[:, :3] / 255.0
                    # Check if depth needs normalization
                    if vis_input[:, 3:4].max() > 1.0 or vis_input[:, 3:4].min() < 0.0:
                        vis_input[:, 3:4] = preprocess_depth(vis_input[:, 3:4])

                elif "rgb" in observations and "depth" in observations:
                    rgb = observations["rgb"].float()
                    if rgb.shape[-1] == 3:
                        rgb = rgb.permute(0, 3, 1, 2)
                    if rgb.max() > 1.0:
                        rgb = rgb / 255.0

                    depth = observations["depth"].float()
                    if depth.shape[-1] == 1:
                        depth = depth.permute(0, 3, 1, 2)
                    depth_norm = preprocess_depth(depth)

                    vis_input = torch.cat([rgb, depth_norm], dim=1)
                else:
                    raise KeyError(f"Expected ('rgb', 'depth') or 'rgbd' in observations, got: {list(observations.keys())}")

            else:
                raise ValueError(f"Unsupported in_channels={self.in_channels}")

        elif torch.is_tensor(observations):
            vis_input = observations.float()
            if vis_input.shape[-1] in (3, 4) and vis_input.ndim == 4:
                vis_input = vis_input.permute(0, 3, 1, 2)
            if vis_input[:, :3].max() > 1.0:
                vis_input[:, :3] = vis_input[:, :3] / 255.0
            if vis_input.shape[1] == 4 and (vis_input[:, 3:4].max() > 1.0 or vis_input[:, 3:4].min() < 0.0):
                vis_input[:, 3:4] = preprocess_depth(vis_input[:, 3:4])
        else:
            raise TypeError(f"Observations must be a dict or Tensor, got {type(observations)}")

        vis_feat = self.fc(self.cnn(vis_input))

        if self.state_fc is not None and state_tensor is not None:
            state_feat = self.state_fc(state_tensor.float())
            return torch.cat([vis_feat, state_feat], dim=1)

        return vis_feat


class Agent(nn.Module):
    """
    Gaussian Actor-Critic Policy with NatureCNN Backbone.
    
    Can be initialized either from a Gymnasium environment or via explicit dimensions.
    """

    def __init__(
        self,
        env: Optional[Any] = None,
        sample_obs: Optional[Any] = None,
        in_channels: int = 4,
        action_dim: Optional[int] = None,
        state_dim: int = 29,
        include_state: bool = True,
        state_activation: str = "none",
    ):
        super().__init__()

        # Infer dimensions from environment if provided
        if env is not None:
            action_space = getattr(env, "single_action_space", getattr(env, "action_space", None))
            if action_space is not None and hasattr(action_space, "shape"):
                action_dim = int(np.prod(action_space.shape))

        if action_dim is None:
            # Default action dim for PickCube pd_ee_delta_pos (dx, dy, dz, dgripper)
            action_dim = 4

        # Infer in_channels from sample_obs if available
        if sample_obs is not None and isinstance(sample_obs, dict):
            if "rgbd" in sample_obs:
                in_channels = 4
            elif "depth" in sample_obs:
                in_channels = 4
            elif "rgb" in sample_obs:
                in_channels = 3

            if "state" in sample_obs:
                state_dim = sample_obs["state"].shape[-1]
                include_state = True

        self.in_channels = in_channels
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.state_activation = str(state_activation).lower().strip()

        # Backbone
        self.feature_net = NatureCNN(
            in_channels=in_channels,
            feature_size=256,
            state_dim=state_dim,
            include_state=include_state,
            state_activation=self.state_activation,
        )
        latent_size = self.feature_net.out_features  # 512 when state included

        # Critic: Linear(512, 512) -> ReLU -> Linear(512, 1)
        self.critic = nn.Sequential(
            layer_init(nn.Linear(latent_size, 512)),
            nn.ReLU(inplace=True),
            layer_init(nn.Linear(512, 1), std=1.0),
        )

        # Actor mean: Linear(512, 512) -> ReLU -> Linear(512, action_dim)
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(latent_size, 512)),
            nn.ReLU(inplace=True),
            layer_init(nn.Linear(512, action_dim), std=0.01 * np.sqrt(2)),
        )

        # Learnable logstd vector initialized to -0.5
        self.actor_logstd = nn.Parameter(torch.ones(1, action_dim) * -0.5)

    def get_features(self, x: Union[Dict[str, torch.Tensor], torch.Tensor]) -> torch.Tensor:
        """Returns the 512-dimensional latent feature vector."""
        return self.feature_net(x)

    def get_value(self, x: Union[Dict[str, torch.Tensor], torch.Tensor]) -> torch.Tensor:
        """Evaluates state value V(s)."""
        feat = self.feature_net(x)
        return self.critic(feat)

    def get_action(
        self,
        x: Union[Dict[str, torch.Tensor], torch.Tensor],
        deterministic: bool = False,
    ) -> torch.Tensor:
        """Samples an action from the policy."""
        feat = self.feature_net(x)
        action_mean = self.actor_mean(feat)
        if deterministic:
            return action_mean
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        return probs.sample()

    def get_action_and_value(
        self,
        x: Union[Dict[str, torch.Tensor], torch.Tensor],
        action: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes action, log probability, entropy, and value estimate for PPO rollout/updates.
        
        Returns:
            (action, log_prob, entropy, value)
        """
        feat = self.feature_net(x)
        action_mean = self.actor_mean(feat)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)

        if action is None:
            action = probs.sample()

        log_prob = probs.log_prob(action).sum(dim=-1)
        entropy = probs.entropy().sum(dim=-1)
        value = self.critic(feat)
        return action, log_prob, entropy, value

    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True):
        """
        Loads state dict with automatic key adaptation for:
        - Legacy checkpoints (CleanRL ppo_rgb.py / train_joint_multiview.py)
        - Differences between bare Linear state_fc and Sequential(Linear, ReLU) state_fc.
        """
        remapped = {}
        for k, v in state_dict.items():
            new_k = k
            # Legacy CleanRL ppo_rgb / ppo_rgbd prefix remappings
            if "feature_net.extractors.rgb.0." in new_k:
                new_k = new_k.replace("feature_net.extractors.rgb.0.", "feature_net.cnn.")
            if "feature_net.extractors.rgb.1.0." in new_k:
                new_k = new_k.replace("feature_net.extractors.rgb.1.0.", "feature_net.fc.0.")
            if "feature_net.extractors.state." in new_k:
                if isinstance(self.feature_net.state_fc, nn.Sequential):
                    new_k = new_k.replace("feature_net.extractors.state.", "feature_net.state_fc.0.")
                else:
                    new_k = new_k.replace("feature_net.extractors.state.", "feature_net.state_fc.")

            # Cross-structure adaptation for state_fc:
            # Checkpoint has state_fc.0.* but model has bare Linear state_fc.*
            if "feature_net.state_fc.0." in new_k and not isinstance(self.feature_net.state_fc, nn.Sequential):
                new_k = new_k.replace("feature_net.state_fc.0.", "feature_net.state_fc.")
            # Checkpoint has bare state_fc.* but model has Sequential state_fc.0.*
            elif "feature_net.state_fc." in new_k and "feature_net.state_fc.0." not in new_k and isinstance(self.feature_net.state_fc, nn.Sequential):
                new_k = new_k.replace("feature_net.state_fc.", "feature_net.state_fc.0.")

            remapped[new_k] = v

        return super().load_state_dict(remapped, strict=strict)
