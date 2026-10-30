"""
Dual-View Environment for Maniwhere: Simultaneous Canonical and Moving Viewpoint Observations.
Renders two synchronized camera sensors in ManiSkill 3:
1. 'cam_canonical': Fixed viewpoint (Canonical Baseline VP 17: -4.7° azimuth, 31.8° elevation)
2. 'cam_moving': Moving viewpoint dynamically sampled across diverse camera angles.
"""

import os
import json
import numpy as np
import torch
import gymnasium as gym

import mani_skill.envs
from mani_skill.envs.tasks.tabletop.pick_cube import PickCubeEnv
from mani_skill.utils.registration import register_env
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils, common
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper

VIEWPOINTS_JSON = "/home/maniskill/phase_1_to_3/sequential_viewpoint_training/viewpoints.json"
CANONICAL_VP_ID = 17


@register_env("PickCubeDualView-v1", max_episode_steps=50)
class PickCubeDualViewEnv(PickCubeEnv):
    def __init__(
        self,
        *args,
        canonical_eye=None,
        moving_eye=None,
        moving_vpid: int = 18,
        **kwargs
    ):
        with open(VIEWPOINTS_JSON) as f:
            self.all_vps = {v["id"]: v for v in json.load(f)}

        if canonical_eye is None:
            self.canonical_eye = self.all_vps[CANONICAL_VP_ID]["world_eye"]
        else:
            self.canonical_eye = canonical_eye

        if moving_eye is None:
            self.moving_eye = self.all_vps[moving_vpid]["world_eye"]
        else:
            self.moving_eye = moving_eye

        super().__init__(*args, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose_canonical = sapien_utils.look_at(eye=self.canonical_eye, target=self.sensor_cam_target_pos)
        pose_moving = sapien_utils.look_at(eye=self.moving_eye, target=self.sensor_cam_target_pos)
        return [
            CameraConfig("cam_canonical", pose_canonical, 128, 128, np.pi / 2, 0.01, 100),
            CameraConfig("cam_moving", pose_moving, 128, 128, np.pi / 2, 0.01, 100),
        ]


class DualViewWrapper(gym.ObservationWrapper):
    """
    Splits multi-camera observation into fixed (canonical) and moving viewpoint observations,
    extracting state and formatting tensors for the Maniwhere multi-view representation loss.
    """
    def __init__(self, env):
        super().__init__(env)

    def observation(self, obs):
        sensor_data = obs.pop("sensor_data")
        if "sensor_param" in obs:
            del obs["sensor_param"]

        state_tensor = common.flatten_state_dict(obs, use_torch=True, device=self.base_env.device)

        canonical_data = sensor_data["cam_canonical"]
        moving_data = sensor_data["cam_moving"]

        return {
            "fixed": {
                "rgb": canonical_data["rgb"],
                "depth": canonical_data["depth"],
                "state": state_tensor,
            },
            "move": {
                "rgb": moving_data["rgb"],
                "depth": moving_data["depth"],
                "state": state_tensor,
            },
        }
