import torch
import numpy as np
import sapien
from typing import Union, Optional, Sequence
from mani_skill.utils.registration import register_env
from mani_skill.envs.tasks.tabletop.pick_cube import PickCubeEnv
from mani_skill.sensors.camera import CameraConfig

@register_env("PickCube45mm-v1", max_episode_steps=50)
class PickCube45mmEnv(PickCubeEnv):
    """
    Subclass of PickCube-v1 configured for the 4.5 cm solid manipulation cube
    with fully customizable camera viewpoints (sensor_cam_eye_pos, sensor_cam_target_pos).
    """
    def __init__(
        self,
        *args,
        robot_uids="panda",
        robot_init_qpos_noise=0.02,
        cube_half_size: float = 0.0225,  # 4.5 cm solid block
        sensor_cam_eye_pos: Optional[Sequence[float]] = None,
        sensor_cam_target_pos: Optional[Sequence[float]] = None,
        **kwargs
    ):
        super().__init__(
            *args,
            robot_uids=robot_uids,
            robot_init_qpos_noise=robot_init_qpos_noise,
            **kwargs
        )
        self.cube_half_size = cube_half_size
        if sensor_cam_eye_pos is not None:
            self.sensor_cam_eye_pos = list(sensor_cam_eye_pos)
        if sensor_cam_target_pos is not None:
            self.sensor_cam_target_pos = list(sensor_cam_target_pos)

    @property
    def _default_sensor_configs(self):
        configs = dict()
        configs["sensor_cam"] = CameraConfig(
            uid="sensor_cam",
            pose=sapien.Pose(
                p=self.sensor_cam_eye_pos,
                q=self._compute_camera_quat(self.sensor_cam_eye_pos, self.sensor_cam_target_pos),
            ),
            width=128,
            height=128,
            fov=np.pi / 2,
            near=0.01,
            far=100,
        )
        return configs

    @staticmethod
    def _compute_camera_quat(eye, target):
        forward = np.array(target) - np.array(eye)
        forward = forward / np.linalg.norm(forward)
        up = np.array([0, 0, 1])
        right = np.cross(forward, up)
        if np.linalg.norm(right) < 1e-6:
            up = np.array([0, 1, 0])
            right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        actual_up = np.cross(right, forward)
        
        R = np.eye(3)
        R[:, 0] = right
        R[:, 1] = actual_up
        R[:, 2] = -forward
        
        import transforms3d.quaternions as quaternions
        return quaternions.mat2quat(R)
