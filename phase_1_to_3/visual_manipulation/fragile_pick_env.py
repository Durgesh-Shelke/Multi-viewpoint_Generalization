"""
FragilePick-v1: Custom ManiSkill Environment for Force-Sensitive Object Manipulation.
Supports 5 distinct objects with realistic physical damage thresholds and object-conditioned observations.
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np
import sapien
import torch

from mani_skill import ASSET_DIR
from mani_skill.agents.robots.panda.panda import Panda
from mani_skill.agents.robots.panda.panda_wristcam import PandaWristCam
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils.randomization.pose import random_quaternions
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.building.actors.ycb import get_ycb_builder
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig

# Object profiles with force thresholds (in Newtons)
OBJECT_PROFILES = {
    "apple": {
        "id": 0,
        "ycb_id": "013_apple",
        "type": "ycb",
        "f_safe": 8.0,    # Bruising begins above 8 N
        "f_max": 18.0,   # Completely crushed above 18 N
        "desc": "Delicate fruit - easily bruised",
    },
    "pear": {
        "id": 1,
        "ycb_id": "016_pear",
        "type": "ycb",
        "f_safe": 7.0,    # Very delicate fruit
        "f_max": 16.0,   # Crushed above 16 N
        "desc": "Very delicate fruit - soft flesh & neck",
    },
    "mustard_bottle": {
        "id": 2,
        "ycb_id": "006_mustard_bottle",
        "type": "ycb",
        "f_safe": 15.0,   # Squeeze squirts / dents bottle
        "f_max": 35.0,   # Permanently crushed
        "desc": "Deformable plastic squeeze bottle",
    },
    "soup_can": {
        "id": 3,
        "ycb_id": "005_tomato_soup_can",
        "type": "ycb",
        "f_safe": 25.0,   # Rigid metal tin can
        "f_max": 60.0,   # Dented / crushed
        "desc": "Rigid metal cylinder - slippery surface",
    },
    "sphere": {
        "id": 4,
        "type": "sphere",
        "radius": 0.025,  # 5 cm diameter smooth ball
        "f_safe": 12.0,   # Needs grip to avoid slip, but sensitive to over-squeeze
        "f_max": 30.0,   # Deformed / crushed
        "desc": "Smooth rigid sphere - rolls and slips easily",
    },
}

OBJECT_NAMES = list(OBJECT_PROFILES.keys())


@register_env("FragilePick-v1", max_episode_steps=50)
class FragilePickEnv(BaseEnv):
    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]
    agent: Union[Panda, PandaWristCam]
    goal_thresh = 0.025

    def __init__(
        self,
        *args,
        robot_uids="panda",
        object_name="apple",  # Can be 'apple', 'pear', 'mustard_bottle', 'soup_can', 'sphere', or 'random'
        robot_init_qpos_noise=0.02,
        num_envs=1,
        **kwargs,
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.object_name_setting = object_name
        super().__init__(
            *args,
            robot_uids=robot_uids,
            num_envs=num_envs,
            **kwargs,
        )

    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                max_rigid_contact_count=2**20, max_rigid_patch_count=2**19
            )
        )

    @property
    def _default_sensor_configs(self):
        # Workstation sensor camera for visual policy input
        return [
            CameraConfig(
                uid="base_camera",
                pose=sapien_utils.look_at(eye=[0.3, 0.0, 0.6], target=[-0.1, 0.0, 0.1]),
                width=128,
                height=128,
                fov=np.pi / 2,
                near=0.01,
                far=100,
            )
        ]

    @property
    def _default_human_render_camera_configs(self):
        # Camera for GUI viewing and video recordings
        return [
            CameraConfig(
                uid="render_camera",
                pose=sapien_utils.look_at(eye=[0.6, 0.7, 0.6], target=[0.0, 0.0, 0.35]),
                width=512,
                height=512,
                fov=1.0,
                near=0.01,
                far=100,
            )
        ]

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(env=self, robot_init_qpos_noise=self.robot_init_qpos_noise)
        self.table_scene.build()

        # Determine which object to spawn
        if self.object_name_setting == "random":
            chosen_name = np.random.choice(OBJECT_NAMES)
        else:
            chosen_name = self.object_name_setting

        self.current_object_name = chosen_name
        self.current_profile = OBJECT_PROFILES[chosen_name]
        self.f_safe = self.current_profile["f_safe"]
        self.f_max = self.current_profile["f_max"]
        self.current_obj_id = self.current_profile["id"]

        # Build object actor
        if self.current_profile["type"] == "ycb":
            builder = get_ycb_builder(self.scene, id=self.current_profile["ycb_id"])
            builder.initial_pose = sapien.Pose(p=[0, 0, 0.1])
            self.obj = builder.build(name="target_object")
        else:
            # Sphere
            builder = self.scene.create_actor_builder()
            builder.add_sphere_collision(radius=self.current_profile["radius"])
            builder.add_sphere_visual(
                radius=self.current_profile["radius"],
                material=sapien.render.RenderMaterial(base_color=[0.1, 0.5, 0.9, 1.0]),
            )
            builder.initial_pose = sapien.Pose(p=[0, 0, 0.1])
            self.obj = builder.build(name="target_object")

        # Goal visual indicator
        self.goal_site = actors.build_sphere(
            self.scene,
            radius=self.goal_thresh,
            color=[0, 1, 0, 0.2],
            name="goal_site",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(p=[0, 0, 0.2]),
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            # Randomize object initial position on table
            xy = torch.rand((b, 2)) * 0.2 - 0.1
            z = 0.05 if self.current_profile["type"] == "sphere" else 0.04
            pos = torch.zeros((b, 3))
            pos[:, :2] = xy
            pos[:, 2] = z
            qs = random_quaternions(b, lock_x=True, lock_y=True)
            self.obj.set_pose(Pose.create_from_pq(p=pos, q=qs))

            # Set target goal position (elevated in air: picking height)
            goal_xyz = torch.zeros((b, 3))
            goal_xyz[:, :2] = torch.rand((b, 2)) * 0.2 - 0.1
            goal_xyz[:, 2] = torch.rand(b) * 0.1 + 0.15
            self.goal_site.set_pose(Pose.create_from_pq(goal_xyz))

            # Reset peak force tracking buffer
            self.peak_force = torch.zeros(b, device=self.device)

    def evaluate(self) -> dict:
        # Measure contact forces on object from left and right gripper pads
        l_contact = self.scene.get_pairwise_contact_forces(self.agent.finger1_link, self.obj)
        r_contact = self.scene.get_pairwise_contact_forces(self.agent.finger2_link, self.obj)
        l_force = torch.linalg.norm(l_contact, axis=-1)
        r_force = torch.linalg.norm(r_contact, axis=-1)
        instant_force = torch.maximum(l_force, r_force)

        # Update peak force experienced during the episode
        self.peak_force = torch.maximum(self.peak_force, instant_force)

        # Continuous damage metric: [0.0% to 100.0%]
        damage_ratio = torch.clamp((self.peak_force - self.f_safe) / (self.f_max - self.f_safe), 0.0, 1.0)
        damage_pct = damage_ratio * 100.0

        is_grasped = self.agent.is_grasping(self.obj)
        obj_to_goal = torch.linalg.norm(self.obj.pose.p - self.goal_site.pose.p, axis=-1)
        is_at_goal = obj_to_goal <= self.goal_thresh
        is_robot_static = self.agent.is_static(0.2)

        success = is_at_goal & is_robot_static
        is_pristine = (damage_pct == 0.0)
        success_pristine = success & is_pristine

        return {
            "success": success,
            "success_pristine": success_pristine,
            "is_grasped": is_grasped,
            "is_pristine": is_pristine,
            "damage_pct": damage_pct,
            "peak_force": self.peak_force,
            "instant_force": instant_force,
            "obj_to_goal_dist": obj_to_goal,
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        tcp_to_obj_dist = torch.linalg.norm(self.obj.pose.p - self.agent.tcp.pose.p, axis=-1)
        reaching_rew = 1 - torch.tanh(5 * tcp_to_obj_dist)

        reward = reaching_rew

        is_grasped = info["is_grasped"]
        reward += is_grasped.float() * 1.0

        if is_grasped.any():
            obj_to_goal_dist = info["obj_to_goal_dist"]
            place_rew = 1 - torch.tanh(5 * obj_to_goal_dist)
            reward += is_grasped.float() * place_rew * 2.0

        # Success bonus
        success = info["success"]
        reward += success.float() * 5.0

        # === Perfection & Fragility Reward Shaping ===
        # 1. Huge Pristine Perfection Bonus: +10.0 for 0% damage
        is_pristine = info["is_pristine"]
        reward += (success & is_pristine).float() * 10.0

        # 2. Steep Non-Linear Damage Penalty: drops sharply even at 1% error
        damage_pct = info["damage_pct"]
        damage_ratio = damage_pct / 100.0
        # Concave penalty: sqrt(ratio) penalizes initial damage heavily
        damage_penalty = 15.0 * torch.sqrt(damage_ratio)
        reward -= is_grasped.float() * damage_penalty

        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 16.0

    def _get_obs_extra(self, info: dict):
        # Provide robot state and 5-dimensional one-hot object identity
        one_hot = torch.zeros((self.num_envs, len(OBJECT_NAMES)), device=self.device)
        one_hot[:, self.current_obj_id] = 1.0

        return dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
            goal_pos=self.goal_site.pose.p,
            is_grasped=info["is_grasped"],
            object_one_hot=one_hot,
            object_id=torch.full((self.num_envs, 1), self.current_obj_id, device=self.device, dtype=torch.float32),
        )
