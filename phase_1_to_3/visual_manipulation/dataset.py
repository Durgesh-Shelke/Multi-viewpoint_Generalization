"""
Dataset loader for visual demonstration trajectories.
Loads RGB camera frames, proprioceptive states, and actions from HDF5 datasets.
"""

import h5py
import torch
from torch.utils.data import Dataset
from mani_skill.utils import common


class VisualPickDataset(Dataset):
    """
    Loads demonstration trajectories replayed with RGB observations.
    """

    def __init__(self, demo_path: str, max_demos: int = None):
        super().__init__()
        print(f"[Dataset] Loading demonstrations from: {demo_path}")
        f = h5py.File(demo_path, "r")
        rgbs, states, actions = [], [], []

        traj_keys = sorted(f.keys())
        if max_demos is not None:
            traj_keys = traj_keys[:max_demos]

        for k in traj_keys:
            traj = f[k]
            act = torch.from_numpy(traj["actions"][:]).float()
            N = len(act)
            rgb = torch.from_numpy(traj["obs"]["sensor_data"]["base_camera"]["rgb"][:N])

            agent_dict = {
                "qpos": torch.from_numpy(traj["obs"]["agent"]["qpos"][:N]),
                "qvel": torch.from_numpy(traj["obs"]["agent"]["qvel"][:N]),
            }
            extra_dict = {
                "is_grasped": torch.from_numpy(traj["obs"]["extra"]["is_grasped"][:N]).unsqueeze(-1),
                "tcp_pose": torch.from_numpy(traj["obs"]["extra"]["tcp_pose"][:N]),
                "goal_pos": torch.from_numpy(traj["obs"]["extra"]["goal_pos"][:N]),
            }
            state = common.flatten_state_dict({"agent": agent_dict, "extra": extra_dict}, use_torch=True).float()

            rgbs.append(rgb)
            states.append(state)
            actions.append(act)

        self.rgbs = torch.cat(rgbs, dim=0)
        self.states = torch.cat(states, dim=0)
        self.actions = torch.cat(actions, dim=0)
        print(f"[Dataset] Loaded {len(self.actions)} transitions from {len(traj_keys)} trajectories.")
        print(f"[Dataset] Shapes -> RGB: {self.rgbs.shape}, State: {self.states.shape}, Action: {self.actions.shape}")

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        return {
            "rgb": self.rgbs[idx],
            "state": self.states[idx],
            "action": self.actions[idx],
        }
