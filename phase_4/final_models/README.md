# Final Models Directory: Checkpoints by Paradigm & Modality

This directory contains the definitive, converged model checkpoints from each major phase of our visual reinforcement learning manipulation research on the Franka Panda arm (`PickCube-v1`, Option A: 4.5 cm solid block).

---

## 1. Summary of Active 5M Parity Models

| Model File | Modality | Channels | Steps | Training View | Front-View (1,000 runs) | Mean Len / Ret | Sweep Mean (13k runs) | Fibonacci Mean (3k runs) | Peak Dome VP |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`pure_rgb_5m_front_view.pt`** | RGB | 3 (`[R, G, B]`) | 5,000,000 | Front (`[0.3, 0.0, 0.6]`) | **97.40%** | 12.82 / 5.46 | 8.89% (Peak: 97.0%) | 3.23% | 75.0% (VP 17) |
| **`clean_rgbd_5m_front_view.pt`** | RGB-D | 4 (`[R, G, B, D]`) | 5,000,000 | Front (`[0.3, 0.0, 0.6]`) | **95.00%** | 14.14 / **5.96** | 8.35% (Peak: 95.0%) | **3.93%** | **87.0% (VP 17)** |

*Note: Pre-5M models and early checkpoints are cleanly archived in `/home/maniskill/phase_4/archive_pre_5m/`.*

---

## 2. Model Loading & Inference Code

### Loading the 3-Channel Pure RGB 5M Model
```python
import torch
import gymnasium as gym
import mani_skill.envs
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper, FlattenActionSpaceWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from ppo_rgb import Agent

env = gym.make("PickCube-v1", num_envs=1, obs_mode="rgb", control_mode="pd_ee_delta_pos", sim_backend="physx_cuda")
env = FlattenRGBDObservationWrapper(env, rgb=True, depth=False, state=True)
env = ManiSkillVectorEnv(env, 1, ignore_terminations=False)
obs, _ = env.reset()

agent = Agent(env, sample_obs=obs).cuda()
agent.load_state_dict(torch.load("/home/maniskill/phase_4/final_models/pure_rgb_5m_front_view.pt"))
agent.eval()

with torch.no_grad():
    action = agent.get_action(obs, deterministic=True)
```

### Loading the 4-Channel RGB-D 5M Model
```python
import torch
import gymnasium as gym
import mani_skill.envs
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper, FlattenActionSpaceWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from ppo_rgbd import Agent

env = gym.make("PickCube-v1", num_envs=1, obs_mode="rgbd", control_mode="pd_ee_delta_pos", sim_backend="physx_cuda")
env = FlattenRGBDObservationWrapper(env, rgb=True, depth=True, state=True, sep_depth=True)
env = ManiSkillVectorEnv(env, 1, ignore_terminations=False)
obs, _ = env.reset()

agent = Agent(env, sample_obs=obs).cuda()
agent.load_state_dict(torch.load("/home/maniskill/phase_4/final_models/clean_rgbd_5m_front_view.pt"))
agent.eval()

with torch.no_grad():
    action = agent.get_action(obs, deterministic=True)
```
