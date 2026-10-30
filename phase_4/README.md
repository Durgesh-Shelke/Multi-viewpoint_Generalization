# Phase 4: RGB vs. RGB-D Head-to-Head Comparative Study (5M-Step Parity)

This directory archives all code, models, training runs, and reports from **Phase 4** of the visual reinforcement learning manipulation project.

---

## 1. Executive Summary & Research Questions

Phase 4 addressed the fundamental question: **Does adding metric depth (4-channel RGB-D) solve the spatial sensitivity, sample efficiency, and trajectory instability of monocular RGB?**

### The 5,000,000-Step Unified Parity Campaign
To eliminate confounding variables (different seeds, different training step budgets, or differing batch sizes), both pure RGB and 4-channel RGB-D were retrained from scratch on the canonical front viewpoint under identical conditions (5,000,000 steps, 256 parallel environments, 16 minibatches, target KL 0.2, GAE truncation).

### Parity Benchmark Matrix (5,000,000 Steps)

| Evaluation Dimension | Pure RGB (3-Channel) | Clean RGB-D (4-Channel) | Advantage / Finding |
| :--- | :---: | :---: | :--- |
| **Front-View In-Domain (1,000 runs)** | **97.40%** | 95.00% | Both achieve near-ceiling picking accuracy. |
| **Mean Pick Episode Length** | 12.82 steps | 14.14 steps | Comparable execution time (~0.65–0.70s). |
| **Mean Trajectory Return** | 5.46 | **5.96** | **RGB-D** achieves cleaner reward accumulation and stability. |
| **13-Angle Sweep (13,000 runs)** | 8.89% (Peak: 97.0%) | 8.35% (Peak: 95.0%) | Both suffer if camera shifts without domain randomization. |
| **30-VP Fibonacci Dome (3,000 runs)** | 3.23% | **3.93%** | RGB-D maintains slightly higher dome baseline. |
| **Peak Dome Viewpoint (VP 17)** | 75.0% | **87.0%** | **RGB-D** retains +12.0% higher grasp success near training pose. |

---

## 2. Directory Structure

```text
phase_4/
├── README.md                                  <- This comprehensive index and guide
├── final_models/                              <- Converged 5M model checkpoints
│   ├── pure_rgb_5m_front_view.pt              <- 5M RGB model (97.4% in-domain)
│   ├── clean_rgbd_5m_front_view.pt            <- 5M RGB-D model (95.0% in-domain)
│   └── README.md                              <- Model loading & inference guide
├── runs/                                      <- Tensorboard and checkpoint directories
│   ├── sim2real_45mm_block_rgb_5m/            <- Pure RGB 5M run data
│   └── sim2real_45mm_block_rgbd_5m/           <- 4-Channel RGB-D 5M run data
├── archive_pre_5m/                            <- Pre-5M baseline models and exploration logs
├── reports/                                   <- Formal research reports 11 through 13
│   ├── 11_rgbd_baseline_benchmark_report.md
│   ├── 12_rgb_vs_rgbd_comparative_research_report.md
│   ├── 13_definitive_5m_rgb_vs_rgbd_comparative_report.md
│   ├── benchmark_1000_rgb_5m_results.json
│   ├── benchmark_1000_rgbd_5m_results.json
│   ├── sweep_results_front_rgbd.json
│   └── consolidated_research_reports_master.md
├── run_5m_training_pipeline.sh                <- Pipeline script to run 5M RGB & RGB-D training
├── evaluate_camera_sweep_rgbd.py              <- Semicircle sweep evaluation script for RGB-D
└── evaluate_fibonacci_hemisphere_rgbd.py      <- 30-viewpoint dome evaluation script for RGB-D
```

---

## 3. How to Load and Evaluate Models

```python
import torch
import gymnasium as gym
import mani_skill.envs
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from ppo_rgbd import Agent

# Load 4-Channel RGB-D model
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
