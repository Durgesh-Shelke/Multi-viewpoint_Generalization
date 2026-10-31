# Multi-Viewpoint Generalization & Sensor Fusion in Deep Visual RL Manipulation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![Framework: ManiSkill 3](https://img.shields.io/badge/Simulator-ManiSkill_3-brightgreen.svg)](https://github.com/haosulab/ManiSkill)
[![Hardware: NVIDIA RTX](https://img.shields.io/badge/GPU-CUDA_12.0+-orange.svg)](https://developer.nvidia.com/cuda-zone)

This repository contains the empirical benchmark suite, continual learning studies, and architectural sensor-fusion ablations investigating **multi-viewpoint generalization**, **catastrophic forgetting**, and **spatial sensor modalities** in deep visual reinforcement learning for robotic manipulation (`PickCube-v1` on a 7-DoF Franka Panda arm).

---

## 1. Executive Summary & Core Scientific Findings

Visual reinforcement learning policies trained on monocular 2D images fail to learn 3D Euclidean world coordinates; instead, they memorize visual coordinate mappings relative to camera frame pixel boundaries. Through a systematic 5-phase empirical campaign spanning over **65,000,000 environment steps** and **75,000+ evaluation episodes**, we uncover the geometric mechanisms governing camera perspective drift and demonstrate the definitive architectural solution:

1. **Extreme Viewpoint Fragility in Pure RGB:**
   - A single-viewpoint monocular RGB policy achieving **97.10% success** on the canonical front view collapses to **0.40%** under a 90° side shift (coordinate axis inversion).
   - In horizontal sweeps, a deviation of merely $\pm 15^\circ$ causes an $>85\%$ collapse (dropping to 4.2%–11.2%).
   - Across a 30-viewpoint Fibonacci equal-area hemisphere dome, global average success is only **5.65%**.
2. **Failure of Continual Sequential Training (Catastrophic Forgetting):**
   - Fine-tuning an agent sequentially across Fibonacci viewpoints overwrites earlier convolutional feature representations. Viewpoint 00 collapsed from 100% to **0.0%**, and dome retention fell to **0.50%**.
3. **The Power of Metric Depth:**
   - Adding a 1-channel metric depth observation provides Euclidean scale invariance, improving trajectory stability and dome peak retention (+12% gain over RGB).
4. **Resolution & Decoupled Asymmetric Sensor Fusion (The Winning Architecture):**
   - **Candidate C (Asymmetric Decoupled: Low-Res $64 \times 64$ RGB $\oplus$ High-Res $256 \times 256$ Depth)** is the overall benchmark champion:
     - **Highest Angular Jitter Robustness (45.77% mean):** 4.03x higher retention at 75° (-15° azimuth perturbation) than symmetric $128 \times 128$ RGB-D.
     - **Superior Photometric Resilience (72.30%):** Downsampling RGB prevents overfitting to illumination shifts, while high-resolution Depth grounds millimeter grasp affordances.
     - **Efficiency:** Achieves a **30.6% parameter reduction** (3.77M vs 5.43M params) and reduces peak VRAM by **771 MiB**.

---

## 2. Master Comparison Matrix: 50M-Step Ablation Study

Each candidate model below was trained for **10,000,000 environment steps** and evaluated across a **6,000-episode 4-tier benchmark battery** (30,000 evaluation episodes total).

| Evaluation Metric / Dimension | Candidate A <br>*(Sym 128×128 RGB-D)* | Candidate B <br>*(Sym 256×256 RGB-D)* | Candidate C <br>*(Asym 64 RGB + 256 D)* | Candidate D <br>*(Pure Depth 256×256)* | Candidate E <br>*(Pure RGB 256×256)* |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Input Modality** | 4-Ch (RGB + Depth) | 4-Ch (RGB + Depth) | Dual-Stream Decoupled | 1-Ch (Depth Only) | 3-Ch (RGB Only) |
| **Resolution** | $128 \times 128$ | $256 \times 256$ | RGB $64 \times 64$ / D $256 \times 256$ | $256 \times 256$ | $256 \times 256$ |
| **Parameter Count** | **2,973,097** | 5,434,921 | **3,771,833** | 5,428,777 | 5,432,873 |
| **Training Speed (Avg SPS)** | **1,689 SPS** | 1,003 SPS | 1,079 SPS | 1,089 SPS | 1,112 SPS |
| **Peak VRAM Allocated** | **1,834 MiB** | 4,725 MiB | 3,954 MiB | 3,866 MiB | 4,077 MiB |
| **Steps to 50% Grasp Success** | 2,713,600 | **2,233,600** *(Fastest)* | 3,033,600 | 6,713,600 | 3,033,600 |
| **Steps to 80% Grasp Success** | 3,033,600 | **2,873,600** *(Fastest)* | 3,673,600 | 8,153,600 | 4,473,600 |
| **Plateau Onset Step** | 3,033,600 | **2,873,600** | 4,153,600 | 8,153,600 | 4,793,600 |
| **Plateau Mean Success** | 93.33% | **95.00%** | 89.53% | 87.50% | 47.08% *(Drift)* |
| **Test 1: In-Domain (1,000 runs)** | **97.70%** | 96.50% | 95.80% | 91.60% | 97.30% |
| **Grasp Trajectory Return** | 5.05 | 5.44 | 5.21 | **6.25** | 5.41 |
| **Mean Pick Steps** | **12.25** | 13.65 | 12.95 | 16.46 | 13.19 |
| **Test 2: Angular Jitter (±15° Mean)**| 35.60% | 44.27% | **45.77%** *(Winner)* | 31.83% | 43.80% |
| - *At 75° (-15° perturbation)* | 6.6% | 20.7% | **26.6%** *(+4.03x gain)* | 5.1% | 13.8% |
| - *At 105° (+15° perturbation)*| 2.7% | 15.7% | 14.6% | 0.8% | **19.6%** |
| **Test 3: Lighting Shift (±30% Mean)** | 64.50% | 59.30% | 72.30% | 89.00% | **94.90%** |
| - *0.7x Darkened Lighting* | 51.0% | 47.1% | 70.1% | 89.1% | **92.7%** |
| - *1.3x Brightened Lighting* | 78.0% | 71.5% | 74.5% | 88.9% | **97.1%** |
| - *Photometric Retention vs Baseline* | 66.0% | 61.5% | 75.5% | **99.33%** *(Immune)* | 97.5% |
| **Inference Latency (per step)** | 0.95 ms | 0.93 ms | 1.15 ms | 0.93 ms | **0.80 ms** |

---

## 3. Repository Architecture

```text
Multi-viewpoint_Generalization/
├── README.md                      <- Master research index & documentation
├── requirements.txt               <- Complete Python dependencies
├── .gitignore                     <- Clean exclusion of temporary/heavy logs
├── envs/                          <- Custom environment registration (4.5cm block & viewpoint API)
│   ├── __init__.py
│   └── custom_pick_cube.py
├── phase_1_to_3/                  <- Baselines, Sweeps, Continual Learning & Catastrophic Forgetting
│   ├── README.md                  <- Phase 1-3 detailed documentation
│   ├── reports/                   <- Research Reports 01 to 10 + Consolidated Master
│   ├── RGB/                       <- Pure RGB archive, checkpoints & demo videos
│   ├── visual_manipulation/       <- Initial wrappers & 90° side-view tests
│   ├── sequential_viewpoint_training/ <- Continual sequential learning pipeline
│   ├── fibonacci_hemisphere_30vp_results/ <- 30-viewpoint dome results & videos
│   ├── joint_multiview_training/  <- Vectorized multi-view training scripts
│   └── joint_multiview_experiment/<- Structured joint multi-view workspace
├── phase_4/                       <- RGB vs. RGB-D Head-to-Head Comparative Study (5M Parity)
│   ├── README.md                  <- Phase 4 detailed documentation
│   ├── reports/                   <- Research Reports 11 to 13
│   ├── final_models/              <- 5M converged checkpoints (RGB & RGB-D)
│   ├── runs/                      <- TensorBoard logs and trajectory histories
│   ├── run_5m_training_pipeline.sh<- Unified 5M training runner
│   ├── evaluate_camera_sweep_rgbd.py
│   └── evaluate_fibonacci_hemisphere_rgbd.py
└── phase_5/                       <- 50M-Step Unbiased Resolution & Sensor Fusion Ablation Study
    ├── README.md                  <- Phase 5 detailed documentation
    ├── reports/                   <- Research Report 14 (Ablation Study)
    ├── checkpoints/               <- Converged best checkpoints (Candidates A, B, C, D, E)
    ├── eval_results/              <- 30,000-episode evaluation datasets & JSONs
    ├── models.py                  <- Multi-modal neural backbones (NatureCNN & Dual-Stream)
    ├── train_candidate.py         <- PPO candidate training harness
    ├── evaluate_battery.py        <- 4-tier evaluation battery runner
    ├── plot_growth_curves.py      <- Plateau & sample efficiency visualization
    ├── run_master_unbiased_benchmark.py <- Autonomous master benchmark pipeline
    ├── status.sh                  <- Real-time telemetry monitor
    ├── pause.sh                   <- Zero-loss checkpointing pause tool
    └── resume.sh                  <- Seamless resumption tool
```

---

## 4. Quickstart & Installation

### Prerequisites
- Ubuntu 20.04 / 22.04 LTS
- NVIDIA GPU with CUDA 12.0+ (Tested on RTX 4060 Laptop GPU, 8GB VRAM)
- Python 3.9, 3.10, or 3.11

### Setup
```bash
# Clone the repository
git clone https://github.com/Durgesh-Shelke/Multi-viewpoint_Generalization.git
cd Multi-viewpoint_Generalization

# Install dependencies
pip install -r requirements.txt
```

---

## 5. Model Loading & Policy Inference

### Loading Candidate C (Asymmetric Decoupled RGB-D Champion)
```python
import torch
import gymnasium as gym
import mani_skill.envs
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from phase_5.models import make_policy_network

# Initialize environment with custom 4.5cm block and front view
env = gym.make("PickCube-v1", num_envs=1, obs_mode="rgbd", control_mode="pd_ee_delta_pos", sim_backend="physx_cuda")
env = ManiSkillVectorEnv(env, 1, ignore_terminations=False)
obs, _ = env.reset()

# Initialize Candidate C Dual-Stream Actor-Critic
policy = make_policy_network("C", env, sample_obs=obs).cuda()
policy.load_state_dict(torch.load("phase_5/checkpoints/candidate_C_best.pt"))
policy.eval()

with torch.no_grad():
    action = policy.get_action(obs, deterministic=True)
```

---

## 6. Running Benchmarks & Control Scripts

### Monitor Live Pipeline Status
```bash
cd phase_5
./status.sh
```

### Run Full 5-Candidate Benchmark Battery
```bash
cd phase_5
python3 run_master_unbiased_benchmark.py --candidates A B C D E --total-timesteps 10000000
```

---

## 7. Research Reports Compendium

All experiments are backed by formal empirical markdown reports:

| Report # | Title & Topic |
| :-: | :--- |
| **01** | [Sim-to-Real 1,000-Run Benchmark (Front View 97.10%)](phase_1_to_3/reports/01_sim2real_1000_run_benchmark.md) |
| **02** | [Shifted 90° Side-View Collapse Analysis](phase_1_to_3/reports/02_side_view_evaluation_report.md) |
| **03** | [20 Hz 1:1 Physical Clock Sync Telemetry](phase_1_to_3/reports/03_live_realtime_evaluation.md) |
| **04** | [Horizontal Semicircle Viewpoint Sweep (13 Angles)](phase_1_to_3/reports/04_camera_viewpoint_sweep_report.md) |
| **05** | [20-Viewpoint Fibonacci Equal-Area Hemisphere Evaluation](phase_1_to_3/reports/05_fibonacci_hemisphere_20vp_report.md) |
| **06** | [30-Viewpoint Fibonacci Equal-Area Hemisphere Evaluation](phase_1_to_3/reports/06_fibonacci_hemisphere_30vp_report.md) |
| **07** | [Sequential Continual Learning Architecture](phase_1_to_3/reports/07_sequential_training_architecture.md) |
| **08** | [Midway Catastrophic Forgetting Evidence](phase_1_to_3/reports/08_catastrophic_forgetting_midway_report.md) |
| **09** | [Final Sequential Continual Training Benchmark](phase_1_to_3/reports/09_final_sequential_training_benchmark_report.md) |
| **10** | [Joint Multi-View Domain Randomization Architecture](phase_1_to_3/reports/10_joint_multiview_training_setup.md) |
| **11** | [4-Channel RGB-D Baseline Benchmark](phase_4/reports/11_rgbd_baseline_benchmark_report.md) |
| **12** | [RGB vs. RGB-D Head-to-Head Comparative Study](phase_4/reports/12_rgb_vs_rgbd_comparative_research_report.md) |
| **13** | [Definitive 5M-Step Parity Study](phase_4/reports/13_definitive_5m_rgb_vs_rgbd_comparative_report.md) |
| **14** | [50M-Step Unbiased Resolution & Sensor Fusion Ablation Study](phase_5/reports/14_resolution_and_asymmetric_depth_ablation_report.md) |
| **18** | Phase 10: OmniView 1,000-Test Multi-Teacher Benchmark |
| **19** | [Continuous Viewpoint Interpolation Collapse Report](reports/research_report_19_interpolated_viewpoints_generalization.md) |
| **20** | Continuous Generalization Master Roadmap & FiLM Conditioning |
| **21** | Phase 12: Continuous Extrinsics FiLM Results & 2D Convolutions Limits |
| **22** | Phase 13: 3D PointNet Metric Spatial Invariance Breakthrough (90.6% SR) |
| **23** | Phase 14: Rigorous Perturbation Robustness Curves & MP4 Video Gallery |
| **24** | [Phase 15: Cross-Embodiment Generalization & Zero-Shot Transfer](reports/research_report_24_phase15_cross_embodiment_zeroshot.md) |
| **25** | [Phase 16: Composite Modular Universal Policy Architecture](reports/research_report_25_phase16_composite_modular_architecture.md) |
| **Master** | [Consolidated Master Research Compendium](phase_1_to_3/reports/consolidated_research_reports_master.md) |

---

## 8. Latest Phase 15 & 16 Scientific Breakthroughs

### Phase 15: Cross-Embodiment Generalization & Zero-Shot Transfer
- **Combined OmniArm Policy:** Trained on joint workspace 3D Point Clouds, achieving **84.9% Success Rate on xArm 6 Robotiq** and **79.2% on Franka Panda** across all 13 camera viewpoints suite.
- **Zero-Shot Transfer:** Achieved **28.9% Success Rate** and **66.3% Grasp Rate** completely zero-shot on an unseen hybrid robot (`xarm6_panda_gripper`).

### Phase 16: Composite Modular Universal Architecture
- **Single Deliverable Policy (`universal_composite_omniarm.pt`):** Decouples Viewpoint Generalization (3D PointNet) from DoF/Kinematic Generalization (Neural IK), assembled via weight surgery into a single deployable module.
- **Master 13-Viewpoint Benchmark (1,300 Episodes):**
  - **xArm 6 Robotiq (6 DoF):** **92.2% Overall SR** (90.5% Anchor, 92.9% In-Between, 96.0% Grasp).
  - **Franka Panda (7 DoF):** **74.6% Overall SR** (80.0% Anchor, 72.2% In-Between, 86.2% Grasp).
  - **Combined Model Average:** **83.4% Overall SR** (91.1% Grasp Rate).

---

## 9. Repository Maintenance & Developer Guide

For complete guidelines on extending this repository, adding new experimental phases (e.g. Phase 6), handling large `.pt` checkpoints, modifying environment wrappers, and maintaining clean git hygiene for future AI agents and contributors, refer to:

👉 **[`MAINTENANCE.md`](MAINTENANCE.md)** — *Operational architecture, dual-workspace conventions, file size guardrails, pause/resume standards, and contributor checklist.*

---

## 9. License

This research repository is licensed under the [MIT License](LICENSE).
