# Sim-to-Real Visual Manipulation Policy: 1,000-Run Benchmark Report

## Executive Summary

The visual reinforcement learning policy was evaluated on **1,000 independent test runs** in SAPIEN simulation with the physical dimensions calibrated for physical parallel-jaw gripper transfer (**Option A: 4.5 cm solid block**, $2 \times 0.0225\text{ m}$).

The trained visuomotor agent achieved a **97.10% pickup success rate** (971 successful picks out of 1,000 runs), comfortably surpassing the $\ge 95.0\%$ target threshold. Training was stopped cleanly to conserve computational resources.

---

## 1,000-Run Benchmark Statistics

| Metric | Measured Value | Target Goal | Status |
| :--- | :--- | :--- | :--- |
| **Total Test Runs** | **1,000 runs** | 1,000 runs | Complete |
| **Successful Picks** | **971 picks** | $\ge 950$ | **PASSED** |
| **Success Rate** | **97.10%** | $\ge 95.00\%$ | **SURPASSED (+2.10%)** |
| **Mean Pick Latency** | **12.9 timesteps (~0.43s)** | $< 50$ steps | Fast & Reactive |
| **Mean Return** | **5.45** | > 4.0 | High Task Efficiency |
| **Checkpoint Evaluated** | `ckpt_401.pt` (5.12M steps) | $\le 50\text{M}$ steps | Converged Early |

```json
{
  "checkpoint": "/home/maniskill/runs/sim2real_45mm_block/ckpt_401.pt",
  "target_episodes": 1000,
  "success_count": 971,
  "success_rate": 97.1,
  "goal_met": true,
  "mean_return": 5.4498,
  "mean_length": 12.95,
  "elapsed_seconds": 45.13
}
```

---

## Policy Execution Demonstration

Below is the recorded execution showing the trained policy picking up the 4.5 cm solid block across randomized table positions:

![Sim-to-Real 4.5cm Block Pick Demonstration](/root/.gemini/antigravity-cli/brain/82673f33-5c96-4154-bb42-cf6de559c7b7/pick_45mm_solid_block_97percent.mp4)

---

## Training & Convergence Trajectory

```mermaid
flowchart LR
    A["Initial Random Policy<br>0% Pick Rate (0 - 1.2M steps)"] --> B["Grasp Acquisition<br>87.5% Success (2.56M steps)"]
    B --> C["1,000-Run Benchmark 1<br>92.50% at ckpt_251 (3.2M steps)"]
    C --> D["1,000-Run Benchmark 2<br>97.10% at ckpt_401 (5.12M steps)"]
    D --> E["Goal Attained & Converged<br>Training Stopped"]
```

- **0 - 1.5M Steps**: Policy explored reach actions, learning spatial depth and end-effector alignment from raw RGB observations.
- **2.56M Steps (`ckpt_201.pt`)**: Reached 87.5% periodic eval success.
- **3.20M Steps (`ckpt_251.pt`)**: Reached **92.50%** across 1,000 runs (close, but just shy of the 95% bar).
- **5.12M Steps (`ckpt_401.pt`)**: Achieved **97.10%** across 1,000 test runs, triggering training completion.

---

## Architecture & Sim-to-Real Specifications

1. **Visual Perception**:
   - Multi-layer NatureCNN backbone processing raw RGB camera feeds.
   - Zero class IDs or symbolic object coordinates required during inference.
2. **End-Effector Control**:
   - Delta pose control mode (`pd_ee_delta_pos`) directly in Cartesian operational space ($[\Delta x, \Delta y, \Delta z, \Delta \text{gripper}]$).
   - Dynamic joint state integration suitable for variable arm embodiments.
3. **Deployment Asset**:
   - Checkpoint saved to: [`/home/mahoraga/maniskill/best_sim2real_policy.pt`](file:///home/mahoraga/maniskill/best_sim2real_policy.pt)
   - Benchmark script: [`/home/maniskill/visual_manipulation/evaluate_1000.py`](file:///home/maniskill/visual_manipulation/evaluate_1000.py)
