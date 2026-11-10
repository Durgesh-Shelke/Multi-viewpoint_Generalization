# AGENT_HANDOFF — Multi-Viewpoint Generalization in Deep Visual Manipulation

> **READ THIS FIRST IN EVERY NEW CHAT OR AGENT SESSION.**
> This file is the single source of truth carrying complete context between chats, subagents, and sessions.
> It prevents wasting tokens on re-indexing the directory.
> **Update it before ending any session:** record new experiments, update decisions, and log results.
> Always write absolute dates (YYYY-MM-DD).

**Last updated:** 2026-11-10  
**Target Milestone:** Bachelor's Thesis Project (BTP) Submission  
**Core Topic:** Continuous Multi-Viewpoint Generalization in Deep Visual Robotic Manipulation  
**Future Work Scope:** DoF & Cross-Embodiment Generalization (Phases 15–16 kept as Future Work / Teaser)

---

## 0. Project Objective & BTP Presentation Strategy

This repository contains the empirical benchmark suite, continual learning studies, and architectural solutions for **continuous multi-viewpoint generalization** in deep visual reinforcement learning for robotic manipulation (`PickCube-v1` on a 7-DoF Franka Panda arm and 6-DoF xArm 6).

### BTP Presentation Strategy (Clarified 2026-11-10):
- **Core Thesis Focus:** **Viewpoint Generalization**. We solve the fundamental problem of camera viewpoint shifts, proving why 2D CNN representations collapse (0.0% SR) and why 3D Metric Point Cloud representations achieve universal spatial invariance (90.6% SR).
- **Future Work Scope:** **Degree-of-Freedom (DoF) & Cross-Embodiment Generalization**. Our multi-arm experiments (Franka 7-DoF vs xArm 6-DoF, reaching 92.2% SR) are framed as an advanced preview / future work validation showing the scalability of our decoupled architecture.
- **Academic Goal:** Deliver an airtight, publication-grade BTP thesis report, presentation deck, high-resolution visual plots, and live video demonstrations.

---

## 1. Workspace Map & System Specifications

| Path | Description | Git Status |
| :--- | :--- | :--- |
| `/home/maniskill/Multi-viewpoint_Generalization/` | **Primary standalone public GitHub repo.** All clean code, weights, reports, and assets live here. | Clean, tracked (`main`), synchronized with `origin/main`. |
| `/home/maniskill/` | Active root workspace containing simulator source (`mani_skill/`) and raw training runs. | Working development area. |
| `/home/maniskill/Multi-viewpoint_Generalization/reports/` | Compendium of formal research reports (Reports 01 to 25). | Tracked in Git. |
| `/home/maniskill/Multi-viewpoint_Generalization/phase_14_robustness_and_videos/videos/` | 5 broadcast-quality MP4 rollout showcase videos. | Tracked in Git. |

### Machine Specifications:
- **GPU:** NVIDIA GeForce RTX 4060 Laptop (8,188 MiB VRAM), Driver 595.99.02, CUDA 13.2.
- **CPU:** AMD Ryzen AI 9 HX 370 w/ Radeon 890M (24 execution threads).
- **RAM:** 29 GiB Host Memory (22+ GiB free), 8.0 GiB Swap.
- **OS:** Linux (Ubuntu-based).
- **Python Environment:** Python 3.9.19, PyTorch 2.6.0+cu124, ManiSkill 3.0.1, Gymnasium 0.29.1.
- **Permissions Rule:** Maintain `chmod -R 777 /home/maniskill/Multi-viewpoint_Generalization` at all times.

---

## 2. Executive Summary of Research Progression & Milestones

```
Phase 1-5 (2D PPO Baseline):
  - Fixed camera: 97.10% SR.
  - Camera shifted by merely ±15°: Drops to 4.2% - 11.2% SR.
  - 90° Side View: Complete collapse to 0.40% SR.
  - 30-Viewpoint Fibonacci Dome: Global average 5.65% SR.

Phase 6-9 (ManiWhere Replication & Multi-Teacher Distillation):
  - Trained on 4 discrete anchor cameras (Front-Mid, Front-Low, Left-Flank, Right-Flank).
  - Replicated ManiWhere discrete representation: ~92.4% SR on those 4 exact viewpoints.

Phase 10-12 (The Critical In-Between Failure Discovery):
  - Evaluated on 9 continuous camera viewpoints interpolated between the 4 trained anchors.
  - Finding: Complete collapse (0.0% Success Rate across all 9 in-between viewpoints!).
  - Extrinsics FiLM conditioning also failed (0.0% SR), proving 2D CNNs cannot interpolate 3D projective transformations.

Phase 13 (The 3D PointNet Breakthrough):
  - Unprojected RGB-D depth into table-frame metric 3D point clouds: P_world = [R|t] * (d * K^-1 [u, v, 1]^T).
  - Trained 512-point PointNet policy:
    * Anchor Viewpoints (4): 93.5% SR
    * Novel In-Between Viewpoints (9): 89.3% SR
    * Overall Generalization: 90.6% SR (Generalization gap closed from 100% to 4.2%!).

Phase 14 (Perturbation Robustness & Video Showcase):
  - 1,000 continuous perturbation stress trials: 75.7% continuous success under ±15cm camera jitter.
  - Recorded 5 broadcast-quality MP4 rollout videos across continuous viewpoints.

Phase 15 (Cross-Embodiment Generalization & Zero-Shot Transfer - Future Work):
  - Evaluated across Franka Panda (7 DoF) and xArm 6 Robotiq (6 DoF).
  - Combined OmniArm Policy: 84.9% on xArm 6, 79.2% on Franka Panda.
  - Zero-Shot Transfer: 28.9% SR and 66.3% Grasp Rate on completely unseen hybrid robot (xarm6_panda_gripper).

Phase 16 (Composite Modular Universal Architecture - Future Work):
  - Assembled decoupled Viewpoint Engine + Embodiment Kinematic Engine into single deliverable.
  - Overall Benchmark: 92.2% on xArm 6 Robotiq, 74.6% on Franka Panda across all 13 viewpoints suite.
```

---

## 3. Master Deliverables & Checkpoints Index

| Deliverable Name | File Path | Parameters / Tensors | Key Metric |
| :--- | :--- | :---: | :--- |
| **Phase 16 Composite Deliverable** | `phase_16_composite_modular_architecture/checkpoints/universal_composite_omniarm.pt` | 29 tensors, 0.70 MB | **92.2% xArm / 74.6% Panda** across all 13 viewpoints |
| **Phase 15 Combined OmniArm Model** | `phase_15_cross_embodiment/combined/combined_omniarm_best.pt` | 14 tensors, 0.61 MB | **84.9% xArm / 79.2% Panda** across all 13 viewpoints |
| **Phase 13 PointNet Generalist** | `phase_13_pointcloud_generalization/student/pointnet_student_best.pt` | 3 tensors (dict), 0.61 MB | **90.6% Overall SR** on Franka Panda across 13 viewpoints |
| **Automated Verification Test Suite** | `test_all_deliverables.py` | — | Runs verification across all checkpoints & rollouts (100% PASS) |

### Key Evaluation Scripts:
- `python3 test_all_deliverables.py`: Automated sanity test suite for all models and environments.
- `python3 evaluate_interpolated_viewpoints.py`: Benchmarks any policy across the 13-viewpoint suite.
- `python3 phase_14_robustness_and_videos/eval_phase14_robustness.py`: Runs continuous jitter & point cloud sparsity stress test.

---

## 4. Primary Visual Artifacts & Publication Figures

| Figure File | Topic | Role in BTP Presentation |
| :--- | :--- | :--- |
| `phase13_pointnet_comparison.png` | Phase 13 PointNet vs 2D CNN Benchmark | **Hero Result Slide:** Shows the 0.0% $\to$ 90.6% breakthrough across 13 views. |
| `phase10_interpolated_viewpoints_benchmark.png` | Phase 10 In-Between Collapse | **Problem Slide:** Proves that 2D CNNs collapse on continuous in-between angles. |
| `phase14_robustness_curves.png` | Phase 14 Jitter & Sparsity Curves | **Robustness Slide:** Proves stability under $\pm 15$ cm camera jitter and down to 64 points. |
| `camera_geometry_diagram.png` | Fibonacci Equal-Area Dome Geometry | **Setup Slide:** Visualizes the 3D continuous camera orbit around the table. |
| `camera_rendered_views.png` | Rendered Multi-Angle Images | **Visual Diversity Slide:** Shows how different the cube and robot look from various angles. |
| `cross_embodiment_master_comparison.png` | Cross-Embodiment Benchmark (Phase 15) | **Future Work Slide:** Shows multi-arm generalization across 7-DoF and 6-DoF robots. |
| `phase16_composite_modular_benchmark.png` | Composite Modular Architecture (Phase 16) | **Future Work Slide:** Shows single deliverable achieving 92.2% on xArm 6. |

---

## 5. Technical Implementation Details & Architectural Rules

### 5.1 3D Table Workspace Filtering
Raw camera depth maps must be unprojected into metric Cartesian space using camera intrinsics $K$ and extrinsics $[R \mid t]$, then clipped to the active workspace bounding box:
```python
def extract_table_workspace_pc(obs: dict, num_points: int = 512) -> torch.Tensor:
    pos = obs["sensor_data"]["base_camera"]["position"].float() / 1000.0  # mm to m
    cam2world = obs["sensor_param"]["base_camera"]["cam2world_gl"].to(pos.device)
    pos_homo = torch.cat([pos, torch.ones_like(pos[..., :1])], dim=-1)
    xyz_world = (pos_homo.reshape(B, -1, 4) @ cam2world.transpose(1, 2))[..., :3]

    # Universal table workspace bounds (discards robot links and background floor)
    mask = (
        (xyz_world[..., 0] >= -0.30) & (xyz_world[..., 0] <= 0.30) &
        (xyz_world[..., 1] >= -0.35) & (xyz_world[..., 1] <= 0.35) &
        (xyz_world[..., 2] >= 0.005) & (xyz_world[..., 2] <= 0.40)
    )
    return sample_points(xyz_world[mask], num_points)
```

### 5.2 Universal Proprioception Interface (12-D)
To decouple spatial perception from robot joint counts:
$$\mathbf{s} = [\mathbf{p}_{\text{tcp}} \in \mathbb{R}^3, \, \mathbf{q}_{\text{tcp}} \in \mathbb{R}^4, \, g_{\text{open}} \in [0, 1], \, \text{grasped} \in \{0, 1\}, \, \mathbf{p}_{\text{goal}} \in \mathbb{R}^3] \in \mathbb{R}^{12}$$

### 5.3 PointNet Spatial Architecture
- Input: Metric coordinates $[B, N=512, 3]$ + Proprioception $[B, 12]$.
- Layer 1: Conv1d(3, 64) $\to$ Conv1d(64, 128) $\to$ Conv1d(128, 256).
- Symmetric Pooling: `torch.max(x, dim=2)[0]` ensures permutation invariance.
- Output: Cartesian end-effector displacement $(\Delta x, \Delta y, \Delta z)$ and semantic grasp $a_g \in [-1, 1]$.

---

## 6. Chronological Session & Phase Log

- **2026-10-30:** Completed Phases 1–9. Replicated ManiWhere multi-teacher distillation across 4 discrete anchor cameras (~92% SR).
- **2026-10-31 (Morning):** Tested interpolated continuous camera angles (Phase 10). Discovered complete collapse (0.0% SR). Tested FiLM camera conditioning (Phase 12), confirmed 2D CNN limitation.
- **2026-10-31 (Midday):** Implemented 3D PointNet Metric Geometry representation (Phase 13). Breakthrough: 90.6% overall SR across all 13 continuous viewpoints.
- **2026-10-31 (Afternoon):** Executed Phase 14 perturbation stress tests (1,000 trials, 75.7% continuous success under jitter) and rendered broadcast rollout videos.
- **2026-10-31 (Evening):** Executed Phase 15 Cross-Embodiment Generalization (Franka Panda vs xArm 6 Robotiq) and Zero-Shot Transfer (28.9% SR on unseen robot).
- **2026-10-31 (Night):** Implemented Phase 16 Composite Modular Architecture. Validated single deliverable model achieving 92.2% on xArm 6 Robotiq and 74.6% on Franka Panda.
- **2026-11-10:** Verified test suite `test_all_deliverables.py` (100% PASS). Synchronized all 25 research reports and established `AGENT_HANDOFF.md` and `TODO.md`. Executed 4 comprehensive viewpoint stress experiments:
  - *Extrinsic calibration noise:* Sustained 82.0% SR at ±2.0 cm hand-eye jitter, 82.0% SR at ±6.0° angular error (`calibration_noise_robustness_curve.png`).
  - *RealSense depth noise:* Sustained 94.0% SR at RealSense D435 baseline σ=2.5mm; sustained 86.0% SR even when 75% of depth points are missing (`realsense_depth_noise_ablation.png`).
  - *Extreme OOD angles:* Maintained 82–92% across ±65° flanks and 0.90m–1.35m zoom; identified gripper self-occlusion at 75° steep overhead (`extreme_ood_viewpoints_benchmark.png`).
  - *Latency profiling:* RTX 4060 runs at 0.13 ms (7,837 FPS / 156.7x 50Hz); AMD Ryzen CPU runs at 0.29 ms (3,445 FPS / 68.9x 50Hz) (`latency_profiling_results.json`).
- **2026-11-10 (Afternoon): Phase 17 Multi-Task Replication:**
  - Automated offline multi-task training across continuous viewpoints for `PushCube-v1` and `PullCube-v1`.
  - Evaluated on all 13 camera viewpoints:
    - `PushCube-v1`: 75.0% Anchor SR, 75.3% In-Between SR, 75.2% Overall SR (Generalization Gap: **-0.3%**).
    - `PullCube-v1`: 44.0% Anchor SR, 48.2% In-Between SR, 46.9% Overall SR (Generalization Gap: **-4.2%**).
  - Proved universal viewpoint invariance across three distinct manipulation interaction regimes (`multitask_viewpoint_generalization_master.png`). Checkpoints saved in `phase_17_multitask_replication/checkpoints/`.

- **2026-11-10 (Evening): Multi-Task Generalized Policy Expansion (Roadmap added to `TODO.md`):**
  - Expanded roadmap to train generalized 3D PointNet models across 8 additional manipulation domains: `StackCube-v1`, `PokeCube-v1`, `PegInsertionSide-v1`, `PlugCharger-v1`, `OpenCabinetDrawer-v1`, `OpenCabinetDoor-v1`, `MoveBucket-v1`, and `ManiSkill-HAB`.
  - Fixed `StackCube-v1` environment bug (reverted cube half-size to 0.02) and verified official PPO teacher checkpoint (`ppo_pd_ee_delta_pose_ckpt.pt`) achieves 100.0% success rate (`eval_success_once_mean = 1.00`).
  - Downloaded 1,000 demonstrations for `PegInsertionSide-v1` (98.0% SR) and `PlugCharger-v1` (motion planning).

- **2026-11-10 (Night): Phase 17 Consistent Scaling Breakthrough (PushCube & PullCube $\to$ 97–98% SR):**
  - Identified the root cause of previous 75.2% (Push) and 46.9% (Pull) accuracy: the `pd_ee_delta_pos` (4-dim) PPO teachers were capped at 82.0% and 58.0% SR, and training was budgeted for only 64K transitions.
  - Re-trained both tasks under the exact consistent regime of PickCube: 1,920 episodes (**96,000 transitions** across continuous dome), **25 epochs**, and verified **100.0% PPO teachers** using `pd_ee_delta_pose` (7-dim action space, 28-dim proprioception).
  - Evaluated on all 13 camera viewpoints:
    - `PushCube-v1`: **98.0% Anchor SR, 96.9% In-Between SR, 97.2% Overall SR (Generalization Gap: 1.1%)**.
    - `PullCube-v1`: **98.0% Anchor SR, 98.0% In-Between SR, 98.0% Overall SR (Generalization Gap: 0.0%)**.
  - Generated updated plots: `pushcube-v1_viewpoint_benchmark_consistent.png`, `pullcube-v1_viewpoint_benchmark_consistent.png`, `multitask_viewpoint_generalization_consistent_master.png`, and updated `multitask_7task_viewpoint_generalization_master.png`. Checkpoints saved to `phase_17_multitask_replication/checkpoints/`.

- **2026-11-10 (Night): Phase 18 PokeCube Consistent Scaling ($\to$ 89.1% SR, 0.6% Gap):**
  - Scaled `PokeCube-v1` to the identical 96,000 transitions / 25 epochs regime using verified 96.0% PPO teacher (`pd_joint_delta_pos`, 8-dim action space, 29-dim proprioception).
  - Script: `phase_18_multitask_extended/train_pokecube_consistent.py`
  - Evaluated on all 13 camera viewpoints:
    - `PokeCube-v1`: **89.5% Anchor SR, 88.9% In-Between SR, 89.1% Overall SR (Generalization Gap: 0.6%)** (jump from 71.2% / 3.8% gap).
  - Checkpoint updated: `phase_18_multitask_extended/checkpoints/pointnet_pokecube-v1_best.pt`.
  - All four core manipulation archetypes are now fully standardized under the 96K transition regime with near-zero viewpoint generalization gap:
    - `PickCube-v1` (Prehensile Pick & Place): **90.6% Overall SR** (Gap: 4.2%)
    - `PushCube-v1` (Planar Sliding Contact): **97.2% Overall SR** (Gap: 1.1%)
    - `PullCube-v1` (Non-Prehensile Hook/Pull): **98.0% Overall SR** (Gap: 0.0%)
    - `PokeCube-v1` (Dynamic Impulsive Striking): **89.1% Overall SR** (Gap: 0.6%)

---

## 7. Instructions for Future Agents

1. **Do not re-train existing phases from scratch:** All checkpoints exist in `checkpoints/` and are fully validated.
2. **Always test deliverables using `python3 test_all_deliverables.py`** before proposing or committing changes.
3. **When doing research experiments on Viewpoint Generalization:** Use the 13-viewpoint suite defined in `viewpoints.json` as the standardized evaluation yardstick.
4. **Before ending any session:** Update this file and `TODO.md`, commit all changes to git, and run `chmod -R 777 /home/maniskill/Multi-viewpoint_Generalization`.
