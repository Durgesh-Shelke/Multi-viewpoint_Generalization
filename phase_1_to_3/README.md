# Phase 1 to Phase 3: Visual RL Baselines, Viewpoint Generalization, & Catastrophic Forgetting

This directory archives all code, models, benchmarks, evaluation videos, and reports from **Phase 1 through Phase 3** of the robotic visual reinforcement learning manipulation research (`PickCube-v1` on Franka Panda arm, picking a 4.5 cm solid block).

---

## 1. Overview of Phases Included

### Phase 1: Pure RGB Baseline & Real-Time Telemetry (Reports 01–03)
- **Goal:** Establish a high-accuracy visual policy on the canonical front viewpoint (`eye = [0.3, 0.0, 0.6]`, `target = [-0.1, 0.0, 0.1]`).
- **Core Results:**
  - **97.10% success** on 1,000 independent test runs with randomized cube and target poses.
  - Native 20 Hz 1:1 physical clock sync telemetry confirming grasp acquisition in ~0.65s (Step 13).

### Phase 2: Camera Viewpoint Sensitivity & 3D Sweeps (Reports 04–06)
- **Goal:** Test spatial generalization when the camera moves away from the training pose.
- **Core Results:**
  - **90° Side View:** Grasping success collapsed from 97.10% to **0.40%** due to visual coordinate frame axis inversion.
  - **13-Angle Semicircular Sweep (13,000 episodes):** A deviation of merely $\pm 15^\circ$ caused an $>85\%$ collapse (dropping to 4.2%–11.2%).
  - **Fibonacci Equal-Area Hemispheres (20 & 30 Viewpoints):** Tested across 3D dome viewpoints. Mean success was only **4.03%** (20 VP) and **5.65%** (30 VP), peaking strictly near the trained front pose (~99%).

### Phase 3: Continual Learning & Catastrophic Forgetting (Reports 07–10)
- **Goal:** Continually fine-tune the agent sequentially across all 30 viewpoints on the Fibonacci dome.
- **Core Results:**
  - Conclusive proof of **Catastrophic Forgetting**: Fine-tuning on new viewpoints completely overwrote earlier feature extractors.
  - Viewpoint 00 collapsed from 100% to **0.0%**; overall dome retention fell to **0.50%**.
  - **Joint Multi-View Training:** Designed architecture distributing 128 parallel vectorized environments concurrently across all 30 viewpoints to prevent forgetting.

---

## 2. Directory Structure

```text
phase_1_to_3/
├── README.md                                  <- This comprehensive index and guide
├── reports/                                   <- Formal research reports 01 through 10
│   ├── 01_sim2real_1000_run_benchmark.md
│   ├── 02_side_view_evaluation_report.md
│   ├── 03_live_realtime_evaluation.md
│   ├── 04_camera_viewpoint_sweep_report.md
│   ├── 05_fibonacci_hemisphere_20vp_report.md
│   ├── 06_fibonacci_hemisphere_30vp_report.md
│   ├── 07_sequential_training_architecture.md
│   ├── 08_catastrophic_forgetting_midway_report.md
│   ├── 09_final_sequential_training_benchmark_report.md
│   ├── 10_joint_multiview_training_setup.md
│   ├── maniskill_feasibility_report.txt
│   └── consolidated_research_reports_master.md
├── RGB/                                       <- Complete Pure RGB research archive
│   ├── checkpoints/                           <- Front-view baseline & sequential checkpoints
│   ├── evaluation_results/                    <- JSON benchmark summaries
│   ├── demonstration_videos/                  <- MP4 trajectory captures
│   └── reports/                               <- Phase 1-3 reports mirror
├── visual_manipulation/                       <- Early visual manipulation tests & wrappers
│   ├── best_sim2real_policy.pt
│   └── benchmark_side_view_1000_results.json
├── sequential_viewpoint_training/             <- Continual sequential training pipeline
│   ├── train_sequential_viewpoints.py
│   ├── viewpoints.json
│   └── backup_original_model.pt
├── fibonacci_hemisphere_30vp_results/         <- 30-viewpoint hemisphere videos & results
├── joint_multiview_training/                  <- Joint multi-view training scripts
└── joint_multiview_experiment/                <- Structured joint multi-view experiment workspace
    ├── configs/
    ├── src/
    └── runs/
```

---

## 3. Key Standalone Scripts

- **`evaluate_camera_sweep.py`:** Evaluates an RGB policy across 13 horizontal semicircle angles.
- **`evaluate_fibonacci_hemisphere.py`:** Evaluates an RGB policy across 30 Fibonacci dome viewpoints.
- **`train_visual_pick_bc.py`:** Behavior cloning pretraining exploration.
