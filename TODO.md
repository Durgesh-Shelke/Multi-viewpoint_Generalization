# TODO — Multi-Viewpoint Generalization & Continuous Spatial Invariance

How to use this list:
- Add ideas the moment they come up, even half-formed ones.
- Tick `[x]` when an item is done and add a one-line result with the date (`→ 2026-11-10: PointNet achieved 90.6% SR across 13 views`).
- Record architectural decisions and reasons in `AGENT_HANDOFF.md`. This file tracks what to do.
- Tags:
  - `[exp]` = Research experiment (produces benchmark metrics/plots).
  - `[btp]` = Presentation, slide deck, or thesis write-up work for tomorrow's submission.
  - `[infra]` = Tooling, test suites, or repository hygiene.
  - `[future]` = Future work (e.g. DoF generalization, real robot transfer).

---

## 1. Setup, Infrastructure & Verification
- [x] `[infra]` Standalone GitHub repository established → 2026-10-30: `https://github.com/Durgesh-Shelke/Multi-viewpoint_Generalization.git`
- [x] `[infra]` Dual-workspace synchronization with `chmod -R 777` permissions enforced → 2026-10-31
- [x] `[infra]` Build automated end-to-end verification test suite → 2026-11-10: `test_all_deliverables.py` (verified 100% PASS across checkpoints, architectures, and simulation rollouts)
- [x] `[infra]` Create `AGENT_HANDOFF.md` as single source of truth for all future agent sessions → 2026-11-10
- [x] `[infra]` Compile compendium of 25 research reports into `reports/` → 2026-11-10

---

## 2. Completed Research Phases (Chronological Summary)
- [x] `[exp]` **Phase 1–5: Baseline RL & Spatial Sensor Fusion Study** → 2026-10-30: Tested 5 sensor architectures over 50M steps; Candidate C (Asymmetric Low-Res RGB + High-Res Depth) was best single-view policy (95.8% in-domain, 45.8% jitter retention).
- [x] `[exp]` **Phase 6–9: Multi-Teacher Distillation & ManiWhere Replication** → 2026-10-30: Distilled 4 discrete anchor camera views (Front-Mid, Front-Low, Left-Flank, Right-Flank) achieving 92.4% SR.
- [x] `[exp]` **Phase 10–12: The Continuous In-Between Viewpoint Collapse** → 2026-10-31: Tested 9 interpolated camera angles on 2D CNNs; discovered complete collapse (0.0% SR). Extrinsics FiLM conditioning also scored 0.0% SR.
- [x] `[exp]` **Phase 13: 3D PointNet Metric Geometry Breakthrough** → 2026-10-31: Unprojected RGB-D into metric 3D point clouds in table frame; achieved 90.6% overall SR across all 13 camera angles (closing the generalization gap from 100% to 4.2%).
- [x] `[exp]` **Phase 14: Perturbation Robustness & Broadcast Video Gallery** → 2026-10-31: Executed 1,000 continuous perturbation stress trials (75.7% continuous success under ±15cm camera jitter); recorded 5 MP4 rollout videos across continuous viewpoints.
- [x] `[exp][future]` **Phase 15: Cross-Embodiment Generalization & Zero-Shot Transfer** → 2026-10-31: Evaluated across Franka Panda (7 DoF) and xArm 6 Robotiq (6 DoF); achieved 84.9% on xArm 6, 79.2% on Franka Panda, and 28.9% SR / 66.3% Grasp zero-shot on unseen hybrid robot.
- [x] `[exp][future]` **Phase 16: Composite Modular Universal Architecture** → 2026-10-31: Decoupled Viewpoint Engine from Kinematic Engine into a single deliverable (`universal_composite_omniarm.pt`); scored 92.2% on xArm 6 and 74.6% on Franka Panda.

---

## 3. High-Impact Research Sprint (Viewpoint Generalization — 1 Day Budget)

These experiments directly strengthen your BTP defense by addressing the exact technical questions professors ask:

- [x] `[exp]` **Experiment A: Sim-to-Real Extrinsic Calibration Noise Sensitivity Curve** → 2026-11-10: Swept 0–10 cm translation & 0–10° rotation error; policy maintains 82.0% SR at typical ±2.0 cm hand-eye jitter and 82.0% SR at ±6.0° angular error (`calibration_noise_robustness_curve.png`).
- [x] `[exp]` **Experiment B: RealSense Depth Sensor Noise & Hole Dropout Stress Test** → 2026-11-10: Swept Gaussian depth jitter 0–20 mm (94% SR at RealSense D435 baseline σ=2.5mm) and point sparsity (maintains 86% SR even when 75% of depth points are missing) (`realsense_depth_noise_ablation.png`).
- [x] `[exp]` **Experiment C: Extreme Out-of-Distribution (OOD) Viewpoint Stress Test** → 2026-11-10: Benchmarked Front Anchor (88%), Left Flank Az=-65° (82%), Right Flank Az=+65° (92%), Close Zoom Dist=0.90m (92%), and Wide Zoom Dist=1.35m (90%). Steep overhead (75°) identified gripper self-occlusion as optical limit (`extreme_ood_viewpoints_benchmark.png`).
- [x] `[exp]` **Experiment D: Forward Pass Latency & Model Profiling for Real-Time Control** → 2026-11-10: RTX 4060 runs at 0.13 ms (7,837 FPS / 156.7x faster than 50Hz); AMD Ryzen CPU runs at 0.29 ms (3,445 FPS / 68.9x faster than 50Hz) (`latency_profiling_results.json`).
- [x] `[exp]` **Experiment E: Multi-Task Viewpoint Generalization Replication (Phase 17)** → 2026-11-10: Trained & benchmarked across 13 camera viewpoints on `PushCube-v1` (75.2% overall SR, -0.3% generalization gap) and `PullCube-v1` (46.9% overall SR, -4.2% generalization gap), proving 3D PointNet metric geometry invariance is task-agnostic and universally reproducible across manipulation interaction regimes (`multitask_viewpoint_generalization_master.png`).

---

## 4. BTP Submission & Presentation Prep Checklist (For Tomorrow)
- [x] `[btp]` Draft Slide-by-Slide BTP Presentation Deck (12–14 slides with visual placement & timing) → 2026-11-10: Created [`reports/btp_presentation_deck.md`](file:///home/maniskill/Multi-viewpoint_Generalization/reports/btp_presentation_deck.md) with word-for-word 12-minute speaking script.
- [x] `[btp]` Prepare Examiner Viva / Defense Q&A Cheat Sheet → 2026-11-10: Created [`reports/btp_defense_viva_cheatsheet.md`](file:///home/maniskill/Multi-viewpoint_Generalization/reports/btp_defense_viva_cheatsheet.md) addressing top 10 difficult committee questions with mathematical and empirical justifications.
- [x] `[btp]` Package presentation media assets → 2026-11-10: 5 MP4 continuous rollout videos + publication-grade PNG plots (`multitask_viewpoint_generalization_master.png`, `calibration_noise_robustness_curve.png`, `realsense_depth_noise_ablation.png`, `extreme_ood_viewpoints_benchmark.png`) mapped to slide decks.

---

## 5. Multi-Task Continuous Viewpoint Generalization Expansion

Goal: Train and benchmark generalized 3D PointNet models across the full suite of manipulation tasks (replicating the success achieved on `PickCube-v1` [90.6% SR], `PushCube-v1` [75.2% SR], and `PullCube-v1` [46.9% SR]):

- [ ] `[exp]` **Task 1: `StackCube-v1` (Dual-Object Relational Stacking)**
  - Train generalized PointNet model for multi-object coordination across continuous spherical dome.
  - Distill from verified 100% PPO teacher (`ppo_pd_ee_delta_pose_ckpt.pt`) using 7-dim action space (`pd_ee_delta_pose`).
  - Benchmark across all 13 camera viewpoints (Anchor SR, In-Between SR, Generalization Gap).
- [ ] `[exp]` **Task 2: `PokeCube-v1` (Planar Contact & Impulsive Interaction)**
  - Utilize 1,000 downloaded demonstrations / PPO teacher (`/root/.maniskill/demos/PokeCube-v1/rl/`).
  - Train PointNet model across continuous camera dome and evaluate 13-viewpoint generalization.
- [ ] `[exp]` **Task 3: `PegInsertionSide-v1` (Sub-Millimeter High-Tolerance Insertion)**
  - Utilize 1,000 downloaded demonstrations (`trajectory.h5` with verified 98.0% success rate).
  - Train PointNet policy in joint action space (`pd_joint_delta_pos`) to demonstrate tight geometric tolerance invariance.
- [ ] `[exp]` **Task 4: `PlugCharger-v1` (Precision Peg-in-Hole Assembly)**
  - Train PointNet policy on the 1,000 motion planning demonstrations (`/root/.maniskill/demos/PlugCharger-v1/motionplanning/`).
- [ ] `[exp]` **Task 5: `OpenCabinetDrawer-v1` (Articulated Prismatic Kinematics)**
  - Train generalized PointNet model for articulated sliding drawer interaction across arbitrary viewpoints.
- [ ] `[exp]` **Task 6: `OpenCabinetDoor-v1` (Articulated Revolute Kinematics)**
  - Train generalized PointNet model for revolute hinge cabinet door manipulation.
- [ ] `[exp]` **Task 7: `MoveBucket-v1` (Constrained Heavy-Load Relocation)**
  - Train generalized PointNet model for lifting and placing bucket objects with physical mass constraints.
- [ ] `[exp]` **Task 8: `ManiSkill-HAB` (Home Assistant Benchmark / Habitat Long-Horizon Mobile Manipulation)**
  - Extend table-frame PointNet to multi-room mobile base navigation and manipulation in Habitat-Matterport 3D environments.

---

## 6. Post-BTP Future Work (DoF & Real-World Generalization)
- [ ] `[future]` Real-world deployment on physical Franka Panda arm using Intel RealSense D435 camera.
- [ ] `[future]` Multi-Arm Foundation Model: Scale Composite Modular Architecture to mobile manipulators (Fetch, Unitree G1 humanoid).
- [ ] `[future]` Multi-Object & Cluttered Manipulation: Extend table-frame PointNet to multi-class pick-and-place and obstacle avoidance.
