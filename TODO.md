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

- [ ] `[exp]` **Experiment A: Sim-to-Real Extrinsic Calibration Noise Sensitivity Curve**
  - *Research Question:* In the real world, camera extrinsic calibration $[R \mid t]$ is never perfect. How much calibration error can our 3D PointNet policy tolerate before performance degrades?
  - *Method:* Systematically inject translational noise $\Delta t \in [0, 1, 2, 5, 8, 10]\text{ cm}$ and rotational noise $\Delta \theta \in [0, 1^\circ, 3^\circ, 5^\circ, 8^\circ, 10^\circ]$ into the unprojection matrix across 100 evaluation episodes per level.
  - *Deliverable:* Publication-grade plot `calibration_noise_robustness_curve.png`.
- [ ] `[exp]` **Experiment B: RealSense Depth Sensor Noise & Hole Dropout Stress Test**
  - *Research Question:* Real-world active stereo/IR depth cameras (Intel RealSense D435) suffer from Gaussian depth jitter and missing black-hole dropouts on specular/reflective surfaces. How resilient is our PointNet policy to depth sensor defects?
  - *Method:* Add zero-mean Gaussian depth noise ($\sigma \in [0, 2, 5, 10, 20]\text{ mm}$) and random point dropout ($[0\%, 10\%, 25\%, 50\%, 75\%]$ missing points) to the tabletop point cloud.
  - *Deliverable:* Publication-grade plot `realsense_depth_noise_ablation.png`.
- [ ] `[exp]` **Experiment C: Extreme Out-of-Distribution (OOD) Viewpoint Stress Test**
  - *Research Question:* Our 13-viewpoint suite covers the frontal hemisphere (Azimuth $[-70^\circ, +70^\circ]$, Elevation $[12^\circ, 50^\circ]$). How does the policy perform on extreme camera angles never seen in training?
  - *Method:* Test on Top-Down Bird's Eye View (Elevation $85^\circ$), Extreme Flank Angles (Azimuth $\pm 85^\circ$), and Extreme Distances ($0.7\text{m}$ close zoom vs $1.8\text{m}$ wide zoom).
  - *Deliverable:* Benchmark table & bar chart comparing 3D PointNet vs 2D CNN under extreme OOD angles.
- [ ] `[exp]` **Experiment D: Forward Pass Latency & Model Profiling for Real-Time Control**
  - *Research Question:* Is 3D PointNet fast enough to run on an embedded robot computer (Raspberry Pi 5 / Jetson Orin) at standard 50 Hz control loops?
  - *Method:* Benchmark CPU vs GPU forward pass latency over 1,000 forward passes (measuring mean, p50, p95, p99 latency in milliseconds and FPS).
  - *Deliverable:* Latency table proving $< 5\text{ms}$ execution time (200+ Hz).
- [ ] `[exp]` **Experiment E: Viewpoint Sample Density Ablation**
  - *Research Question:* How many training viewpoints are necessary to achieve universal viewpoint generalization?
  - *Method:* Compare policies trained on 1 view vs 2 views vs 4 views vs continuous dome sampling.

---

## 4. BTP Submission & Presentation Prep Checklist (For Tomorrow)
- [ ] `[btp]` Draft Slide-by-Slide BTP Presentation Deck (12–14 slides with visual placement & timing).
- [ ] `[btp]` Compile Speaker Speaking Script (word-for-word narrative connecting Problem $\to$ 2D Failure $\to$ 3D Solution $\to$ Results $\to$ Future Work).
- [ ] `[btp]` Draft Formal BTP Thesis Chapter / LaTeX-ready Report Section.
- [ ] `[btp]` Prepare Examiner Viva / Defense Q&A Cheat Sheet (Top 10 hard questions and mathematically grounded answers).
- [ ] `[btp]` Package presentation media assets: 5 MP4 rollout videos + high-res PNG plots mapped to each slide.

---

## 5. Post-BTP Future Work (DoF & Real-World Generalization)
- [ ] `[future]` Real-world deployment on physical Franka Panda arm using Intel RealSense D435 camera.
- [ ] `[future]` Multi-Arm Foundation Model: Scale Composite Modular Architecture to mobile manipulators (Fetch, Unitree G1 humanoid).
- [ ] `[future]` Multi-Object & Cluttered Manipulation: Extend table-frame PointNet to multi-class pick-and-place and obstacle avoidance.
