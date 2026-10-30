# Phase 6: 8-Viewpoint Individual Specialist Benchmark (128x128 RGB-D)

This directory contains the independent viewpoint training pipeline, models, and 8x8 cross-viewpoint transfer evaluation matrix for robotic visual reinforcement learning (`PickCube-v1`).

---

## 1. Experimental Overview

- **Input Modality:** 4-Channel $128 \times 128$ RGB-D (Candidate A NatureCNN visual encoder + 29-dim proprioception MLP).
- **Initialization:** Orthogonal Random Initialization from scratch for each viewpoint (measuring pure viewpoint learnability without front-view bias).
- **Core Viewpoints (8 Strategic Anchors):**
  - **VP 17:** Canonical Front-Mid ($-4.7^\circ$ az, $31.8^\circ$ el) — Baseline Control
  - **VP 00:** Front-Low Graze ($0.0^\circ$ az, $12.3^\circ$ el) — Shallow Tabletop Perspective
  - **VP 28:** Front-High Tilt ($+2.9^\circ$ az, $46.2^\circ$ el) — Steep Downward Perspective
  - **VP 18:** Left Flank $90^\circ$ ($-89.7^\circ$ az, $33.1^\circ$ el) — Orthogonal Axis Inversion
  - **VP 27:** Right Flank $90^\circ$ ($+87.9^\circ$ az, $44.6^\circ$ el) — Opposing Flank
  - **VP 20:** Diagonal Front-Left ($-37.2^\circ$ az, $35.0^\circ$ el) — Diagonal $-45^\circ$
  - **VP 19:** Diagonal Front-Right ($+47.8^\circ$ az, $33.7^\circ$ el) — Diagonal $+45^\circ$
  - **VP 06:** Low Flank Right ($+72.6^\circ$ az, $18.3^\circ$ el) — Low-grazing Flank
- **Training Budget:** Up to 2,500,000 steps per viewpoint with target early stopping at $\ge 95.0\%$ eval success.
- **Cross-Evaluation:** Comprehensive $8 \times 8$ evaluation matrix (each model tested on all 8 viewpoints $\times$ 200 episodes) producing cross-transfer heatmaps and JSON metrics.

---

## 2. Directory Structure

```text
phase_6_viewpoint_specialists/
├── README.md                          <- This documentation
├── models.py                          <- 128x128 RGB-D modular policy architecture
├── train_viewpoint_specialist.py      <- Single-viewpoint training harness
├── run_master_phase6.py               <- Master 8-viewpoint orchestrator
├── evaluate_cross_transfer_matrix.py  <- 8x8 cross-evaluation matrix script
├── run_phase6_pipeline.sh             <- Runner script
├── status.sh                          <- Live telemetry monitor
├── pause.sh                           <- Non-destructive pause tool
├── resume.sh                          <- Seamless resumption tool
├── checkpoints/                       <- Converged weights (specialist_vp_{id}_best.pt)
├── eval_results/                      <- Trajectory metrics and cross-transfer heatmap
├── reports/                           <- Formal Research Report 15
└── tb_logs/                           <- TensorBoard logs per viewpoint
```

---

## 3. Control & Telemetry Commands

- **Check Progress:** `./status.sh`
- **Pause Pipeline:** `./pause.sh`
- **Resume Pipeline:** `./resume.sh`
- **Live Logs:** `tail -f master_execution.log`
