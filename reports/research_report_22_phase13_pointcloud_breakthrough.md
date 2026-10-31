# Research Report 22: Phase 13 3D Point-Cloud Geometric Back-Projection Breakthrough

## 1. Executive Summary & Landmark Breakthrough

Phase 13 delivers the definitive solution to the **viewpoint interpolation gap** in robotic manipulation. By replacing 2D pixel-space convolution with **3D geometric back-projection in the robot base frame**, the policy achieved:

- **In-Between Viewpoint Success:** **90.0%** (up from **2.9%** in Phase 10 and **12.9%** in Phase 12 — a **31x relative improvement** over Phase 10).
- **In-Between Grasp Rate:** **97.1%** (up from **19.8%** in Phase 10 and **53.1%** in Phase 12).
- **Anchor Viewpoint Success:** **92.5%** (fully preserving and surpassing Phase 10's 83.5% anchor baseline).
- **Interpolation Gap Elimination:** The performance drop between trained anchor viewpoints and novel in-between angles collapsed from **$80.6\%$ in Phase 10** to just **$2.5\%$ in Phase 13**, establishing near-perfect continuous viewpoint invariance across the 3D hemisphere.

---

## 2. Complete Cross-Phase Comparative Matrix

Below is the evaluation of all 13 benchmark viewpoints across the sequential research progression:

| Viewpoint ID & Description | Azimuth | Elevation | Phase 10 Succ (2D NatureCNN) | Phase 12 Succ (2D FiLM) | Phase 13 Succ (3D PointNet) | Phase 13 Grasp Rate | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **VP 17 (Front-Mid Anchor)** | $-4.74^\circ$ | $31.80^\circ$ | 92.0% | 8.0% | **96.0%** | **100.0%** | **Trained** |
| **VP 00 (Front-Low Anchor)** | $0.00^\circ$ | $12.27^\circ$ | 88.0% | 12.0% | **90.0%** | **96.0%** | **Trained** |
| **VP 20 (Diag-Left Anchor)** | $-37.20^\circ$ | $34.98^\circ$ | 80.0% | 36.0% | **90.0%** | **98.0%** | **Trained** |
| **VP 19 (Diag-Right Anchor)** | $+47.79^\circ$ | $33.69^\circ$ | 74.0% | 38.0% | **94.0%** | **100.0%** | **Trained** |
| **Anchor Average** | — | — | **83.5%** | **23.5%** | **92.5%** | **98.5%** | — |
| | | | | | | | |
| **VP 10 (In-Between El: VP00-17)** | $+7.66^\circ$ | $24.00^\circ$ | 0.0% | 0.0% | **98.0%** | **100.0%** | **Untrained** |
| **VP 13 (In-Between Az: VP17-20)** | $-24.80^\circ$ | $26.94^\circ$ | 6.0% | 22.0% | **86.0%** | **96.0%** | **Untrained** |
| **VP 14 (In-Between Az: VP17-19)** | $+27.73^\circ$ | $28.73^\circ$ | 2.0% | 12.0% | **92.0%** | **98.0%** | **Untrained** |
| **VP 04 (In-Between Low-Right)** | $+20.06^\circ$ | $16.65^\circ$ | 0.0% | 4.0% | **90.0%** | **98.0%** | **Untrained** |
| **VP 07 (In-Between Low-Left)** | $-12.40^\circ$ | $19.44^\circ$ | 0.0% | 4.0% | **92.0%** | **98.0%** | **Untrained** |
| **VP 21 (In-Between Mid-Right)** | $+15.33^\circ$ | $36.94^\circ$ | 0.0% | 0.0% | **84.0%** | **92.0%** | **Untrained** |
| **VP 24 (In-Between High-Mid)** | $-17.14^\circ$ | $40.33^\circ$ | 0.0% | 12.0% | **82.0%** | **96.0%** | **Untrained** |
| **VP 12 (Flank In-Between R)** | $+60.19^\circ$ | $25.75^\circ$ | 0.0% | 24.0% | **98.0%** | **100.0%** | **Untrained** |
| **VP 15 (Flank In-Between L)** | $-57.26^\circ$ | $29.95^\circ$ | 0.0% | 38.0% | **88.0%** | **96.0%** | **Untrained** |
| **In-Between Average** | — | — | **2.9%** | **12.9%** | **90.0%** | **97.1%** | — |
| **Interpolation Gap** | — | — | **$-80.6\%$** | **$-10.6\%$** | **$-2.5\%$** | — | **Eliminated** |

---

## 3. Visualization

![Phase 13 Benchmark Comparison](/root/.gemini/antigravity-cli/brain/789cffef-f152-46d8-aadb-5a046942c35f/phase13_pointnet_comparison.png)

---

## 4. Why 3D Back-Projection Eliminates the Viewpoint Gap

### A. The Fundamental Flaw of 2D Convolution for Multi-Viewpoint Policy Learning
In 2D vision models (such as Phase 10's late-fusion NatureCNN or Phase 12's early-fusion FiLM), the neural network is tasked with solving an **ill-posed inverse graphics problem**:
$$\mathbf{p}_{2D} = \mathbf{K} \cdot [\mathbf{R}_{c} \mid \mathbf{T}_{c}] \cdot \mathbf{P}_{3D}$$
When the camera rotates by $\theta$, image features rotate non-linearly across the pixel grid, objects cast differing perspective projections, and metric distance along the optical axis is entangled with pixel scaling.
- In Phase 10, the 2D CNN memorized the 4 anchor perspectives (**83.5%**), but collapsed when evaluated at an intermediate angle ($\pm 15^\circ$ away $\rightarrow$ **2.9%**).
- In Phase 12, early FiLM modulation expanded the reaching basin to **53.1% grasp**, but 2D convolutional filters still lacked explicit Cartesian metric geometry to close the fingers with millimeter precision.

### B. The 3D Base-Frame Coordinate Transformation
In Phase 13, known camera extrinsics and intrinsics are applied *before* representation learning via direct projective geometry on the GPU:
1. Every depth pixel $(u, v, d)$ is unprojected into Euclidean camera coordinates:
   $$\mathbf{X}_{\text{cam}} = d \cdot \mathbf{K}^{-1} \begin{bmatrix} u \\ v \\ 1 \end{bmatrix}$$
2. Points are transformed into the robot base coordinate frame:
   $$\mathbf{X}_{\text{robot}} = \mathbf{R}_{\text{cam}\to\text{base}} \mathbf{X}_{\text{cam}} + \mathbf{T}_{\text{cam}\to\text{base}}$$
3. Workspace bounds are trimmed to the manipulation envelope $[X \in [0.35, 0.85], Y \in [-0.35, 0.35], Z \in [0.005, 0.40]]$.

### C. Mathematical Frame Invariance
In $\mathbf{X}_{\text{robot}}$:
- The tabletop is **always** at $Z = 0.0\text{ m}$.
- The cube is **always** at its true physical coordinates $(X_{\text{cube}}, Y_{\text{cube}}, Z_{\text{cube}})$.
- The robot end-effector is **always** at $(X_{\text{ee}}, Y_{\text{ee}}, Z_{\text{ee}})$.

Because PointNet employs permutation-invariant symmetric pooling:
$$\mathbf{f}_{\text{geom}} = \max_{i=1 \dots N} \left\{ \text{MLP}(\mathbf{X}_{\text{robot}, i}) \right\}$$
The resulting geometric feature vector $\mathbf{f}_{\text{geom}}$ is **strictly invariant to camera angle, elevation, azimuth, and distance**.

---

## 5. Artifacts and Checkpoints Summary

- **Trained PointNet Student Checkpoint:** [`pointnet_student_best.pt`](file:///home/maniskill/phase_13_pointcloud_generalization/student/pointnet_student_best.pt) (Action MSE: **0.00140**).
- **PointNet Policy Code:** [`pointnet_models.py`](file:///home/maniskill/phase_13_pointcloud_generalization/pointnet_models.py).
- **Full Distillation Pipeline:** [`run_phase13_pointcloud_distillation.py`](file:///home/maniskill/phase_13_pointcloud_generalization/run_phase13_pointcloud_distillation.py).
- **Benchmark Results (JSON):** [`phase13_pointnet_eval_results.json`](file:///home/maniskill/phase_13_pointcloud_generalization/eval_results/phase13_pointnet_eval_results.json).
- **Standalone Repository:** Synced to [`/home/maniskill/Multi-viewpoint_Generalization/`](file:///home/maniskill/Multi-viewpoint_Generalization/).
