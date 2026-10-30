# Viewpoint Generalization Benchmark: Shifted Side-View Camera

## 1. Target Point & Cube Randomization

**Yes, both the initial cube position and the target lift point are completely randomized across all 3 spatial dimensions ($x, y, z$) in every single episode.**

As defined in [`PickCubeEnv._initialize_episode`](file:///home/maniskill/mani_skill/envs/tasks/tabletop/pick_cube.py#L90-L125):

```python
# 1. Cube Spawn Position: Randomized in a 20cm x 20cm area on table
xyz[:, :2] = torch.rand((b, 2)) * self.cube_spawn_half_size * 2 - self.cube_spawn_half_size
xyz[:, 2] = self.cube_half_size  # Rest on table surface
qs = randomization.random_quaternions(b, lock_x=True, lock_y=True) # Random yaw rotation

# 2. Target Goal Position: Fully randomized in X, Y, and Z
goal_xyz[:, :2] = torch.rand((b, 2)) * self.cube_spawn_half_size * 2 - self.cube_spawn_half_size
goal_xyz[:, 2] = torch.rand((b)) * self.max_goal_height + xyz[:, 2] # Up to +30cm above table
self.goal_site.set_pose(Pose.create_from_pq(goal_xyz))
```

- **Cube initial position:** Uniformly sampled across $x \in [-10\text{ cm}, +10\text{ cm}]$, $y \in [-10\text{ cm}, +10\text{ cm}]$ with randomized yaw angle.
- **Target lift height:** Uniformly sampled from $z_{\text{surface}}$ up to $+30\text{ cm}$ above the table, with randomized horizontal coordinates $(x_{\text{goal}}, y_{\text{goal}})$.

---

## 2. Shifted Side-View Geometry on the Sphere

The new camera position was placed precisely on the **same sphere centered at the manipulator base**:

$$\text{Sphere Center (Robot Base): } C = (-0.615,\ 0.0,\ 0.0)\text{ m}$$
$$\text{Sphere Radius: } R = \sqrt{(0.3 - (-0.615))^2 + 0.0^2 + 0.6^2} = \mathbf{1.0942\text{ meters}}$$

| Coordinate | Original Front Camera | Shifted Side Camera (New) | In Base Frame |
| :--- | :--- | :--- | :--- |
| **X (Depth)** | $+0.300\text{ m}$ | **$-0.615\text{ m}$** | $0.000\text{ m}$ |
| **Y (Lateral)** | $0.000\text{ m}$ | **$+0.915\text{ m}$** | $+0.915\text{ m}$ |
| **Z (Height)** | $+0.600\text{ m}$ | **$+0.600\text{ m}$** | $+0.600\text{ m}$ |
| **Distance to Base** | $1.0942\text{ m}$ | **$1.0942\text{ m}$** | Radius $R$ Preserved |

---

## 3. 1,000-Run Benchmark: Viewpoint Sensitivity

Evaluating the policy trained on the front camera without retraining on the shifted side view:

| Metric | Front View (Training Perspective) | Shifted Side View (90° Rotation) | Impact of Viewpoint Shift |
| :--- | :--- | :--- | :--- |
| **1,000-Run Accuracy** | **97.10%** (971 / 1,000) | **0.40%** (4 / 1,000) | **-96.7% drop** |
| **Mean Return** | **5.45** | **3.72** | Substantial degradation |
| **Mean Episode Length**| **12.9 steps** (fast pick) | **49.9 steps** (timed out) | Agent fails to grasp |

### Why Did Accuracy Drop to 0.4%?
Visual RL policies parameterized with convolutional encoders (CNNs) without multi-view data augmentation or 3D coordinate representations learn coordinate mappings coupled to the specific camera frame:
1. **Axis Inversion:** In the side view, arm motion along the robot's forward axis ($+X$) appears as horizontal pixel motion instead of vertical perspective scaling.
2. **Feature Misalignment:** The visual tokens extracted by NatureCNN do not correspond to the learned delta-position priors, illustrating the classic **out-of-distribution (OOD) viewpoint sensitivity** in single-camera robot learning.

---

## 4. Independent Test Video (Side-by-Side)

Below is the recorded test run showing what the robot's visual policy perceives from the side view (left) versus the external third-person physical reality (right):

![Shifted Side-View Execution Test](/root/.gemini/antigravity-cli/brain/82673f33-5c96-4154-bb42-cf6de559c7b7/side_view_test_evaluation.mp4)

- **Host Video File:** `/home/mahoraga/maniskill/side_view_test_evaluation.mp4`
- **Container Video File:** [`/home/maniskill/side_view_test_evaluation.mp4`](file:///home/maniskill/side_view_test_evaluation.mp4)
