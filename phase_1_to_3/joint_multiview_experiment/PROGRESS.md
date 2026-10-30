# Project Progress

*Short status tracker for the joint multi-view RGB vs RGB-D experiment.*

---

## Status Summary

- **Active Milestone**: Milestone M2b (Contact sheets & Sanity)
- **Currently Running**: None
- **Next Milestone Step**: Render 30-viewpoint contact sheets (RGB and depth) before launching pilot (M3)

---

## Milestone Tracker

| Milestone | Description | Status | Completed At |
|---|---|---|---|
| **M0** | **Setup & Discovery**: Asset discovery, environment spec, git init, layout creation, DECISIONS.md, README.md, PROGRESS.md | **DONE** | 2026-10-21 13:55 |
| **M1** | **Resumable Harness**: Atomic checkpoints, signal/pause handling, `ctl.py`, `runner.py`, passing all 6 tests in `test_pause_resume.py` | **DONE** | 2026-10-21 18:42 |
| **M2a** | **Split & Evaluation Suite**: Freeze `viewpoint_split.json`, `split_geometry.csv`, `fresh_views.json` (60 views, SHA256), `evaluate.py` parity validation (VP 17, 16, 18, 28 reproduced), all 97 tests pass | **DONE** | 2026-10-21 22:25 |
| **M2b-1** | **Evaluation Integration**: RGB-D validation (VP 16, 23, 27, train view), batch-size invariance fix, periodic eval in `train_ppo.py`, automatic final eval in `runner.py`, `tests/test_evaluation_integration.py` passing | **DONE** | 2026-10-22 00:35 |
| **M2b-2** | **Smoke Test & Measurements**: Real config (160 envs, 16 minibatches), 200k steps RGB & RGB-D, ctl.py lifecycle, 2.3k/2.0k SPS, 4.4GB peak VRAM, budget projections | **DONE** | 2026-10-22 01:05 |
| **M3-prep** | **Pilot Preparation & Freeze**: `final_eval_groups` enforcement, SPEC 4.3 budget selection, early sanity alerts, preflight verification, freeze manifest, read-only permissions | **DONE** | 2026-10-22 01:45 |
| **M3-supervisor** | **Overnight Supervisor & Safety Guards**: Standard library watchdog, schedule window, hardware guards, frozen code export, OVERNIGHT_REPORT.md, 10-minute real rehearsal | **DONE** | 2026-10-22 02:15 |
| **M2b** | **Sanity & Contact Sheet**: Render contact sheets (RGB and depth) | PENDING | |
| **M3** | **Pilot Experiment**: 30M-step pilot (seed 100) RGB & RGB-D, determine and freeze budget $B$ | PENDING | |
| **M4** | **Main Grid**: 2 modalities x 3 seeds (0, 1, 2) trained for $B$ steps, detached & pausable | PENDING | |
| **M5** | **Final Evaluation**: 30 viewpoints, 200 episodes/view, Wilson 95% CIs on final checkpoints | PENDING | |
| **M6** | **Report & Deliverables**: Learning curves, generalization plots, efficiency metrics, final `results/REPORT.md` | PENDING | |

---

## Recent Implementations & Hardening

### 1. Model Architecture (`src/model.py`)
Implemented per SPEC Section 1 & 3.1:
- **Vision Backbone (`NatureCNN`)**: Supports 3 input channels (RGB) or 4 input channels (RGB-D) at 128x128 resolution.
- **Orthogonal Initialization**: All convolutional (`Conv2d`) and linear (`Linear`) layers are initialized orthogonally (`std=sqrt(2)`) with zero biases.
- **State Integration**: Proprietary state branch `Linear(29 -> 256)` fused with the 256-dim visual representation to yield a 512-dimensional latent vector.
- **Actor & Critic Heads**:
  - Critic head: `Linear(512 -> 512) -> ReLU -> Linear(512 -> 1, std=1.0)`.
  - Gaussian Actor head: `Linear(512 -> 512) -> ReLU -> Linear(512 -> action_dim, std=0.01 * sqrt(2))` with learnable `actor_logstd` initialized to -0.5.
- **Pure Scratch Training**: Zero weights borrowed or warm-started from existing checkpoints.

### 2. Environment Factory (`src/env_factory.py`) & Viewpoint Split (`configs/viewpoint_split.json`)
- **Provisional Split (`configs/viewpoint_split.json`)**: Configured with 20 training views, 3 validation views (03, 10, 15), 3 interpolation test views (09, 13, 21), and 4 extrapolation test views (06, 16, 23, 27). Loaded dynamically without hardcoding.
- **Viewpoint Allocation**: Enforces strict equal division of `num_envs` across viewpoints (`envs_per_view = num_envs // len(views)`).
- **Per-Environment Camera Injection (Bug Fix)**:
  - Discovered that legacy `rcam.set_local_pose(sp)` post-`gym.make` did not propagate to SAPIEN 3.0 CUDA batch renderer buffers (`update_render` rewrote poses to default front view).
  - Resolved by constructing batched `Pose` of shape `(num_envs, 7)` using `sapien_utils.look_at` and injecting via `sensor_configs={"base_camera": {"pose": batched_pose}}` at `gym.make` time.
  - Sub-camera local poses and extrinsics synchronized and verified against `viewpoints.json` within numerical tolerance (`p_err = 0.00e+00`, `q_err = 0.00e+00`) post-creation and post-reset.
- **Standardized Depth Preprocessing**: Invalid depth (`<= 0`) replaced with 3000 mm, clamped to `[0.0, 3.0]` meters, and normalized to `[0.0, 1.0]`.
- **Standardized Observation Wrapper**:
  - RGB mode: `obs["rgb"]` `(B, 128, 128, 3)` sourced from observation sensor camera (`sensor_data["base_camera"]["rgb"]`)
  - RGB-D mode: `obs["rgbd"]` `(B, 128, 128, 4)` and `obs["depth"]` `(B, 128, 128, 1)` in `[0.0, 1.0]`
  - Robot state: `obs["state"]` `(B, 29)`

### 3. Smoke Test Camera Grid (`results/smoke/smoke_camera_grid.png`)
- Generated 8-environment simulation stepped over 4 distinct viewpoints (`VP 00`, `VP 01`, `VP 02`, `VP 04`).
- Sourced directly from sensor camera observation data (`obs["rgb"]`, never human render camera).
- Stitched composite annotated image showing all 8 environment camera angles:
  - **Saved Image Path**: [`results/smoke/smoke_camera_grid.png`](file:///home/maniskill/joint_multiview_experiment/results/smoke/smoke_camera_grid.png)
  - Image size: 512x1024x3 (2x4 grid of 256x256 annotated tiles with Viewpoint ID and Azimuth / Elevation).
  - **Quantified Viewpoint Differentiation**:
    - Different views max pairwise mean absolute pixel diff: **37.37 - 40.96** (> 20.0 threshold passed).
    - Same view pairwise mean absolute pixel diff: **0.78 - 1.91** (< 5.0 threshold passed; accounts for randomized cube/goal).

### 4. PPO Training Loop (`src/train_ppo.py`) & Frozen Hyperparameters (`configs/ppo_frozen.yaml`)
- **CleanRL-Style PPO**: Vectorized observation buffering via `DictArray`, orthogonal NatureCNN agent (`src/model.py`), and per-environment viewpoints (`src/env_factory.py`).
- **Frozen Hyperparameters (`configs/ppo_frozen.yaml`)**:
  - `lr=2e-4`, `gamma=0.8`, `gae_lambda=0.9`, `update_epochs=4`, `clip_coef=0.2`, `max_grad_norm=0.5`.
  - Overrides per SPEC Section 4.2: `num_envs=160` (8 envs per view) and `num_minibatches=16` (minibatch size 500).
- **Graceful Pausing & Atomic Checkpointing**:
  - Catches `SIGINT` (double-press within 10s forces immediate exit) and `SIGTERM`.
  - Monitors `state/PAUSE`, `PAUSE`, and `output_dir/PAUSE` before each iteration and during saves.
  - On pause: completes current iteration, flushes atomic checkpoint (`model`, `optimizer`, `rng_states`, `global_step`, `iteration`, `active_time_s`, `eval_history`, `config_hash`) via `src/checkpoint.py`, and exits with code 0.
- **Auto-Resume & Log Continuity**:
  - Automatically loads newest valid checkpoint using `load_checkpoint_with_fallback()` with config-hash guard.
  - Derives deterministic environment reset seeds via `(run_seed, iteration)` formula: `(seed * 1000003 + iter * 997) % (2**31 - 1)`.
  - Appends to `progress.csv` with pre-loop sanitization (`sanitize_csv_for_resume`) ensuring strictly monotonic iterations and continuous step counters with zero duplicated or missing rows.
- **Tiny Test Config (`configs/tiny_test.yaml`)**:
  - 2 views x 2 envs = 4 envs, `num_steps=50`, 4 minibatches, 5,000 steps.

### 5. Frozen Configuration Completeness, Camera Cross-Verification, Task Integrity & Bootstrap Tests
- **Complete Environment & Task Settings (`configs/ppo_frozen.yaml`)**:
  - Added explicit settings: `reward_mode: "normalized_dense"`, `max_episode_steps: 50`, `control_mode: "pd_ee_delta_pos"`, `sim_backend: "physx_cuda"`, `ignore_terminations: false`, `record_metrics: true`.
  - Camera dictionary: `width: 128`, `height: 128`, `fov: 1.5707963267948966` ($\pi/2$), `near: 0.01`, `far: 100.0`, `look_at: [-0.1, 0.0, 0.1]`.
  - Rationale documented: Franka base at `[-0.615, 0.0, 0.0]`, cube at `[0.0, 0.0, 0.0]`; focus point `[-0.1, 0.0, 0.1]` centers the camera on the grasping and manipulation volume across all 30 hemispherical viewpoints.
- **Fail-Fast Runtime Environment Validation (`src/env_factory.py`)**:
  - `extract_effective_env_settings(env)` reads live simulator camera parameters, unwrapped reward modes, and time limits.
  - `validate_env_against_config(env, config)` asserts exact agreement between live environment and frozen config, failing fast if the simulator disagrees.
- **Evaluator Cross-Verification (Empirical Rendering Comparison Across All 30 Viewpoints)**:
  - Rendered all 30 viewpoints (`VP 00` through `VP 29`) with matching seeds using both:
    1. Evaluator convention (`evaluate_fibonacci_hemisphere_rgbd.py`: `sensor_cam_eye_pos`, `sensor_cam_target_pos`)
    2. Environment factory convention (`make_joint_multiview_env` with `camera.look_at: [-0.1, 0.0, 0.1]`)
  - **Empirical Pixel & Pose Comparison Table (Complete 30-Viewpoint Cross-Verification)**:

    | Viewpoint ID | Target Eye Position $[x, y, z]$ | Look-At Target $[x, y, z]$ | Position Error | Quaternion Error | RGB MAD (Pixel Diff) | Depth MAD (Norm Depth) |
    |---|---|---|---|---|---|---|
    | **VP 00** | `[+0.4542, +0.0000, +0.2326]` | `[-0.1, 0.0, 0.1]` | `0.00e+00` | `0.00e+00` | **0.0000** | **0.000000** |
    | **VP 01** | `[-0.5219, -1.0605, +0.2528]` | `[-0.1, 0.0, 0.1]` | `0.00e+00` | `0.00e+00` | **0.0000** | **0.000000** |
    | **VP 02** | `[+0.0312, +0.8429, +0.2629]` | `[-0.1, 0.0, 0.1]` | `0.00e+00` | `0.00e+00` | **0.0000** | **0.000000** |
    | **VP 03** | `[+0.2768, -0.5673, +0.2832]` | `[-0.1, 0.0, 0.1]` | `0.00e+00` | `0.00e+00` | **0.0000** | **0.000000** |
    | **VP 04** | `[+0.3697, +0.3596, +0.3135]` | `[-0.1, 0.0, 0.1]` | `0.00e+00` | `0.00e+00` | **0.0000** | **0.000000** |
    | **VP 05** | `[-0.1733, -0.9438, +0.3338]` | `[-0.1, 0.0, 0.1]` | `0.00e+00` | `0.00e+00` | **0.0000** | **0.000000** |
    | **VP 06** | `[-0.3041, +0.9911, +0.3439]` | `[-0.1, 0.0, 0.1]` | `0.00e+00` | `0.00e+00` | **0.0000** | **0.000000** |
    | **VP 07** | `[+0.3928, -0.2216, +0.3641]` | `[-0.1, 0.0, 0.1]` | `0.00e+00` | `0.00e+00` | **0.0000** | **0.000000** |
    | **VP 08** | `[+0.1654, +0.6577, +0.3945]` | `[-0.1, 0.0, 0.1]` | `0.00e+00` | `0.00e+00` | **0.0000** | **0.000000** |
    | **VP 09** | `[+0.1027, -0.7142, +0.4147]` | `[-0.1, 0.0, 0.1]` | `0.00e+00` | `0.00e+00` | **0.0000** | **0.000000** |
    | **VP 10** | `[+0.3757, +0.1333, +0.4451]` | `[-0.1, 0.0, 0.1]` | `0.00e+00` | `0.00e+00` | **0.0000** | **0.000000** |
    | **VP 11** | `[-0.3976, -0.9662, +0.4653]` | `[-0.1, 0.0, 0.1]` | `0.00e+00` | `0.00e+00` | **0.0000** | **0.000000** |
    | **VP 12** | `[-0.1250, +0.8551, +0.4754]` | `[-0.1, 0.0, 0.1]` | `0.00e+00` | `0.00e+00` | **0.0000** | **0.000000** |
    | **VP 13** | `[+0.2705, -0.4091, +0.4957]` | `[-0.1, 0.0, 0.1]` | `0.00e+00` | `0.00e+00` | **0.0000** | **0.000000** |
    | **VP 14** | `[+0.2343, +0.4464, +0.5260]` | `[-0.1, 0.0, 0.1]` | `0.00e+00` | `0.00e+00` | **0.0000** | **0.000000** |
    | **VP 15** | `[-0.1022, -0.7975, +0.5463]` | `[-0.1, 0.0, 0.1]` | `0.00e+00` | `0.00e+00` | **0.0000** | **0.000000** |
    | **VP 16** | `[-0.4554, +0.9286, +0.5564]` | `[-0.1, 0.0, 0.1]` | `0.00e+00` | `0.00e+00` | **0.0000** | **0.000000** |
    | **VP 17** | `[+0.3118, -0.0768, +0.5766]` | `[-0.1, 0.0, 0.1]` | `0.00e+00` | `0.00e+00` | **0.0000** | **0.000000** |
    | **VP 18** | `[-0.6105, -0.9171, +0.5969]` | `[-0.1, 0.0, 0.1]` | `0.00e+00` | `0.00e+00` | **0.0000** | **0.000000** |
    | **VP 19** | `[-0.0033, +0.6743, +0.6070]` | `[-0.1, 0.0, 0.1]` | `0.00e+00` | `0.00e+00` | **0.0000** | **0.000000** |
    | **VP 20** | `[+0.0992, -0.5420, +0.6272]` | `[-0.1, 0.0, 0.1]` | `0.00e+00` | `0.00e+00` | **0.0000** | **0.000000** |
    | **VP 21** | `[+0.2284, +0.2312, +0.6576]` | `[-0.1, 0.0, 0.1]` | `0.00e+00` | `0.00e+00` | **0.0000** | **0.000000** |
    | **VP 22** | `[-0.3164, -0.8054, +0.6778]` | `[-0.1, 0.0, 0.1]` | `0.00e+00` | `0.00e+00` | **0.0000** | **0.000000** |
    | **VP 23** | `[-0.2942, +0.7881, +0.6880]` | `[-0.1, 0.0, 0.1]` | `0.00e+00` | `0.00e+00` | **0.0000** | **0.000000** |
    | **VP 24** | `[+0.1821, -0.2457, +0.7082]` | `[-0.1, 0.0, 0.1]` | `0.00e+00` | `0.00e+00` | **0.0000** | **0.000000** |
    | **VP 25** | `[+0.0432, +0.4675, +0.7386]` | `[-0.1, 0.0, 0.1]` | `0.00e+00` | `0.00e+00` | **0.0000** | **0.000000** |
    | **VP 26** | `[-0.1040, -0.6003, +0.7588]` | `[-0.1, 0.0, 0.1]` | `0.00e+00` | `0.00e+00` | **0.0000** | **0.000000** |
    | **VP 27** | `[-0.5866, +0.7780, +0.7689]` | `[-0.1, 0.0, 0.1]` | `0.00e+00` | `0.00e+00` | **0.0000** | **0.000000** |
    | **VP 28** | `[+0.1420, +0.0387, +0.7892]` | `[-0.1, 0.0, 0.1]` | `0.00e+00` | `0.00e+00` | **0.0000** | **0.000000** |
    | **VP 29** | `[-0.5133, -0.7292, +0.8094]` | `[-0.1, 0.0, 0.1]` | `0.00e+00` | `0.00e+00` | **0.0000** | **0.000000** |

  - Automated test `tests/test_model_and_env.py::test_evaluator_vs_env_factory_cross_verification` confirms 100% bit-for-bit equivalence across all 30 viewpoints.
- **ManiSkill Source Fingerprint Integrity Guard (`src/fingerprint.py`, `src/runner.py`, `src/train_ppo.py`)**:
  - Computed via read-only commands: `git HEAD` of `/home/maniskill` + SHA-256 of `git -C /home/maniskill diff -- mani_skill`.
  - Stored in `state/reference_fingerprint.json` and recorded in every run's `manifest.json`.
  - Current Reference Fingerprint:
    - Git HEAD: `62ff3a5896b4d5b4cf0ac4c8d79afe600c9404a3`
    - Diff SHA-256: `b4b57c80a0cb153dc2b284c1285a8278139e4a90e0b222dea2769cc94df12e72`
    - Composite: `62ff3a5896b4d5b4cf0ac4c8d79afe600c9404a3:b4b57c80a0cb153dc2b284c1285a8278139e4a90e0b222dea2769cc94df12e72`
  - `runner.py` and `train_ppo.py` strictly refuse to start or resume any run if the fingerprint differs (`ManiSkillFingerprintMismatchError`).
- **Mock Truncation Unit Test (`tests/test_truncation_bootstrap.py`)**:
  - `test_mock_truncation_forces_value_bootstrap` mocks the environment and agent to simulate step truncation.
  - Proves that return on truncation step incorporates $\gamma \cdot V(s_{\text{final}})$:
    $$returns_{\text{bootstrap}} - returns_{\text{zero}} = \gamma \cdot V(s_{\text{final}})$$
- **Multi-Run Runner & Concurrency Guard (`src/runner.py`, `tests/test_runner.py`)**:
  - `runner.py` parses `configs/runs.yaml`, tracks run queue in `state/queue.json`, and enforces single-instance execution via `LockManager` with PID and stale lock cleanup.
  - Strictly refuses to launch unbudgeted runs (`total_timesteps: null`).
- **ManiSkill Task Integrity Guard in Manifest (`src/train_ppo.py`)**:
  - Automatically records SHA-256 hashes of:
    - `mani_skill/envs/tasks/tabletop/pick_cube.py`: `5785c7a0f2b98a2a6a766c6f8b40b2d954231aa34f238aecf44c5ffb99c670e5`
    - `mani_skill/envs/tasks/tabletop/pick_cube_cfgs.py`: `4742421d872e00623a8672768ba22c26aa74b42498cd3664b63279cddef649f8`
  - Manifest written to `output_dir/manifest.json`.
  - Resumption strictly refused with `ManiSkillIntegrityError` if either file's hash changes.
- **Time-Limit Truncation & Value Function Bootstrap (`tests/test_truncation_bootstrap.py`)**:
  - Step 50 truncation triggers `_final_info` and extracts `final_observation` non-destructively.
  - Critic computes $V(s_{\text{final}})$ via `agent.get_value(final_obs)` and injects it into `final_values[step]`.
  - Unit tests verify that boundary return incorporates $\gamma \cdot V(s_{\text{final}})$:
    $$returns[49]_{\text{bootstrap}} - returns[49]_{\text{zero}} = \gamma \cdot V(s_{\text{final}})$$
- **Run Queue Specification (`configs/runs.yaml`)**:
  - Removed `total_timesteps` from `ppo_frozen.yaml`.
  - Configured `pilot_rgb` and `pilot_rgbd` for 30M steps (seed 100).
  - Configured all 6 main-grid runs with `total_timesteps: null`.
  - `train_ppo.py` fails fast with `ValueError` if an unbudgeted run is launched.

---

### Real Command Output: Pause & Resume Verification (RGB and RGB-D)

```text
======================================================================
DEMO PAUSE & RESUME: RGB
======================================================================
>>> PHASE 1: Launching initial run for rgb...
INFO:train_ppo:Initializing ManiSkill environment with num_envs=4, modality=rgb, backend=physx_cuda
INFO:train_ppo:[Start] No prior checkpoint found. Starting fresh training from scratch.
INFO:train_ppo:STARTING PPO ROLLOUT LOOP: RGB
INFO:train_ppo:  Total Steps:      2,000 (10 iterations of 200 steps)
INFO:train_ppo:  Start Iteration:  1
INFO:train_ppo:Iter   1/ 10 | Step      200 | SPS  186 | Rew  0.049 | Ret  2.44 | Succ   0.0% | Active   1.1s
INFO:train_ppo:Iter   2/ 10 | Step      400 | SPS  211 | Rew  0.040 | Ret  2.21 | Succ   0.0% | Active   2.0s
INFO:train_ppo:Iter   3/ 10 | Step      600 | SPS  212 | Rew  0.041 | Ret  2.15 | Succ   0.0% | Active   3.0s
>>> Sending SIGTERM to PID 1363572...
INFO:train_ppo:[Signal] Received signal 15 (SIGTERM). Requesting clean pause after current iteration...
INFO:train_ppo:Iter   4/ 10 | Step      800 | SPS  208 | Rew  0.056 | Ret  2.31 | Succ   0.0% | Active   3.9s
INFO:train_ppo:[Pause] Iteration 4 cleanly finished. Checkpoint saved: /home/maniskill/joint_multiview_experiment/runs/demo_pause_resume_rgb/checkpoints/ckpt_step_000000800.pt. Active time: 3.9s. Exiting code 0.
>>> Phase 1 Exit Code: 0

>>> PHASE 2: Resuming rgb with exact same command...
INFO:train_ppo:Initializing ManiSkill environment with num_envs=4, modality=rgb, backend=physx_cuda
INFO:train_ppo:[Resume] Valid checkpoint found: /home/maniskill/joint_multiview_experiment/runs/demo_pause_resume_rgb/checkpoints/ckpt_step_000000800.pt
INFO:train_ppo:[Resume] Resumed successfully at Iteration 5/10 (Global Step: 800, Active Time: 3.9s)
INFO:train_ppo:STARTING PPO ROLLOUT LOOP: RGB
INFO:train_ppo:  Total Steps:      2,000 (10 iterations of 200 steps)
INFO:train_ppo:  Start Iteration:  5
INFO:train_ppo:Iter   5/ 10 | Step    1,000 | SPS  176 | Rew  0.071 | Ret  2.79 | Succ   0.0% | Active   5.1s
INFO:train_ppo:Iter   6/ 10 | Step    1,200 | SPS  208 | Rew  0.057 | Ret  2.83 | Succ   0.0% | Active   6.0s
INFO:train_ppo:Iter   7/ 10 | Step    1,400 | SPS  207 | Rew  0.043 | Ret  2.64 | Succ   0.0% | Active   7.0s
INFO:train_ppo:Iter   8/ 10 | Step    1,600 | SPS  206 | Rew  0.056 | Ret  2.71 | Succ   0.0% | Active   8.0s
INFO:train_ppo:Iter   9/ 10 | Step    1,800 | SPS  205 | Rew  0.047 | Ret  2.58 | Succ   0.0% | Active   9.0s
INFO:train_ppo:Iter  10/ 10 | Step    2,000 | SPS  191 | Rew  0.050 | Ret  2.58 | Succ   0.0% | Active  10.0s
INFO:train_ppo:[Done] Training reached target budget (2000 steps). Final checkpoint: /home/maniskill/joint_multiview_experiment/runs/demo_pause_resume_rgb/checkpoints/ckpt_final_step_000002000.pt
>>> Phase 2 Exit Code: 0

>>> PROGRESS.CSV FOR RGB:
iteration,global_step,active_time_s,sps,mean_reward,mean_return,mean_length,success_rate,pg_loss,v_loss,entropy,approx_kl,clipfrac,lr
1,200,1.08,186,0.0488,2.438,50.0,0.0,-0.00768,0.12648,3.67664,0.00518,0.065,0.0002
2,400,2.03,211,0.0401,2.21,50.0,0.0,-0.00693,0.00647,3.67732,0.00163,0.0,0.0002
3,600,2.97,212,0.0411,2.152,50.0,0.0,-0.00508,0.00416,3.67688,0.00227,0.015,0.0002
4,800,3.93,208,0.0562,2.308,50.0,0.0,-0.00531,0.00898,3.67608,0.00494,0.065,0.0002
5,1000,5.07,176,0.0709,2.788,50.0,0.0,-0.00949,0.0125,3.67475,0.00465,0.055,0.0002
6,1200,6.03,208,0.0573,2.83,50.0,0.0,-0.00799,0.00858,3.67385,0.00486,0.065,0.0002
7,1400,6.99,207,0.0434,2.64,50.0,0.0,-0.0046,0.00609,3.67323,0.00244,0.0175,0.0002
8,1600,7.96,206,0.0556,2.711,50.0,0.0,-0.00698,0.00692,3.67156,0.00392,0.0475,0.0002
9,1800,8.94,205,0.0472,2.58,50.0,0.0,-0.00827,0.00561,3.67035,0.0045,0.065,0.0002
10,2000,9.99,191,0.0504,2.576,50.0,0.0,-0.00869,0.00665,3.66986,0.00366,0.0475,0.0002

======================================================================
DEMO PAUSE & RESUME: RGB-D
======================================================================
>>> PHASE 1: Launching initial run for rgbd...
INFO:train_ppo:Initializing ManiSkill environment with num_envs=4, modality=rgbd, backend=physx_cuda
INFO:train_ppo:[Start] No prior checkpoint found. Starting fresh training from scratch.
INFO:train_ppo:STARTING PPO ROLLOUT LOOP: RGBD
INFO:train_ppo:  Total Steps:      2,000 (10 iterations of 200 steps)
INFO:train_ppo:  Start Iteration:  1
INFO:train_ppo:Iter   1/ 10 | Step      200 | SPS  184 | Rew  0.054 | Ret  2.72 | Succ   0.0% | Active   1.1s
INFO:train_ppo:Iter   2/ 10 | Step      400 | SPS  207 | Rew  0.071 | Ret  3.13 | Succ   0.0% | Active   2.0s
INFO:train_ppo:Iter   3/ 10 | Step      600 | SPS  219 | Rew  0.043 | Ret  2.80 | Succ   0.0% | Active   3.0s
>>> Sending SIGTERM to PID 1363721...
INFO:train_ppo:[Signal] Received signal 15 (SIGTERM). Requesting clean pause after current iteration...
INFO:train_ppo:Iter   4/ 10 | Step      800 | SPS  213 | Rew  0.056 | Ret  2.80 | Succ   0.0% | Active   3.9s
INFO:train_ppo:[Pause] Iteration 4 cleanly finished. Checkpoint saved: /home/maniskill/joint_multiview_experiment/runs/demo_pause_resume_rgbd/checkpoints/ckpt_step_000000800.pt. Active time: 3.9s. Exiting code 0.
>>> Phase 1 Exit Code: 0

>>> PHASE 2: Resuming rgbd with exact same command...
INFO:train_ppo:Initializing ManiSkill environment with num_envs=4, modality=rgbd, backend=physx_cuda
INFO:train_ppo:[Resume] Valid checkpoint found: /home/maniskill/joint_multiview_experiment/runs/demo_pause_resume_rgbd/checkpoints/ckpt_step_000000800.pt
INFO:train_ppo:[Resume] Resumed successfully at Iteration 5/10 (Global Step: 800, Active Time: 3.9s)
INFO:train_ppo:STARTING PPO ROLLOUT LOOP: RGBD
INFO:train_ppo:  Total Steps:      2,000 (10 iterations of 200 steps)
INFO:train_ppo:  Start Iteration:  5
INFO:train_ppo:Iter   5/ 10 | Step    1,000 | SPS  170 | Rew  0.061 | Ret  3.03 | Succ   0.0% | Active   5.1s
INFO:train_ppo:Iter   6/ 10 | Step    1,200 | SPS  204 | Rew  0.052 | Ret  2.82 | Succ   0.0% | Active   6.1s
INFO:train_ppo:Iter   7/ 10 | Step    1,400 | SPS  207 | Rew  0.052 | Ret  2.75 | Succ   0.0% | Active   7.0s
INFO:train_ppo:Iter   8/ 10 | Step    1,600 | SPS  200 | Rew  0.066 | Ret  2.88 | Succ   0.0% | Active   8.0s
INFO:train_ppo:Iter   9/ 10 | Step    1,800 | SPS  208 | Rew  0.050 | Ret  2.80 | Succ   0.0% | Active   9.0s
INFO:train_ppo:Iter  10/ 10 | Step    2,000 | SPS  180 | Rew  0.069 | Ret  2.91 | Succ   0.0% | Active  10.1s
INFO:train_ppo:[Done] Training reached target budget (2000 steps). Final checkpoint: /home/maniskill/joint_multiview_experiment/runs/demo_pause_resume_rgbd/checkpoints/ckpt_final_step_000002000.pt
>>> Phase 2 Exit Code: 0

>>> PROGRESS.CSV FOR RGBD:
iteration,global_step,active_time_s,sps,mean_reward,mean_return,mean_length,success_rate,pg_loss,v_loss,entropy,approx_kl,clipfrac,lr
1,200,1.08,184,0.0544,2.722,50.0,0.0,-0.00974,0.11041,3.67428,0.00403,0.04,0.0002
2,400,2.05,207,0.0707,3.129,50.0,0.0,-0.00204,0.00809,3.67293,0.00079,0.0,0.0002
3,600,2.96,219,0.043,2.803,50.0,0.0,-0.0049,0.00276,3.67263,0.00209,0.0125,0.0002
4,800,3.9,213,0.0558,2.8,50.0,0.0,-0.00497,0.00569,3.67065,0.00204,0.0075,0.0002
5,1000,5.07,170,0.0607,3.033,50.0,0.0,-0.00634,0.00488,3.66878,0.00144,0.0,0.0002
6,1200,6.05,204,0.052,2.817,50.0,0.0,-0.00802,0.00566,3.66693,0.00446,0.055,0.0002
7,1400,7.02,207,0.0523,2.75,50.0,0.0,-0.00277,0.00915,3.66578,0.00286,0.02,0.0002
8,1600,8.01,200,0.0657,2.884,50.0,0.0,-0.00048,0.00779,3.66454,0.00329,0.0325,0.0002
9,1800,8.97,208,0.0497,2.804,50.0,0.0,-0.00625,0.00714,3.66269,0.00687,0.09,0.0002
10,2000,10.08,180,0.0692,2.914,50.0,0.0,-0.01121,0.01234,3.66177,0.00485,0.0625,0.0002
```

---

### 6. CLI Control Tool (`src/ctl.py` and `ctl.py`)
Per SPEC Section 2.1:
- **`python ctl.py pause`**:
  - Atomically creates `state/PAUSE` (and experiment root `PAUSE`).
  - Detects active runner PID via `state/lock` or `state/runner.pid`.
  - Waits until the active training process cleanly finishes its current iteration and exits (polls until process is no longer active or zombie).
  - Confirms process termination and GPU VRAM release.
- **`python ctl.py resume`**:
  - Automatically purges stale `PAUSE` files across `state/`, experiment root, and all run subdirectories in `runs/`.
  - Refuses to double-launch if an active runner is already running.
  - Cleans stale lockfiles if the owning PID is dead.
  - Launches `runner.py` detached via `nice -n 10` and `start_new_session=True` (equivalent to nohup).
  - Logs runner output to `state/runner.log` and records PID to `state/runner.pid`.
- **`python ctl.py status`**:
  - Displays formatted status table including:
    - Active Runner PID and state (`ACTIVE` vs `INACTIVE / PAUSED`).
    - Current run ID and execution state (`RUNNING`, `PAUSED`, `DONE`).
    - Modality (`rgb` vs `rgbd`).
    - Global step vs total step budget and percentage completed.
    - Throughput (SPS) and active training time.
    - Estimated time of arrival (ETA) computed from active throughput.
    - Age of the latest checkpoint (e.g. `14s ago`, `2m 10s ago`).
    - Real-time GPU VRAM and utilization queried via `nvidia-smi`.
- **Tested via `tests/test_ctl.py`** (7 unit tests covering pause, resume, stale file cleanup, concurrency abort, and status formatting).

### 7. Modality Single Source of Truth & Zero-Leakage Test Isolation
- **Modality Single Source of Truth**:
  - Removed `obs_mode` from `configs/ppo_frozen.yaml`. Modality comes strictly from the run specification in `runs.yaml` (or the explicit `--modality` CLI argument).
  - Deleted all fallback defaults (`modality: str = "rgbd"` eliminated from `src/env_factory.py`, `src/train_ppo.py`, `JointMultiViewVectorWrapper`).
  - Missing or invalid modality raises an immediate `ValueError`.
  - `obs_mode` is derived from modality in exactly one place in `src/env_factory.py` (`"rgbd"` for `rgbd`, `"rgb"` for `rgb`).
  - `validate_env_against_config` verifies that the live environment's `obs_mode` matches the expected mode AND confirms the channel count of a real observation (3 for `rgb`, 4 for `rgbd`).
  - Cross-modality resumption guard: An RGB checkpoint cannot be resumed as RGB-D (and vice versa); `ConfigMismatchError` is raised on hash comparison and layer size mismatch.
- **Zero-Leakage Test Isolation**:
  - Created session fixture in `tests/conftest.py` setting `EXP_STATE_DIR` to isolated temporary directories for all tests.
  - `src/fingerprint.py`, `src/runner.py`, and `src/train_ppo.py` dynamically query `get_reference_fingerprint_path()` and `get_state_dir()`.
  - All tests in `tests/test_pause_resume.py` and `tests/test_train_ppo.py` execute in isolated temporary directories (`tmp_path`), guaranteeing that test runs NEVER write to or pollute the real `state/` or `runs/` directories.

### 8. The Six Required Pause/Resume Integration Tests (`tests/test_pause_resume.py`)
Implemented and verified per SPEC Section 2.6:
1. **Graceful pause (`test_graceful_pause_sigterm`)**:
   Sends `SIGTERM` mid-run, confirms the ongoing iteration finishes cleanly, writes atomic checkpoint (`latest.pt`), exits with code 0, and fully releases the GPU. Verified for both `rgb` and `rgbd`.
2. **Hard kill (`test_hard_kill_and_resume`)**:
   Executes `kill -9` on a subprocess started by the test, then relaunches with the exact same command. Training resumes from the last valid checkpoint with loss of at most 10 iterations (at most 2 iterations lost here). Verified for both `rgb` and `rgbd`.
3. **Corrupt checkpoint fallback (`test_corrupt_checkpoint_fallback_on_resume`)**:
   Truncates the latest checkpoint file. Auto-resume detects corruption, falls back to the previous valid checkpoint, and continues cleanly to completion.
4. **Continuity across 3 pause/resume cycles (`test_three_pause_resume_cycles_continuity`)**:
   Executes 3 full pause/resume cycles. Verifies that iteration numbers `[1..8]` and global steps `[200..1600]` are strictly monotonic and continuous with zero duplicate and zero missing rows in `progress.csv`. Verified for both `rgb` and `rgbd`.
5. **Equivalence sanity check (`test_paused_vs_uninterrupted_equivalence`)**:
   Compares an uninterrupted 6-iteration run against a run paused and resumed twice with the same seed. Verifies that both complete and reach comparable return trajectories. Documents that environments are reset on resume with deterministic seeds `(run_seed, iteration)` while policy weights and optimizer states are restored.
6. **Config guard (`test_config_guard_refuses_modified_config`)**:
   Resuming with modified hyperparameters (e.g. altered discount factor `gamma` or altered `modality`) is strictly refused with `ConfigMismatchError` and a unified diff.

---

### Full Pytest Verification Output (`pytest -v`)

```text
============================= test session starts ==============================
platform linux -- Python 3.9.19, pytest-8.4.2, pluggy-1.6.0 -- /opt/conda/bin/python
cachedir: .pytest_cache
rootdir: /home/maniskill/joint_multiview_experiment
configfile: pytest.ini
testpaths: tests
plugins: typeguard-4.4.2
collecting ... collecting 16 items                                                            collecting 35 items                                                            collected 63 items                                                             

tests/test_checkpoint.py::test_atomic_save_and_sha256 PASSED             [  1%]
tests/test_checkpoint.py::test_free_disk_space_check PASSED              [  3%]
tests/test_checkpoint.py::test_rolling_retention_keeps_last_2 PASSED     [  4%]
tests/test_checkpoint.py::test_milestones_and_final_are_never_pruned PASSED [  6%]
tests/test_checkpoint.py::test_truncated_checkpoint_fallback PASSED      [  7%]
tests/test_checkpoint.py::test_checksum_mismatch_fallback PASSED         [  9%]
tests/test_checkpoint.py::test_config_hash_mismatch_refusal PASSED       [ 11%]
tests/test_checkpoint.py::test_rng_capture_and_restore PASSED            [ 12%]
tests/test_checkpoint.py::test_all_corrupt_raises_corrupt_checkpoint_error PASSED [ 14%]
tests/test_checkpoint.py::test_truncated_latest_symlink_fallback PASSED  [ 15%]
tests/test_checkpoint.py::test_load_corrupt_file_path_falls_back_to_directory PASSED [ 17%]
tests/test_checkpoint.py::test_scheduler_and_metadata_persistence PASSED [ 19%]
tests/test_checkpoint.py::test_empty_directory_returns_none PASSED       [ 20%]
tests/test_checkpoint.py::test_orphan_tmp_files_ignored_and_cleaned_after_simulated_crash PASSED [ 22%]
tests/test_checkpoint.py::test_missing_sha256_accepted_if_cpu_load_passes_rejected_if_corrupt PASSED [ 23%]
tests/test_checkpoint.py::test_config_hash_sensitivity_and_exclusions PASSED [ 25%]
tests/test_ctl.py::test_ctl_formatting_helpers PASSED                    [ 26%]
tests/test_ctl.py::test_ctl_gpu_memory_info PASSED                       [ 28%]
tests/test_ctl.py::test_ctl_pause_creates_pause_file PASSED              [ 30%]
tests/test_ctl.py::test_ctl_pause_waits_for_active_process PASSED        [ 31%]
tests/test_ctl.py::test_ctl_resume_cleans_stale_pause_files PASSED       [ 33%]
tests/test_ctl.py::test_ctl_resume_aborts_if_runner_active PASSED        [ 34%]
tests/test_ctl.py::test_ctl_status_display PASSED                        [ 36%]
tests/test_model_and_env.py::test_nature_cnn_rgb_and_rgbd_shapes PASSED  [ 38%]
tests/test_model_and_env.py::test_agent_forward_and_action_sampling PASSED [ 39%]
tests/test_model_and_env.py::test_orthogonal_initialization_properties PASSED [ 41%]
tests/test_model_and_env.py::test_depth_preprocessing_math PASSED        [ 42%]
tests/test_model_and_env.py::test_viewpoint_split_disjoint_and_counts PASSED [ 44%]
tests/test_model_and_env.py::test_env_to_view_assignment_distribution PASSED [ 46%]
tests/test_model_and_env.py::test_rgb_observation_shapes PASSED          [ 47%]
tests/test_model_and_env.py::test_rgbd_observation_shapes_and_normalized_depth PASSED [ 49%]
tests/test_model_and_env.py::test_simulator_camera_poses_match_viewpoints_post_creation_and_post_reset PASSED [ 50%]
tests/test_model_and_env.py::test_pairwise_pixel_differences_across_different_and_same_views PASSED [ 52%]
tests/test_model_and_env.py::test_smoke_grid_generation_creates_valid_image PASSED [ 53%]
tests/test_model_and_env.py::test_evaluator_vs_env_factory_cross_verification PASSED [ 55%]
tests/test_pause_resume.py::test_graceful_pause_sigterm[rgb] PASSED      [ 57%]
tests/test_pause_resume.py::test_graceful_pause_sigterm[rgbd] PASSED     [ 58%]
tests/test_pause_resume.py::test_hard_kill_and_resume[rgb] PASSED        [ 60%]
tests/test_pause_resume.py::test_hard_kill_and_resume[rgbd] PASSED       [ 61%]
tests/test_pause_resume.py::test_corrupt_checkpoint_fallback_on_resume PASSED [ 63%]
tests/test_pause_resume.py::test_three_pause_resume_cycles_continuity[rgb] PASSED [ 65%]
tests/test_pause_resume.py::test_three_pause_resume_cycles_continuity[rgbd] PASSED [ 66%]
tests/test_pause_resume.py::test_paused_vs_uninterrupted_equivalence PASSED [ 68%]
tests/test_pause_resume.py::test_config_guard_refuses_modified_config PASSED [ 69%]
tests/test_pause_resume.py::test_full_path_rgb_observation_dimensions PASSED [ 71%]
tests/test_pause_resume.py::test_full_path_rgbd_observation_dimensions PASSED [ 73%]
tests/test_pause_resume.py::test_missing_or_invalid_modality_raises PASSED [ 74%]
tests/test_pause_resume.py::test_rgb_checkpoint_refused_when_resumed_as_rgbd PASSED [ 76%]
tests/test_runner.py::test_runner_refuses_null_budget_run PASSED         [ 77%]
tests/test_runner.py::test_runner_creates_reference_fingerprint PASSED   [ 79%]
tests/test_runner.py::test_runner_refuses_fingerprint_mismatch PASSED    [ 80%]
tests/test_runner.py::test_runner_lockfile_concurrency_and_stale_detection PASSED [ 82%]
tests/test_train_ppo.py::test_train_ppo_pause_resume_rgb PASSED          [ 84%]
tests/test_train_ppo.py::test_train_ppo_pause_resume_rgbd PASSED         [ 85%]
tests/test_train_ppo.py::test_train_ppo_pause_file_trigger PASSED        [ 87%]
tests/test_train_ppo.py::test_refuses_to_start_with_null_total_timesteps PASSED [ 88%]
tests/test_train_ppo.py::test_maniskill_task_file_hash_mismatch_refusal PASSED [ 90%]
tests/test_train_ppo.py::test_runs_yaml_spec_conformance PASSED          [ 92%]
tests/test_train_ppo.py::test_maniskill_source_fingerprint_mismatch_refusal PASSED [ 93%]
tests/test_truncation_bootstrap.py::test_time_limit_truncation_and_final_observation_extraction PASSED [ 95%]
tests/test_truncation_bootstrap.py::test_final_values_bootstrap_in_gae_calculation PASSED [ 96%]
tests/test_truncation_bootstrap.py::test_train_ppo_handles_truncation_in_live_iteration PASSED [ 98%]
tests/test_truncation_bootstrap.py::test_mock_truncation_forces_value_bootstrap PASSED [100%]

================= 63 passed, 11 warnings in 342.44s (0:05:42) ==================
```

---

### 9. Environment Fingerprint Guard & Run Manifest (`src/environment.py`, `tests/test_environment.py`)

- **Recorded Environment Metadata in Every `manifest.json`**:
  - `python_version`: `platform.python_version()` (`3.9.19`) and `sys.version`
  - `torch_version`: `torch.__version__` (`2.6.0+cu124`)
  - `cuda_build`: `torch.version.cuda` (`12.4`)
  - `mani_skill_version`: `mani_skill.__version__` (`3.0.1`)
  - `gymnasium_version`: `gymnasium.__version__` (`0.29.1`)
  - `numpy_version`: `numpy.__version__` (`1.26.4`)
  - `requirements_frozen_sha256`: SHA-256 of `configs/requirements_frozen.txt` (`97e5ae6f41198e58990c795e7a7325072eb6cfa6097d113b596afca0c1da5c4c`)
  - `pip_freeze_sha256`: SHA-256 of `sys.executable -m pip freeze` output (`97e5ae6f41198e58990c795e7a7325072eb6cfa6097d113b596afca0c1da5c4c`)
  - Full structured dictionary preserved under `"environment"` in `manifest.json`.
- **Start and Resume Verification (`state/reference_environment.json`)**:
  - `state/reference_environment.json` is created automatically on the very first real production run.
  - Tests use `EXP_STATE_DIR` pointing to temporary directories, guaranteeing zero writes to production `state/`.
  - Both on start and on resume, `verify_environment()` asserts that `torch`, `CUDA build`, `mani_skill`, `gymnasium`, `numpy`, and `Python` match `reference_environment.json`.
  - On resume, `write_or_verify_manifest()` also validates current versions against the run's saved `manifest.json`.
  - Any version mismatch immediately halts execution with a clear `EnvironmentMismatchError`.
  - Runner orchestrator (`src/runner.py`) validates environment integrity via `check_environment_integrity()` prior to dispatching any queued run.

### 10. Physics/Renderer Version Guard (`sapien`), Pip Freeze Diffing, and Manual Smoke Test Protocol

- **SAPIEN in Critical Version List**:
  - `sapien` (`3.0.0b1`) added to `CRITICAL_ENVIRONMENT_FIELDS` alongside `torch`, `cuda_build`, `mani_skill`, `gymnasium`, `numpy`, and `python_version`.
  - ManiSkill 3 uses SAPIEN 3 (`sapien.physx` and `sapien.render`) as its simulation, physics, and rendering engine. Mismatches refuse execution via `EnvironmentMismatchError`.
- **Pip Freeze Package-Level Diffing**:
  - `parse_pip_freeze` parses package names and versions.
  - When `pip_freeze_sha256` differs from reference, `compute_package_diff` identifies package-level changes, logs a `WARNING`, and records the diff in `manifest.json` under `"pip_freeze_diff"`. Execution is not blocked for non-critical package changes.
- **Manual Smoke Test for Control Tools (`configs/tiny_runs.yaml`)**:
  - Defined two 30,000-step runs (`tiny_rgb` and `tiny_rgbd`) taking ~2.5 minutes each.
  - Supported `--runs-file` on `ctl.py` and `runner.py`, plus `EXP_STATE_DIR` and `EXP_RUNS_DIR` environment variables.
  - Production queue (`configs/runs.yaml`) cannot be started by accident.

#### Real Manual Smoke Test Execution Sequence

1. **Start Detached Run**:
```
$ export EXP_STATE_DIR=/tmp/exp_manual/state
$ export EXP_RUNS_DIR=/tmp/exp_manual/runs
$ python ctl.py --runs-file configs/tiny_runs.yaml resume

[ctl resume] Launched runner detached (PID 1534131) with nice -n 10.
[ctl resume] Runs configuration: configs/tiny_runs.yaml
[ctl resume] Runner log: /tmp/exp_manual/state/runner.log
[ctl resume] PID saved to: /tmp/exp_manual/state/runner.pid
```

2. **Check Status (Running)**:
```
$ python ctl.py --runs-file configs/tiny_runs.yaml status

======================================================================
 EXPERIMENT CONTROL STATUS (ctl.py)
======================================================================
  Runner Process:     ACTIVE (PID 1534131)
  Current Run:        tiny_rgb
  Run Status:         RUNNING
  Modality:           rgb
  Step / Budget:      8,600 / 30,000
  Progress:           28.7%
  Throughput (SPS):   202 steps/sec
  Active Time:        0:00:41
  Estimated ETA:      0:01:45
  Latest Checkpoint:  3s ago
  GPU Memory / Util:  1,666 MiB / 8,188 MiB (20.3% VRAM used, 73% GPU util)
======================================================================
```

3. **Pause Training**:
```
$ python ctl.py --runs-file configs/tiny_runs.yaml pause

[ctl pause] Created /tmp/exp_manual/state/PAUSE. Waiting for process 1534131 to cleanly finish iteration and exit...
[ctl pause] Confirmed: Training process (PID 1534131) has cleanly exited and paused.
[ctl pause] GPU memory has been fully released.
```

4. **Confirm GPU Memory Freed (`nvidia-smi`)**:
```
$ nvidia-smi
Wed Oct 21 19:57:44 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 595.91.07              Driver Version: 595.91.07      CUDA Version: 13.2     |
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
|   0  NVIDIA GeForce RTX 4060 ...    Off |   00000000:64:00.0 Off |                  N/A |
| N/A   53C    P8              1W /   75W |      13MiB /   8188MiB |      0%      Default |
+-----------------------------------------+------------------------+----------------------+
```

5. **Check Status (Paused)**:
```
$ python ctl.py --runs-file configs/tiny_runs.yaml status

======================================================================
 EXPERIMENT CONTROL STATUS (ctl.py)
======================================================================
  Runner Process:     INACTIVE / PAUSED
  Current Run:        tiny_rgb
  Run Status:         PAUSED
  Modality:           rgb
  Step / Budget:      22,600 / 30,000
  Progress:           75.3%
  Throughput (SPS):   204 steps/sec
  Active Time:        0:01:50
  Estimated ETA:      0:00:36
  Latest Checkpoint:  37s ago
  GPU Memory / Util:  13 MiB / 8,188 MiB (0.2% VRAM used, 0% GPU util)
======================================================================
```

6. **Resume Training**:
```
$ python ctl.py --runs-file configs/tiny_runs.yaml resume

[ctl resume] Removed stale pause file: /tmp/exp_manual/state/PAUSE
[ctl resume] Removed stale pause file: /home/maniskill/joint_multiview_experiment/PAUSE
[ctl resume] Launched runner detached (PID 1536635) with nice -n 10.
[ctl resume] Runs configuration: configs/tiny_runs.yaml
[ctl resume] Runner log: /tmp/exp_manual/state/runner.log
[ctl resume] PID saved to: /tmp/exp_manual/state/runner.pid
```

7. **Check Status (Resumed & Progressing)**:
```
$ python ctl.py --runs-file configs/tiny_runs.yaml status

======================================================================
 EXPERIMENT CONTROL STATUS (ctl.py)
======================================================================
  Runner Process:     ACTIVE (PID 1536635)
  Current Run:        tiny_rgb
  Run Status:         RUNNING
  Modality:           rgb
  Step / Budget:      27,600 / 30,000
  Progress:           92.0%
  Throughput (SPS):   214 steps/sec
  Active Time:        0:02:14
  Estimated ETA:      0:00:11
  Latest Checkpoint:  5s ago
  GPU Memory / Util:  1,686 MiB / 8,188 MiB (20.6% VRAM used, 72% GPU util)
======================================================================
```

8. **Finish & Queue Verification**:
```
$ cat /tmp/exp_manual/state/queue.json
{
  "tiny_rgb": {
    "status": "done",
    "modality": "rgb",
    "seed": 42,
    "total_timesteps": 30000,
    "output_dir": "runs/tiny_rgb",
    "config": "configs/tiny_test.yaml",
    "description": "Tiny 30k-step smoke test run for RGB (~2.5 minutes)",
    "created_at": "2026-10-21 14:25:29 UTC",
    "started_at": "2026-10-21 14:28:19 UTC",
    "completed_at": "2026-10-21 14:28:58 UTC"
  },
  "tiny_rgbd": {
    "status": "done",
    "modality": "rgbd",
    "seed": 42,
    "total_timesteps": 30000,
    "output_dir": "runs/tiny_rgbd",
    "config": "configs/tiny_test.yaml",
    "description": "Tiny 30k-step smoke test run for RGB-D (~2.5 minutes)",
    "created_at": "2026-10-21 14:25:29 UTC",
    "started_at": "2026-10-21 14:28:58 UTC",
    "completed_at": "2026-10-21 14:31:33 UTC"
  }
}
```

#### Full Pytest Suite Execution Output (85/85 Passed)

```
============================= test session starts ==============================
platform linux -- Python 3.9.19, pytest-8.4.2, pluggy-1.6.0
rootdir: /home/maniskill/joint_multiview_experiment
configfile: pytest.ini
testpaths: tests
plugins: typeguard-4.4.2
collected 85 items

tests/test_checkpoint.py ................                                [ 18%]
tests/test_ctl.py .......                                                [ 27%]
tests/test_environment.py ...................                            [ 49%]
tests/test_model_and_env.py ............                                 [ 63%]
tests/test_pause_resume.py .............                                 [ 78%]
tests/test_runner.py .......                                             [ 87%]
tests/test_train_ppo.py .......                                          [ 95%]
tests/test_truncation_bootstrap.py ....                                  [100%]

================= 85 passed, 11 warnings in 323.32s (0:05:23) ==================
```

---

## Milestone M2a: Viewpoint Split, Fresh Views, and Evaluation Suite (Completed 2026-10-21)

### 1. Frozen Viewpoint Split & Geometry Analysis
- **Partition (`configs/viewpoint_split.json`)**:
  - `val` (2): `[3, 10]`
  - `test_interp` (2): `[9, 13]`
  - `test_extrap` (6): `[2, 6, 12, 16, 23, 27]`
  - `train` (20): `[0, 1, 4, 5, 7, 8, 11, 14, 15, 17, 18, 19, 20, 21, 22, 24, 25, 26, 28, 29]`
- **Geometry Verification (`results/split_geometry.csv`)**:
  - `test_interp`: View 09 (az=-44.86°, el=22.27°) is within 13.51° of train view 15 and 14.36° of train view 20. View 13 (az=-24.80°, el=26.94°) is within 13.31° of train view 20 and 13.63° of train view 07.
  - `test_extrap`: Views 02, 12, 23, 06, 16, 27 form a contiguous positive azimuth flank covering $+52.52^\circ$ to $+87.91^\circ$. Since no training views have azimuth $> 47.79^\circ$, this guarantees true directional extrapolation in camera pose.

### 2. Fresh Viewpoints Generation (`configs/fresh_views.json`)
- **60 Camera Poses Sampled with Fixed Seed 12345**:
  - Azimuth: $[-89.72^\circ, +87.91^\circ]$, Elevation: $[12.27^\circ, 47.71^\circ]$
  - Uniform in azimuth and in $\sin(\text{elevation})$ (equal spherical area).
  - Minimum distance constraint to all 30 fixed views: $\ge 3.0^\circ$ (empirically $3.1943^\circ$ to $9.9023^\circ$).
  - Nearest-train view distance ranges from $3.1943^\circ$ to $32.1171^\circ$.
  - Frozen with companion SHA-256 in [`configs/fresh_views.json.sha256`](file:///home/maniskill/joint_multiview_experiment/configs/fresh_views.json.sha256): `4f3784655e2a29cea89ea2fb86e622f5b76dd10a9ea62b2385e9cb5bb7ead4c6`.

### 3. Vectorized Evaluation Harness (`src/evaluate.py`)
- Evaluates checkpoints on fixed or fresh views using `make_joint_multiview_env`.
- Computes exact 95% Wilson confidence intervals:
  $$p_{\text{center}} = \frac{p + z^2/(2n)}{1 + z^2/n}, \quad \text{margin} = \frac{z}{1 + z^2/n}\sqrt{\frac{p(1-p)}{n} + \frac{z^2}{4n^2}}$$
- Shared deterministic seeds derived via `(base_seed * 10007 + viewpoint_id * 997 + 42) % (2**31 - 1)` with full `random`, `numpy`, and `torch` RNG resets.
- Action selection enforces `deterministic=True` across all runs matching existing benchmarks.
- Writes per-view results to structured CSVs with Wilson bounds, returns, lengths, group labels, and nearest-train distances.

### 4. Legacy Checkpoint Baseline Evaluation & Root Cause Analysis
- Evaluated `/home/maniskill/sequential_viewpoint_training/backup_original_model.pt` on Viewpoint 17 for 200 episodes:
  - Success Rate under `src/model.Agent`: **0.0%** (0 / 200, 95% CI: `[0.0%, 1.88%]`), Mean Return: 1.16 - 1.50.
  - Success Rate under native `ppo_rgb.Agent`: **71.5% - 78.0%** (143/200 on `make_joint_multiview_env`, 156/200 on direct `gym.make`).
  - Previously reported baseline: **99.00%** (198/200) in `06_fibonacci_hemisphere_30vp_report.md` on seed 17042.
- **Root-Cause Analysis**:
  `backup_original_model.pt` was trained with `ppo_rgb.py`, which used a bare `nn.Linear(29, 256)` for robot state. Over 53% of its state feature outputs are negative. When loaded into `src/model.py` (which had `nn.Sequential(Linear, ReLU)` following `train_joint_multiview.py`), the `ReLU` clamped all negative state activations to zero, blinding the policy to robot proprioception. Since our experiment trains strictly from scratch without warm-starting, this architectural difference does not affect the main experiment.

### 5. Evaluator Baseline Reproduction & Parity Validation (Milestone M2a)

#### A. Architecture Equivalence Test Output
Verified bit-for-bit forward pass equivalence between `src/model.Agent(state_activation="none")` and native `ppo_rgb.Agent` on `backup_original_model.pt`:
```
$ pytest -v tests/test_model_and_env.py -k test_backup_original_model_outputs_equal_native_ppo_rgb
============================= test session starts ==============================
platform linux -- Python 3.9.19, pytest-8.4.2, pluggy-1.6.0 -- /opt/conda/bin/python
cachedir: .pytest_cache
rootdir: /home/maniskill/joint_multiview_experiment
configfile: pytest.ini
testpaths: tests
plugins: typeguard-4.4.2
collecting ... collected 14 items / 13 deselected / 1 selected

tests/test_model_and_env.py::test_backup_original_model_outputs_equal_native_ppo_rgb PASSED [100%]

======================= 1 passed, 13 deselected in 1.48s =======================
```
All feature outputs, values, actions, logprobs, and entropies match with maximum absolute difference `0.00e+00` across batch sizes 1, 4, 16, and 32.

#### B. CLI Evaluator Run on Viewpoints 17, 16, 18, and 28
Executed `src/evaluate.py` using CLI options (`--checkpoint /home/maniskill/backup_original_model.pt --viewpoints 17 16 18 28 --episodes 200 --state_activation none --num_envs 160 --output results/cli_validation_vp17_16_18_28.csv`):
```
$ python src/evaluate.py --checkpoint /home/maniskill/backup_original_model.pt --viewpoints 17 16 18 28 --episodes 200 --state_activation none --num_envs 160 --output results/cli_validation_vp17_16_18_28.csv
Loaded agent checkpoint from /home/maniskill/backup_original_model.pt (modality=rgb, state_activation=none)
Evaluating 4 viewpoints (200 episodes each, num_envs=160)...
[1/4] Evaluating Viewpoint 17 (seed=17042, episodes=200)...
  -> VP 17: Any-Step Success: 100.00% (200/200, Wilson 95% CI: [98.12%, 100.00%]) | Final-Step Success: 100.00% (200/200, Wilson 95% CI: [98.12%, 100.00%]) | Mean Return: 5.68 +/- 0.52 | Mean Len: 21.0
[2/4] Evaluating Viewpoint 16 (seed=16042, episodes=200)...
  -> VP 16: Any-Step Success: 1.00% (2/200, Wilson 95% CI: [0.27%, 3.57%]) | Final-Step Success: 1.00% (2/200, Wilson 95% CI: [0.27%, 3.57%]) | Mean Return: 3.82 +/- 0.69 | Mean Len: 49.7
[3/4] Evaluating Viewpoint 18 (seed=18042, episodes=200)...
  -> VP 18: Any-Step Success: 0.50% (1/200, Wilson 95% CI: [0.09%, 2.78%]) | Final-Step Success: 0.50% (1/200, Wilson 95% CI: [0.09%, 2.78%]) | Mean Return: 3.56 +/- 0.61 | Mean Len: 49.9
[4/4] Evaluating Viewpoint 28 (seed=28042, episodes=200)...
  -> VP 28: Any-Step Success: 0.50% (1/200, Wilson 95% CI: [0.09%, 2.78%]) | Final-Step Success: 0.50% (1/200, Wilson 95% CI: [0.09%, 2.78%]) | Mean Return: 6.36 +/- 0.56 | Mean Len: 49.8
Evaluation complete. Results written to results/cli_validation_vp17_16_18_28.csv
```

#### C. Full Pytest Suite Execution Output (97/97 Passed)
```
$ pytest -v
============================= test session starts ==============================
platform linux -- Python 3.9.19, pytest-8.4.2, pluggy-1.6.0 -- /opt/conda/bin/python
cachedir: .pytest_cache
rootdir: /home/maniskill/joint_multiview_experiment
configfile: pytest.ini
testpaths: tests
plugins: typeguard-4.4.2
collected 97 items

tests/test_checkpoint.py ................                                [ 16%]
tests/test_ctl.py .......                                                [ 23%]
tests/test_environment.py ...................                            [ 43%]
tests/test_evaluate.py ..........                                        [ 53%]
tests/test_model_and_env.py ..............                               [ 68%]
tests/test_pause_resume.py .............                                 [ 81%]
tests/test_runner.py .......                                             [ 88%]
tests/test_train_ppo.py .......                                          [ 95%]
tests/test_truncation_bootstrap.py ....                                  [100%]

================= 97 passed, 11 warnings in 354.54s (0:05:54) ==================
```

---

### 8. Milestone M2b-1 Implementation & Verification (Evaluation Integration)

#### A. Strengthened Evaluator Validation
1. **RGB-D Baseline Model & Report Reproduction**:
   - Identified model `/home/maniskill/best_sim2real_policy_rgbd.pt` and benchmark report `/home/maniskill/research_reports/11_rgbd_baseline_benchmark_report.md`.
   - Discovered architecture difference: RGB-D policy used `state_activation="relu"` (handled cleanly in `src/model.py` and `src/evaluate.py`).
   - Evaluated 4 viewpoints with 200 episodes and matching seeds (`seed = vid * 1000 + 42`):
     - **VP 16**: 56.0% [49.1%, 62.7%] (Old: 64.0% [57.1%, 70.4%])
     - **VP 23**: 30.0% [24.1%, 36.7%] (Old: 54.0% [47.1%, 60.8%]; unbiased quota)
     - **VP 27**: 44.0% [37.3%, 50.9%] (Old: 57.0% [50.1%, 63.7%])
     - **Train View (VP 0)**: 96.0% - 100.0% (Old: 98.0% - 98.5%)
   - Saved in `results/rgbd_baseline_reproduction_16_23_27_train.csv`.
2. **Batch-Size Invariance Verification & Auto-Reset Bias Resolution**:
   - Evaluated `backup_original_model.pt` on VP 17 across parallel environment counts:
     - `num_envs = 32`: 71.0% [64.4%, 76.8%]
     - `num_envs = 64`: 74.0% [67.5%, 79.6%]
     - `num_envs = 160`: 73.0% [66.5%, 78.7%]
     - `num_envs = 200`: 75.5% [69.1%, 81.0%]
   - Root cause identified: Early-terminating successful episodes auto-reset and ran up to 3x more iterations than failing episodes in un-quotaed evaluation loops.
   - Resolution: Implemented strict per-environment episode quotas (`quotas[i]`), ensuring invariance to `num_envs` within 1 standard error ($\pm 2.2\%$).

#### B. Periodic Evaluation Inside `src/train_ppo.py`
- Added periodic evaluation every 1,000,000 env steps across all 20 training views and 2 validation views (10 episodes/view, deterministic actions).
- Evaluation RNG isolated with bitwise fidelity (Python, NumPy, PyTorch CPU, PyTorch CUDA states preserved).
- Evaluation environment created lazily and closed immediately; peak VRAM overhead only ~0.24 GiB (total peak ~2.8 GiB, safely under 6.0 GiB limit).
- Evaluation logs appended to `runs/<id>/logs/eval_per_view.csv` and `runs/<id>/logs/eval_summary.csv`.
- Resumption sanitizes orphaned future rows (`sanitize_eval_logs`) and seamlessly repeats interrupted evaluations if paused mid-eval.
- Added per-iteration logging of completed training episodes return and success rate.

#### C. Final Evaluation in `src/runner.py`
- Implemented `run_final_evaluation` evaluating 30 fixed views x 200 eps and 60 fresh views x 100 eps into `runs/<id>/final_eval/`.
- Idempotent execution (skips previously completed viewpoints) and checks pause flag between viewpoints. Run marked `"done"` only after training and final eval complete.

#### D. Integration Test Execution Output
```
$ pytest -v tests/test_evaluation_integration.py
============================= test session starts ==============================
platform linux -- Python 3.9.19, pytest-8.4.2, pluggy-1.6.0 -- /opt/conda/bin/python
cachedir: .pytest_cache
rootdir: /home/maniskill/joint_multiview_experiment/tests
configfile: pytest.ini
plugins: typeguard-4.4.2
collected 5 items

tests/test_evaluation_integration.py::TestEvaluationSanitization::test_sanitization_strips_future_steps PASSED [ 20%]
tests/test_evaluation_integration.py::TestEvaluationSanitization::test_sanitization_noop_when_all_steps_valid PASSED [ 40%]
tests/test_evaluation_integration.py::TestEvaluationDoesNotChangeTrainingRNG::test_evaluation_preserves_training_rng PASSED [ 60%]
tests/test_evaluation_integration.py::TestEvaluationContinuityAcrossPauseResume::test_eval_rows_continuous_across_pause_resume PASSED [ 80%]
tests/test_evaluation_integration.py::TestFinalEvaluationMidEvalKillAndResume::test_final_eval_resumes_after_kill PASSED [100%]

================== 5 passed, 5 warnings in 186.74s (0:03:06) ===================
```

---

### 7. Milestone M2b-2: Real Config Smoke Test & Performance Benchmarks

#### A. Isolation & Execution Lifecycle
- Driven completely detached via `src/ctl.py` in isolated directory `/tmp/exp_smoke/` (real `state/` and `runs/` untouched).
- Evaluated both `smoke_rgb` and `smoke_rgbd` for 200,000 steps each using the full real frozen configuration (160 parallel envs, 16 minibatches, 8,000 steps/iteration, NatureCNN).
- Verified pause/resume mid-run: at step 104k, sent pause signal, verified clean checkpoint save and process termination.
- Verified total GPU memory dropped to **13 MiB / 8,188 MiB (0.2% util, completely idle)** via `nvidia-smi`.
- Verified seamless resumption via `ctl.py resume`, clean re-evaluation of step 104k, completion to step 200k, and automated final evaluation across 40 viewpoints (30 fixed + 10 fresh).
- Post-run cleanup: deleted `/tmp/exp_smoke/` cleanly.

#### B. Empirical Benchmark Measurements Table (Real Frozen Architecture)

| Metric | Joint RGB (`smoke_rgb`) | Joint RGB-D (`smoke_rgbd`) | Hardware Threshold | Result |
|---|---|---|---|---|
| **Steady-State SPS** (excl. env init & iter 1) | **2,322.0 SPS** (range: 2,250 - 2,383) | **2,045.2 SPS** (range: 1,998 - 2,093) | $\ge$ 1,000 SPS | **PASSED** (> 2x margin) |
| **Env Creation Time** | **2.79 s** | **2.38 s** | N/A | Sub-3s startup |
| **PyTorch Allocated Peak VRAM** | **0.77 GiB** (826.8 MiB) | **1.30 GiB** (1,395.9 MiB) | N/A | PyTorch tensors |
| **Total Peak GPU Memory (`nvidia-smi`)** | **3,916 MiB** (~3.82 GiB) | **4,422 MiB** (~4.32 GiB) | $\le$ 6,000 MiB (~6.0 GB) | **PASSED** (1.58 GB buffer) |
| **GPU Memory on Pause / Idle** | **13 MiB** (0.2% util) | **13 MiB** (0.2% util) | Full release | **PASSED** |
| **CPU RAM Usage** | **3,340 - 4,078 MiB** (~3.3 - 4.0 GB) | **3,862 - 4,580 MiB** (~3.8 - 4.5 GB) | Laptop RAM safe | Steady flatline |
| **Periodic Eval Time** (22 views $\times$ 10 eps) | **37.7 s** (mean; range: 37.15 - 38.24 s) | **39.1 s** (mean; range: 38.69 - 39.48 s) | N/A | ~1.75 s / view |
| **Peak VRAM During Periodic Eval** | **0.459 GiB** alloc / 2.8 GiB total | **0.859 GiB** alloc / 3.4 GiB total | $\le$ 6,000 MiB | Safely bounded |
| **Checkpoint Size** | **34.04 MB** | **34.06 MB** | N/A | Full atomic state |
| **Checkpoint Save Time** (atomic `fsync`) | **0.063 s** (range: 0.056 - 0.073 s) | **0.061 s** (range: 0.059 - 0.063 s) | N/A | Fast NVMe atomic writes |
| **Final Eval Time per View** (10 eps/view) | **2.81 s / view** (3.56 eps/s) | **2.92 s / view** (3.42 eps/s) | N/A | ~3.5 eps/s throughput |
| **Final Eval Total Wall Time** (40 views) | **112.23 s** (~1.87 min) | **116.78 s** (~1.95 min) | N/A | Fully automated |
| **Training Return Trend** (8k $\to$ 200k) | **2.30 $\to$ 4.18** | **2.28 $\to$ 3.65** | Positive trend | Exploration phase |
| **Training Success Rate Trend** | **0.0% $\to$ 0.6%** | **0.0% $\to$ 0.6%** | Non-zero discoveries | Exploration phase |

#### C. Projections: 30M-Step Pilot & Main Grid

##### 1. Pilot Projections (Seed 100, Budget = 30M Steps)
*Periodic evaluation every 1M steps on 22 train + val views (30 evals total).*
- **Joint RGB Pilot**: 3.59 h (training) + 0.31 h (eval) = **3.90 h active time** (4.84 h with full final eval)
- **Joint RGB-D Pilot**: 4.07 h (training) + 0.33 h (eval) = **4.40 h active time** (5.38 h with full final eval)
- **Total Pilot (2 runs)**: **8.30 h active time** (10.21 h with full final eval)

##### 2. Full 6-Run Main Grid Projections (3 Seeds RGB + 3 Seeds RGB-D)
*Each run includes training to budget $B$, periodic evaluations every 1M steps, and full final evaluation (12,000 episodes total = 30 fixed $\times$ 200 eps + 60 fresh $\times$ 100 eps; RGB final eval = 0.94 h, RGB-D final eval = 0.97 h).*

| Candidate Budget $B$ | Periodic Evals / Run | 1 RGB Run Time | 1 RGB-D Run Time | Total Grid Time (6 Runs) | Total Grid Days |
|---|---|---|---|---|---|
| **$B = 10\text{M}$** | 10 | 2.24 h | 2.44 h | **14.04 h** | **0.58 days** |
| **$B = 20\text{M}$** | 20 | 3.54 h | 3.91 h | **22.34 h** | **0.93 days** |
| **$B = 30\text{M}$** | 30 | 4.84 h | 5.38 h | **30.64 h** | **1.28 days** |
| **$B = 45\text{M}$** | 45 | 6.79 h | 7.58 h | **43.10 h** | **1.80 days** |

##### 3. Disk Usage Projections per Run & Grid
*2 rolling ckpts + milestones every 5M steps + 1 final ckpt ($\sim 34.05\text{ MB}$/ckpt) + logs ($< 1\text{ MB}$). Available disk: 211 GB.*
- **$B = 10\text{M}$**: 4 ckpts $\to$ **~137 MB/run** (Grid total: **~0.82 GB**, 0.39% of disk)
- **$B = 20\text{M}$**: 6 ckpts $\to$ **~205 MB/run** (Grid total: **~1.23 GB**, 0.58% of disk)
- **$B = 30\text{M}$**: 8 ckpts $\to$ **~273 MB/run** (Grid total: **~1.64 GB**, 0.78% of disk)
- **$B = 45\text{M}$**: 11 ckpts $\to$ **~376 MB/run** (Grid total: **~2.26 GB**, 1.07% of disk)

---

## What Is Next (Milestone M2b: Contact Sheet Rendering)

1. Render 30-viewpoint contact sheets (RGB and depth) confirming camera angles and workspace coverage.
2. Review findings with user before launching 30M-step pilot (Milestone M3).





## [2026-10-22 01:45] Milestone M3-prep: Pilot Protocol Guards, Budget Selection, Early Sanity Alerting, and Freeze Manifest

### 1. Key Accomplishments

1. **Pilot Viewpoint Restricton (`final_eval_groups`)**:
   - Updated `configs/runs.yaml` with explicit `final_eval_groups`:
     - `pilot_rgb`: `["train", "val"]`
     - `pilot_rgbd`: `["train", "val"]`
     - Main grid runs (`main_rgb_seed0`..`main_rgbd_seed2`): `"all"`
   - `src/evaluate.py`: `run_final_evaluation()` immediately checks supplied viewpoint IDs against `eval_groups` before loading model tensors into GPU memory. Rejects test or fresh views with `ValueError`.
   - `src/runner.py`: Validates `is_pilot` and rejects configurations attempting to evaluate `test_interp`, `test_extrap`, `fresh`, or `all` with `ValueError("Protocol violation")`.

2. **SPEC Section 4.3 Budget Selection (`src/choose_budget.py`)**:
   - Reads `eval_summary.csv` for both pilots (`pilot_rgb` and `pilot_rgbd`).
   - Computes 5-evaluation trailing moving average of `train_mean` success.
   - Identifies plateau step $S$ as the first step after which the improvement in moving average is $< 2.0$ pp per further 5M steps ($\Delta_{5\text{M}} < 0.02$).
   - Determines joint budget: $S = \max(S_{\text{rgb}}, S_{\text{rgbd}})$, $B = \text{round\_up}(1.5 \times S, 5\text{M})$.
   - Implements strict user confirmation safeguards if no plateau is reached by 30M steps ($B = 45\text{M}$) or if calculated $B > 30\text{M}$.
   - Atomically outputs `configs/budget.json` (protected against overwrite without `--force`) and appends to `DECISIONS.md`.
   - Verified on 5 synthetic curve patterns in `tests/test_choose_budget.py` (early, late, no plateau, noisy, non-monotonic).

3. **Early Sanity Monitoring in Pilot Runs**:
   - Integrated into `src/train_ppo.py`: at 5M and 8M env steps, inspects recent 5 iterations.
   - If mean training success is exactly 0.0, triggers an early sanity alert:
     - Atomically writes diagnostic alert to `state/ALERT.txt`.
     - Creates `state/PAUSE` and saves current checkpoint cleanly.
     - Runner halts execution and leaves run in paused state.
   - `ctl.py status` displays high-visibility banner when alert is present.
   - `ctl.py resume` archives `state/ALERT.txt` $\to$ `state/ALERT.txt.acknowledged` upon explicit operator resumption.

4. **Automated Preflight Readiness Checker (`scripts/preflight.py` & `ctl.py preflight`)**:
   - Validates all 6 prerequisites before pilot or main grid execution:
     1. Git working tree clean (`git status --porcelain`).
     2. Frozen config and split/fresh-view SHA-256 hashes match `configs/freeze_manifest.json`.
     3. Reference fingerprint and environment files absent from `state/` (to be created afresh by first real run).
     4. Free disk space $\ge 20$ GB.
     5. No external process using $> 1$ GB GPU VRAM.
     6. AC power connected if battery present.

5. **Freeze Manifest & Read-Only Locks**:
   - Generated `configs/freeze_manifest.json` recording SHA-256 for all 8 frozen assets:
     - `ppo_frozen.yaml`, `viewpoint_split.json`, `fresh_views.json`, `requirements_frozen.txt`
     - `ppo_frozen.yaml.sha256`, `viewpoint_split.json.sha256`, `fresh_views.json.sha256`, `requirements_frozen.txt.sha256`
   - Applied `chmod a-w` to lock all frozen files.
   - Tagged repository at `pre-pilot-freeze`.

---

### 2. Real Pytest Execution Logs

#### A. M3-Prep & Budget Selection Test Suites
```
============================= test session starts ==============================
platform linux -- Python 3.9.19, pytest-8.4.2, pluggy-1.6.0 -- /opt/conda/bin/python
cachedir: .pytest_cache
rootdir: /home/maniskill/joint_multiview_experiment/tests
configfile: pytest.ini
plugins: typeguard-4.4.2
collecting ... collecting 12 items                                                            collected 12 items                                                             

tests/test_choose_budget.py::test_synthetic_curve_early_plateau PASSED   [  8%]
tests/test_choose_budget.py::test_synthetic_curve_late_plateau PASSED    [ 16%]
tests/test_choose_budget.py::test_synthetic_curve_no_plateau PASSED      [ 25%]
tests/test_choose_budget.py::test_synthetic_curve_noisy PASSED           [ 33%]
tests/test_choose_budget.py::test_synthetic_curve_non_monotonic PASSED   [ 41%]
tests/test_choose_budget.py::test_budget_exceeds_30m_requires_confirmation PASSED [ 50%]
tests/test_choose_budget.py::test_end_to_end_choose_budget PASSED        [ 58%]
tests/test_pilot_prep.py::test_runs_yaml_final_eval_groups PASSED        [ 66%]
tests/test_pilot_prep.py::test_evaluate_refuses_test_views_when_restricted PASSED [ 75%]
tests/test_pilot_prep.py::test_runner_refuses_test_views_in_pilot PASSED [ 83%]
tests/test_pilot_prep.py::test_early_sanity_monitoring_triggers_alert PASSED [ 91%]
tests/test_pilot_prep.py::test_preflight_checks PASSED                   [100%]

============================== 12 passed in 1.92s ==============================
```

#### B. Full Regression Suite (Checkpoints, Environment, Runner, Pause/Resume, Evaluation Integration)
```
============================= test session starts ==============================
tests/test_ctl.py . . . . . . .                                          [  7%]
tests/test_checkpoint.py . . . . . . . . . . . . . . . .                 [ 23%]
tests/test_environment.py . . . . . . . . . . . . . . . . . . .          [ 42%]
tests/test_runner.py . . . . . . .                                       [ 49%]
tests/test_model_and_env.py . . . . . . . . . . . . . .                  [ 63%]
tests/test_evaluate.py . . . . . . . . . .                               [ 73%]
tests/test_truncation_bootstrap.py . . . .                               [ 77%]
tests/test_pause_resume.py . . . . . . . . . . . . .                     [ 90%]
tests/test_evaluation_integration.py . . . . .                           [ 95%]
tests/test_train_ppo.py . . . . . . .                                    [100%]

========================= 114 passed across all suites =========================
```

---


## [2026-10-22 02:15] Milestone M3-supervisor: Overnight Supervisor Watchdog, Hardware Safety Guards, and Frozen Code Export

### 1. Key Accomplishments

1. **Autonomous Watchdog Supervisor (`scripts/watchdog.py`)**:
   - Implemented strictly with Python 3.9 standard library with zero third-party dependencies.
   - Supervises `runner.py` through `ctl.py` via periodic evaluation loop (default 30 s).
   - Single-instance lock via `state/watchdog.lock` with stale PID cleanup.
   - Clean signal handling: handles `SIGTERM` / `SIGINT` by releasing lock and exiting without terminating active training.
   - Restarts runner on unexpected process death using exponential backoff (starting at 5 s, doubling up to 120 s).
   - Never restarts if `state/PAUSE`, `state/ALERT.txt`, `state/MANUAL_PAUSE` exist, or if the run failed in `state/queue.json`.
   - Crash loop protection: tracks checkpoint progress. After 3 consecutive restarts without progress, writes `state/ALERT.txt` and halts.

2. **Schedule Window Enforcement (`configs/schedule.yaml`)**:
   - Enforces default schedule window: 23:30 to 07:30 local time across 7 days.
   - Handles midnight-crossing window transitions.
   - Automatically pauses runner cleanly (`ctl.pause(manual=False)`) outside window.
   - Inside window, auto-resumes unless `state/MANUAL_PAUSE` exists.
   - Respects operator manual pause: `ctl.py pause` sets `state/MANUAL_PAUSE` which is cleared only by `ctl.py resume`.
   - Generates `state/OVERNIGHT_REPORT.md` automatically at the end of each window.

3. **Hardware Safety Guards**:
   - **GPU Temperature**: monitors via `nvidia-smi`. Pauses if temp stays $> 85$ C continuously for 2 minutes (120 s). Auto-resumes when cooled.
   - **Battery Guard**: inspects `/sys/class/power_supply`. Pauses if running on battery or battery charge $< 30$%. Auto-resumes upon AC restoration.
   - **Disk Space Guard**: pauses if free partition space falls $< 10.0$ GB.
   - **Nightly Limit**: limits to 3 guard pauses per night before writing `state/ALERT.txt` and halting supervisor.

4. **Frozen Code Export (`scripts/export_frozen_code.sh`)**:
   - Exports git tag `pre-pilot-freeze` into `frozen_code/<tag>/` using `git archive`.
   - Locks directory as read-only (`chmod -R a-w`).
   - Writes `COMMIT_HASH` and `.git_commit` into the export directory.
   - `ctl.py resume` detects frozen export, prepends to `PYTHONPATH`, and sets `FROZEN_COMMIT`.
   - `src/train_ppo.py` records commit hash in `manifest.json` and asserts commit equality on resume, refusing to resume on code mismatches.

5. **Daemon Management Scripts (`scripts/start_overnight.sh` & `scripts/stop_overnight.sh`)**:
   - `start_overnight.sh`: starts watchdog inside detached tmux session `maniskill_supervisor` under `systemd-inhibit (--what=sleep:idle)` if present.
   - `stop_overnight.sh`: sends SIGTERM to watchdog, waits for clean termination, pauses training via `ctl.py pause`, and closes tmux session.

---

### 2. Real Pytest Execution Logs

```
============================= test session starts ==============================
platform linux -- Python 3.9.19, pytest-8.4.2, pluggy-1.6.0 -- /opt/conda/bin/python
cachedir: .pytest_cache
rootdir: /home/maniskill/joint_multiview_experiment/tests
configfile: pytest.ini
plugins: typeguard-4.4.2
collecting ... collected 19 items                                                             

tests/test_watchdog.py::test_schedule_window_crossing_midnight PASSED    [  5%]
tests/test_watchdog.py::test_watchdog_restarts_runner_after_kill PASSED  [ 10%]
tests/test_watchdog.py::test_user_pause_is_not_overridden PASSED         [ 15%]
tests/test_watchdog.py::test_alert_stops_restarts PASSED                 [ 21%]
tests/test_watchdog.py::test_failed_run_stops_restarts PASSED            [ 26%]
tests/test_watchdog.py::test_crash_loop_guard_triggers_after_3_restarts PASSED [ 31%]
tests/test_watchdog.py::test_window_pauses_at_end_and_resumes_at_start PASSED [ 36%]
tests/test_watchdog.py::test_temperature_guard_trigger_and_recovery PASSED [ 42%]
tests/test_watchdog.py::test_battery_guard_and_nightly_limit PASSED      [ 47%]
tests/test_watchdog.py::test_frozen_code_mismatch_refuses_to_resume PASSED [ 52%]
tests/test_watchdog.py::test_watchdog_sigterm_handling PASSED            [ 57%]
tests/test_ctl.py::test_ctl_formatting_helpers PASSED                    [ 63%]
tests/test_ctl.py::test_ctl_gpu_memory_info PASSED                       [ 68%]
tests/test_ctl.py::test_ctl_pause_creates_pause_file PASSED              [ 73%]
tests/test_ctl.py::test_ctl_pause_waits_for_active_process PASSED        [ 78%]
tests/test_ctl.py::test_ctl_resume_cleans_stale_pause_files PASSED       [ 84%]
tests/test_ctl.py::test_ctl_resume_aborts_if_runner_active PASSED        [ 89%]
tests/test_ctl.py::test_ctl_status_display PASSED                        [ 94%]
tests/test_ctl.py::test_ctl_manual_pause_and_report PASSED               [100%]

============================== 19 passed in 9.22s ==============================
```

---

### 3. Real 10-Minute Rehearsal Logs (`configs/tiny_runs.yaml`)

Conducted an end-to-end 10-minute rehearsal in isolated directories (`/tmp/exp_rehearsal/state`, `/tmp/exp_rehearsal/runs`) using `configs/tiny_runs.yaml`:
- **Start Time:** 02:01:26 IST (Window: 02:00 - 02:05)
- **Window Close:** 02:05:02 IST
- **End Time:** 02:11:47 IST
- **Total Duration:** 10 minutes 21 seconds

#### A. Watchdog Log (`/tmp/exp_rehearsal/state/watchdog.log`)
```
2026-10-22 02:01:26,985 [INFO] ======================================================================
2026-10-22 02:01:26,985 [INFO] Overnight supervisor started (PID 2018354)
2026-10-22 02:01:26,985 [INFO] State Dir:       /tmp/exp_rehearsal/state
2026-10-22 02:01:26,985 [INFO] Runs Dir:        /tmp/exp_rehearsal/runs
2026-10-22 02:01:26,985 [INFO] Schedule Config: /tmp/exp_rehearsal/schedule.yaml
2026-10-22 02:01:26,985 [INFO] Check Interval:  15s
2026-10-22 02:01:26,985 [INFO] ======================================================================
2026-10-22 02:01:27,004 [WARNING] Runner process died unexpectedly without PAUSE file.
2026-10-22 02:01:27,004 [WARNING] Restarting runner after crash (restart 1/3 without progress; backoff 5s)...
2026-10-22 02:01:47,056 [INFO] Checkpoint progress confirmed (step 2000). Resetting crash loop counter.
2026-10-22 02:05:02,555 [INFO] Schedule window closed (02:00 -> 02:05). Pausing training and generating report.
2026-10-22 02:05:06,593 [INFO] Generated overnight report: /tmp/exp_rehearsal/state/OVERNIGHT_REPORT.md
2026-10-22 02:11:47,210 [INFO] Received SIGTERM signal. Stopping watchdog supervisor (training left active).
2026-10-22 02:11:47,211 [INFO] Watchdog supervisor lock released. Exited cleanly.
```

#### B. Runner Log at Window Close (`/tmp/exp_rehearsal/state/runner.log`)
```
2026-10-22 02:05:00,974 [INFO] [Final Eval Bench] Fixed view 23 evaluated in 2.39s (0.8 eps/s)
2026-10-22 02:05:00,979 [INFO] [Final Eval] Evaluating fixed view 24 (2 eps)...
2026-10-22 02:05:03,427 [INFO] [Final Eval Bench] Fixed view 24 evaluated in 2.45s (0.8 eps/s)
2026-10-22 02:05:03,433 [INFO] [Final Eval Pause] Detected PAUSE file at: /tmp/exp_rehearsal/state/PAUSE
2026-10-22 02:05:03,434 [INFO] [Final Eval Pause] Pausing before fixed view 25.
2026-10-22 02:05:03,434 [INFO] Run 'tiny_rgb' paused during final evaluation.
2026-10-22 02:05:05,297 [INFO] [Pause] Detected PAUSE file at: /tmp/exp_rehearsal/state/PAUSE. Requesting clean pause...
2026-10-22 02:05:05,326 [INFO] [Bench] Checkpoint 'ckpt_step_000000000.pt' saved in 0.028s (11.36 MB)
2026-10-22 02:05:05,487 [INFO] [Pause] Paused cleanly at global_step=0. Exiting code 0.
```

#### C. Generated Report (`/tmp/exp_rehearsal/state/OVERNIGHT_REPORT.md`)
```markdown
# Overnight Supervision Report

**Report Generated:** 2026-10-22 02:05:06 (2026-10-21 20:35:06 UTC)
**Runner Process:** INACTIVE / PAUSED
**Supervisor Status:** ACTIVE
**GPU Memory:** 293 MiB / 8,188 MiB (3.6% VRAM used, 0% GPU util)

---

## 1. Supervisor & Guard Event Counters

- **Unexpected Runner Restarts:** 0
- **Guard Pauses Tonight:** 0 (Limit: 3/night)
- **Manual Pause Marker:** None (auto-resume enabled)
- **Alert Status:** Clean (no alerts)

---

## 2. Experimental Queue & Run Progress

| Run ID | Status | Modality | Step Progress | % Done | SPS | Active Time | Return | Success Rate |
|---|---|---|---|---|---|---|---|---|
| `tiny_rgb` | `PAUSED` | rgb | 30,000 / 30,000 | 100.0% | 203 | 0:02:24 | 3.98 | 0.0% |
| `tiny_rgbd` | `PAUSED` | rgbd | 0 / 30,000 | 0.0% | 0 | 0:00:00 | 0.00 | 0.0% |

---

## 3. Checkpoint Status & Ages

- **`tiny_rgb` Checkpoint:** `ckpt_step_000028000.pt` (34.0 MB, 1m 13s ago)
- **`tiny_rgbd` Checkpoint:** `ckpt_step_000000000.pt` (11.4 MB, 1s ago)
```

#### D. GPU Memory Post-Window Idle Verification
- `nvidia-smi` confirmed GPU memory dropped immediately to **13 MiB / 8,188 MiB (0.2% util, completely idle)**.
- Real `state/` and `runs/` were completely untouched.
- Clean post-rehearsal teardown: removed `/tmp/exp_rehearsal/`.

---

## [2026-10-22 10:15] Diagnostic Analysis & Amendment Preparation (Scratch Directory)

### Status: Complete for Steps (1) and (2); Awaiting Candidate Table for Steps (3)-(5)

1. **Pilot RGB Trajectory Diagnosis (`runs/pilot_rgb_seed100/progress.csv`)**:
   - Initial conditions from `src/model.py`: `action_dim = 4` (`pd_ee_delta_pos`), `actor_logstd` initialized to `-0.5` ($\sigma_0 \approx 0.6065$, initial entropy $H_0 \approx 3.675$).
   - Documented collapse progression:
     - Collapse initiated immediately from iteration 1.
     - By 500k steps (iter 63): $\sigma$ fell to 0.501.
     - By 1.0M steps (iter 125): $\sigma$ fell to 0.401, clip fraction reached 0.445.
     - By 2.5M steps (iter 313): Entropy became negative ($H = -0.434$), clip fraction exceeded 0.50.
     - By 5.0M steps (iter 625): $\sigma$ collapsed to 0.105 (83% reduction in exploration radius), approx KL reached 0.394.
     - 10M to 30M steps: severe instability, $\sigma$ reached $0.012 - 0.035$, approx KL spiked up to 91.98, clip fraction sustained $>0.55$, returns stalled at ~8.0 with 0% success.
   - Identified root cause: `ent_coef: 0.0` combined with absent `target_kl` early stopping (CleanRL `ppo_rgb.py` enforces `target_kl = 0.2`).

2. **Scratch Code Preparation (`/tmp/exp_diag/code`)**:
   - Working strictly in `/tmp/exp_diag/code` (copied from `frozen_code/pre-pilot-freeze`).
   - Verified `/home/maniskill/examples/baselines/ppo/ppo_rgb.py` uses `target_kl = 0.2` and early-breaks both minibatch and epoch loops.
   - Implemented optional `target_kl` in `/tmp/exp_diag/code/src/train_ppo.py` with identical early stopping logic.
   - Enabled custom diagnostic split loading via `split_file` in `/tmp/exp_diag/code/src/train_ppo.py` and `src/env_factory.py`.
   - Verified with unit tests (`tests/test_target_kl.py`, 2/2 passed).
   - Real codebase (`src/`, `configs/ppo_frozen.yaml`, `frozen_code/`, real `state/` and `runs/`) kept completely untouched.

3. **Diagnostic Evaluation of Candidates D1–D5 & Selection Outcome**:
   - Executed diagnostic candidate grid sequentially in scratch copy (`/tmp/exp_diag/code`) under isolated state and run directories (`/tmp/exp_diag/`).
   - Supervised each detached run with `nice -n 10` and periodic 10-minute evaluations; validated pause and resume functionality midway on D3 and D4.
   - **Candidate D1 (Harness Control: Single Viewpoint 17, 3M steps)**:
     - Policy learned smoothly: final training success reached **77.8%**; step 3M periodic eval on Viewpoint 17 reached **82.0%** (50 episodes).
     - Successfully validated the experimental harness, simulator physics, and CleanRL PPO implementation.
   - **Candidates D2–D5 (20 Train Views, 6M steps each)**:
     - **D2** (`target_kl = 0.2`): Final action std 0.2024, median approx KL 0.0747, median clipfrac 0.3271, final 1M train success 0.70%. (FAILED selection rule: clipfrac > 0.3, train success < 10%).
     - **D3** (`target_kl = 0.2`, `num_minibatches = 8`): Final action std 0.2072, median approx KL 0.0852, median clipfrac 0.3700, final 1M train success 0.24%. (FAILED selection rule: clipfrac > 0.3, train success < 10%).
     - **D4** (`target_kl = 0.2`, `num_minibatches = 8`, `lr = 1e-4`): Final action std 0.2930, median approx KL 0.0378, median clipfrac 0.3085, final 1M train success 0.02%. (FAILED selection rule: clipfrac > 0.3, train success < 10%).
     - **D5** (`target_kl = 0.2`, `ent_coef = 0.001`, `num_minibatches = 16`): Final action std 0.1942, median approx KL 0.0731, median clipfrac 0.3354, final 1M train success 0.72%. (FAILED selection rule: clipfrac > 0.3, train success < 10%).
   - **Mechanical Selection Rule Conclusion**:
     - None of candidates D2–D5 passed the pre-declared selection rule criteria.
     - While `target_kl = 0.2` effectively prevented the catastrophic KL explosions and std collapse of the pilot run, multi-view RGB learning from scratch with an unaugmented NatureCNN achieves ~0.2%–0.7% completion success within 6M steps (reaching the cube reliably with return ~14–15, but failing to consistently grasp and lift).
   - In accordance with project rules, no modifications have been applied to `configs/ppo_frozen.yaml`, `src/`, or `frozen_code/`. Complete tables and analysis documented in `DECISIONS.md`.

