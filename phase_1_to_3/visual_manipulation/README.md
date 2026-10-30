# Visual Manipulation Pipeline for ManiSkill

An end-to-end pipeline for training vision-based robotic manipulation policies using a combination of **Imitation Learning (Behavioral Cloning)** and **Visual Reinforcement Learning (PPO)**.

---

## Architecture Overview

- **Visual Input**: `128x128x3` RGB images rendered in real-time from the robot workstation's `base_camera`.
- **Proprioceptive Input**: 29-dimensional state vector (robot joint angles, joint velocities, TCP pose, goal position, and grasp flag).
- **Vision Backbone**: `NatureCNN` (3 convolutional layers with ReLU activations) mapping images to a 256-dimensional embedding, concatenated with a 256-dimensional proprioceptive feature to form a 512-dimensional latent representation.
- **Control Space**: `pd_ee_delta_pos` (4-DoF end-effector controller: $[dx, dy, dz, \text{gripper}]$).
- **Hardware Profile**: Optimized for consumer GPUs (e.g., RTX 4060 with 8GB VRAM) using parallel GPU simulation in ManiSkill/SAPIEN.

---

## Directory Structure

```
visual_manipulation/
├── README.md               # This documentation
├── model.py                # Visual Actor-Critic & NatureCNN neural network architecture
├── dataset.py              # PyTorch Dataset for loading demonstration trajectories
├── prepare_dataset.py      # Automated demonstration downloader & RGB replay utility
├── train_bc.py             # Behavioral Cloning (Imitation Learning) pre-trainer
├── train_rl.py             # Visual PPO Reinforcement Learning fine-tuner
└── evaluate.py             # Evaluation & video recording tool
```

---

## Quickstart Guide

### 1. Data Preparation (Generate Visual Demonstrations)
Downloads official expert motion planning demonstrations and replays them through the simulator to capture rendered camera observations:

```bash
python3 prepare_dataset.py --env_id PickCube-v1 --count 100 --control_mode pd_ee_delta_pos
```

This generates an RGB dataset file:
`~/.maniskill/demos/PickCube-v1/motionplanning/trajectory.rgb.pd_ee_delta_pos.physx_cpu.h5`

---

### 2. Behavioral Cloning Pre-training
Pre-trains the visual policy directly on expert demonstrations to learn basic reaching, grasping, and lifting behaviors in ~20-30 seconds:

```bash
python3 train_bc.py \
  --epochs 30 \
  --batch_size 128 \
  --lr 1e-3 \
  --save_dir checkpoints/bc
```

Output checkpoints:
- `checkpoints/bc/best_val_model.pt` (best validation loss)
- `checkpoints/bc/final_bc_model.pt` (final epoch)

---

### 3. Visual PPO Fine-Tuning
Loads the pre-trained weights into the visual actor-critic and optimizes policy performance using reward feedback across 128 parallel GPU environments:

```bash
python3 train_rl.py \
  --checkpoint checkpoints/bc/final_bc_model.pt \
  --num_envs 128 \
  --total_timesteps 1000000 \
  --eval_freq 5 \
  --exp_name visual_ppo_finetune
```

*Note: For an 8GB GPU, `--num_envs 128` achieves ~1,400 simulation steps/sec while keeping VRAM comfortably under 5.5 GB.*

---

### 4. Evaluation & Video Recording
Evaluates any saved checkpoint on randomized environment seeds and records `.mp4` videos:

```bash
python3 evaluate.py \
  --checkpoint checkpoints/bc/final_bc_model.pt \
  --episodes 5 \
  --video_dir eval_videos
```

Evaluation videos are saved to `eval_videos/` (e.g., `0.mp4`, `1.mp4`).
