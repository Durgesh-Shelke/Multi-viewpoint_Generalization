# Sequential Viewpoint Continual Training Architecture

## 1. Directory Structure & Assets

All training files and checkpoints are stored in the host-mounted workspace:

```text
/home/mahoraga/maniskill/sequential_viewpoint_training/
├── backup_original_model.pt     <- Untouched backup of the original 97.1% front-view model
├── viewpoints.json              <- The 30 equal-area Fibonacci hemispherical camera coordinates
├── training_progress.json       <- Live state tracker updated every epoch (enables seamless resume)
├── train_sequential_viewpoints.py <- Continual fine-tuning script
└── checkpoints/
    ├── latest_in_progress.pt    <- Instant checkpoint saved after every PPO iteration
    ├── ckpt_vp_00.pt            <- Checkpoint after completing Viewpoint 00
    ├── ckpt_vp_01.pt            <- Checkpoint after completing Viewpoint 01
    └── ... (up to ckpt_vp_29.pt)
```

---

## 2. Safety & Model Preservation

- **Original Base Model:** Permanently preserved at:
  - Host: `/home/mahoraga/maniskill/sequential_viewpoint_training/backup_original_model.pt`
  - Workspace: [`/home/maniskill/sequential_viewpoint_training/backup_original_model.pt`](file:///home/maniskill/sequential_viewpoint_training/backup_original_model.pt)
  - Also mirrored at: `/home/mahoraga/maniskill/best_sim2real_policy.pt`
- Training operates on downstream weights and checkpoints, ensuring zero risk of losing the original policy.

---

## 3. Resumability & Progress Persistence

Every iteration (every ~2 seconds), the system flushes the latest model weights and writes an updated state snapshot to [`training_progress.json`](file:///home/maniskill/sequential_viewpoint_training/training_progress.json):

```json
{
  "current_viewpoint_idx": 0,
  "total_viewpoints": 30,
  "status": "training",
  "current_iteration": 6,
  "total_iterations_per_vp": 25,
  "latest_checkpoint": ".../checkpoints/latest_in_progress.pt",
  "completed_viewpoints": [],
  "last_updated": "2026-10-20 19:45:13"
}
```

### Stopping & Resuming
- **To pause/stop training:** You can interrupt the process anytime with `Ctrl+C` (or signal termination). The script handles `SIGINT` / `SIGTERM`, saves the latest weights to `latest_in_progress.pt`, updates `training_progress.json`, and exits cleanly.
- **To resume training:** Simply execute:
  ```bash
  python3 /home/mahoraga/maniskill/sequential_viewpoint_training/train_sequential_viewpoints.py
  ```
  It will automatically detect `training_progress.json`, load `latest_in_progress.pt`, and continue from the exact viewpoint and iteration where it was stopped.
