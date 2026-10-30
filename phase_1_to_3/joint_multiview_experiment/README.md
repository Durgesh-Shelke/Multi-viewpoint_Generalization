# Joint Multi-View RGB vs RGB-D Experiment

Benchmarking visual generalization across camera viewpoints on `PickCube-v1` (ManiSkill3, Franka Panda, 4.5 cm cube, `pd_ee_delta_pos` control mode).

---

## Command Reference (Harness CLI)

All long jobs run detached in the background with lowered priority (`nice -n 10`).

### 1. Status Check
Check current run progress, step budget, SPS, ETA, checkpoint age, and VRAM:
```bash
python src/ctl.py status
```

### 2. Pause Training
Gracefully pause the active run (finishes current iteration, saves atomic checkpoint, releases GPU):
```bash
python src/ctl.py pause
```
Or create a pause file directly:
```bash
touch state/PAUSE
```

### 3. Resume Training
Resume interrupted training from the latest verified checkpoint:
```bash
python src/ctl.py resume
```

---

## Safe Manual Smoke Test for Control Tools (`configs/tiny_runs.yaml`)

This protocol allows verifying the full control workflow (`start` -> `status` -> `pause` -> `nvidia-smi` -> `resume` -> `finish`) in ~3 minutes per run without touching production queues or data.

### Queue Safety Guard
When `--runs-file configs/tiny_runs.yaml` is provided, `ctl.py` passes the file directly to `runner.py`. The runner executes **only** the runs defined in `configs/tiny_runs.yaml` (`tiny_rgb` and `tiny_rgbd`, using `configs/tiny_test.yaml`). The real production queue (`configs/runs.yaml`) can never be started by accident. Furthermore, setting `EXP_STATE_DIR` and `EXP_RUNS_DIR` guarantees that all state locks, queue metadata, checkpoints, and logs write exclusively to temporary storage (`/tmp/exp_manual/`) and never touch the repository's `state/` or `runs/` directories.

### Step-by-Step Manual Test Instructions

#### 1. Configure Isolated Temporary Directories
```bash
export EXP_STATE_DIR=/tmp/exp_manual/state
export EXP_RUNS_DIR=/tmp/exp_manual/runs
mkdir -p /tmp/exp_manual/state /tmp/exp_manual/runs
```

#### 2. Start Training Detached
Launch the runner detached in the background with `nice -n 10` using the tiny configuration:
```bash
python ctl.py --runs-file configs/tiny_runs.yaml resume
```

#### 3. Check Status
Inspect active progress, current iteration, throughput (SPS), estimated ETA, and GPU VRAM:
```bash
python ctl.py --runs-file configs/tiny_runs.yaml status
```

#### 4. Pause Training
Gracefully pause the training process (allows current iteration to finish, writes atomic checkpoint, and shuts down):
```bash
python ctl.py --runs-file configs/tiny_runs.yaml pause
```

#### 5. Confirm GPU Memory Is Fully Freed
Inspect `nvidia-smi` to verify that the training process has cleanly terminated and freed all GPU VRAM:
```bash
nvidia-smi
```

#### 6. Resume Training
Resume training seamlessly from the latest saved checkpoint:
```bash
python ctl.py --runs-file configs/tiny_runs.yaml resume
```

#### 7. Monitor & Finish
Check status to monitor the run until completion:
```bash
python ctl.py --runs-file configs/tiny_runs.yaml status
```
When finished, `Run Status` will display `DONE` and `Step / Budget` will reach `30,000 / 30,000`.

---

### 4. Running the Harness / Queue Detached
Launch the experiment runner queue detached in background:
```bash
nohup nice -n 10 python src/runner.py > state/runner.log 2>&1 &
```

### 5. Running Verification Tests
Execute the pause/resume test suite before running any job longer than 5 minutes:
```bash
pytest tests/test_pause_resume.py -v
```

### 6. Viewpoint Geometry & Split Verification
Verify the train/val/test viewpoint split geometry and angular distances:
```bash
python src/env_factory.py --verify-split
```

### 7. Evaluation
Run evaluation benchmark across all 30 viewpoints with Wilson confidence intervals:
```bash
python src/evaluate.py --checkpoint runs/<run_id>/checkpoints/final.pt --episodes-per-view 200
```

### 8. Report Generation
Generate tables, learning curves, and generalization reports:
```bash
python src/report.py --results-dir results/
```
