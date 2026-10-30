# Agent Task Spec: Joint Multi-View Training, RGB vs RGB-D (ManiSkill3 PickCube)

You are an autonomous coding/experiment agent working on my personal laptop (RTX 4060 Laptop, 8 GB VRAM). Read this whole file before doing anything. Follow it as the source of truth. If something here conflicts with existing code, the spec wins; log the conflict in `DECISIONS.md`.

---

## 0. GOAL (what "done" means)

**Primary goal.** Train and fairly compare **joint multi-view RGB** and **joint multi-view RGB-D** visuomotor PPO policies for `PickCube-v1` (4.5 cm block), and measure how well each generalizes to **held-out camera viewpoints**.

**The task is complete only when ALL of the following are true:**

- [ ] **G1. Pause/resume works at any time** (Section 2). Verified by automated tests, including a hard-kill test.
- [ ] **G2. Frozen experimental protocol**: viewpoint split, hyperparameters, step budget, seeds are written to config files and not changed after the pilot (Sections 3 and 4).
- [ ] **G3. Pilot completed** and the fixed step budget `B` chosen by the written rule in Section 4.3.
- [ ] **G4. Main grid completed**: 2 modalities (RGB, RGB-D) x 3 seeds (0, 1, 2), each trained from scratch for exactly `B` environment steps.
- [ ] **G5. Final evaluation completed** on all 30 viewpoints for every final checkpoint (Section 5), with confidence intervals.
- [ ] **G6. Report generated** (Section 7): tables, figures, and a cautious written analysis, plus the reproducibility manifest.

**Secondary goals (do not start until G1-G6 are done):**
- S1. Ablation: depth clamp at 1.5 m instead of 3.0 m (RGB-D only, 3 seeds, same budget).
- S2. Make the training harness modular so a federated (FedAvg) client/server loop can reuse the same env, model, eval, and checkpoint code later. **Do not implement federation now.**

**Non-goals:** No sim-to-real work, no new tasks, no architecture changes, no hyperparameter tuning beyond what Section 4 allows.

---

## 1. Context

- Research question: does adding depth help a single policy generalize across camera viewpoints when trained jointly on many views? (Later phase: does federated training approach joint training?)
- Prior single-view results (do not re-run): policies trained on one camera work only within roughly +/-15 deg and collapse elsewhere. A previous RGB vs RGB-D comparison was confounded (different training cameras, different PPO configs, single seed), so **this experiment replaces it. Do not cite those earlier RGB-vs-RGB-D numbers as evidence.**
- Existing assets (locate them, do not modify or delete; treat as read-only):
  - Joint multi-view training script: `/home/maniskill/joint_multiview_training/train_joint_multiview.py`
  - 30 Fibonacci viewpoints: `viewpoints.json` (under the `sequential_viewpoint_training` directory; paths in my notes are inconsistent between `/home/maniskill/` and `/home/mahoraga/maniskill/`, so find the real one and record it in `DECISIONS.md`)
  - RGB-D evaluator: `/home/maniskill/evaluate_fibonacci_hemisphere_rgbd.py`
  - Backups such as `backup_original_model.pt`, `best_sim2real_policy.pt`: never overwrite.
- Create all new work under: `/home/maniskill/joint_multiview_experiment/` (call it `$EXP`).

Environment facts (verify against the code, then record):
- ManiSkill3, `PickCube-v1`, Franka Panda, 4.5 cm solid block, control mode `pd_ee_delta_pos`, 20 Hz, episode length 50.
- Camera sphere centre `[-0.615, 0.0, 0.0]`, radius `1.0942 m`, 128x128 images.
- Model: NatureCNN (3 or 4 input channels) -> FC 256, state `Linear(29 -> 256)`, fused 512-d, PPO (CleanRL-style, GPU-vectorized), orthogonal init.
- Depth preprocessing (keep as is for the main grid): `clamp(depth_mm / 1000, 0, 3.0) / 3.0`.

---

## 2. HARD REQUIREMENT: pause and resume at any time

This is my personal laptop and I need it for other work. I must be able to stop training whenever I want and resume later without losing meaningful progress or corrupting results. **Pausing must fully release the GPU** (the process exits).

### 2.1 Pause mechanisms (implement all)
1. **Signals:** handle `SIGINT` (Ctrl+C) and `SIGTERM`. On receipt, set a flag; finish the current PPO iteration; write a checkpoint; write the state file; exit with code 0. Second `SIGINT` within 10 s: exit immediately after a quick checkpoint attempt.
2. **PAUSE file:** if `$EXP/PAUSE` exists, behave as above (check at the end of every PPO iteration). Resuming deletes it.
3. **CLI control script** `ctl.py`:
   - `python ctl.py pause` creates `PAUSE` and waits until the process exits.
   - `python ctl.py resume` (re)launches the runner in the background (`nohup`), detached from the terminal.
   - `python ctl.py status` prints: current run, step / budget, % done, SPS, ETA (active time), last checkpoint age, GPU memory.
4. **Optional auto-pause (flag, default off):** pause when on battery or battery < 25 %, and optionally when free VRAM < 1.5 GB.

### 2.2 Checkpointing
- Save at the end of an iteration when **either** 5 minutes of wall-clock **or** 10 iterations have passed since the last save, whichever comes first, **and** always on pause/exit.
- A checkpoint must contain everything needed to continue: model, optimizer, LR scheduler state, `global_step`, `iteration`, all RNG states (python, numpy, torch CPU, torch CUDA), any observation/value normalization statistics, evaluation history so far, cumulative *active* training time, and the config hash.
- **Atomic writes:** write to `*.tmp`, `fsync`, then `os.replace`. Never leave a half-written checkpoint as "latest".
- Keep the last **2** rolling checkpoints plus periodic milestone checkpoints (every 5M steps) plus the final one. On startup, validate the latest checkpoint (load test + checksum); if corrupt, fall back to the previous one and log it.
- Check free disk space before writing; pause cleanly with a clear message if space is low.

### 2.3 Resume semantics
- Running the same command again **auto-resumes** from the newest valid checkpoint. No manual flags needed.
- Refuse to resume if the config hash differs from the checkpoint's (print a diff). Never silently change settings mid-run.
- Environments cannot be restored mid-episode; on resume, reset envs. Derive env seeds from `(run_seed, iteration)` so a resumed run is reproducible. This is acceptable and must be documented.
- Logs (CSV/JSONL/TensorBoard) are **appended**, never overwritten. After resume, steps must continue monotonically with no gaps or duplicates.
- Track and log **active training time** separately from wall time (paused time excluded).

### 2.4 Multi-run orchestration
- A `runner.py` executes a queue of runs (pilot, then main grid) in a fixed order. Per-run status in `$EXP/state/queue.json`: `pending | running | paused | done | failed`.
- Resuming the runner continues the interrupted run, then proceeds to pending ones. Completed runs are never repeated.
- Only one runner instance at a time (lock file with PID; detect stale locks).

### 2.5 Being a good citizen on a shared laptop
- Keep VRAM use **under ~6 GB** and log it. Make `num_envs` configurable (but frozen per Section 4 for the experiment).
- Run with lowered CPU priority (`nice -n 10`).
- Never start anything GPU-heavy besides the active run. Evaluation jobs run inside the runner, not concurrently.

### 2.6 Required tests (part of goal G1)
Write and run `tests/test_pause_resume.py` (use a tiny config, e.g. 2 envs, a few thousand steps):
1. **Graceful pause**: send SIGTERM mid-run, confirm checkpoint + state written, process exits 0, GPU freed.
2. **Hard kill**: `kill -9` mid-run, then restart; must resume from the last good checkpoint and continue. Loss of at most ~5 min / 10 iterations of progress is acceptable.
3. **Corrupt checkpoint**: truncate the latest checkpoint file; resume must fall back to the previous one.
4. **Continuity**: step counter, eval history, and log files are continuous (no duplicates/gaps) across 3 pause/resume cycles.
5. **Equivalence sanity check**: a run paused and resumed twice reaches a comparable final result to an uninterrupted run with the same seed (exact bitwise equality is NOT required; document differences).
6. **Config guard**: resuming with a modified config is refused.

Do not start any long run until all six tests pass.

---

## 3. Experimental design

### 3.1 Arms
| Arm | Input | Notes |
|---|---|---|
| `joint_rgb` | RGB 128x128x3 | from scratch |
| `joint_rgbd` | RGB + depth 128x128x4 | from scratch, **no warm-start** from any RGB weights |

Both arms: orthogonal init, identical PPO hyperparameters, identical `num_envs`, minibatches, step budget, seeds, and train/eval views. **The only difference is the observation channels.**

### 3.2 Viewpoint split (30 Fibonacci viewpoints, IDs as in `viewpoints.json`)
Proposed default (edit before freezing, then never change):

| Set | Viewpoint IDs | Purpose |
|---|---|---|
| `train` (20) | all IDs not listed in the three rows below | training |
| `val` (3) | 03, 10, 15 | monitoring only (no tuning on test) |
| `test_interp` (3) | 09, 13, 21 | interpolation: surrounded by train views |
| `test_extrap` (4) | 06, 16, 23, 27 | extrapolation: contiguous far-left flank (azimuth >= about +67 deg) |

Rules:
- Build `train` programmatically as the complement of the other three sets (expected size 20), and assert the four sets are disjoint and cover all 30.
- Verification script must print, for each `test_interp` view, its angular distance to the nearest 2 train views (expect <= ~20 deg), and for each `test_extrap` view the distance to the nearest train view (expect noticeably larger). If the geometry does not match the intent, report it in `DECISIONS.md` and propose a corrected split; **do not silently change it.**
- Save as `$EXP/configs/viewpoint_split.json` and freeze it.
- Training envs are distributed **evenly across train views** (use `num_envs` divisible by 20, default **160** = 8 envs per view).
- Cube/goal randomization stays as in the existing task.

### 3.3 Seeds
Main grid seeds: `0, 1, 2`. Pilot uses seed `100` (not reused in the main grid).

---

## 4. Protocol: budget, hyperparameters, stopping

### 4.1 No accuracy-threshold stopping
Do **not** stop on success rate. Every main run trains for exactly `B` environment steps. The only other termination conditions: wall-clock cap per run (default 12 h of active time), NaN/inf loss (halt, mark `failed`, report), and my pause.

### 4.2 Hyperparameters
Extract PPO hyperparameters from the existing joint training script, write them to `$EXP/configs/ppo_frozen.yaml`, and use the identical file for both arms. Defaults to set if the script lacks them: `num_envs=160`, `num_steps=50` (rollout batch 8,000), 16 minibatches, otherwise inherit the existing values. **No tuning is permitted after the pilot.** If training is clearly broken (e.g., NaNs, no learning at all in the pilot), stop and report to me instead of tuning silently.

### 4.3 Choosing the budget `B` (pilot)
1. Run one `joint_rgb` and one `joint_rgbd` pilot (seed 100) up to **30M steps** each. Evaluate on `train` + `val` views only. **Never look at test views during the pilot.**
2. Let `S` be the first step at which the 5-evaluation moving average of mean `train` success improves by less than 2 percentage points per further 5M steps (the plateau). Take the later `S` of the two arms.
3. `B = round_up(1.5 * S, to nearest 5M)`.
4. If no plateau by 30M, set `B = 45M` and **ask me for confirmation** before continuing (cost grows). If `B > 30M` in general, also ask me first.
5. Write `B`, the pilot curves, and the reasoning to `$EXP/configs/budget.json` and `DECISIONS.md`. Freeze.

Rough cost (at ~2,000-2,500 SPS): 20M steps is about 2.5-3 h. Main grid at 20M is about 17 h active time total. Because of pause/resume it can be spread over many days.

---

## 5. Evaluation

**Periodic (during training, every ~1M steps):** `train` and `val` views, 10 episodes per view, per-view success logged. This is for learning curves and diagnostics only.

**Final (once per run, on the final checkpoint at step `B`, no checkpoint picking):**
- All 30 viewpoints, **200 episodes per view**, fixed evaluation seeds shared by both arms.
- Per view: success rate, 95 % Wilson confidence interval, mean return, mean episode length.
- Group aggregates: `train`, `val`, `test_interp`, `test_extrap`, and `all`. Aggregate across seeds as mean +/- std (n = 3) and report per-seed values too.

**Efficiency metrics:** SPS, peak VRAM, active training time, and steps to reach 50 % and 80 % mean `train` success (from learning curves).

Evaluation must use the same evaluator code path for both arms (only the observation mode differs).

---

## 6. Repository layout

```
$EXP/
  DECISIONS.md            # every assumption, deviation, conflict, with timestamp
  README.md               # exact commands: start, pause, resume, status, evaluate, report
  configs/
    ppo_frozen.yaml
    viewpoint_split.json
    budget.json
    runs.yaml             # queue: pilot_rgb, pilot_rgbd, then rgb/rgbd x seeds 0,1,2
  src/
    env_factory.py        # builds envs, assigns viewpoints to env indices, RGB / RGB-D obs
    model.py              # NatureCNN with 3 or 4 channels
    train_ppo.py          # training loop, resumable
    checkpoint.py         # atomic save/load/validate
    evaluate.py           # per-view evaluation with Wilson CIs
    runner.py  ctl.py     # queue, lock, pause/resume
    report.py             # tables + figures
  tests/
  state/                  # queue.json, lock, PAUSE
  runs/<run_id>/          # checkpoints/, logs/, evals/, config_snapshot.yaml, manifest.json
  results/                # final CSVs, figures, REPORT.md
```

Each run's `manifest.json`: git commit hash (initialize a git repo in `$EXP` if none), package versions (ManiSkill, torch, CUDA), GPU, config hash, seeds, viewpoint split hash, start/stop/resume timestamps.

---

## 7. Deliverables and report rules

`results/REPORT.md` with:
1. Setup summary (arms, split, budget `B` and how it was chosen, seeds).
2. Table: mean success (+/- std over seeds) for `train`, `val`, `test_interp`, `test_extrap`, `all`, RGB vs RGB-D. Also per-seed table.
3. Per-viewpoint table (30 rows) with CIs.
4. Figures: learning curves (mean +/- std across seeds) per arm; scatter of the 30 viewpoints on azimuth-elevation axes colored by success, one panel per arm; bar chart of group means with error bars.
5. Efficiency table (SPS, VRAM, active time, steps-to-threshold).
6. Analysis, written under these rules:
   - Report the **final** checkpoint, not the best. No cherry-picking of seeds or views.
   - Describe differences as supported only if the seed-to-seed spread and CIs allow it; with n = 3, say "suggests" rather than "proves".
   - State limitations: simulation only, noise-free depth, one task, one backbone, depth clamp choice, held-out split choice.
   - Do not claim "viewpoint invariance" or "categorically superior"; report what was measured.
   - Mention which results may be explained by view proximity (distance from each test view to nearest train view).

---

## 8. Milestones (work in this order; check off in `DECISIONS.md`)

- [ ] **M0 Setup & discovery:** locate existing scripts/assets, write `README.md` skeleton and `DECISIONS.md`, set up git, record environment versions.
- [ ] **M1 Resumable harness:** implement Section 2; all six tests pass. *(Gate for everything else.)*
- [ ] **M2 Split & sanity:** freeze `viewpoint_split.json`; render a contact sheet (one RGB and one depth image per view) to verify cameras look at the workspace; smoke-test both arms for ~50k steps each.
- [ ] **M3 Pilot** (Section 4.3) and freeze `budget.json`. Report to me the chosen `B` and projected total time before continuing.
- [ ] **M4 Main grid:** 6 runs, resumable, queued. Provide `ctl.py status` summaries.
- [ ] **M5 Final evaluation** (Section 5).
- [ ] **M6 Report** (Section 7) and a final checklist confirming goals G1-G6.

---

## 9. Agent behavior rules

- **Do not change the protocol on your own** (split, budget, hyperparameters, seeds, preprocessing). If you think a change is needed, write the case in `DECISIONS.md` and ask me.
- Ask me only when blocked or for the explicit confirmation points above (`B > 30M`, split geometry problems, training broken). Otherwise make a reasonable choice, log it, and continue.
- Never delete or overwrite pre-existing models, scripts, or results outside `$EXP`.
- Keep a short running `PROGRESS.md` (what is done, what is running, what is next) so I can read the status at a glance after returning from other work.
- When I pause, you stop quickly and cleanly. When I resume, you continue without asking.
- Do not run long jobs while a test or debug session is using the GPU.
- Prefer simple, readable code over cleverness; I need to explain this in my BTP thesis.
