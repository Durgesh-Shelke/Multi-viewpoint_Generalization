# Repository Maintenance & Developer Guide for Future Agents and Contributors

This document defines the operational architecture, development workflow, and safety guidelines for maintaining and extending the `Multi-viewpoint_Generalization` repository. **Future AI agents and human developers must consult this document before adding new experiments, modifying environments, or pushing commits.**

---

## 1. Dual-Workspace Architecture

This project operates with a clean separation between the **simulation execution runtime** and the **version-controlled research repository**:

```text
/home/
├── maniskill/                         <- Simulation & Execution Engine (ManiSkill 3)
│   ├── mani_skill/                    <- SAPIEN/PhysX simulation backend (Keep clean/unmodified)
│   ├── phase_1_to_3/                  <- Local working copies
│   ├── phase_4/                       <- Local working copies
│   └── phase_5/                       <- Local working copies & live checkpoints
│
└── Multi-viewpoint_Generalization/    <- Standalone Git Research Repository (GitHub remote)
    ├── envs/                          <- Self-contained task registration & wrappers
    ├── phase_1_to_3/                  <- Viewpoint sweeps, catastrophic forgetting & reports
    ├── phase_4/                       <- RGB vs. RGB-D parity comparison (5M steps)
    ├── phase_5/                       <- 50M-step resolution & sensor fusion study
    ├── README.md                      <- Master publication & benchmark documentation
    ├── MAINTENANCE.md                 <- This operational manual
    ├── requirements.txt               <- Frozen dependencies
    └── .gitignore                     <- Strict exclusion of heavy training artifacts
```

### Key Operational Rule
- **Run heavy GPU training and simulations** in `/home/maniskill/` or inside a dedicated phase directory.
- **Commit and version-control deliverables** inside `/home/Multi-viewpoint_Generalization/`.
- When experiments conclude, sync deliverables (code, JSON summaries, reports, plots, and best models) into `/home/Multi-viewpoint_Generalization/`.

---

## 2. File Size & Git Safety Guardrails (CRITICAL)

GitHub strictly rejects files $>100\text{ MB}$ and issues repository health warnings for files $>50\text{ MB}$. Reinforcement learning runs generate gigabytes of weights and logs.

### The Storage Hierarchy

| Artifact Category | Examples | Tracked in Git? | Where Stored / Handled |
| :--- | :--- | :---: | :--- |
| **Code & Scripts** | `*.py`, `*.sh`, `*.json`, `*.csv` | **YES** | Tracked directly in Git |
| **Documentation & Reports** | `*.md`, `*.png` | **YES** | Tracked directly in Git |
| **Best Model Checkpoints** | `candidate_*_best.pt` (~12–21 MB) | **YES** | Tracked directly in Git |
| **Optimizer Checkpoints** | `*_latest.pt` (~63 MB) | **NO** | Excluded via `.gitignore` (kept locally) |
| **Intermediate Training Iterations**| `ckpt_*.pt` (10s of GBs) | **NO** | Excluded via `.gitignore` (kept locally) |
| **TensorBoard Logs** | `events.out.tfevents.*` | **NO** | Excluded via `.gitignore` (kept locally) |
| **Heavy Video Sweeps** | Render batches (>50 MB) | **NO** | Excluded; keep only sample demo MP4s |

### Mandatory Pre-Commit Check for Agents
Before making any `git commit`, an agent **must** run:
```bash
find /home/Multi-viewpoint_Generalization -type f -size +45M
```
If any file is returned:
1. Do **not** commit it directly.
2. If it is an intermediate checkpoint, ensure it matches a rule in `.gitignore`.
3. If it is a necessary final checkpoint, convert it to a weights-only model (`torch.save(agent.state_dict(), ...)` without optimizer state) to reduce its size below 25 MB.

---

## 3. How to Modify or Add Environments

**Never modify the internal ManiSkill library files directly** (e.g. do not edit `/home/maniskill/mani_skill/envs/tasks/tabletop/pick_cube.py`). Modifying upstream library code breaks reproducibility when external researchers install ManiSkill via `pip`.

### Proper Workflow: Subclassing & Registration
1. Add new environment variations inside [`/home/Multi-viewpoint_Generalization/envs/`](file:///home/Multi-viewpoint_Generalization/envs/).
2. Subclass the base task and use ManiSkill's `@register_env` decorator:

```python
from mani_skill.envs.tasks.tabletop.pick_cube import PickCubeEnv
from mani_skill.utils.registration import register_env

@register_env("PickCubeCustom-v1", max_episode_steps=50)
class PickCubeCustomEnv(PickCubeEnv):
    def __init__(self, *args, cube_half_size=0.0225, **kwargs):
        super().__init__(*args, **kwargs)
        self.cube_half_size = cube_half_size
```

3. Expose the environment in [`envs/__init__.py`](file:///home/Multi-viewpoint_Generalization/envs/__init__.py) so importing `envs` automatically registers the task.

---

## 4. How to Structure a New Experiment (e.g., Phase 6)

When embarking on a new research hypothesis (e.g., multi-task stacking, domain randomization, or architectural extensions):

1. **Create a Dedicated Phase Directory:**
   ```bash
   mkdir -p /home/Multi-viewpoint_Generalization/phase_6_stack_cube_or_domain_rand/{checkpoints,eval_results,reports}
   ```
2. **Follow the Standard Component Structure:**
   - `models.py`: Neural architecture definitions (inheriting from `torch.nn.Module`).
   - `train_*.py`: Vectorized PPO training loop with SAPIEN GPU simulation (`physx_cuda`).
   - `evaluate_battery.py`: Multi-tier evaluation suite (in-domain, perturbations, physical clock latency).
   - `status.sh` / `pause.sh` / `resume.sh`: Non-destructive control tools.
   - `README.md`: Directory overview, candidate matrix, and execution instructions.
3. **Formal Reporting Convention:**
   - Write a formal markdown report following the numbering sequence:
     - Phase 1–3: Reports `01_` through `10_`
     - Phase 4: Reports `11_` through `13_`
     - Phase 5: Report `14_resolution_and_asymmetric_depth_ablation_report.md`
     - Phase 6: Report `15_<descriptive_title>.md`
   - Archive the report in `phase_6_.../reports/` and update the master README index.

---

## 5. Training Control Standards (Pause, Resume, Status)

All training scripts designed for overnight or long-horizon execution should conform to our **zero-loss stateful interruption pattern**:

1. **The `.pause` Flag Mechanism:**
   - The shell script [`pause.sh`](file:///home/Multi-viewpoint_Generalization/phase_5/pause.sh) touches a `.pause` file in the experiment directory.
   - The training loop checks for `os.path.exists(".pause")` at the boundary of each iteration.
   - When detected, the script saves `candidate_{name}_latest.pt` (including model weights, optimizer states, iteration index, and global steps), prints a resumption log, and enters a low-CPU sleep loop or exits cleanly with code 0.
2. **Resumption:**
   - [`resume.sh`](file:///home/Multi-viewpoint_Generalization/phase_5/resume.sh) removes `.pause` and launches the runner.
   - The runner detects the checkpoint and restores `agent.load_state_dict(ckpt["model_state"])` and `optimizer.load_state_dict(ckpt["optimizer_state"])`, resuming from the exact step.
3. **Telemetry:**
   - Maintain a [`status.sh`](file:///home/Multi-viewpoint_Generalization/phase_5/status.sh) script reporting:
     - Current pipeline state (`[RUNNING]`, `[PAUSED]`, or `[COMPLETED]`)
     - Steps completed vs target budget
     - Live eval success rates
     - GPU utilization, temperature, and VRAM (`nvidia-smi`)

---

## 6. Git Synchronization Workflow for Future Agents

When changes or new deliverables are produced, agents should follow this exact sequence:

```bash
cd /home/Multi-viewpoint_Generalization

# 1. Verify working directory status
git status

# 2. Safety check: ensure no file > 45MB is staged
find . -type f -size +45M

# 3. Stage changes cleanly
git add envs/ phase_*/ README.md MAINTENANCE.md requirements.txt .gitignore

# 4. Commit using Conventional Commits format
git commit -m "feat(phase-6): add domain-randomized stacking experiment and report 15"

# 5. Push changes
git push origin main
```

### Commit Message Conventions
- `feat(...)`: A new model candidate, evaluation battery, or research phase.
- `fix(...)`: Bug fix in environment wrapping, observation tensor shape, or loss calculation.
- `docs(...)`: Updating research reports, benchmark matrices, or README documentation.
- `refactor(...)`: Restructuring code, cleaning imports, or modularizing models.

---

## 7. Quick Agent Checklist

Before completing any user turn that modifies the repository, verify:
- [ ] Are all new files located inside their appropriate `phase_X/` or `envs/` directory?
- [ ] Have all temporary `.tfevents*` or `.pt` files larger than 45 MB been excluded via `.gitignore`?
- [ ] Are script internal paths relative or using the standard workspace path?
- [ ] Does `git status` show a clean or properly staged state?
- [ ] If an experiment was run, was a corresponding structured report or JSON summary generated?
