# Project rules (read SPEC.md for the full protocol)

- SPEC.md is the source of truth. Never change the viewpoint split, hyperparameters,
  step budget, seeds, or preprocessing on your own. Log proposals in DECISIONS.md and ask me.
- This is my personal laptop. Training must always be pausable and resumable (SPEC Section 2).
- NEVER run training in the foreground. Launch long jobs detached via ctl.py / runner.py.
- NEVER use broad kill commands (pkill python, killall). Only kill PIDs that you started
  and recorded.
- Keep VRAM under ~6 GB. Run training with nice -n 10.
- Do not start any run longer than ~5 minutes until the pause/resume tests pass and I approve.
- Never modify or delete files outside the experiment directory.
- Update PROGRESS.md after every milestone and log assumptions in DECISIONS.md.
- Start every new session by reading AGENTS.md, SPEC.md, PROGRESS.md, DECISIONS.md.
- Python is 3.9. Do not use `X | Y` type unions, `match` statements, or other 3.10+ syntax.
- Run every test and show the real output before claiming it passes.
- Never run git checkout/restore/reset/clean/stash in /home/maniskill (parent ManiSkill repo has uncommitted user changes). Run git commands only inside the experiment folder.
- The Python environment is conda base and is frozen (configs/requirements_frozen.txt). Never run pip install, pip uninstall, pip upgrade, or conda install/update. If a new package seems necessary, stop and ask me.
