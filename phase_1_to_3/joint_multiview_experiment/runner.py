#!/usr/bin/env python3
"""
Multi-Run Training Orchestrator Entrypoint forwarding to src/runner.py.
"""
import os
import sys
from pathlib import Path

# If frozen code directory exists and not already in PYTHONPATH, prefer it
frozen_dir = Path(__file__).resolve().parent / "frozen_code" / "v2-freeze"
if frozen_dir.is_dir() and str(frozen_dir) not in sys.path:
    sys.path.insert(0, str(frozen_dir))
    os.environ["FROZEN_CODE_DIR"] = str(frozen_dir)
    commit_file = frozen_dir / "COMMIT_HASH"
    if commit_file.is_file():
        os.environ["FROZEN_COMMIT"] = commit_file.read_text().strip()

from src.runner import main

if __name__ == "__main__":
    main()
