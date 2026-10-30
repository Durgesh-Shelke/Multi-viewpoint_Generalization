"""
Unit tests for CLI control tool src/ctl.py.

Validates per SPEC Section 2.1:
1. ctl.py pause creates state/PAUSE and waits for active training process to exit.
2. ctl.py resume deletes stale PAUSE files across state, experiment root, and run output dirs.
3. ctl.py resume refuses to launch if runner is already active.
4. ctl.py status prints formatted overview with step/budget, % done, SPS, ETA, and GPU memory.

Python 3.9 compatible only.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from src.ctl import (
    format_age,
    format_duration,
    generate_overnight_report,
    get_active_runner_pid,
    get_gpu_memory_info,
    is_pid_alive,
    pause,
    report,
    resume,
    status,
)


def test_ctl_formatting_helpers():
    """Verifies duration and age formatting utilities."""
    assert format_duration(0) == "0:00:00"
    assert format_duration(3665) == "1:01:05"
    assert format_duration(75) == "0:01:15"

    assert format_age(15) == "15s ago"
    assert format_age(125) == "2m 5s ago"
    assert format_age(3670) == "1h 1m ago"


def test_ctl_gpu_memory_info():
    """Verifies GPU memory querying via nvidia-smi."""
    info = get_gpu_memory_info()
    assert "raw" in info
    assert isinstance(info["raw"], str)
    if info["total_mib"] is not None:
        assert info["total_mib"] > 0


def test_ctl_pause_creates_pause_file(tmp_path):
    """
    Protocol Requirement:
    ctl.py pause creates state/PAUSE. If no active process, prints confirmation and returns.
    """
    state_dir = tmp_path / "state"
    res = pause(state_dir=state_dir, timeout=5.0)
    assert res is True

    pause_file = state_dir / "PAUSE"
    assert pause_file.is_file()
    with open(pause_file, "r") as f:
        content = f.read()
    assert "PAUSE requested by ctl.py" in content


def test_ctl_pause_waits_for_active_process(tmp_path):
    """
    Protocol Requirement:
    ctl.py pause waits until active process exits cleanly.
    """
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True, exist_ok=True)

    # Spawn a dummy short-lived subprocess (sleeps 1.5s then exits)
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(1.5)"])
    pid = proc.pid

    # Record PID in lockfile
    lock_file = state_dir / "lock"
    with open(lock_file, "w") as f:
        json.dump({"pid": pid, "acquired_at": "test_time"}, f)

    assert is_pid_alive(pid)

    # Call pause: must wait until process exits
    start_t = time.time()
    res = pause(state_dir=state_dir, timeout=10.0)
    elapsed = time.time() - start_t

    assert res is True
    assert elapsed >= 1.0  # Proves it waited for the process
    assert not is_pid_alive(pid)


def test_ctl_resume_cleans_stale_pause_files(tmp_path):
    """
    Protocol Requirement:
    ctl.py resume deletes stale PAUSE files across state, experiment root, and run output dirs.
    """
    state_dir = tmp_path / "state"
    runs_dir = tmp_path / "runs"
    state_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    # Create dummy stale PAUSE files
    state_pause = state_dir / "PAUSE"
    state_pause.write_text("pause\n")

    run1_pause = runs_dir / "run1" / "PAUSE"
    run1_pause.parent.mkdir(parents=True, exist_ok=True)
    run1_pause.write_text("pause\n")

    # Mock subprocess.Popen to prevent actually starting a full runner in this unit test
    with patch("subprocess.Popen") as mock_popen:
        mock_proc = mock_popen.return_value
        mock_proc.pid = 99999

        res = resume(state_dir=state_dir, runs_dir=runs_dir)
        assert res is True

    # Assert all stale PAUSE files were deleted
    assert not state_pause.exists()
    assert not run1_pause.exists()

    # Assert runner PID was written
    pid_file = state_dir / "runner.pid"
    assert pid_file.is_file()
    assert pid_file.read_text().strip() == "99999"


def test_ctl_resume_aborts_if_runner_active(tmp_path):
    """
    Protocol Requirement:
    ctl.py resume aborts if another runner instance is already active.
    """
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True, exist_ok=True)

    # Record current running process PID as active
    lock_file = state_dir / "lock"
    with open(lock_file, "w") as f:
        json.dump({"pid": os.getpid(), "acquired_at": "test_time"}, f)

    res = resume(state_dir=state_dir)
    assert res is False, "Resume should have aborted due to active runner PID"


def test_ctl_status_display(tmp_path, capsys):
    """
    Protocol Requirement:
    ctl.py status prints formatted status block including step, budget, % done, SPS, ETA.
    """
    state_dir = tmp_path / "state"
    runs_dir = tmp_path / "runs"
    state_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    # Create queue.json
    queue_data = {
        "pilot_rgb": {
            "status": "running",
            "modality": "rgb",
            "seed": 100,
            "total_timesteps": 30000000,
            "output_dir": str(runs_dir / "pilot_rgb"),
        }
    }
    with open(state_dir / "queue.json", "w") as f:
        json.dump(queue_data, f)

    # Create progress.csv in output_dir
    run_dir = runs_dir / "pilot_rgb"
    run_dir.mkdir(parents=True, exist_ok=True)
    csv_file = run_dir / "progress.csv"
    with open(csv_file, "w") as f:
        f.write("iteration,global_step,active_time_s,sps,mean_reward,mean_return,mean_length,success_rate,pg_loss,v_loss,entropy,approx_kl,clipfrac,lr\n")
        f.write("5,1000000,500.0,2000.0,0.5,25.0,50.0,0.8,-0.005,0.01,3.5,0.002,0.05,0.0002\n")

    # Run status
    status(state_dir=state_dir, runs_dir=runs_dir)
    captured = capsys.readouterr()

    assert "EXPERIMENT CONTROL STATUS (ctl.py)" in captured.out
    assert "pilot_rgb" in captured.out
    assert "1,000,000 / 30,000,000" in captured.out
    assert "3.3%" in captured.out
    assert "2000 steps/sec" in captured.out


def test_ctl_manual_pause_and_report(tmp_path: Path):
    """Verifies manual pause flag behavior and overnight report generation."""
    from src.ctl import generate_overnight_report
    state_dir = tmp_path / "state"
    runs_dir = tmp_path / "runs"
    state_dir.mkdir(parents=True)
    runs_dir.mkdir(parents=True)

    # 1. Test pause with manual=True creates MANUAL_PAUSE
    pause(state_dir=state_dir, timeout=1.0, manual=True)
    assert (state_dir / "PAUSE").is_file()
    assert (state_dir / "MANUAL_PAUSE").is_file()

    # 2. Test resume clears MANUAL_PAUSE
    with patch("subprocess.Popen") as mock_popen:
        mock_proc = mock_popen.return_value
        mock_proc.pid = 99999
        res = resume(state_dir=state_dir, runs_dir=runs_dir)
        assert res is True
    assert not (state_dir / "PAUSE").exists()
    assert not (state_dir / "MANUAL_PAUSE").exists()

    # 3. Test pause with manual=False does NOT create MANUAL_PAUSE
    pause(state_dir=state_dir, timeout=1.0, manual=False)
    assert (state_dir / "PAUSE").is_file()
    assert not (state_dir / "MANUAL_PAUSE").exists()

    # 4. Test report generation
    report_text = generate_overnight_report(state_dir=state_dir, runs_dir=runs_dir)
    assert "# Overnight Supervision Report" in report_text
    assert (state_dir / "OVERNIGHT_REPORT.md").is_file()


def test_ctl_report_and_status_with_null_budget(tmp_path: Path, capsys):
    """
    Verifies that generate_overnight_report, report, and status handle queues
    containing null-budget (total_timesteps is None or not set) runs gracefully
    without raising TypeError, and format the budget as 'budget not set'.
    """
    state_dir = tmp_path / "state"
    runs_dir = tmp_path / "runs"
    state_dir.mkdir(parents=True)
    runs_dir.mkdir(parents=True)

    # Queue with a mix of budgeted, null-budget, and missing-budget runs
    queue_data = {
        "budgeted_run": {
            "status": "running",
            "modality": "rgb",
            "seed": 100,
            "total_timesteps": 30000000,
            "output_dir": str(runs_dir / "budgeted_run"),
        },
        "null_budget_run": {
            "status": "pending",
            "modality": "rgb",
            "seed": 0,
            "total_timesteps": None,
            "output_dir": str(runs_dir / "null_budget_run"),
        },
        "missing_budget_run": {
            "status": "pending",
            "modality": "rgbd",
            "seed": 1,
            "output_dir": str(runs_dir / "missing_budget_run"),
        },
    }

    queue_file = state_dir / "queue.json"
    with open(queue_file, "w") as f:
        json.dump(queue_data, f)

    # 1. Test generate_overnight_report with null-budget runs
    report_content = generate_overnight_report(state_dir=state_dir, runs_dir=runs_dir)
    assert "# Overnight Supervision Report" in report_content
    assert "| `budgeted_run` | `RUNNING` | rgb | 0 / 30,000,000 | 0.0% |" in report_content
    assert "| `null_budget_run` | `PENDING` | rgb | 0 / budget not set | 0.0% |" in report_content
    assert "| `missing_budget_run` | `PENDING` | rgbd | 0 / budget not set | 0.0% |" in report_content

    # 2. Test report() CLI command output
    capsys.readouterr()  # clear buffer
    report(state_dir=state_dir, runs_dir=runs_dir)
    captured = capsys.readouterr()
    assert "0 / budget not set" in captured.out
    assert "budgeted_run" in captured.out
    assert "null_budget_run" in captured.out

    # 3. Test status() when active/focused run has null budget
    # Update queue so null_budget_run is the active run (running status)
    queue_data["budgeted_run"]["status"] = "done"
    queue_data["null_budget_run"]["status"] = "running"
    with open(queue_file, "w") as f:
        json.dump(queue_data, f)

    status(state_dir=state_dir, runs_dir=runs_dir)
    status_out = capsys.readouterr().out
    assert "Current Run:        null_budget_run" in status_out
    assert "Step / Budget:      0 / budget not set" in status_out
    assert "Progress:           0.0%" in status_out


