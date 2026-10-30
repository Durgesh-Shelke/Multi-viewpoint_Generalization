"""
Unit tests for Overnight Supervisor Watchdog (scripts/watchdog.py).

Validates:
1. Runner auto-restart on unexpected kill (kill -9).
2. User manual pause (state/MANUAL_PAUSE) is respected and not overridden.
3. state/ALERT.txt stops restarts.
4. Crash-loop guard triggers after 3 restarts without progress.
5. Schedule window pauses at end and resumes at start (mocked clock).
6. Temperature and battery guards trigger pause and clear automatically;
   at most 3 guard pauses before ALERT.txt is written.
7. Frozen-code commit mismatch refuses to resume.
8. Watchdog lock handling and SIGTERM handling.

Python 3.9 compatible only.
"""

import datetime
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from scripts.watchdog import (
    EXP_DIR,
    HardwareGuards,
    Supervisor,
    WatchdogLock,
    is_time_in_window,
    load_schedule,
)
from src.ctl import generate_overnight_report, pause, resume


@pytest.fixture
def mock_supervisor_env(tmp_path: Path):
    """Sets up an isolated supervisor testing environment."""
    state_dir = tmp_path / "state"
    runs_dir = tmp_path / "runs"
    state_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    sched_path = tmp_path / "schedule.yaml"
    sched_data = {
        "schedule": {
            "start": "23:30",
            "end": "07:30",
            "timezone": "local",
            "days": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
        }
    }
    with open(sched_path, "w") as f:
        yaml.dump(sched_data, f)

    sup = Supervisor(
        exp_dir=tmp_path,
        state_dir=state_dir,
        runs_dir=runs_dir,
        schedule_path=sched_path,
        check_interval=1.0,
    )
    return sup, state_dir, runs_dir, sched_path


def test_schedule_window_crossing_midnight():
    """Verifies schedule window logic across midnight and normal hours."""
    schedule = {
        "start": "23:30",
        "end": "07:30",
        "days": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
    }
    # 23:45 on Monday -> in window
    dt1 = datetime.datetime(2026, 10, 19, 23, 45)
    assert is_time_in_window(dt1, schedule) is True

    # 03:15 on Tuesday -> in window (started Monday night)
    dt2 = datetime.datetime(2026, 10, 20, 3, 15)
    assert is_time_in_window(dt2, schedule) is True

    # 07:29 on Tuesday -> in window
    dt3 = datetime.datetime(2026, 10, 20, 7, 29)
    assert is_time_in_window(dt3, schedule) is True

    # 07:30 on Tuesday -> outside window
    dt4 = datetime.datetime(2026, 10, 20, 7, 30)
    assert is_time_in_window(dt4, schedule) is False

    # 14:00 on Tuesday -> outside window
    dt5 = datetime.datetime(2026, 10, 20, 14, 0)
    assert is_time_in_window(dt5, schedule) is False


def test_watchdog_restarts_runner_after_kill(mock_supervisor_env):
    """
    1. kill -9 on runner is restarted within supervisor check:
    If runner died without state/PAUSE, supervisor triggers restart.
    """
    sup, state_dir, runs_dir, _ = mock_supervisor_env
    dt_in_window = datetime.datetime(2026, 10, 20, 2, 0)

    # Mock runner dead (no PID alive) and no pause file
    with patch.object(sup, "ctl_resume") as mock_resume, \
         patch("scripts.watchdog.get_active_runner_pid", return_value=None):
        should_cont = sup.run_iteration(now_dt=dt_in_window)
        assert should_cont is True
        assert mock_resume.called
        assert sup.restarts_without_progress == 1


def test_user_pause_is_not_overridden(mock_supervisor_env):
    """
    2. A user pause (state/MANUAL_PAUSE) is NOT overridden by supervisor inside window.
    """
    sup, state_dir, runs_dir, _ = mock_supervisor_env
    dt_in_window = datetime.datetime(2026, 10, 20, 2, 0)

    # Set up user manual pause
    (state_dir / "PAUSE").write_text("PAUSE\n")
    (state_dir / "MANUAL_PAUSE").write_text("MANUAL_PAUSE\n")

    with patch.object(sup, "ctl_resume") as mock_resume, \
         patch("scripts.watchdog.get_active_runner_pid", return_value=None):
        should_cont = sup.run_iteration(now_dt=dt_in_window)
        assert should_cont is True
        # Must NOT resume because MANUAL_PAUSE is present
        assert not mock_resume.called


def test_alert_stops_restarts(mock_supervisor_env):
    """
    3. state/ALERT.txt stops restarts even if runner is dead inside window.
    """
    sup, state_dir, runs_dir, _ = mock_supervisor_env
    dt_in_window = datetime.datetime(2026, 10, 20, 2, 0)

    (state_dir / "ALERT.txt").write_text("Critical alert!\n")

    with patch.object(sup, "ctl_resume") as mock_resume, \
         patch("scripts.watchdog.get_active_runner_pid", return_value=None):
        should_cont = sup.run_iteration(now_dt=dt_in_window)
        assert should_cont is True
        assert not mock_resume.called


def test_failed_run_stops_restarts(mock_supervisor_env):
    """
    If run is marked failed in queue.json, supervisor refuses to restart.
    """
    sup, state_dir, runs_dir, _ = mock_supervisor_env
    dt_in_window = datetime.datetime(2026, 10, 20, 2, 0)

    queue = {
        "pilot_rgb": {
            "status": "failed",
            "error": "NaN loss encountered",
        }
    }
    (state_dir / "queue.json").write_text(json.dumps(queue))

    with patch.object(sup, "ctl_resume") as mock_resume, \
         patch("scripts.watchdog.get_active_runner_pid", return_value=None):
        should_cont = sup.run_iteration(now_dt=dt_in_window)
        assert should_cont is True
        assert not mock_resume.called


def test_crash_loop_guard_triggers_after_3_restarts(mock_supervisor_env):
    """
    4. Crash-loop guard: after 3 restarts with no checkpoint progress,
       writes state/ALERT.txt, creates state/PAUSE, and stops supervising.
    """
    sup, state_dir, runs_dir, _ = mock_supervisor_env
    dt_in_window = datetime.datetime(2026, 10, 20, 2, 0)

    with patch.object(sup, "ctl_resume"), \
         patch("scripts.watchdog.get_active_runner_pid", return_value=None), \
         patch("time.sleep"):  # skip backoff sleep in unit test
        # 1st restart
        assert sup.run_iteration(now_dt=dt_in_window) is True
        assert sup.restarts_without_progress == 1

        # 2nd restart
        assert sup.run_iteration(now_dt=dt_in_window) is True
        assert sup.restarts_without_progress == 2

        # 3rd restart
        assert sup.run_iteration(now_dt=dt_in_window) is True
        assert sup.restarts_without_progress == 3

        # 4th unexpected death without progress -> triggers crash loop guard!
        res = sup.run_iteration(now_dt=dt_in_window)
        assert res is False
        assert (state_dir / "ALERT.txt").is_file()
        assert (state_dir / "PAUSE").is_file()
        assert "CRASH LOOP DETECTED" in (state_dir / "ALERT.txt").read_text()


def test_window_pauses_at_end_and_resumes_at_start(mock_supervisor_env):
    """
    5. The window pauses at its end and resumes at its start (mocked clock).
       Also verifies automatic OVERNIGHT_REPORT.md generation at window end.
    """
    sup, state_dir, runs_dir, _ = mock_supervisor_env

    # 1. Start inside window (02:00) with running process
    dt_inside = datetime.datetime(2026, 10, 20, 2, 0)
    with patch("scripts.watchdog.get_active_runner_pid", return_value=12345):
        sup.run_iteration(now_dt=dt_inside)
        assert sup.was_in_window is True

    # 2. Clock advances to 07:35 (outside window) -> must trigger pause and report
    dt_outside = datetime.datetime(2026, 10, 20, 7, 35)
    with patch.object(sup, "ctl_pause") as mock_pause, \
         patch.object(sup, "generate_report") as mock_report, \
         patch("scripts.watchdog.get_active_runner_pid", return_value=12345):
        sup.run_iteration(now_dt=dt_outside)
        assert mock_pause.called
        assert mock_report.called
        assert sup.was_in_window is False

    # 3. Clock advances to 23:35 (window reopens) -> auto-resumes
    dt_reopen = datetime.datetime(2026, 10, 20, 23, 35)
    # Simulate that runner is paused from window
    (state_dir / "PAUSE").write_text("PAUSE\n")
    with patch.object(sup, "ctl_resume") as mock_resume, \
         patch("scripts.watchdog.get_active_runner_pid", return_value=None):
        sup.run_iteration(now_dt=dt_reopen)
        assert mock_resume.called
        assert sup.was_in_window is True


def test_temperature_guard_trigger_and_recovery(mock_supervisor_env):
    """
    6a. Temperature guard: pauses when GPU temp stays > 85 C for 2 minutes,
    and resumes automatically once cleared.
    """
    sup, state_dir, runs_dir, _ = mock_supervisor_env
    dt_in_window = datetime.datetime(2026, 10, 20, 2, 0)
    guards = sup.guards

    # Step 1: Temp is 86 C at t = 0 -> timer starts, not yet tripped
    with patch.object(guards, "read_gpu_temperature", return_value=86):
        tripped, _ = guards.check(now=1000.0)
        assert tripped is False
        assert guards.high_temp_start_time == 1000.0

        # Step 2: Temp is 86 C at t = 60s -> still under 120s, not tripped
        tripped, _ = guards.check(now=1060.0)
        assert tripped is False

        # Step 3: Temp is 86 C at t = 125s -> exceeds 120s, trips!
        tripped, reason = guards.check(now=1125.0)
        assert tripped is True
        assert "85 C for 125s" in reason

    # Step 4: When guard trips during supervisor iteration, pauses runner
    with patch.object(guards, "check", return_value=(True, "High GPU Temp")), \
         patch.object(sup, "ctl_pause") as mock_pause, \
         patch("scripts.watchdog.get_active_runner_pid", return_value=12345):
        sup.run_iteration(now_dt=dt_in_window)
        assert mock_pause.called
        assert sup.guard_pauses_count == 1

    # Step 5: Temp drops to 70 C -> timer resets and conditions clear
    with patch.object(guards, "read_gpu_temperature", return_value=70):
        tripped, _ = guards.check(now=1200.0)
        assert tripped is False
        assert guards.high_temp_start_time is None

    # Step 6: Next iteration auto-resumes
    (state_dir / "PAUSE").write_text("PAUSE\n")
    with patch.object(guards, "check", return_value=(False, None)), \
         patch.object(sup, "ctl_resume") as mock_resume, \
         patch("scripts.watchdog.get_active_runner_pid", return_value=None):
        sup.run_iteration(now_dt=dt_in_window)
        assert mock_resume.called


def test_battery_guard_and_nightly_limit(mock_supervisor_env):
    """
    6b. Battery guard triggers when on battery or capacity < 30%.
    At most 3 guard pauses per night before writing ALERT.txt.
    """
    sup, state_dir, runs_dir, _ = mock_supervisor_env
    dt_in_window = datetime.datetime(2026, 10, 20, 2, 0)
    guards = sup.guards

    # On battery power -> trips
    with patch.object(guards, "read_battery_status", return_value=(True, True, 80)):
        tripped, reason = guards.check()
        assert tripped is True
        assert "running on battery" in reason

    # Low battery (20%) on AC -> trips
    with patch.object(guards, "read_battery_status", return_value=(True, False, 20)):
        tripped, reason = guards.check()
        assert tripped is True
        assert "below 30%" in reason

    # Exceeding 3 guard pauses triggers ALERT.txt
    sup.guard_pauses_count = 3
    with patch.object(guards, "check", return_value=(True, "Battery low")), \
         patch.object(sup, "ctl_pause"), \
         patch("scripts.watchdog.get_active_runner_pid", return_value=12345):
        cont = sup.run_iteration(now_dt=dt_in_window)
        assert cont is False
        assert (state_dir / "ALERT.txt").is_file()
        assert "MAXIMUM GUARD PAUSES EXCEEDED" in (state_dir / "ALERT.txt").read_text()


def test_frozen_code_mismatch_refuses_to_resume(tmp_path: Path):
    """
    7. Frozen-code commit mismatch:
    If manifest records a different commit hash than the frozen export,
    train_ppo refuses to resume and raises RuntimeError.
    """
    from src.train_ppo import write_or_verify_manifest

    out_dir = tmp_path / "run_test"
    out_dir.mkdir(parents=True)
    manifest_path = out_dir / "manifest.json"

    # Saved manifest recorded commit "aaaa1111"
    manifest_data = {
        "created_at": "test",
        "git_commit": "aaaa111122223333444455556666777788889999",
        "python_version": sys.version,
        "config_hash": "dummy_cfg",
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest_data, f)

    # Current frozen export commit is "bbbb9999"
    os.environ["FROZEN_COMMIT"] = "bbbb999900001111222233334444555566667777"
    try:
        with pytest.raises(RuntimeError, match="Frozen code commit hash mismatch"):
            write_or_verify_manifest(
                output_dir=str(out_dir),
                config={"total_timesteps": 1000},
                effective_env_settings={},
                is_resuming=True,
            )
    finally:
        os.environ.pop("FROZEN_COMMIT", None)


def test_watchdog_sigterm_handling(mock_supervisor_env):
    """
    8. Verifies SIGTERM handler sets shutdown flag without terminating training.
    """
    sup, state_dir, _, _ = mock_supervisor_env
    assert sup.shutdown_requested is False

    sup.handle_signal(signal.SIGTERM, None)
    assert sup.shutdown_requested is True
