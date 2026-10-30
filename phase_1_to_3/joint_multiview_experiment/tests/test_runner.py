"""
Unit tests for multi-run runner and source fingerprint integrity guards.

Per SPEC Section 2.4, 2.5:
1. Runner refuses to start any run whose total_timesteps is null.
2. Runner records reference fingerprint in state/reference_fingerprint.json on first run.
3. Runner refuses to start or resume any run if ManiSkill source fingerprint differs.
4. Runner lockfile enforces single active instance and automatically purges stale locks.

Python 3.9 compatible only.
"""

import json
import os
import shutil
import tempfile
import pytest
import yaml
from pathlib import Path
from unittest.mock import patch

from src.fingerprint import (
    ManiSkillFingerprintMismatchError,
    compute_maniskill_source_fingerprint,
    verify_maniskill_source_fingerprint,
    save_reference_fingerprint,
)
from src.runner import (
    Runner,
    LockManager,
    RunnerLockedError,
)


@pytest.fixture
def tmp_state_dir():
    d = tempfile.mkdtemp(prefix="test_runner_state_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


def test_runner_refuses_null_budget_run(tmp_state_dir, tmp_path):
    """
    Protocol Requirement:
    The runner must strictly refuse to start any run with a null or unbudgeted step count.
    """
    dummy_runs_path = tmp_path / "dummy_null_runs.yaml"
    dummy_runs_path.write_text(yaml.dump({
        "runs": {
            "unbudgeted_test_run": {
                "modality": "rgb",
                "seed": 0,
                "total_timesteps": None,
                "output_dir": "runs/unbudgeted_test_run",
                "final_eval_groups": "all",
            }
        }
    }))
    runner = Runner(
        runs_config_path=str(dummy_runs_path),
        base_config_path="configs/ppo_frozen.yaml",
        state_dir=tmp_state_dir,
    )

    with pytest.raises(ValueError) as exc_info:
        runner.run_single("unbudgeted_test_run")

    assert "The budget B has not been frozen yet; refusing to start unbudgeted main-grid run" in str(exc_info.value)


def test_runner_creates_reference_fingerprint(tmp_state_dir):
    """
    Protocol Requirement:
    On initial run, state/reference_fingerprint.json is created atomically
    and stores git HEAD + diff SHA256 of /home/maniskill.
    """
    ref_file = os.path.join(tmp_state_dir, "reference_fingerprint.json")
    assert not os.path.exists(ref_file)

    runner = Runner(
        runs_config_path="configs/runs.yaml",
        base_config_path="configs/ppo_frozen.yaml",
        state_dir=tmp_state_dir,
    )

    fp = runner.check_fingerprint_integrity()
    assert os.path.isfile(ref_file)
    assert "git_head" in fp
    assert "mani_skill_diff_sha256" in fp
    assert "fingerprint" in fp
    assert fp["fingerprint"] == f"{fp['git_head']}:{fp['mani_skill_diff_sha256']}"

    # Verify content in saved file
    with open(ref_file, "r") as f:
        saved_fp = json.load(f)
    assert saved_fp["fingerprint"] == fp["fingerprint"]


def test_runner_refuses_fingerprint_mismatch(tmp_state_dir):
    """
    Protocol Requirement:
    If /home/maniskill source code or commit HEAD differs from reference,
    the runner must raise ManiSkillFingerprintMismatchError and refuse execution.
    """
    ref_file = os.path.join(tmp_state_dir, "reference_fingerprint.json")
    tampered_fp = {
        "git_head": "0000000000000000000000000000000000000000",
        "mani_skill_diff_sha256": "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
        "fingerprint": "0000000000000000000000000000000000000000:ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
    }
    with open(ref_file, "w") as f:
        json.dump(tampered_fp, f)

    runner = Runner(
        runs_config_path="configs/runs.yaml",
        base_config_path="configs/ppo_frozen.yaml",
        state_dir=tmp_state_dir,
    )

    with pytest.raises(ManiSkillFingerprintMismatchError) as exc_info:
        runner.check_fingerprint_integrity()

    assert "ManiSkill source fingerprint mismatch" in str(exc_info.value)
    assert "Refusing to start or resume any run with altered ManiSkill source repository" in str(exc_info.value)


def test_runner_lockfile_concurrency_and_stale_detection(tmp_state_dir):
    """
    Protocol Requirement:
    - Only one runner instance may run at a time (PID lock).
    - Concurrent launch fails with RunnerLockedError.
    - Stale locks from dead PIDs are automatically detected and cleared.
    """
    lock_path = os.path.join(tmp_state_dir, "lock")
    lock1 = LockManager(lock_path=lock_path)
    lock2 = LockManager(lock_path=lock_path)

    # 1. Initial lock acquisition succeeds
    lock1.acquire()
    assert os.path.isfile(lock_path)
    with open(lock_path, "r") as f:
        data = json.load(f)
    assert data["pid"] == os.getpid()

    # 2. Second lock acquisition with active PID must raise RunnerLockedError
    with pytest.raises(RunnerLockedError) as exc_info:
        lock2.acquire()
    assert f"Another runner instance is already active (PID {os.getpid()})" in str(exc_info.value)

    # 3. Clean release
    lock1.release()
    assert not os.path.exists(lock_path)

    # 4. Stale lock recovery: write dead PID
    dead_pid = 9999999
    with open(lock_path, "w") as f:
        json.dump({"pid": dead_pid, "acquired_at": "old_timestamp"}, f)

    # lock2 should detect dead PID, remove stale lock, and acquire cleanly
    lock2.acquire()
    assert os.path.isfile(lock_path)
    with open(lock_path, "r") as f:
        data = json.load(f)
    assert data["pid"] == os.getpid()
    lock2.release()


def test_runner_loads_custom_runs_file(tmp_state_dir):
    """Verifies that Runner with configs/tiny_runs.yaml loads only tiny runs."""
    runner = Runner(
        runs_config_path="configs/tiny_runs.yaml",
        base_config_path="configs/tiny_test.yaml",
        state_dir=tmp_state_dir,
    )
    queue = runner.load_or_init_queue()
    assert "tiny_rgb" in queue
    assert "tiny_rgbd" in queue
    assert "pilot_rgb" not in queue
    assert "main_rgb_seed0" not in queue
    assert queue["tiny_rgb"]["total_timesteps"] == 30000


def test_runner_exp_runs_dir_reroutes_output(tmp_path, monkeypatch):
    """Verifies that EXP_RUNS_DIR re-roots output directories."""
    from src.runner import resolve_output_dir
    fake_runs = str(tmp_path / "exp_manual_runs")
    monkeypatch.setenv("EXP_RUNS_DIR", fake_runs)

    resolved = resolve_output_dir("runs/tiny_rgb")
    assert resolved == os.path.join(fake_runs, "tiny_rgb")

    resolved_bare = resolve_output_dir("tiny_rgb")
    assert resolved_bare == os.path.join(fake_runs, "tiny_rgb")


def test_ctl_runs_file_and_exp_env_forwarding(tmp_path, monkeypatch):
    """Verifies that ctl.py forwards --runs-file and respects EXP_STATE_DIR & EXP_RUNS_DIR."""
    from src.ctl import get_state_dir, get_runs_dir, resolve_output_dir
    manual_state = tmp_path / "manual_state"
    manual_runs = tmp_path / "manual_runs"
    monkeypatch.setenv("EXP_STATE_DIR", str(manual_state))
    monkeypatch.setenv("EXP_RUNS_DIR", str(manual_runs))

    assert get_state_dir() == manual_state
    assert get_runs_dir() == manual_runs
    assert resolve_output_dir("runs/tiny_rgb") == manual_runs / "tiny_rgb"
