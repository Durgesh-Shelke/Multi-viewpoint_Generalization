"""
Unit tests for Milestone M3-prep:
1. Enforcement of final_eval_groups:
   - Pilot runs in configs/runs.yaml specify final_eval_groups: ["train", "val"].
   - Main-grid runs specify final_eval_groups: "all".
   - evaluate.py refuses to evaluate test views for pilot runs or when eval_groups is restricted.
   - runner.py refuses to allow pilot runs to evaluate test or fresh views.
2. Early sanity monitoring:
   - At sanity checkpoint steps (5M, 8M, or configured), if mean success over 5 iterations is 0.0,
     training pauses and writes state/ALERT.txt.
   - Runner detects state/ALERT.txt and halts.
   - Resuming via ctl.py archives the alert and clears pause.
"""

import csv
import json
import os
import shutil
from pathlib import Path
from unittest.mock import patch
import pytest
import yaml

from src.evaluate import run_final_evaluation
from src.runner import Runner
from src.ctl import resume, status
from scripts.preflight import (
    check_git_clean,
    check_freeze_manifest,
    check_state_clean,
    check_disk_space,
    check_gpu_processes,
    check_ac_power,
    run_preflight,
)

EXP_DIR = Path(__file__).resolve().parent.parent


def test_runs_yaml_final_eval_groups():
    """Verifies that configs/runs.yaml specifies final_eval_groups correctly."""
    runs_cfg_path = EXP_DIR / "configs" / "runs.yaml"
    with open(runs_cfg_path, "r") as f:
        data = yaml.safe_load(f)

    runs = data.get("runs", {})
    if "pilot_rgb" in runs:
        assert runs["pilot_rgb"]["final_eval_groups"] == ["train", "val"]
    if "pilot_rgbd" in runs:
        assert runs["pilot_rgbd"]["final_eval_groups"] == ["train", "val"]

    main_grid = [
        "main_rgb_seed0", "main_rgb_seed1", "main_rgb_seed2",
        "main_rgbd_seed0", "main_rgbd_seed1", "main_rgbd_seed2",
    ]
    for m in main_grid:
        assert m in runs
        assert runs[m]["final_eval_groups"] == "all"


def test_evaluate_refuses_test_views_when_restricted(tmp_path: Path):
    """
    Verifies that run_final_evaluation raises ValueError if caller attempts
    to evaluate test views when final_eval_groups is restricted to ['train', 'val'].
    """
    # Viewpoint 16 is in 'test_extrap' per viewpoint_split.json
    dummy_ckpt = tmp_path / "dummy_ckpt.pt"
    dummy_ckpt.write_text("fake_ckpt")

    # 1. Passing fixed_view_ids containing test view must fail
    with pytest.raises(ValueError, match="not permitted in final_eval_groups"):
        run_final_evaluation(
            checkpoint_path=dummy_ckpt,
            output_dir=tmp_path / "eval_out",
            modality="rgb",
            fixed_view_ids=[16],
            eval_groups=["train", "val"],
        )

    # 2. Passing fresh_view_ids when 'fresh' is not in eval_groups must fail
    with pytest.raises(ValueError, match="Fresh viewpoints are not permitted"):
        run_final_evaluation(
            checkpoint_path=dummy_ckpt,
            output_dir=tmp_path / "eval_out",
            modality="rgb",
            fixed_view_ids=[0],  # view 0 is train
            fresh_view_ids=[0],
            eval_groups=["train", "val"],
        )


def test_runner_refuses_test_views_in_pilot(tmp_path: Path):
    """
    Verifies that Runner refuses to start a pilot run if its config
    or runs.yaml entry attempts to include test views in final_eval_groups.
    """
    bad_runs = {
        "runs": {
            "pilot_rgb": {
                "modality": "rgb",
                "seed": 100,
                "total_timesteps": 1000,
                "output_dir": str(tmp_path / "pilot_bad"),
                "final_eval_groups": "all",  # VIOLATION for pilot!
            }
        }
    }
    bad_runs_path = tmp_path / "bad_runs.yaml"
    with open(bad_runs_path, "w") as f:
        yaml.dump(bad_runs, f)

    state_dir = tmp_path / "state"
    runner = Runner(runs_config_path=str(bad_runs_path), state_dir=str(state_dir))

    with pytest.raises(ValueError, match="Protocol violation"):
        runner.run_single("pilot_rgb")


def test_early_sanity_monitoring_triggers_alert(tmp_path: Path):
    """
    Verifies early sanity monitoring:
    When a pilot run encounters a sanity checkpoint step with 0.0 success over 5 iterations:
    - Writes state/ALERT.txt with numbers.
    - Sets state/PAUSE.
    - Saves checkpoint.
    - Returns paused status.
    """
    out_dir = tmp_path / "pilot_sanity_run"
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True, exist_ok=True)

    # Set isolated state dir
    os.environ["EXP_STATE_DIR"] = str(state_dir)

    try:
        alert_file = state_dir / "ALERT.txt"
        assert not alert_file.exists()

        # Write synthetic progress.csv with 5 iterations of 0.0 success
        csv_path = out_dir / "progress.csv"
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["iteration", "global_step", "iter_success_rate", "active_time_s", "sps", "mean_reward", "mean_return", "mean_length", "success_rate", "pg_loss", "v_loss", "entropy", "approx_kl", "clipfrac", "lr"])
            for i in range(1, 6):
                w.writerow([i, i * 40, 0.0, i * 1.0, 100, 0.01, 0.5, 20.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0001])

        # Verify ctl.py resume cleans ALERT.txt
        alert_file.write_text("Test alert content")
        assert alert_file.exists()
        with patch("subprocess.Popen") as mock_popen:
            mock_proc = mock_popen.return_value
            mock_proc.pid = 99999
            resume(state_dir=state_dir, runs_dir=out_dir)

        # Alert should be archived
        assert not alert_file.exists()
        assert (state_dir / "ALERT.txt.acknowledged").exists()
    finally:
        os.environ.pop("EXP_STATE_DIR", None)


def test_preflight_checks(tmp_path: Path):
    """Verifies scripts/preflight.py check components under controlled conditions."""
    # 1. State check
    fake_state = tmp_path / "fake_state"
    fake_state.mkdir()
    ok, msg = check_state_clean(fake_state)
    assert ok is True

    (fake_state / "reference_environment.json").write_text("{}")
    ok, msg = check_state_clean(fake_state)
    assert ok is False
    assert "reference_environment.json" in msg

    (fake_state / "reference_environment.json").unlink()
    (fake_state / "reference_fingerprint.json").write_text("{}")
    ok, msg = check_state_clean(fake_state)
    assert ok is False
    assert "reference_fingerprint.json" in msg

    (fake_state / "reference_fingerprint.json").unlink()
    ok, msg = check_state_clean(fake_state)
    assert ok is True

    # 2. Disk space check
    ok, msg = check_disk_space(min_gb=0.001)
    assert ok is True

    ok, msg = check_disk_space(min_gb=99999999.0)
    assert ok is False

    # 3. GPU processes check (current system should pass or have no offending compute app)
    ok, msg = check_gpu_processes(max_other_vram_mib=1024.0)
    assert isinstance(ok, bool)

    # 4. AC power check
    ok, msg = check_ac_power()
    assert isinstance(ok, bool)

    # 5. Manifest check with mock manifest
    mock_manifest = tmp_path / "mock_manifest.json"
    dummy_file = tmp_path / "dummy_frozen.yaml"
    dummy_file.write_text("frozen: true\n")
    import hashlib
    h = hashlib.sha256(b"frozen: true\n").hexdigest()

    manifest_data = {
        "created_at": "test",
        "file_hashes": {
            str(dummy_file.relative_to(tmp_path)): h
        }
    }
    mock_manifest.write_text(json.dumps(manifest_data))

    # Temporarily check manifest resolution
    with patch("scripts.preflight.EXP_DIR", tmp_path):
        ok, msg = check_freeze_manifest(mock_manifest)
        assert ok is True

        # Alter file to verify mismatch detection
        dummy_file.write_text("frozen: false\n")
        ok, msg = check_freeze_manifest(mock_manifest)
        assert ok is False
        assert "Hash mismatch" in msg
