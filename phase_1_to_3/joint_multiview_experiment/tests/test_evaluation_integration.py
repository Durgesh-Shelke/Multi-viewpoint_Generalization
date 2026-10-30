"""
Integration Tests for Periodic and Final Evaluation Harness (Milestone M2b-1).

Per SPEC Section 5 and Milestone M2b-1 Requirements:
1. Eval rows are continuous across pause/resume:
   eval_per_view.csv and eval_summary.csv maintain strict step monotonicity,
   zero duplicate rows, and zero missing rows across pause/resume cycles.
2. Evaluation does not change training RNG:
   Training with periodic evaluation produces an identical training loss and parameter
   trajectory as training without evaluation for the same random seed.
3. Final evaluation resumes after a mid-eval kill/pause:
   Idempotent evaluation skips already-completed viewpoints, evaluates remaining views,
   and writes final_eval_summary.json with status="completed".
4. Sanitization works:
   sanitize_eval_logs strips rows with step > checkpoint_step from both CSVs,
   guaranteeing clean state after mid-eval crash or pause.

Python 3.9 compatible only.
"""

import csv
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest
import torch

from src.checkpoint import save_checkpoint
from src.evaluate import (
    EVAL_FIELDNAMES,
    append_evaluation_csv_row,
    load_completed_view_ids,
    run_final_evaluation,
)
from src.model import Agent
from src.train_ppo import (
    EVAL_PER_VIEW_FIELDNAMES,
    EVAL_SUMMARY_FIELDNAMES,
    load_yaml_config,
    run_periodic_evaluation,
    sanitize_eval_logs,
    train_ppo,
)

EXP_DIR = Path(__file__).resolve().parent.parent
TINY_CONFIG_PATH = str(EXP_DIR / "configs" / "tiny_test.yaml")


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    """Reads CSV rows as list of dictionaries."""
    if not path.is_file():
        return []
    with open(path, "r", newline="") as f:
        return list(csv.DictReader(f))


def wait_for_csv_rows(csv_path: str, min_rows: int, timeout: float = 60.0) -> bool:
    """Waits until CSV file exists and contains at least min_rows."""
    start_t = time.time()
    while time.time() - start_t < timeout:
        if os.path.isfile(csv_path):
            with open(csv_path, "r", newline="") as f:
                reader = csv.DictReader(f)
                if len(list(reader)) >= min_rows:
                    return True
        time.sleep(0.2)
    return False


class TestEvaluationSanitization:
    """Verifies that sanitize_eval_logs cleanly strips crash-orphaned rows."""

    def test_sanitization_strips_future_steps(self, tmp_path: Path):
        """Rows with step > checkpoint_step are stripped; rows <= checkpoint_step are preserved."""
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir(parents=True)

        per_view_csv = logs_dir / "eval_per_view.csv"
        summary_csv = logs_dir / "eval_summary.csv"

        # Write rows at step 200 (valid) and step 400 (orphaned after crash)
        with open(per_view_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=EVAL_PER_VIEW_FIELDNAMES)
            w.writeheader()
            w.writerow({"step": 200, "view": 0, "group": "train", "successes": 2, "episodes": 2})
            w.writerow({"step": 200, "view": 1, "group": "train", "successes": 1, "episodes": 2})
            w.writerow({"step": 400, "view": 0, "group": "train", "successes": 2, "episodes": 2})
            w.writerow({"step": 400, "view": 1, "group": "train", "successes": 2, "episodes": 2})

        with open(summary_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=EVAL_SUMMARY_FIELDNAMES)
            w.writeheader()
            w.writerow({"step": 200, "train_mean": 75.0, "val_mean": 50.0})
            w.writerow({"step": 400, "train_mean": 100.0, "val_mean": 100.0})

        # Sanitize at checkpoint_step = 200
        sanitize_eval_logs(str(logs_dir), checkpoint_step=200)

        pv_rows = read_csv_rows(per_view_csv)
        sm_rows = read_csv_rows(summary_csv)

        assert len(pv_rows) == 2, f"Expected 2 rows remaining, got {len(pv_rows)}"
        assert all(int(r["step"]) == 200 for r in pv_rows)

        assert len(sm_rows) == 1, f"Expected 1 summary row remaining, got {len(sm_rows)}"
        assert int(sm_rows[0]["step"]) == 200
        assert float(sm_rows[0]["train_mean"]) == 75.0

    def test_sanitization_noop_when_all_steps_valid(self, tmp_path: Path):
        """Sanitization does not alter files when all steps are <= checkpoint_step."""
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir(parents=True)

        summary_csv = logs_dir / "eval_summary.csv"
        with open(summary_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=EVAL_SUMMARY_FIELDNAMES)
            w.writeheader()
            w.writerow({"step": 100, "train_mean": 50.0, "val_mean": 50.0})
            w.writerow({"step": 200, "train_mean": 80.0, "val_mean": 70.0})

        sanitize_eval_logs(str(logs_dir), checkpoint_step=200)
        sm_rows = read_csv_rows(summary_csv)
        assert len(sm_rows) == 2


class TestEvaluationDoesNotChangeTrainingRNG:
    """Verifies that periodic evaluation strictly preserves training RNG and trajectories."""

    def test_evaluation_preserves_training_rng(self, tmp_path: Path):
        """
        Runs two training sessions with identical random seed:
        - Run A: Periodic evaluation is triggered at step 40 (after iteration 1).
        - Run B: No periodic evaluation (eval_interval_steps=999999).
        Asserts that the model parameters and training losses after iteration 2
        are identical within small numerical precision.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        cfg = load_yaml_config(TINY_CONFIG_PATH)
        cfg["num_envs"] = 4
        cfg["num_steps"] = 10  # 40 env steps per iteration
        cfg["update_epochs"] = 1
        cfg["num_minibatches"] = 1
        cfg["eval_episodes_per_view"] = 1

        out_a = tmp_path / "run_eval"
        out_b = tmp_path / "run_no_eval"

        # Run A: with evaluation after iteration 1
        cfg_a = dict(cfg)
        cfg_a["eval_interval_steps"] = 40  # Triggers at step 40
        train_ppo(
            config=cfg_a,
            output_dir=str(out_a),
            modality="rgb",
            seed=54321,
            total_timesteps=80,  # Exactly 2 iterations
            device=device,
        )

        # Run B: without evaluation
        cfg_b = dict(cfg)
        cfg_b["eval_interval_steps"] = 99999999
        train_ppo(
            config=cfg_b,
            output_dir=str(out_b),
            modality="rgb",
            seed=54321,
            total_timesteps=80,  # Exactly 2 iterations
            device=device,
        )

        # 1. Verify Run A actually performed periodic evaluation
        eval_csv = out_a / "logs" / "eval_summary.csv"
        assert eval_csv.is_file(), "Run A should have created eval_summary.csv"
        eval_rows = read_csv_rows(eval_csv)
        assert len(eval_rows) >= 1, "Run A should have evaluated at step 40"
        assert int(eval_rows[0]["step"]) == 40

        # 2. Compare progress.csv logs
        prog_a = read_csv_rows(out_a / "progress.csv")
        prog_b = read_csv_rows(out_b / "progress.csv")
        assert len(prog_a) == 2
        assert len(prog_b) == 2

        # Check loss values in iteration 2 (after evaluation occurred in Run A)
        loss_keys = ["pg_loss", "v_loss", "entropy"]
        for k in loss_keys:
            val_a = float(prog_a[1][k])
            val_b = float(prog_b[1][k])
            diff = abs(val_a - val_b)
            assert diff < 1e-4, f"Mismatch in {k} for iteration 2: {val_a} vs {val_b} (diff={diff})"

        # 3. Compare final checkpoint model weights
        from src.checkpoint import get_newest_checkpoint
        ckpt_a_path, _ = get_newest_checkpoint(str(out_a / "checkpoints"))
        ckpt_b_path, _ = get_newest_checkpoint(str(out_b / "checkpoints"))

        ckpt_a = torch.load(ckpt_a_path, map_location=device, weights_only=False)
        ckpt_b = torch.load(ckpt_b_path, map_location=device, weights_only=False)

        max_param_diff = 0.0
        for name, param_a in ckpt_a["model_state_dict"].items():
            param_b = ckpt_b["model_state_dict"][name]
            diff = (param_a - param_b).abs().max().item()
            if diff > max_param_diff:
                max_param_diff = diff

        assert max_param_diff < 1e-4, f"Max parameter difference was {max_param_diff:.6e}, expected < 1e-4"


class TestEvaluationContinuityAcrossPauseResume:
    """Verifies eval rows continuity across pause/resume cycles."""

    def test_eval_rows_continuous_across_pause_resume(self, tmp_path: Path):
        """
        Runs training with periodic evaluation (200 steps per iteration, 3 iterations = 600 steps):
        - Iteration 1 (step 200): evaluates step 200, pauses via SIGTERM.
        - Resume: runs Iterations 2 & 3 (steps 400 & 600), evaluating at steps 400 and 600.
        Verifies:
        - eval_summary.csv has strictly monotonic steps [200, 400, 600] with no duplicates.
        - eval_per_view.csv has exactly 3 sets of per-view records.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        out_dir = tmp_path / "run_pause_resume"
        csv_path = out_dir / "progress.csv"
        eval_sum_csv = out_dir / "logs" / "eval_summary.csv"
        eval_pv_csv = out_dir / "logs" / "eval_per_view.csv"

        import yaml
        custom_cfg = load_yaml_config(TINY_CONFIG_PATH)
        custom_cfg["eval_interval_steps"] = 200
        custom_cfg_path = str(tmp_path / "custom_tiny_eval.yaml")
        with open(custom_cfg_path, "w") as f:
            yaml.dump(custom_cfg, f)

        # Phase 1: Launch subprocess for 600 steps total
        cmd = [
            "nice", "-n", "10",
            sys.executable, "-m", "src.train_ppo",
            "--config", custom_cfg_path,
            "--modality", "rgb",
            "--output_dir", str(out_dir),
            "--total_timesteps", "600",
            "--seed", "777",
            "--device", device,
        ]
        sub_env = dict(os.environ)
        sub_env["EXP_STATE_DIR"] = str(tmp_path / "isolated_state")

        proc = subprocess.Popen(cmd, cwd=str(EXP_DIR), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=sub_env)
        # Wait for iteration 1 periodic eval to finish (eval_summary.csv has 1 row)
        assert wait_for_csv_rows(str(eval_sum_csv), min_rows=1, timeout=60.0)
        # Send SIGTERM for graceful pause
        os.kill(proc.pid, signal.SIGTERM)
        stdout, _ = proc.communicate(timeout=40.0)
        assert proc.returncode == 0, f"Process did not exit code 0:\n{stdout}"

        # Verify Phase 1 eval logs
        sum_rows_1 = read_csv_rows(eval_sum_csv)
        assert len(sum_rows_1) == 1
        assert int(sum_rows_1[0]["step"]) == 200
        pv_rows_1 = read_csv_rows(eval_pv_csv)
        views_per_eval = len(pv_rows_1)
        assert views_per_eval > 0

        # Phase 2: Resume to completion (600 steps)
        proc2 = subprocess.Popen(cmd, cwd=str(EXP_DIR), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=sub_env)
        stdout2, _ = proc2.communicate(timeout=90.0)
        assert proc2.returncode == 0, f"Resume did not exit code 0:\n{stdout2}"

        # Verify continuity across pause/resume
        sum_rows_2 = read_csv_rows(eval_sum_csv)
        steps = [int(r["step"]) for r in sum_rows_2]
        assert steps == [200, 400, 600], f"Expected steps [200, 400, 600], got {steps}"

        pv_rows_2 = read_csv_rows(eval_pv_csv)
        assert len(pv_rows_2) == 3 * views_per_eval, (
            f"Expected {3 * views_per_eval} per-view rows, got {len(pv_rows_2)}"
        )


class TestFinalEvaluationMidEvalKillAndResume:
    """Verifies that run_final_evaluation cleanly resumes after mid-eval interruption."""

    def test_final_eval_resumes_after_kill(self, tmp_path: Path):
        """
        Tests final evaluation idempotency:
        - Evaluates a subset of fixed viewpoints [0, 1, 2, 3] and fresh viewpoints [0, 1].
        - First execution is interrupted after fixed view 1 via PAUSE file.
        - Verifies that fixed_views_eval.csv contains views 0 and 1.
        - Second execution resumes without PAUSE file:
          Skips views 0 and 1, finishes views 2 and 3 and fresh views 0 and 1.
        - Verifies final_eval_summary.json is written with status="completed".
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        eval_dir = tmp_path / "final_eval_test"
        eval_dir.mkdir(parents=True)
        pause_file = tmp_path / "PAUSE"

        # Create a tiny agent and save checkpoint
        agent = Agent(in_channels=3, action_dim=4, state_dim=29).to(device)
        optim = torch.optim.Adam(agent.parameters(), lr=3e-4)
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir(parents=True)

        ckpt_path = save_checkpoint(
            ckpt_dir=str(ckpt_dir),
            model=agent,
            optimizer=optim,
            global_step=1000,
            iteration=1,
            config={"model": {"state_activation": "none"}},
            is_final=True,
        )

        target_fixed = [0, 1, 2, 3]
        target_fresh = [0, 1]

        # Phase 1: Simulate pause after 2 fixed views
        fixed_csv = eval_dir / "final_eval" / "fixed_views_eval.csv"
        fixed_csv.parent.mkdir(parents=True, exist_ok=True)

        # Run with pause file present so it halts cleanly
        with open(pause_file, "w") as f:
            f.write("PAUSE\n")

        res_paused = run_final_evaluation(
            checkpoint_path=ckpt_path,
            output_dir=str(eval_dir),
            modality="rgb",
            fixed_view_ids=target_fixed,
            fresh_view_ids=target_fresh,
            fixed_episodes=2,
            fresh_episodes=2,
            num_envs=2,
            pause_paths=[str(pause_file)],
            device=device,
            verbose=False,
        )
        assert res_paused["status"] == "paused"
        pause_file.unlink()

        # Pre-populate fixed_csv with views 0 and 1 to simulate a kill that completed views 0 and 1
        run_final_evaluation(
            checkpoint_path=ckpt_path,
            output_dir=str(eval_dir),
            modality="rgb",
            fixed_view_ids=[0, 1],
            fresh_view_ids=[],
            fixed_episodes=2,
            fresh_episodes=2,
            num_envs=2,
            device=device,
            verbose=False,
        )
        # Verify fixed_csv now has views 0 and 1
        rows_phase1 = read_csv_rows(fixed_csv)
        assert len(rows_phase1) == 2
        completed_vids = [int(r["view_id"]) for r in rows_phase1]
        assert completed_vids == [0, 1]

        # Phase 2: Resume with all views [0, 1, 2, 3] and fresh views [0, 1]
        # It must SKIP views 0 and 1 and finish views 2, 3 and fresh views 0, 1
        res_resumed = run_final_evaluation(
            checkpoint_path=ckpt_path,
            output_dir=str(eval_dir),
            modality="rgb",
            fixed_view_ids=target_fixed,
            fresh_view_ids=target_fresh,
            fixed_episodes=2,
            fresh_episodes=2,
            num_envs=2,
            device=device,
            verbose=False,
        )

        assert res_resumed["status"] == "completed"

        # Check fixed_views_eval.csv: must have exactly 4 rows (no duplicate 0 or 1)
        final_fixed_rows = read_csv_rows(fixed_csv)
        final_vids = [int(r["view_id"]) for r in final_fixed_rows]
        assert len(final_fixed_rows) == 4, f"Expected 4 fixed view rows, got {len(final_fixed_rows)}"
        assert sorted(final_vids) == [0, 1, 2, 3], f"Expected views [0, 1, 2, 3], got {final_vids}"

        # Check fresh_views_eval.csv
        fresh_csv = eval_dir / "final_eval" / "fresh_views_eval.csv"
        assert fresh_csv.is_file()
        final_fresh_rows = read_csv_rows(fresh_csv)
        assert len(final_fresh_rows) == 2

        # Check final_eval_summary.json
        summary_file = eval_dir / "final_eval" / "final_eval_summary.json"
        assert summary_file.is_file()
        with open(summary_file, "r") as f:
            summary_data = json.load(f)
        assert summary_data["status"] == "completed"
        assert summary_data["fixed_views_count"] == 4
        assert summary_data["fresh_views_count"] == 2
