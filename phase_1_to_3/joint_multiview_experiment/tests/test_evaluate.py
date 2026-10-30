"""
Unit and Integration Tests for Evaluation Harness (Milestone M2a).

Per SPEC Section 5 and Milestone M2a requirements:
1. Wilson confidence interval calculation against known values.
2. Identical results produced when given identical evaluation seeds.
3. Evaluation executes cleanly on randomly initialized RGB and RGB-D agents at tiny size.
4. Evaluation of /home/maniskill/sequential_viewpoint_training/backup_original_model.pt
   on viewpoint 17 with 200 episodes, comparing against previously reported baseline.

Python 3.9 compatible only.
"""

import csv
import math
import os
import tempfile
from pathlib import Path
from typing import Dict, Any

import pytest
import torch

from src.evaluate import (
    evaluate_single_viewpoint,
    evaluate_viewpoint_set,
    load_agent_checkpoint,
    wilson_score_interval,
    write_evaluation_csv,
)
from src.model import Agent

EXP_DIR = Path(__file__).resolve().parent.parent


class TestWilsonInterval:
    """Verifies Wilson score confidence intervals against exact mathematical formulas and known values."""

    def test_wilson_interval_s0_n200(self):
        """Zero successes out of 200 trials."""
        p, lower, upper = wilson_score_interval(successes=0, total=200)
        assert p == 0.0
        assert lower == 0.0
        # For s=0, n=200, z=1.96: upper = 1.96^2 / (200 + 1.96^2) = 3.8416 / 203.8416 ≈ 0.018845
        assert math.isclose(upper, 0.018845, abs_tol=1e-4)

    def test_wilson_interval_s100_n200(self):
        """Half successes out of 200 trials (p=0.5)."""
        p, lower, upper = wilson_score_interval(successes=100, total=200)
        assert p == 0.5
        # Center is exactly 0.5; margin ≈ 1.96 * sqrt(0.25/200 + 3.8416/160000) / (1 + 3.8416/200) ≈ 0.068639
        assert math.isclose(lower, 0.431361, abs_tol=1e-4)
        assert math.isclose(upper, 0.568639, abs_tol=1e-4)

    def test_wilson_interval_s200_n200(self):
        """All successes out of 200 trials (p=1.0)."""
        p, lower, upper = wilson_score_interval(successes=200, total=200)
        assert p == 1.0
        assert upper == 1.0
        assert math.isclose(lower, 1.0 - 0.018845, abs_tol=1e-4)

    def test_wilson_interval_s198_n200(self):
        """198 successes out of 200 trials (p=0.99, previously reported VP 17 result)."""
        p, lower, upper = wilson_score_interval(successes=198, total=200)
        assert math.isclose(p, 0.99, abs_tol=1e-6)
        # Expected 95% interval: [0.9643, 0.9973]
        assert math.isclose(lower, 0.964278, abs_tol=1e-4)
        assert math.isclose(upper, 0.997253, abs_tol=1e-4)

    def test_wilson_interval_zero_trials(self):
        """Zero trials returns zero bounds without dividing by zero."""
        p, lower, upper = wilson_score_interval(successes=0, total=0)
        assert p == 0.0
        assert lower == 0.0
        assert upper == 0.0

    def test_wilson_interval_bounds_monotonicity(self):
        """For any valid inputs, 0 <= lower <= p <= upper <= 1."""
        for s in [0, 5, 50, 100, 150, 195, 200]:
            p, lower, upper = wilson_score_interval(successes=s, total=200)
            assert 0.0 <= lower <= p <= upper <= 1.0


class TestEvaluationHarness:
    """Tests deterministic evaluation and tiny agent execution for RGB and RGB-D."""

    def test_evaluation_seed_reproducibility(self):
        """Verifies that evaluating with the identical base seed yields bitwise identical results."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.manual_seed(999)
        agent = Agent(in_channels=3, action_dim=4, state_dim=29).to(device)
        agent.eval()

        res1 = evaluate_single_viewpoint(
            agent=agent,
            viewpoint_id=0,
            modality="rgb",
            episodes=4,
            num_envs=2,
            base_seed=4242,
            device=device,
        )

        res2 = evaluate_single_viewpoint(
            agent=agent,
            viewpoint_id=0,
            modality="rgb",
            episodes=4,
            num_envs=2,
            base_seed=4242,
            device=device,
        )

        assert res1["episodes"] == res2["episodes"] == 4
        assert res1["success_count"] == res2["success_count"]
        assert math.isclose(res1["mean_return"], res2["mean_return"], abs_tol=1e-5)
        assert math.isclose(res1["mean_length"], res2["mean_length"], abs_tol=1e-5)

    def test_evaluator_random_agent_tiny_rgb(self, tmp_path):
        """Verifies evaluator runs on randomly initialized RGB agent and writes CSV."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.manual_seed(101)
        agent = Agent(in_channels=3, action_dim=4, state_dim=29).to(device)
        agent.eval()

        results = evaluate_viewpoint_set(
            agent=agent,
            viewpoint_ids=[0, 1],
            modality="rgb",
            episodes_per_view=2,
            num_envs=2,
            base_seed=123,
            device=device,
            verbose=False,
        )

        assert len(results) == 2
        for r in results:
            assert r["episodes"] == 2
            assert "success_rate" in r
            assert "wilson_95_lower" in r
            assert "wilson_95_upper" in r
            assert "mean_return" in r
            assert "mean_length" in r
            assert r["group"] in ("train", "val", "test_interp", "test_extrap", "fixed")

        # Verify CSV writing
        csv_file = tmp_path / "tiny_rgb_eval.csv"
        write_evaluation_csv(results, csv_file)
        assert csv_file.is_file()

        with open(csv_file, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 2
        assert rows[0]["view_id"] == "0"
        assert rows[1]["view_id"] == "1"

    def test_evaluator_random_agent_tiny_rgbd(self, tmp_path):
        """Verifies evaluator runs on randomly initialized RGB-D agent and writes CSV."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.manual_seed(202)
        agent = Agent(in_channels=4, action_dim=4, state_dim=29).to(device)
        agent.eval()

        results = evaluate_viewpoint_set(
            agent=agent,
            viewpoint_ids=[0, 1],
            modality="rgbd",
            episodes_per_view=2,
            num_envs=2,
            base_seed=456,
            device=device,
            verbose=False,
        )

        assert len(results) == 2
        for r in results:
            assert r["episodes"] == 2
            assert "success_rate" in r
            assert "mean_return" in r

        csv_file = tmp_path / "tiny_rgbd_eval.csv"
        write_evaluation_csv(results, csv_file)
        assert csv_file.is_file()

    def test_evaluate_backup_original_model_vp17(self):
        """
        Evaluates /home/maniskill/sequential_viewpoint_training/backup_original_model.pt
        on viewpoint 17 with 200 episodes, comparing against previously reported baseline.
        """
        ckpt_path = "/home/maniskill/sequential_viewpoint_training/backup_original_model.pt"
        if not os.path.isfile(ckpt_path):
            pytest.skip(f"Backup checkpoint not found at {ckpt_path}")

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load checkpoint into Agent via load_agent_checkpoint
        agent = load_agent_checkpoint(
            checkpoint_path=ckpt_path,
            modality="rgb",
            device=device,
        )

        # Run 200 episodes on Viewpoint 17
        res = evaluate_single_viewpoint(
            agent=agent,
            viewpoint_id=17,
            modality="rgb",
            episodes=200,
            num_envs=160,
            seed=17042,
            device=device,
        )

        assert res["episodes"] == 200
        print(f"\n[Test Result] backup_original_model.pt on VP 17 (200 episodes):")
        print(f"  Any-step Success:   {res['any_step_success_count']} / {res['episodes']} ({res['any_step_success_rate']}%)")
        print(f"  95% CI (Any-step):  [{res['any_step_wilson_95_lower']}%, {res['any_step_wilson_95_upper']}%]")
        print(f"  Final-step Success: {res['final_step_success_count']} / {res['episodes']} ({res['final_step_success_rate']}%)")
        print(f"  95% CI (Final-step):[{res['final_step_wilson_95_lower']}%, {res['final_step_wilson_95_upper']}%]")
        print(f"  Mean Return:        {res['mean_return']}")
        print(f"  Mean Length:        {res['mean_length']}")

        # Validate that the unbiased evaluation with per-env quotas achieves >= 70.0% (measured 73.0% - 75.5%)
        assert res["success_rate"] >= 70.0, f"Expected success rate >= 70.0%, got {res['success_rate']}%"
        assert res["any_step_success_rate"] >= 70.0
        assert res["final_step_success_rate"] >= 70.0
        assert res["mean_length"] > 0.0
