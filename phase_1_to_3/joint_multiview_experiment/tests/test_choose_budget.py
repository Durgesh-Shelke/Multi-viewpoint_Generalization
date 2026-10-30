"""
Unit and Synthetic Curve Tests for Budget Selection (src/choose_budget.py).

SPEC Section 4.3 requirements:
- Read eval_summary.csv of both pilots.
- 5-evaluation moving average of mean train success.
- Plateau step S: first step after which moving average improvement < 2 percentage points per further 5M steps.
- Later S of the two arms: S = max(S_rgb, S_rgbd).
- B = round_up(1.5 * S, to nearest 5M).
- If no plateau by 30M: B = 45M, requires user confirmation.
- If B > 30M: requires user confirmation.
- Never overwrite existing configs/budget.json without --force.
- Tested on at least 5 synthetic curves: early plateau, late plateau, no plateau, noisy, non-monotonic.
"""

import csv
import json
import pytest
from pathlib import Path
from typing import List, Tuple

from src.choose_budget import (
    compute_moving_average,
    find_plateau_step,
    compute_budget_from_plateaus,
    round_up_to_multiple,
    choose_budget,
    load_eval_summary,
)


def make_eval_csv(path: Path, steps: List[int], train_means: List[float]):
    """Helper creating synthetic eval_summary.csv."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "train_mean", "val_mean"])
        for s, tm in zip(steps, train_means):
            writer.writerow([s, tm, tm * 0.9])


# ======================================================================
# 1. Curve 1: Early Plateau (climbs to 70% by 8M, then flat to 30M)
# ======================================================================
def test_synthetic_curve_early_plateau():
    steps = [i * 1_000_000 for i in range(1, 31)]  # 1M to 30M
    train_means = [
        0.05, 0.15, 0.30, 0.45, 0.58, 0.65, 0.69, 0.70,  # 1M - 8M (climb)
        0.702, 0.701, 0.703, 0.702, 0.704, 0.705, 0.704, 0.706,
        0.705, 0.707, 0.706, 0.708, 0.707, 0.708, 0.709, 0.708,
        0.709, 0.710, 0.709, 0.710, 0.711, 0.710          # 9M - 30M (flat < 1% gain total)
    ]
    pcts = [v * 100.0 for v in train_means]
    ma = compute_moving_average(pcts, window=5)
    s = find_plateau_step(steps, ma)

    assert s is not None, "Plateau should be detected for early plateau curve"
    # 5-eval MA absorbs plateau at 10M steps
    assert s == 10_000_000, f"Expected MA plateau at 10M, got {s}"

    b, s_comb, req_conf, reason = compute_budget_from_plateaus(s, s)
    # 1.5 * 10M = 15M -> rounded up to 15M
    assert b == 15_000_000, f"Expected B = 15M, got {b}"
    assert req_conf is False, "B <= 30M should not require user confirmation"


# ======================================================================
# 2. Curve 2: Late Plateau (climbs steadily to 80% by 20M, then flat to 30M)
# ======================================================================
def test_synthetic_curve_late_plateau():
    steps = [i * 1_000_000 for i in range(1, 31)]
    train_means = []
    # Steady climb to 80% at 20M (gain ~4% per 1M steps)
    for i in range(1, 21):
        train_means.append(0.04 * i)
    # Flat from 21M to 30M (80% - 80.8%)
    for i in range(21, 31):
        train_means.append(0.80 + 0.0008 * (i - 20))

    pcts = [v * 100.0 for v in train_means]
    ma = compute_moving_average(pcts, window=5)
    s = find_plateau_step(steps, ma)

    assert s is not None, "Plateau should be detected for late plateau curve"
    # MA absorbs late plateau at 23M steps
    assert s == 23_000_000, f"Expected MA plateau at 23M, got {s}"

    b, s_comb, req_conf, reason = compute_budget_from_plateaus(s, s)
    # 1.5 * 23M = 34.5M -> rounded up to 35M
    assert b == 35_000_000, f"Expected B = 35M, got {b}"
    assert req_conf is True, "B > 30M must require user confirmation"


# ======================================================================
# 3. Curve 3: No Plateau by 30M (climbs continuously)
# ======================================================================
def test_synthetic_curve_no_plateau():
    steps = [i * 1_000_000 for i in range(1, 31)]
    # Continuous linear improvement: 3% every 1M steps (15% per 5M steps >> 2% threshold)
    train_means = [0.03 * i for i in range(1, 31)]

    pcts = [v * 100.0 for v in train_means]
    ma = compute_moving_average(pcts, window=5)
    s = find_plateau_step(steps, ma)

    assert s is None, f"Expected no plateau for constantly improving curve, got {s}"

    b, s_comb, req_conf, reason = compute_budget_from_plateaus(s, 10_000_000)
    assert b == 45_000_000, f"Expected B = 45M for no plateau, got {b}"
    assert req_conf is True, "No plateau must require user confirmation"
    assert "User confirmation is required" in reason


# ======================================================================
# 4. Curve 4: Noisy Curve (reaches 75% at 12M, then noisy +/- 1.5%)
# ======================================================================
def test_synthetic_curve_noisy():
    steps = [i * 1_000_000 for i in range(1, 31)]
    # Climb up to 12M
    train_means = [0.06 * i for i in range(1, 13)]  # 0.06 to 0.72
    # Noisy oscillations around 75% (magnitude up to +/- 1.5%)
    noise_pattern = [0.015, -0.012, 0.014, -0.015, 0.010, -0.008, 0.012, -0.014, 0.009, -0.011,
                     0.013, -0.015, 0.011, -0.009, 0.014, -0.012, 0.010, -0.008]
    for n in noise_pattern:
        train_means.append(0.75 + n)

    pcts = [v * 100.0 for v in train_means]
    ma = compute_moving_average(pcts, window=5)
    s = find_plateau_step(steps, ma)

    assert s is not None, "Moving average should filter out noise and detect plateau"
    assert s == 16_000_000, f"Expected MA plateau at 16M, got {s}"

    b, _, req_conf, _ = compute_budget_from_plateaus(s, s)
    # 1.5 * 16M = 24M -> rounded up to 25M
    assert b == 25_000_000
    assert req_conf is False


# ======================================================================
# 5. Curve 5: Non-Monotonic Curve (climbs to 78% at 14M, dips to 76%, flat)
# ======================================================================
def test_synthetic_curve_non_monotonic():
    steps = [i * 1_000_000 for i in range(1, 31)]
    train_means = [
        0.05, 0.12, 0.22, 0.35, 0.48, 0.58, 0.65, 0.70, 0.73, 0.75,  # 1M - 10M
        0.76, 0.77, 0.78, 0.78,                                        # 11M - 14M (peak 78%)
        0.77, 0.76, 0.76, 0.76, 0.758, 0.761,                         # 15M - 20M (dip to 76%)
        0.760, 0.762, 0.761, 0.763, 0.762, 0.764, 0.763, 0.765, 0.764, 0.765 # 21M - 30M
    ]
    pcts = [v * 100.0 for v in train_means]
    ma = compute_moving_average(pcts, window=5)
    s = find_plateau_step(steps, ma)

    assert s is not None, "Plateau should be detected for non-monotonic curve"
    assert 10_000_000 <= s <= 15_000_000, f"Expected plateau in peak region, got {s}"

    b, _, req_conf, _ = compute_budget_from_plateaus(s, s)
    assert b <= 25_000_000


# ======================================================================
# 6. Budget > 30M requires confirmation
# ======================================================================
def test_budget_exceeds_30m_requires_confirmation():
    s_rgb = 22_000_000  # 1.5 * 22M = 33M -> B = 35M > 30M
    s_rgbd = 20_000_000
    b, s, req_conf, reason = compute_budget_from_plateaus(s_rgb, s_rgbd)

    assert b == 35_000_000
    assert req_conf is True
    assert "user confirmation is strictly required" in reason


# ======================================================================
# 7. End-to-End File Persistence, --force Guard & Decisions Logging
# ======================================================================
def test_end_to_end_choose_budget(tmp_path: Path):
    steps = [i * 1_000_000 for i in range(1, 31)]
    # RGB plateaus at 10M raw (13M MA)
    rgb_means = [min(0.075 * i, 0.75) for i in range(1, 31)]
    # RGB-D plateaus at 14M raw (17M MA)
    rgbd_means = [min(0.06 * i, 0.80) for i in range(1, 31)]

    rgb_csv = tmp_path / "rgb_eval.csv"
    rgbd_csv = tmp_path / "rgbd_eval.csv"
    out_json = tmp_path / "budget.json"
    dec_file = tmp_path / "DECISIONS.md"

    make_eval_csv(rgb_csv, steps, rgb_means)
    make_eval_csv(rgbd_csv, steps, rgbd_means)
    dec_file.write_text("# Existing Decisions\n")

    info = choose_budget(
        rgb_eval_summary=rgb_csv,
        rgbd_eval_summary=rgbd_csv,
        output_path=out_json,
        force=False,
        decisions_path=dec_file,
    )

    # RGB plateau at 13M, RGB-D plateau at 17M -> S = 17M
    # 1.5 * 17M = 25.5M -> B = 30M
    assert info["plateau_step_rgb"] == 13_000_000
    assert info["plateau_step_rgbd"] == 17_000_000
    assert info["plateau_step_combined"] == 17_000_000
    assert info["budget_steps"] == 30_000_000
    assert info["requires_user_confirmation"] is False

    assert out_json.is_file()
    saved = json.loads(out_json.read_text())
    assert saved["budget_steps"] == 30_000_000

    # Overwrite without force must raise FileExistsError
    with pytest.raises(FileExistsError):
        choose_budget(
            rgb_eval_summary=rgb_csv,
            rgbd_eval_summary=rgbd_csv,
            output_path=out_json,
            force=False,
        )

    # Overwrite with force succeeds
    info2 = choose_budget(
        rgb_eval_summary=rgb_csv,
        rgbd_eval_summary=rgbd_csv,
        output_path=out_json,
        force=True,
    )
    assert info2["budget_steps"] == 30_000_000

    # Verify DECISIONS.md was appended
    dec_content = dec_file.read_text()
    assert "Milestone M3: Budget Selection Protocol Resolution" in dec_content
    assert "30M" in dec_content

