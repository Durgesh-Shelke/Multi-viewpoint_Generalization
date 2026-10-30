"""
Budget Selection Module (SPEC Section 4.3).

Analyzes pilot evaluations for joint RGB and RGB-D policies to mathematically determine
the fixed training budget B:
1. Reads eval_summary.csv of both pilots (evaluated every 1M steps on train + val views).
2. Computes the 5-evaluation moving average of mean train success.
3. Finds the plateau step S for each arm: first step after which the improvement of the moving
   average is below 2 percentage points per further 5M steps.
4. Takes the later S of the two arms: S = max(S_rgb, S_rgbd).
5. Sets B = round_up(1.5 * S, to nearest 5M steps).
6. If no plateau by 30M steps, sets B = 45M steps and flags that user confirmation is required.
   If B > 30M steps in general, also requires user confirmation.
7. Writes configs/budget.json (never overwriting without --force) and appends to DECISIONS.md.
"""

import argparse
import csv
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger("choose_budget")
EXP_DIR = Path(__file__).resolve().parent.parent


def load_eval_summary(csv_path: Union[str, Path]) -> Tuple[List[int], List[float]]:
    """
    Loads eval_summary.csv containing columns: step, train_mean, val_mean.
    Returns (steps, train_means_pct) sorted by step.
    Values in train_mean are converted to percentage points (0.0 to 100.0%).
    """
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Evaluation summary file not found: {path}")

    steps: List[int] = []
    train_means: List[float] = []

    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row or not row.get("step"):
                continue
            step = int(row["step"])
            val = float(row["train_mean"])
            steps.append(step)
            train_means.append(val)

    if not steps:
        raise ValueError(f"No valid evaluation rows found in {path}")

    # Sort by step
    combined = sorted(zip(steps, train_means), key=lambda x: x[0])
    sorted_steps = [x[0] for x in combined]
    sorted_means = [x[1] for x in combined]

    # Convert ratio (0.0 to 1.0) to percentage points (0.0 to 100.0%) if needed
    max_val = max(sorted_means) if sorted_means else 0.0
    if max_val <= 1.0:
        sorted_means = [v * 100.0 for v in sorted_means]

    return sorted_steps, sorted_means


def compute_moving_average(values: List[float], window: int = 5) -> List[float]:
    """
    Computes trailing moving average.
    For initial indices i < window - 1, uses expanding window mean.
    For i >= window - 1, uses mean over values[i - window + 1 : i + 1].
    """
    if not values:
        return []
    ma: List[float] = []
    for i in range(len(values)):
        if i < window - 1:
            window_vals = values[: i + 1]
        else:
            window_vals = values[i - window + 1 : i + 1]
        ma.append(float(sum(window_vals) / len(window_vals)))
    return ma


def find_plateau_step(
    steps: List[int],
    moving_averages: List[float],
    min_horizon_steps: int = 5_000_000,
    threshold_pct: float = 2.0,
    max_step_limit: int = 30_000_000,
) -> Optional[int]:
    """
    Finds plateau step S per SPEC Section 4.3:
    The first step after which the improvement of the 5-evaluation moving average
    is below 2 percentage points per further 5M steps.

    Specifically, for candidate step S_i:
    From step S_i onward:
    1. For every future evaluation step S_k (where k >= i):
       For all steps S_l >= S_k + min_horizon_steps, the gain MA[l] - MA[k] < threshold_pct.
    2. Over any remaining horizon to the end of data (up to max_step_limit),
       the gain normalized per 5M steps remains < threshold_pct:
       (MA[j] - MA[i]) / max((steps[j] - steps[i]) / 5M, 1.0) < threshold_pct for all j > i.

    Returns the first step S that meets this criterion, or None if no plateau occurs
    by max_step_limit.
    """
    if len(steps) < 2 or len(steps) != len(moving_averages):
        return None

    n = len(steps)
    for i in range(n):
        s_i = steps[i]
        if s_i > max_step_limit:
            break

        # Check if from step s_i onwards, all subsequent improvements are < threshold_pct per 5M steps
        is_plateau = True

        # Check all 5M-horizon intervals starting at or after index i
        found_at_least_one_check = False
        for k in range(i, n):
            s_k = steps[k]
            # Compare with all points at least 5M steps ahead
            for l in range(k + 1, n):
                s_l = steps[l]
                delta_s = s_l - s_k
                if delta_s >= min_horizon_steps:
                    found_at_least_one_check = True
                    gain = moving_averages[l] - moving_averages[k]
                    # Gain over this horizon scaled per 5M steps
                    gain_per_5m = gain / (delta_s / float(min_horizon_steps))
                    if gain_per_5m >= threshold_pct:
                        is_plateau = False
                        break
            if not is_plateau:
                break

        # Check total remaining gain from s_i to any future point
        if is_plateau:
            for j in range(i + 1, n):
                s_j = steps[j]
                delta_s = s_j - s_i
                gain = moving_averages[j] - moving_averages[i]
                norm_gain = gain / max(delta_s / float(min_horizon_steps), 1.0)
                if norm_gain >= threshold_pct:
                    is_plateau = False
                    break

        if is_plateau and found_at_least_one_check:
            return s_i

    return None


def round_up_to_multiple(value: float, multiple: int = 5_000_000) -> int:
    """Rounds up value to the nearest multiple (default: 5,000,000)."""
    return int(math.ceil(value / float(multiple))) * multiple


def compute_budget_from_plateaus(
    plateau_rgb: Optional[int],
    plateau_rgbd: Optional[int],
    multiplier: float = 1.5,
    budget_unit: int = 5_000_000,
) -> Tuple[int, Optional[int], bool, str]:
    """
    Applies SPEC Section 4.3 rules to compute budget B:
    - If either arm has no plateau by 30M: B = 45M (requires confirmation).
    - Else: S = max(S_rgb, S_rgbd), B = round_up(1.5 * S, 5M).
    - If B > 30M: requires user confirmation.
    Returns (budget_b, plateau_s, requires_confirmation, reasoning).
    """
    if plateau_rgb is None or plateau_rgbd is None:
        b = 45_000_000
        req_conf = True
        missing = []
        if plateau_rgb is None:
            missing.append("RGB")
        if plateau_rgbd is None:
            missing.append("RGB-D")
        reason = (
            f"No plateau detected by 30M steps for {', '.join(missing)}. "
            f"Budget set to B = 45,000,000 steps (45M) per SPEC Section 4.3 rule 4. "
            "User confirmation is required before proceeding with the main grid."
        )
        return b, None, req_conf, reason

    s = max(plateau_rgb, plateau_rgbd)
    raw_b = multiplier * float(s)
    b = round_up_to_multiple(raw_b, budget_unit)

    if b > 30_000_000:
        req_conf = True
        reason = (
            f"Plateau identified at S = {s:,} steps (RGB: {plateau_rgb:,}, RGB-D: {plateau_rgbd:,}). "
            f"Raw budget is {multiplier} * {s:,} = {raw_b:,.0f} steps, rounded up to B = {b:,} steps ({b//1_000_000}M). "
            f"Because B > 30M steps, user confirmation is strictly required before proceeding per SPEC Section 4.3 rule 4."
        )
    else:
        req_conf = False
        reason = (
            f"Plateau identified at S = {s:,} steps (RGB: {plateau_rgb:,}, RGB-D: {plateau_rgbd:,}). "
            f"Raw budget is {multiplier} * {s:,} = {raw_b:,.0f} steps, rounded up to B = {b:,} steps ({b//1_000_000}M). "
            f"Budget satisfies B <= 30M steps."
        )

    return b, s, req_conf, reason


def append_decisions_log(budget_info: Dict[str, Any], decisions_path: Union[str, Path] = "DECISIONS.md") -> None:
    """Appends authoritative budget decision entry to DECISIONS.md."""
    dec_file = Path(decisions_path)
    timestamp = time.strftime("%Y-%m-%d %H:%M", time.localtime())

    b_steps = budget_info["budget_steps"]
    b_m = b_steps // 1_000_000
    s_steps = budget_info.get("plateau_step_combined")
    s_m = f"{s_steps // 1_000_000}M" if s_steps else "None (No plateau)"
    rgb_s = f"{budget_info.get('plateau_step_rgb', 0) // 1_000_000}M" if budget_info.get("plateau_step_rgb") else "None"
    rgbd_s = f"{budget_info.get('plateau_step_rgbd', 0) // 1_000_000}M" if budget_info.get("plateau_step_rgbd") else "None"

    conf_str = "YES (User Confirmation Required)" if budget_info.get("requires_user_confirmation") else "NO (Approved by Protocol)"

    entry = f"""
---

## [{timestamp}] Milestone M3: Budget Selection Protocol Resolution (`configs/budget.json`)

### 1. SPEC Section 4.3 Plateau Analysis & Calculations
- **Joint RGB Pilot Plateau Step ($S_{{rgb}}$)**: `{rgb_s}` ({budget_info.get('plateau_step_rgb', 'N/A')} steps)
- **Joint RGB-D Pilot Plateau Step ($S_{{rgbd}}$)**: `{rgbd_s}` ({budget_info.get('plateau_step_rgbd', 'N/A')} steps)
- **Selected Plateau Step ($S = \\max(S_{{rgb}}, S_{{rgbd}})$)**: `{s_m}` ({s_steps if s_steps else 'N/A'} steps)
- **Budget Multiplier**: `{budget_info.get('multiplier', 1.5)}x`
- **Computed Step Budget ($B$)**: **`{b_m}M`** (`{b_steps:,}` environment steps)
- **User Confirmation Required**: **`{conf_str}`**
- **Reasoning**: {budget_info.get('reasoning')}

### 2. Experimental Implications
- All 6 main grid runs (3 seeds RGB + 3 seeds RGB-D) will be executed for exactly **`{b_m}M steps`** each.
- Frozen in `configs/budget.json`.
"""
    with open(dec_file, "a") as f:
        f.write(entry)
    logger.info("Appended budget decision entry to %s", dec_file)


def choose_budget(
    rgb_eval_summary: Union[str, Path],
    rgbd_eval_summary: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    force: bool = False,
    decisions_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Main function to execute budget selection from pilot CSVs.
    """
    # 1. Load data
    rgb_steps, rgb_means = load_eval_summary(rgb_eval_summary)
    rgbd_steps, rgbd_means = load_eval_summary(rgbd_eval_summary)

    # 2. Compute 5-eval moving averages
    rgb_ma = compute_moving_average(rgb_means, window=5)
    rgbd_ma = compute_moving_average(rgbd_means, window=5)

    # 3. Find plateaus
    s_rgb = find_plateau_step(rgb_steps, rgb_ma)
    s_rgbd = find_plateau_step(rgbd_steps, rgbd_ma)

    # 4. Compute budget
    b, s_combined, req_conf, reason = compute_budget_from_plateaus(s_rgb, s_rgbd)

    budget_info: Dict[str, Any] = {
        "budget_steps": b,
        "budget_millions": b // 1_000_000,
        "plateau_step_rgb": s_rgb,
        "plateau_step_rgbd": s_rgbd,
        "plateau_step_combined": s_combined,
        "multiplier": 1.5,
        "requires_user_confirmation": req_conf,
        "frozen": not req_conf,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "rgb_summary_file": str(rgb_eval_summary),
        "rgbd_summary_file": str(rgbd_eval_summary),
        "reasoning": reason,
    }

    # 5. Write configs/budget.json
    if output_path:
        out_p = Path(output_path)
        if out_p.is_file() and not force:
            raise FileExistsError(f"Target budget file {out_p} already exists! Use --force to overwrite.")

        out_p.parent.mkdir(parents=True, exist_ok=True)
        tmp_p = str(out_p) + f".tmp.{os.getpid()}"
        with open(tmp_p, "w") as f:
            json.dump(budget_info, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_p, str(out_p))
        logger.info("Saved budget configuration to %s", out_p)

    # 6. Update DECISIONS.md
    if decisions_path:
        append_decisions_log(budget_info, decisions_path)

    return budget_info


def parse_args():
    parser = argparse.ArgumentParser(description="Choose Training Budget B from Pilot Evaluations (SPEC Section 4.3)")
    parser.add_argument("--rgb-eval", type=str, required=True, help="Path to pilot_rgb eval_summary.csv")
    parser.add_argument("--rgbd-eval", type=str, required=True, help="Path to pilot_rgbd eval_summary.csv")
    parser.add_argument("--output", type=str, default=str(EXP_DIR / "configs" / "budget.json"), help="Output path for budget.json")
    parser.add_argument("--force", action="store_true", help="Overwrite existing budget.json")
    parser.add_argument("--decisions-file", type=str, default=str(EXP_DIR / "DECISIONS.md"), help="Path to DECISIONS.md")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    try:
        info = choose_budget(
            rgb_eval_summary=args.rgb_eval,
            rgbd_eval_summary=args.rgbd_eval,
            output_path=args.output,
            force=args.force,
            decisions_path=args.decisions_file,
        )
        print("\n" + "=" * 60)
        print(" BUDGET SELECTION RESULT")
        print("=" * 60)
        print(f" Plateau Step RGB:      {info['plateau_step_rgb']:,} steps" if info['plateau_step_rgb'] else " Plateau Step RGB:      None (No Plateau)")
        print(f" Plateau Step RGB-D:    {info['plateau_step_rgbd']:,} steps" if info['plateau_step_rgbd'] else " Plateau Step RGB-D:    None (No Plateau)")
        print(f" Combined Plateau S:    {info['plateau_step_combined']:,} steps" if info['plateau_step_combined'] else " Combined Plateau S:    None")
        print(f" Chosen Budget B:       {info['budget_steps']:,} steps ({info['budget_millions']}M)")
        print(f" Requires Confirmation: {'YES' if info['requires_user_confirmation'] else 'NO'}")
        print(f" Reasoning:             {info['reasoning']}")
        print("=" * 60 + "\n")
        if info["requires_user_confirmation"]:
            print(">>> ACTION REQUIRED: User confirmation is required before proceeding with the main grid!\n")
    except Exception as e:
        logger.error("Budget selection failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
