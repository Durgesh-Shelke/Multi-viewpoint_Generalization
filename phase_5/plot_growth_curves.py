"""
Publication-Quality Growth Curve and Plateau Analysis Plotting Tool.
Reads candidate training histories and visualizes convergence, sample efficiency, and plateau dynamics.
"""

import argparse
import csv
import json
import os
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CANDIDATE_META = {
    "A": {"label": "Cand A: Sym 128x128 RGB-D", "color": "#1f77b4", "marker": "o"},
    "B": {"label": "Cand B: Sym 256x256 RGB-D", "color": "#d62728", "marker": "s"},
    "C": {"label": "Cand C: Asym (64 RGB + 256 D)", "color": "#2ca02c", "marker": "^"},
    "D": {"label": "Cand D: Pure Depth 256x256", "color": "#9467bd", "marker": "D"},
    "E": {"label": "Cand E: Pure RGB 256x256", "color": "#ff7f0e", "marker": "v"},
}


def analyze_plateau(history_rows: list) -> dict:
    if not history_rows:
        return {}

    steps = [r["global_step"] for r in history_rows]
    rates = [r["eval_success_rate"] for r in history_rows]

    peak_success = max(rates) if rates else 0.0
    step_50 = None
    step_80 = None
    plateau_start_step = None

    plateau_threshold = 0.90 * peak_success if peak_success > 30.0 else 50.0

    for s, r in zip(steps, rates):
        if step_50 is None and r >= 50.0:
            step_50 = s
        if step_80 is None and r >= 80.0:
            step_80 = s
        if plateau_start_step is None and r >= plateau_threshold:
            plateau_start_step = s

    # Average performance on the plateau (from plateau_start_step to end)
    plateau_vals = [r for s, r in zip(steps, rates) if plateau_start_step is not None and s >= plateau_start_step]
    plateau_mean = np.mean(plateau_vals) if plateau_vals else peak_success

    return {
        "peak_success": peak_success,
        "step_to_50_pct": step_50,
        "step_to_80_pct": step_80,
        "plateau_start_step": plateau_start_step,
        "plateau_mean_success": round(float(plateau_mean), 2),
    }


def generate_plots(
    eval_dir: str = "/home/maniskill/phase_5/eval_results",
    output_png: str = "/home/maniskill/phase_5/growth_curves_comparison.png",
    brain_dir: str = "/root/.gemini/antigravity-cli/brain/82673f33-5c96-4154-bb42-cf6de559c7b7",
):
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=300)

    candidates = ["A", "B", "C", "D", "E"]
    plateau_stats = {}

    ax_success = axes[0, 0]
    ax_reward = axes[0, 1]
    ax_plateau = axes[1, 0]
    ax_tradeoff = axes[1, 1]

    peak_successes = []
    avg_sps_vals = []
    cand_labels = []

    for cand in candidates:
        csv_path = os.path.join(eval_dir, f"candidate_{cand}_history.csv")
        json_path = os.path.join(eval_dir, f"candidate_{cand}_history.json")

        rows = []
        if os.path.exists(csv_path):
            with open(csv_path) as f:
                reader = csv.DictReader(f)
                for r in reader:
                    rows.append({
                        "iteration": int(r["iteration"]),
                        "global_step": int(r["global_step"]),
                        "eval_success_rate": float(r["eval_success_rate"]),
                        "mean_reward": float(r["mean_reward"]),
                        "sps": int(r.get("sps", 0)),
                    })
        elif os.path.exists(json_path):
            with open(json_path) as f:
                rows = json.load(f)

        if not rows:
            print(f"[Warning] No history found for Candidate {cand} at {csv_path}")
            continue

        meta = CANDIDATE_META[cand]
        steps_m = [r["global_step"] / 1e6 for r in rows]
        success_rates = [r["eval_success_rate"] for r in rows]
        rewards = [r["mean_reward"] for r in rows]
        sps_list = [r["sps"] for r in rows if r["sps"] > 0]
        avg_sps = int(np.mean(sps_list)) if sps_list else 1000

        # Plot Success Rate
        ax_success.plot(steps_m, success_rates, label=meta["label"], color=meta["color"], marker=meta["marker"], markersize=4, linewidth=2, alpha=0.9)

        # Plot Reward
        ax_reward.plot(steps_m, rewards, label=meta["label"], color=meta["color"], marker=meta["marker"], markersize=4, linewidth=2, alpha=0.9)

        # Plateau Analysis
        stats = analyze_plateau(rows)
        plateau_stats[cand] = stats
        if stats.get("plateau_start_step"):
            plat_m = stats["plateau_start_step"] / 1e6
            ax_success.axvline(plat_m, color=meta["color"], linestyle="--", alpha=0.5)

        peak_successes.append(stats.get("peak_success", 0.0))
        avg_sps_vals.append(avg_sps)
        cand_labels.append(f"Cand {cand}")

    # Panel 1: Success Rate Styling
    ax_success.set_title("Policy Success Rate (%) vs. Timesteps (0 to 10M)", fontsize=13, fontweight="bold")
    ax_success.set_xlabel("Environment Interaction Steps (Millions)", fontsize=11)
    ax_success.set_ylabel("Evaluation Success Rate (%)", fontsize=11)
    ax_success.set_ylim(-2, 105)
    ax_success.axhline(80, color="gray", linestyle=":", alpha=0.7, label="80% Grasp Proficiency")
    ax_success.axhline(95, color="green", linestyle=":", alpha=0.7, label="95% Production Target")
    ax_success.legend(loc="lower right", frameon=True, fontsize=9)

    # Panel 2: Episode Reward Styling
    ax_reward.set_title("Mean Episode Reward Growth vs. Timesteps", fontsize=13, fontweight="bold")
    ax_reward.set_xlabel("Environment Interaction Steps (Millions)", fontsize=11)
    ax_reward.set_ylabel("Mean Episode Reward", fontsize=11)
    ax_reward.legend(loc="lower right", frameon=True, fontsize=9)

    # Panel 3: Plateau Onset & Sample Efficiency
    bar_width = 0.35
    cand_keys = [c for c in candidates if c in plateau_stats]
    x_pos = np.arange(len(cand_keys))
    step80_vals = [plateau_stats[c].get("step_to_80_pct") / 1e6 if plateau_stats[c].get("step_to_80_pct") else 0 for c in cand_keys]
    plat_vals = [plateau_stats[c].get("plateau_start_step") / 1e6 if plateau_stats[c].get("plateau_start_step") else 0 for c in cand_keys]

    ax_plateau.bar(x_pos - bar_width / 2, step80_vals, bar_width, label="Steps to 80% Success", color="#3498db", alpha=0.85)
    ax_plateau.bar(x_pos + bar_width / 2, plat_vals, bar_width, label="Plateau Onset Step", color="#e67e22", alpha=0.85)
    ax_plateau.set_title("Sample Efficiency & Plateau Onset Comparison", fontsize=13, fontweight="bold")
    ax_plateau.set_ylabel("Steps to Milestone (Millions)", fontsize=11)
    ax_plateau.set_xticks(x_pos)
    ax_plateau.set_xticklabels([f"Cand {c}" for c in cand_keys], fontsize=11)
    ax_plateau.legend(loc="upper left", frameon=True, fontsize=9)

    # Panel 4: Pareto Tradeoff (Peak Success vs. Throughput)
    for c_id, peak, sps in zip(cand_labels, peak_successes, avg_sps_vals):
        cand_key = c_id.replace("Cand ", "")
        color = CANDIDATE_META[cand_key]["color"]
        ax_tradeoff.scatter(sps, peak, s=250, color=color, alpha=0.85, edgecolors="black", zorder=5)
        ax_tradeoff.annotate(c_id, (sps + 15, peak - 1.5), fontsize=10, fontweight="bold")

    ax_tradeoff.set_title("Pareto Efficiency: Grasp Performance vs. Training Speed", fontsize=13, fontweight="bold")
    ax_tradeoff.set_xlabel("Training Throughput (SPS)", fontsize=11)
    ax_tradeoff.set_ylabel("Peak Success Rate (%)", fontsize=11)
    ax_tradeoff.set_ylim(-2, 105)

    plt.suptitle("ManiSkill Robot Arm Manipulation: Resolution & Asymmetric Decoupling Study (10M Steps)", fontsize=15, fontweight="bold", y=0.99)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    fig.savefig(output_png, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[Plot] Generated growth curves comparison plot saved to: {output_png}")

    # Mirror to brain directory
    if os.path.exists(brain_dir):
        brain_png = os.path.join(brain_dir, "growth_curves_comparison.png")
        shutil.copy(output_png, brain_png)
        print(f"[Plot] Mirrored plot to brain directory: {brain_png}")

    # Save Plateau Summary JSON
    stats_json = os.path.join(eval_dir, "plateau_summary.json")
    with open(stats_json, "w") as f:
        json.dump(plateau_stats, f, indent=2)
    print(f"[Stats] Saved plateau summary to: {stats_json}")

    return plateau_stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", type=str, default="/home/maniskill/phase_5/eval_results")
    parser.add_argument("--output-png", type=str, default="/home/maniskill/phase_5/growth_curves_comparison.png")
    args = parser.parse_args()

    generate_plots(eval_dir=args.eval_dir, output_png=args.output_png)
