"""
Phase 15: Cross-Embodiment Comprehensive Plot & Master Benchmark Synthesis.

Loads:
- specialists_accuracy_benchmark.json
- combined_omniarm_benchmark.json
- cross_embodiment_zeroshot_benchmark.json

Generates:
- cross_embodiment_master_comparison.png (publication-grade multi-bar comparison)
- research_report_24_phase15_cross_embodiment_zeroshot.md
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = "/home/maniskill/phase_15_cross_embodiment"
EVAL_DIR = os.path.join(BASE_DIR, "eval_results")
SPEC_JSON = os.path.join(EVAL_DIR, "specialists_accuracy_benchmark.json")
COMB_JSON = os.path.join(EVAL_DIR, "combined_omniarm_benchmark.json")
ZERO_JSON = os.path.join(EVAL_DIR, "cross_embodiment_zeroshot_benchmark.json")
MASTER_PLOT = os.path.join(EVAL_DIR, "cross_embodiment_master_comparison.png")


def synthesize_results():
    if not os.path.exists(SPEC_JSON):
        print(f"Waiting for {SPEC_JSON}...")
        return False

    with open(SPEC_JSON, "r") as f:
        spec_data = json.load(f)

    comb_data = None
    if os.path.exists(COMB_JSON):
        with open(COMB_JSON, "r") as f:
            comb_data = json.load(f)

    zero_data = None
    if os.path.exists(ZERO_JSON):
        with open(ZERO_JSON, "r") as f:
            zero_data = json.load(f)

    panda_spec = [r["success_rate"] for r in spec_data["benchmark_records"]["panda"]]
    xarm6_spec = [r["success_rate"] for r in spec_data["benchmark_records"]["xarm6"]]
    vp_names = [f"VP {r['id']}\n({r['type'][:3]})" for r in spec_data["benchmark_records"]["panda"]]

    num_vps = len(vp_names)
    x = np.arange(num_vps)

    fig, ax = plt.subplots(figsize=(16, 7))

    width = 0.18
    ax.bar(x - 2*width, panda_spec, width, label=f"Panda Specialist on Panda (Mean: {np.mean(panda_spec):.1f}%)", color="#1f77b4", alpha=0.9)
    ax.bar(x - width, xarm6_spec, width, label=f"xArm 6 Specialist on xArm 6 (Mean: {np.mean(xarm6_spec):.1f}%)", color="#2ca02c", alpha=0.9)

    if comb_data:
        p_comb = [r["success_rate"] for r in comb_data["panda"]]
        x_comb = [r["success_rate"] for r in comb_data["xarm6"]]
        ax.bar(x, p_comb, width, label=f"Combined Policy on Panda (Mean: {np.mean(p_comb):.1f}%)", color="#00bcd4", alpha=0.9)
        ax.bar(x + width, x_comb, width, label=f"Combined Policy on xArm 6 (Mean: {np.mean(x_comb):.1f}%)", color="#8bc34a", alpha=0.9)

    if zero_data:
        p_zero = [r["success_rate"] for r in zero_data["panda_zeroshot_on_xarm6"]]
        ax.bar(x + 2*width, p_zero, width, label=f"Zero-Shot: Panda Spec -> xArm 6 (Mean: {np.mean(p_zero):.1f}%)", color="#ff9800", alpha=0.9)

    ax.axvline(3.5, color="red", linestyle="--", linewidth=1.5, label="Boundary: Trained Anchors vs Novel In-Betweens")
    ax.set_ylabel("Task Success Rate (%)", fontsize=12, fontweight="bold")
    ax.set_title("Cross-Embodiment Generalization Benchmark across 13 Continuous Viewpoints\n(Franka Panda vs UFactory xArm 6 Robotiq | Specialists vs Combined vs Zero-Shot Transfer)", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(vp_names, fontsize=10)
    ax.set_ylim(0, 115)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.legend(loc="lower right", fontsize=10, ncol=2)

    plt.tight_layout()
    plt.savefig(MASTER_PLOT, dpi=200)
    plt.close()
    print(f"[OK] Master comparison plot saved to {MASTER_PLOT}")
    return True


if __name__ == "__main__":
    synthesize_results()
