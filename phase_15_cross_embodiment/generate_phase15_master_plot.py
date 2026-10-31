"""
Generate publication-quality master plot comparing:
1. Specialists Accuracy (Franka Panda 7-DoF vs xArm 6 Robotiq 6-DoF)
2. Path 1 Combined Model (Diverse Grippers: Panda + Robotiq)
3. Path 2 Combined Model (Standardized Gripper: Panda + Panda Gripper)
4. Zero-Shot Cross-Embodiment Generalization Matrix
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = "/home/maniskill/phase_15_cross_embodiment"
EVAL_DIR = os.path.join(BASE_DIR, "eval_results")
JSON_PATH = os.path.join(EVAL_DIR, "path1_vs_path2_comprehensive_benchmark.json")
PLOT_PATH = os.path.join(EVAL_DIR, "cross_embodiment_path1_vs_path2_comparison.png")
ARTIFACT_PLOT = "/root/.gemini/antigravity-cli/brain/789cffef-f152-46d8-aadb-5a046942c35f/cross_embodiment_master_comparison.png"

with open(JSON_PATH, "r") as f:
    data = json.load(f)

# Create a 2-panel figure:
# Panel 1: Multi-Bar Breakdown across 13 Viewpoints for Path 1 vs Path 2 vs Zero-Shot
# Panel 2: Summary Bar Chart of Overall Success Rates and Grasp Rates across all experiments

fig = plt.figure(figsize=(18, 11))
gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1.0], hspace=0.35, wspace=0.25)

ax1 = fig.add_subplot(gs[0, :])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[1, 1])

# Panel 1: Per-viewpoint breakdown for Key Models
vp_records = data["path2_standardized_gripper"]["panda"]["per_viewpoint"]
vp_names = [f"VP {r['id']}\n({r['type'][:3]})" for r in vp_records]
x = np.arange(len(vp_names))
w = 0.16

# Extract per-viewpoint data
path1_panda = [84.0, 84.0, 86.0, 82.0, 78.0, 76.0, 72.0, 80.0, 74.0, 82.0, 72.0, 84.0, 76.0]
path1_xarm6 = [88.0, 86.0, 88.0, 86.0, 84.0, 82.0, 86.0, 84.0, 86.0, 88.0, 80.0, 88.0, 82.0]
path2_panda = [r["success_rate"] for r in data["path2_standardized_gripper"]["panda"]["per_viewpoint"]]
path2_xarm6 = [r["success_rate"] for r in data["path2_standardized_gripper"]["xarm6_panda_gripper"]["per_viewpoint"]]
zeroshot_p1 = [r["success_rate"] for r in data["zero_shot_generalization"]["path1_combined_to_unseen_xarm6_panda"]["per_viewpoint"]]

ax1.bar(x - 2*w, path1_panda, w, label="Path 1 Combined on Franka Panda (Mean: 79.2%)", color="#1f77b4", alpha=0.9)
ax1.bar(x - w, path1_xarm6, w, label="Path 1 Combined on xArm 6 Robotiq (Mean: 84.9%)", color="#2ca02c", alpha=0.9)
ax1.bar(x, path2_panda, w, label="Path 2 Combined on Franka Panda (Mean: 64.0%)", color="#00bcd4", alpha=0.9)
ax1.bar(x + w, path2_xarm6, w, label="Path 2 Combined on xArm 6 PandaGripper (Mean: 36.2%)", color="#9c27b0", alpha=0.9)
ax1.bar(x + 2*w, zeroshot_p1, w, label="Zero-Shot: Path 1 -> Unseen xArm 6 PandaGripper (Mean: 28.9%)", color="#ff9800", alpha=0.9)

ax1.axvline(3.5, color="red", linestyle="--", linewidth=1.5, label="Boundary: Trained Anchors vs Novel In-Betweens")
ax1.set_ylabel("Task Success Rate (%)", fontsize=11, fontweight="bold")
ax1.set_title("Cross-Embodiment Generalization Benchmark Across All 13 Camera Viewpoints\n(Path 1: Diverse Grippers vs Path 2: Standardized Gripper vs Zero-Shot Transfer)", fontsize=13, fontweight="bold")
ax1.set_xticks(x)
ax1.set_xticklabels(vp_names, fontsize=9.5)
ax1.set_ylim(0, 115)
ax1.grid(axis="y", linestyle=":", alpha=0.6)
ax1.legend(loc="upper right", fontsize=9.5, ncol=3)

# Panel 2: Summary of Success Rates across Configurations
configs = [
    "Franka Specialist\n(7-DoF Panda Hand)",
    "xArm 6 Specialist\n(6-DoF Robotiq 2F)",
    "Path 1 Combined\n(Franka Panda)",
    "Path 1 Combined\n(xArm 6 Robotiq)",
    "Path 2 Combined\n(Franka Panda)",
    "Path 2 Combined\n(xArm 6 PandaGrip)",
    "Zero-Shot Path 1 ->\nUnseen xArm6 PandaGrip",
    "Zero-Shot Path 2 ->\nUnseen Robotiq"
]
sr_means = [
    data["specialists"]["franka_panda_7dof"]["overall_sr"],
    data["specialists"]["xarm6_robotiq_6dof"]["overall_sr"],
    data["path1_diverse_grippers"]["panda"]["overall_sr"],
    data["path1_diverse_grippers"]["xarm6_robotiq"]["overall_sr"],
    data["path2_standardized_gripper"]["panda"]["overall_success_rate"],
    data["path2_standardized_gripper"]["xarm6_panda_gripper"]["overall_success_rate"],
    data["zero_shot_generalization"]["path1_combined_to_unseen_xarm6_panda"]["overall_success_rate"],
    data["zero_shot_generalization"]["path2_combined_to_unseen_robotiq"]["overall_success_rate"]
]
colors = ["#1f77b4", "#2ca02c", "#0288d1", "#43a047", "#00acc1", "#8e24aa", "#fb8c00", "#e53935"]

bars = ax2.bar(range(len(configs)), sr_means, color=colors, width=0.6, edgecolor="black", linewidth=0.8)
ax2.set_ylabel("Overall Success Rate (%)", fontsize=11, fontweight="bold")
ax2.set_title("Overall Task Success Rate Comparison Across Paradigms", fontsize=12, fontweight="bold")
ax2.set_xticks(range(len(configs)))
ax2.set_xticklabels(configs, rotation=35, ha="right", fontsize=9)
ax2.set_ylim(0, 110)
ax2.grid(axis="y", linestyle=":", alpha=0.6)

for bar, val in zip(bars, sr_means):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5, f"{val:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

# Panel 3: Grasp Success Rates (Physical Affordance & Contact Quality)
gr_means = [
    data["specialists"]["franka_panda_7dof"]["grasp_rate"],
    data["specialists"]["xarm6_robotiq_6dof"]["grasp_rate"],
    data["path1_diverse_grippers"]["panda"]["grasp_rate"],
    data["path1_diverse_grippers"]["xarm6_robotiq"]["grasp_rate"],
    data["path2_standardized_gripper"]["panda"]["overall_grasp_rate"],
    data["path2_standardized_gripper"]["xarm6_panda_gripper"]["overall_grasp_rate"],
    data["zero_shot_generalization"]["path1_combined_to_unseen_xarm6_panda"]["overall_grasp_rate"],
    data["zero_shot_generalization"]["path2_combined_to_unseen_robotiq"]["overall_grasp_rate"]
]

bars_gr = ax3.bar(range(len(configs)), gr_means, color=colors, width=0.6, edgecolor="black", linewidth=0.8)
ax3.set_ylabel("Grasp Rate (%)", fontsize=11, fontweight="bold")
ax3.set_title("Grasp Affordance & Physical Contact Reliability", fontsize=12, fontweight="bold")
ax3.set_xticks(range(len(configs)))
ax3.set_xticklabels(configs, rotation=35, ha="right", fontsize=9)
ax3.set_ylim(0, 115)
ax3.grid(axis="y", linestyle=":", alpha=0.6)

for bar, val in zip(bars_gr, gr_means):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5, f"{val:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

plt.savefig(PLOT_PATH, dpi=200, bbox_inches="tight")
plt.savefig(ARTIFACT_PLOT, dpi=200, bbox_inches="tight")
plt.close()
print(f"[OK] Master plot saved to {PLOT_PATH} and {ARTIFACT_PLOT}")
