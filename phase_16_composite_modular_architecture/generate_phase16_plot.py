"""
Phase 16: Composite Modular Architecture Benchmark Plot.
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = "/home/maniskill/phase_16_composite_modular_architecture"
EVAL_DIR = os.path.join(BASE_DIR, "eval_results")
JSON_PATH = os.path.join(EVAL_DIR, "composite_architecture_benchmark.json")
PLOT_PATH = os.path.join(EVAL_DIR, "phase16_composite_modular_benchmark.png")
ARTIFACT_PLOT = "/root/.gemini/antigravity-cli/brain/789cffef-f152-46d8-aadb-5a046942c35f/phase16_composite_modular_benchmark.png"

with open(JSON_PATH, "r") as f:
    data = json.load(f)

fig = plt.figure(figsize=(18, 10))
gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1.0], hspace=0.35, wspace=0.25)

ax1 = fig.add_subplot(gs[0, :])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[1, 1])

# Panel 1: Per-viewpoint breakdown
vp_records = data["panda"]
vp_names = [f"VP {r['id']}\n({r['type'][:3]})" for r in vp_records]
x = np.arange(len(vp_names))
w = 0.35

panda_sr = [r["success_rate"] for r in data["panda"]]
xarm6_sr = [r["success_rate"] for r in data["xarm6"]]

b1 = ax1.bar(x - w/2, panda_sr, w, label=f"Composite Deliverable on Franka Panda (7 DoF) | Mean: {np.mean(panda_sr):.1f}%", color="#1f77b4", alpha=0.9)
b2 = ax1.bar(x + w/2, xarm6_sr, w, label=f"Composite Deliverable on xArm 6 Robotiq (6 DoF) | Mean: {np.mean(xarm6_sr):.1f}%", color="#2ca02c", alpha=0.9)

ax1.axvline(3.5, color="red", linestyle="--", linewidth=1.5, label="Boundary: Trained Anchors vs Novel In-Betweens")
ax1.set_ylabel("Task Success Rate (%)", fontsize=11, fontweight="bold")
ax1.set_title("Phase 16: Single Unified Composite Deliverable ('UniversalCompositeOmniPolicy')\nEvaluated on Franka Panda (7 DoF) and xArm 6 (6 DoF) across All 13 Camera Viewpoints", fontsize=13, fontweight="bold")
ax1.set_xticks(x)
ax1.set_xticklabels(vp_names, fontsize=9.5)
ax1.set_ylim(0, 115)
ax1.grid(axis="y", linestyle=":", alpha=0.6)
ax1.legend(loc="upper right", fontsize=10)

# Add values above bars in Panel 1
for bar in b1:
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.0, f"{bar.get_height():.0f}", ha="center", va="bottom", fontsize=8)
for bar in b2:
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.0, f"{bar.get_height():.0f}", ha="center", va="bottom", fontsize=8)

# Panel 2: Summary Comparison (Anchor vs In-Between vs Overall)
categories = ["Anchor Viewpoints (4)", "Novel In-Betweens (9)", "Overall Average (13)"]
x_cat = np.arange(len(categories))
w_cat = 0.35

panda_means = [80.0, 72.2, 74.6]
xarm6_means = [90.5, 92.9, 92.2]

ax2.bar(x_cat - w_cat/2, panda_means, w_cat, label="Franka Panda (7 DoF)", color="#1f77b4", alpha=0.9)
ax2.bar(x_cat + w_cat/2, xarm6_means, w_cat, label="xArm 6 Robotiq (6 DoF)", color="#2ca02c", alpha=0.9)

ax2.set_ylabel("Success Rate (%)", fontsize=11, fontweight="bold")
ax2.set_title("Generalization Across Anchor vs. Novel In-Between Views", fontsize=12, fontweight="bold")
ax2.set_xticks(x_cat)
ax2.set_xticklabels(categories, fontsize=10, fontweight="bold")
ax2.set_ylim(0, 110)
ax2.grid(axis="y", linestyle=":", alpha=0.6)
ax2.legend(loc="lower right", fontsize=10)

for idx, (p, x_val) in enumerate(zip(panda_means, xarm6_means)):
    ax2.text(idx - w_cat/2, p + 1.5, f"{p:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=10)
    ax2.text(idx + w_cat/2, x_val + 1.5, f"{x_val:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=10)

# Panel 3: Grasp Reliability & Contact Physics
panda_grasp = [r["grasp_rate"] for r in data["panda"]]
xarm6_grasp = [r["grasp_rate"] for r in data["xarm6"]]
grasp_cats = ["Franka Panda (7 DoF)", "xArm 6 Robotiq (6 DoF)", "Combined Average"]
grasp_vals = [np.mean(panda_grasp), np.mean(xarm6_grasp), (np.mean(panda_grasp)+np.mean(xarm6_grasp))/2]
grasp_colors = ["#1f77b4", "#2ca02c", "#00bcd4"]

bars_g = ax3.bar(range(len(grasp_cats)), grasp_vals, color=grasp_colors, width=0.55, edgecolor="black")
ax3.set_ylabel("Grasp Success Rate (%)", fontsize=11, fontweight="bold")
ax3.set_title("Physical Grasp & Affordance Reliability Across Robots", fontsize=12, fontweight="bold")
ax3.set_xticks(range(len(grasp_cats)))
ax3.set_xticklabels(grasp_cats, fontsize=10, fontweight="bold")
ax3.set_ylim(0, 115)
ax3.grid(axis="y", linestyle=":", alpha=0.6)

for bar, val in zip(bars_g, grasp_vals):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5, f"{val:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

plt.savefig(PLOT_PATH, dpi=200, bbox_inches="tight")
plt.savefig(ARTIFACT_PLOT, dpi=200, bbox_inches="tight")
plt.close()
print(f"[OK] Phase 16 plot saved to {PLOT_PATH} and {ARTIFACT_PLOT}")
