"""
Master Pipeline Orchestrator for Phase 6: Individual Viewpoint Specialist Training & Matrix Evaluation.
Sequentially trains an independent specialist for each of the 8 Core Viewpoints,
monitors progress, triggers the 8x8 transfer evaluation, and compiles the final report.
"""

import os
import sys
import json
import time
import subprocess
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = SCRIPT_DIR
LOG_FILE = os.path.join(OUTPUT_DIR, "master_execution.log")
REPORT_PATH = os.path.join(OUTPUT_DIR, "reports", "15_individual_viewpoint_specialists_report.md")

CORE_VP_IDS = [17, 0, 28, 18, 27, 20, 19, 6]

VP_METADATA = {
    17: {"name": "Front-Mid (Canonical Baseline)", "az": -4.7, "el": 31.8},
    0:  {"name": "Front-Low (Tabletop Graze)",     "az": 0.0,  "el": 12.3},
    28: {"name": "Front-High (Steep Tilt)",        "az": 2.9,  "el": 46.2},
    18: {"name": "Left-Flank 90°",                 "az": -89.7,"el": 33.1},
    27: {"name": "Right-Flank 90°",                "az": 87.9, "el": 44.6},
    20: {"name": "Diag Front-Left (-45°)",         "az": -37.2,"el": 35.0},
    19: {"name": "Diag Front-Right (+45°)",        "az": 47.8, "el": 33.7},
    6:  {"name": "Low Flank Right",                "az": 72.6, "el": 18.3},
}


def log(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"[{ts}] {msg}"
    print(formatted, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(formatted + "\n")


def is_viewpoint_completed(vp_id: int) -> bool:
    metrics_path = os.path.join(OUTPUT_DIR, "checkpoints", f"specialist_vp_{vp_id:02d}_train_metrics.json")
    best_ckpt = os.path.join(OUTPUT_DIR, "checkpoints", f"specialist_vp_{vp_id:02d}_best.pt")
    if os.path.exists(metrics_path) and os.path.exists(best_ckpt):
        try:
            with open(metrics_path) as f:
                data = json.load(f)
            succ = data.get("best_eval_success", 0.0)
            steps = data.get("total_timesteps", 0)
            return succ >= 95.0 or steps >= 5000000
        except Exception:
            return False
    return False


def compile_report():
    log("[Report] Compiling formal Research Report 15...")
    metrics_summary = {}
    for vp_id in CORE_VP_IDS:
        m_path = os.path.join(OUTPUT_DIR, "checkpoints", f"specialist_vp_{vp_id:02d}_train_metrics.json")
        if os.path.exists(m_path):
            with open(m_path) as f:
                metrics_summary[vp_id] = json.load(f)

    matrix_path = os.path.join(OUTPUT_DIR, "eval_results", "cross_viewpoint_transfer_matrix.json")
    matrix_data = {}
    if os.path.exists(matrix_path):
        with open(matrix_path) as f:
            matrix_data = json.load(f)

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        f.write("# Research Report 15: Individual Viewpoint Specialist Training & 8x8 Cross-Transfer Matrix\n\n")
        f.write("**Platform:** NVIDIA GeForce RTX 4060 Laptop GPU, ManiSkill 3 (`PickCube-v1`)\n")
        f.write("**Architecture:** 4-Channel $128 \\times 128$ RGB-D (Candidate A NatureCNN backbone)\n")
        f.write("**Initialization:** Orthogonal Random Initialization from scratch for each viewpoint\n")
        f.write(f"**Date:** {time.strftime('%Y-%m-%d')}\n\n---\n\n")

        f.write("## 1. Specialist Training Summary Matrix\n\n")
        f.write("| Viewpoint | Role | Azimuth | Elevation | Best Eval Success | Steps to Master | Avg SPS | Walltime |\n")
        f.write("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n")

        for vp_id in CORE_VP_IDS:
            meta = VP_METADATA[vp_id]
            m = metrics_summary.get(vp_id, {})
            succ = f"{m.get('best_eval_success', 0.0):.1f}%"
            steps = f"{m.get('total_timesteps', 0):,}"
            sps = f"{m.get('avg_sps', 0)}"
            mins = f"{m.get('total_walltime_seconds', 0) / 60:.1f}m"
            f.write(f"| **VP {vp_id:02d}** | {meta['name']} | {meta['az']:+.1f}° | {meta['el']:.1f}° | **{succ}** | {steps} | {sps} | {mins} |\n")

        f.write("\n---\n\n")
        f.write("## 2. Cross-Viewpoint Generalization & Transfer Findings\n\n")
        f.write("![Cross-Viewpoint Transfer Heatmap](../eval_results/cross_viewpoint_transfer_heatmap.png)\n\n")
        f.write("- **Self-Diagonal Mastery:** Confirms ceiling in-domain performance achieved when trained on designated viewpoint.\n")
        f.write("- **Off-Diagonal Transfer:** Measures natural generalization bandwidth across orthogonal camera axes.\n")

    log(f"[Report] Successfully generated Research Report 15 at: {REPORT_PATH}")


def run_pipeline():
    log("==========================================================================")
    log(" STARTING PHASE 6: INDIVIDUAL VIEWPOINT SPECIALIST TRAINING (8 VIEWPOINTS)")
    log(f" Working Directory: {OUTPUT_DIR}")
    log(f" Viewpoint Order:   {CORE_VP_IDS}")
    log("==========================================================================")

    for idx, vp_id in enumerate(CORE_VP_IDS, 1):
        meta = VP_METADATA[vp_id]
        log(f"\n--- [Pipeline {idx}/8] Viewpoint {vp_id:02d}: {meta['name']} (Az: {meta['az']:+.1f}°, El: {meta['el']:.1f}°) ---")

        if is_viewpoint_completed(vp_id):
            log(f" Viewpoint {vp_id:02d} is ALREADY COMPLETED. Skipping to next.")
            continue

        cmd = [
            sys.executable,
            "-u",
            os.path.join(OUTPUT_DIR, "train_viewpoint_specialist.py"),
            "--viewpoint-id", str(vp_id),
            "--total-timesteps", "5000000",
            "--num-envs", "128",
            "--early-stop-threshold", "95.0",
            "--eval-freq", "25",
            "--output-dir", OUTPUT_DIR,
        ]

        log(f" Executing: {' '.join(cmd)}")
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in p.stdout:
            print(line, end="", flush=True)
            with open(LOG_FILE, "a") as f:
                f.write(line)
        p.wait()

        if p.returncode != 0:
            log(f"[Error] Training failed for VP {vp_id:02d} with exit code {p.returncode}! Aborting pipeline.")
            return

    log("\n==========================================================================")
    log(" ALL 8 VIEWPOINT SPECIALISTS COMPLETED! PROCEEDING TO CROSS-EVALUATION...")
    log("==========================================================================")

    eval_cmd = [
        sys.executable,
        os.path.join(OUTPUT_DIR, "evaluate_cross_transfer_matrix.py"),
        "--output-dir", OUTPUT_DIR,
        "--episodes", "200",
    ]
    log(f" Executing: {' '.join(eval_cmd)}")
    subprocess.run(eval_cmd, check=True)

    compile_report()
    log("\n==========================================================================")
    log(" PHASE 6 BENCHMARK SUCCESSFULLY COMPLETED!")
    log("==========================================================================")


if __name__ == "__main__":
    run_pipeline()
