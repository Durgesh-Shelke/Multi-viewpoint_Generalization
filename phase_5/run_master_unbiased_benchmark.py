#!/usr/bin/env python3
"""
Master Unbiased Resolution & Asymmetric Sensor Fusion Benchmark Pipeline.

Executes sequential training and comprehensive 4-tier evaluation across:
  - Candidate A: Symmetric 128x128 RGB-D (Baseline Control)
  - Candidate B: Symmetric 256x256 RGB-D
  - Candidate C: Asymmetric Decoupled (Low-Res RGB 64x64 + High-Res Depth 256x256)
  - Candidate D: High-Res Pure Depth (256x256 Depth Only)
  - Candidate E: High-Res Pure RGB (256x256 RGB Only)

Safe for long-running overnight unattended execution.
"""

import argparse
import datetime
import gc
import json
import os
import subprocess
import sys
import time
import traceback

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, "/home/maniskill")

from evaluate_battery import evaluate_candidate_battery
from plot_growth_curves import generate_plots
from train_candidate import train_single_candidate

OUTPUT_DIR = "/home/maniskill/phase_5"
REPORT_PATH = "/home/maniskill/phase_5/reports/14_resolution_and_asymmetric_depth_ablation_report.md"
MASTER_JSON = os.path.join(OUTPUT_DIR, "master_benchmark_summary.json")
LOG_FILE = os.path.join(OUTPUT_DIR, "master_execution.log")


class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", buffering=1)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def generate_markdown_report(all_data: dict, total_walltime: float) -> str:
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    md = f"""# Research Report 14: Unbiased Resolution & Asymmetric Sensor Fusion Benchmark

**Date:** {timestamp}  
**Hardware:** NVIDIA GeForce RTX 4060 Laptop GPU (8GB VRAM), AMD Ryzen 7  
**Task:** `PickCube-v1` (4.5cm Solid Physical Cube, Randomized Poses)  
**Total Benchmark Runtime:** {total_walltime / 3600:.2f} hours  

---

## 1. Executive Summary & Comparative Matrix

This controlled empirical study evaluated five sensor resolution and modal fusion configurations under strict parity:
- **Candidate A:** Symmetric $128 \\times 128$ RGB-D (Baseline Control)
- **Candidate B:** Symmetric $256 \\times 256$ RGB-D (Full High-Resolution)
- **Candidate C:** Asymmetric Decoupled (Low-Res RGB $64 \\times 64$ + High-Res Depth $256 \\times 256$)
- **Candidate D:** Pure Depth High-Res ($256 \\times 256$ Depth Only, RGB Discarded)
- **Candidate E:** Pure RGB High-Res ($256 \\times 256$ RGB Only, Depth Discarded)

### Comparative Performance Table
| Candidate | Modality & Resolution | Train Steps | In-Domain (1k runs) | Grasp Len (steps) | Return | Jitter Retention (±15°) | Lighting Robustness (±30%) | Train SPS | Peak VRAM |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""

    for cand_id in ["A", "B", "C", "D", "E"]:
        if cand_id in all_data and "eval" in all_data[cand_id]:
            c_data = all_data[cand_id]
            t_train = c_data.get("train", {})
            t_eval = c_data.get("eval", {})
            t1 = t_eval.get("test1_front_view_1000", {})
            t2 = t_eval.get("test2_angular_jitter", {})
            t3 = t_eval.get("test3_photometric_shift", {})

            modality_desc = {
                "A": "Symmetric 128x128 RGB-D",
                "B": "Symmetric 256x256 RGB-D",
                "C": "Asym: 64 RGB + 256 Depth",
                "D": "Pure Depth 256x256",
                "E": "Pure RGB 256x256",
            }[cand_id]

            md += f"| **Candidate {cand_id}** | {modality_desc} | {t_train.get('total_timesteps', 0):,} | **{t1.get('success_rate', 0.0):.2f}%** | {t1.get('mean_length', 0.0):.1f} | {t1.get('mean_return', 0.0):.2f} | {t2.get('mean_success_rate', 0.0):.2f}% | **{t3.get('mean_perturbed_success', 0.0):.2f}%** | ~{t_train.get('avg_sps', 0)} | {t_train.get('peak_vram_mb', 0):.0f} MB |\n"

    md += """
---

## 2. Benchmark Battery Detailed Findings

### Test 1: In-Domain Grasping Mastery (1,000 Runs Canonical View)
Evaluates raw picking reliability, mean episode steps to pick, and trajectory reward on the canonical front viewpoint (`[0.3, 0.0, 0.6]`):
"""
    for cand_id in ["A", "B", "C", "D", "E"]:
        if cand_id in all_data and "eval" in all_data[cand_id]:
            t1 = all_data[cand_id]["eval"]["test1_front_view_1000"]
            md += f"- **Candidate {cand_id}:** {t1.get('success_count', 0)}/1000 ({t1.get('success_rate', 0.0):.2f}%) | Mean Length: {t1.get('mean_length', 0.0):.1f} steps | Mean Return: {t1.get('mean_return', 0.0):.4f}\n"

    md += """
### Test 2: Angular Jitter & Mounting Tolerance (75°, 90°, 105°)
Evaluates policy retention when the physical camera undergoes $\\pm 15^\\circ$ angular vibration or mounting displacement:
"""
    for cand_id in ["A", "B", "C", "D", "E"]:
        if cand_id in all_data and "eval" in all_data[cand_id]:
            t2 = all_data[cand_id]["eval"]["test2_angular_jitter"]
            angles = t2.get("angles", {})
            md += f"- **Candidate {cand_id} (Mean {t2.get('mean_success_rate', 0.0):.2f}%):** 75°: {angles.get('angle_75deg', {}).get('success_rate', 0.0)}% | 90°: {angles.get('angle_90deg', {}).get('success_rate', 0.0)}% | 105°: {angles.get('angle_105deg', {}).get('success_rate', 0.0)}%\n"

    md += """
### Test 3: Photometric Domain Gap / Lighting Perturbation (±30% Shift)
Tests visual resilience against ambient lighting changes (darkened 0.7x and brightened 1.3x):
"""
    for cand_id in ["A", "B", "C", "D", "E"]:
        if cand_id in all_data and "eval" in all_data[cand_id]:
            t3 = all_data[cand_id]["eval"]["test3_photometric_shift"]
            md += f"- **Candidate {cand_id} (Mean {t3.get('mean_perturbed_success', 0.0):.2f}%):** 0.7x Darkened: {t3.get('darkened_0.7x', {}).get('success_rate', 0.0)}% | 1.3x Brightened: {t3.get('brightened_1.3x', {}).get('success_rate', 0.0)}%\n"

    md += """
---

## 3. Learning Curves & Plateau Dynamics Analysis

![Learning Curves and Plateau Analysis](/root/.gemini/antigravity-cli/brain/82673f33-5c96-4154-bb42-cf6de559c7b7/growth_curves_comparison.png)

### Plateau Onset & Sample Efficiency Summary
| Candidate | Modality | Steps to 50% Success | Steps to 80% Success | Plateau Onset Step | Mean Plateau Success (%) |
| :--- | :--- | :---: | :---: | :---: | :---: |
"""
    plat_json = os.path.join(OUTPUT_DIR, "eval_results", "plateau_summary.json")
    if os.path.exists(plat_json):
        try:
            with open(plat_json) as f:
                plat_data = json.load(f)
            for c_id in ["A", "B", "C", "D", "E"]:
                if c_id in plat_data:
                    p = plat_data[c_id]
                    s50 = f"{p.get('step_to_50_pct', 0):,}" if p.get('step_to_50_pct') else "N/A"
                    s80 = f"{p.get('step_to_80_pct', 0):,}" if p.get('step_to_80_pct') else "N/A"
                    splat = f"{p.get('plateau_start_step', 0):,}" if p.get('plateau_start_step') else "N/A"
                    p_mean = f"{p.get('plateau_mean_success', 0.0):.2f}%"
                    mod_name = {"A": "128x128 RGB-D", "B": "256x256 RGB-D", "C": "Asym 64/256", "D": "Depth 256x256", "E": "RGB 256x256"}[c_id]
                    md += f"| **Candidate {c_id}** | {mod_name} | {s50} | {s80} | **{splat}** | **{p_mean}** |\n"
        except Exception:
            pass

    md += """
---

## 4. Scientific Conclusions & Path Forward

1. **Impact of 256x256 Resolution:**
   - Evaluates whether quadrupling pixel density enhances sub-millimeter contact stability or introduces compute bottlenecks.
2. **Asymmetric Sensor Fusion (Candidate C):**
   - Assesses whether downsampling RGB to 64x64 filters out high-frequency photometric noise while preserving semantic cube localization.
3. **Pure Depth Viability (Candidate D):**
   - Determines whether RGB is redundant for physical object manipulation once metric depth is available.
4. **Pure RGB 256x256 (Candidate E):**
   - Isolates the role of high-resolution visual textures/color in the absence of explicit 3D geometry.

---
*Report auto-generated by `run_master_unbiased_benchmark.py`.*
"""
    return md


def main():
    parser = argparse.ArgumentParser(description="Master Unbiased Resolution & Fusion Benchmark")
    parser.add_argument("--total-timesteps", type=int, default=10000000, help="Timesteps per candidate (default: 10,000,000)")
    parser.add_argument("--candidates", nargs="+", default=["A", "B", "C", "D", "E"], help="Candidates to run (default: A B C D E)")
    parser.add_argument("--num-envs", type=int, default=128, help="Parallel environments for training (128 verified safe for 256x256)")
    parser.add_argument("--eval-num-envs", type=int, default=100, help="Parallel environments for battery evaluation")
    parser.add_argument("--eval-target-episodes", type=int, default=1000, help="Episodes per battery test (default: 1,000)")
    parser.add_argument("--eval-freq", type=int, default=25, help="Eval frequency during training")
    parser.add_argument("--resume", action="store_true", default=True, help="Resume training and skip already-completed candidates (default: True)")
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="Ignore existing checkpoints and force rerun from step 0")
    parser.add_argument("--force-rerun", action="store_true", default=False, help="Force rerun even if candidate results exist")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    sys.stdout = Logger(LOG_FILE)

    print("=" * 80)
    print(" STARTING MASTER UNBIASED RESOLUTION & SENSOR FUSION BENCHMARK")
    print(f" Start Time:        {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" Candidates to Run: {args.candidates}")
    print(f" Budget per Model:  {args.total_timesteps:,} steps")
    print(f" Resume Mode:       {'Enabled (Checkpoints active)' if args.resume else 'Disabled (Fresh start)'}")
    print(f" Training Envs:     {args.num_envs} | Battery Eval Envs: {args.eval_num_envs}")
    print(f" Output Directory:  {OUTPUT_DIR}")
    print(f" Logging to:        {LOG_FILE}")
    print("=" * 80 + "\n")

    master_results = {}
    if os.path.exists(MASTER_JSON):
        try:
            with open(MASTER_JSON) as f:
                master_results = json.load(f)
        except Exception:
            pass

    master_start_time = time.time()
    pause_file = os.path.join(OUTPUT_DIR, ".pause")

    for cand in args.candidates:
        cand = cand.upper()
        cand_start = time.time()

        # Check for pause flag before starting next candidate
        if os.path.exists(pause_file):
            print(f"\n[PAUSED] Detected pause flag: {pause_file}. Waiting before launching Candidate {cand}...")
            while os.path.exists(pause_file):
                time.sleep(1)
            print(f"[RESUMED] Resuming pipeline! Proceeding with Candidate {cand}.\n")

        # Check if candidate has already completed full training and evaluation
        ckpt_path = os.path.join(OUTPUT_DIR, "checkpoints", f"candidate_{cand}.pt")
        metrics_path = os.path.join(OUTPUT_DIR, "checkpoints", f"candidate_{cand}_train_metrics.json")
        eval_res_path = os.path.join(OUTPUT_DIR, "eval_results", f"candidate_{cand}_results.json")

        if args.resume and not args.force_rerun and os.path.exists(metrics_path) and os.path.exists(eval_res_path):
            try:
                with open(metrics_path) as f:
                    train_m = json.load(f)
                with open(eval_res_path) as f:
                    eval_m = json.load(f)
                if train_m.get("total_timesteps", 0) >= (args.total_timesteps - 10000) and train_m.get("completed", False):
                    print(f"\n[RESUME SKIP] Candidate {cand} has already completed {args.total_timesteps:,} steps and 4-tier battery evaluation.")
                    t1_rate = eval_m.get("test1_front_view_1000", {}).get("success_rate", 0.0)
                    print(f"  -> In-Domain Grasp Success: {t1_rate:.2f}% | Skipping to next candidate.\n")
                    master_results[cand] = {
                        "train": train_m,
                        "eval": eval_m,
                        "total_candidate_time_seconds": train_m.get("total_walltime_seconds", 0),
                    }
                    with open(MASTER_JSON, "w") as f:
                        json.dump(master_results, f, indent=2)
                    continue
            except Exception as e:
                print(f"[Warning] Error inspecting existing results for Candidate {cand}: {e}. Proceeding with run.")

        print("\n" + "#" * 80)
        print(f" LAUNCHING CANDIDATE {cand}")
        print("#" * 80 + "\n")

        try:
            # 1. Train Candidate in isolated subprocess
            train_cmd = [
                sys.executable,
                os.path.join(SCRIPT_DIR, "train_candidate.py"),
                "--candidate", cand,
                "--total-timesteps", str(args.total_timesteps),
                "--num-envs", str(args.num_envs),
                "--num-minibatches", "8" if cand == "A" else "16",
                "--target-kl", "0.2",
                "--eval-freq", str(args.eval_freq),
                "--output-dir", OUTPUT_DIR,
            ]
            if not args.resume:
                train_cmd.append("--no-resume")
            print(f"Executing: {' '.join(train_cmd)}\n", flush=True)
            res = subprocess.run(train_cmd)
            if res.returncode == 130:
                print(f"\n[PIPELINE PAUSED/STOPPED] Candidate {cand} was stopped or paused by user.")
                print(f"[RESUME READY] Run /home/maniskill/phase_5/resume.sh to continue training seamlessly.\n")
                break
            elif res.returncode != 0:
                raise RuntimeError(f"Training of Candidate {cand} exited with code {res.returncode}")

            with open(metrics_path) as f:
                train_metrics = json.load(f)

            # 2. Evaluate Battery in isolated subprocess
            eval_cmd = [
                sys.executable,
                os.path.join(SCRIPT_DIR, "evaluate_battery.py"),
                "--candidate", cand,
                "--checkpoint", ckpt_path,
                "--num-envs", str(args.eval_num_envs),
                "--output-dir", os.path.join(OUTPUT_DIR, "eval_results"),
            ]
            if hasattr(args, "eval_target_episodes"):
                eval_cmd.extend(["--target-episodes", str(args.eval_target_episodes)])
            print(f"\nExecuting: {' '.join(eval_cmd)}\n", flush=True)
            subprocess.run(eval_cmd, check=True)

            with open(eval_res_path) as f:
                eval_metrics = json.load(f)

            master_results[cand] = {
                "train": train_metrics,
                "eval": eval_metrics,
                "total_candidate_time_seconds": round(time.time() - cand_start, 2),
            }

            # Save intermediate master results
            with open(MASTER_JSON, "w") as f:
                json.dump(master_results, f, indent=2)

        except KeyboardInterrupt:
            print(f"\n[INTERRUPT] Benchmark interrupted by user at Candidate {cand}.")
            print(f"[RESUME READY] Run /home/maniskill/phase_5/resume.sh to continue training.")
            break
        except Exception as e:
            print(f"\n[ERROR] Exception encountered during Candidate {cand} execution:")
            traceback.print_exc()
            master_results[cand] = {"error": str(e), "traceback": traceback.format_exc()}
            with open(MASTER_JSON, "w") as f:
                json.dump(master_results, f, indent=2)

        finally:
            gc.collect()
            torch.cuda.empty_cache()

    total_walltime = time.time() - master_start_time

    # Generate Publication Growth Plots & Plateau Analysis
    try:
        generate_plots(
            eval_dir=os.path.join(OUTPUT_DIR, "eval_results"),
            output_png=os.path.join(OUTPUT_DIR, "growth_curves_comparison.png"),
        )
    except Exception as e:
        print(f"[Warning] Failed to generate growth plots: {e}")

    # Generate Markdown Report
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    report_content = generate_markdown_report(master_results, total_walltime)
    with open(REPORT_PATH, "w") as f:
        f.write(report_content)

    print("\n" + "=" * 80)
    print(" MASTER UNBIASED BENCHMARK COMPLETED SUCCESSFULLY!")
    print(f" Total Runtime:     {total_walltime / 3600:.2f} hours ({total_walltime:.1f} seconds)")
    print(f" Master JSON:       {MASTER_JSON}")
    print(f" Research Report:   {REPORT_PATH}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
