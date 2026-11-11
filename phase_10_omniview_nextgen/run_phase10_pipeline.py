"""
Master Orchestrator Pipeline for Phase 10 OmniView.
Executes the full pipeline end-to-end:
1. Trains all 8 Teacher Specialists for EXACTLY 5,000,000 steps each (no early stopping).
2. Distills the 8 teachers into the Next-Gen OmniView Student (3-frame stack + extrinsics conditioning).
3. Executes 1,000-Trial Randomized Viewpoint Benchmark.
4. Generates publication figures and summary reports.
"""

import os
import sys
sys.path.append("/home/maniskill")
import time
import json

TEACHERS_DIR = "/home/maniskill/Multi-viewpoint_Generalization/phase_10_omniview_nextgen/teachers"
STUDENT_DIR = "/home/maniskill/Multi-viewpoint_Generalization/phase_10_omniview_nextgen/student"
VIEWPOINTS_JSON = "/home/maniskill/Multi-viewpoint_Generalization/phase_1_to_3/sequential_viewpoint_training/viewpoints.json"
CORE_TEACHERS = [17, 0, 28, 18, 27, 20, 19, 6]


def run_full_phase10_pipeline():
    t_start = time.time()
    os.makedirs(TEACHERS_DIR, exist_ok=True)
    os.makedirs(STUDENT_DIR, exist_ok=True)

    with open(VIEWPOINTS_JSON) as f:
        vps = json.load(f)
    vp_dict = {vp["id"]: vp for vp in vps}

    print("\n" + "=" * 80)
    print(" >>> STARTING MASTER PHASE 10 OMNIVIEW PIPELINE <<<")
    print(f" Viewpoint Anchors: {CORE_TEACHERS}")
    print(f" Budget per Teacher: EXACTLY 5,000,000 steps (Total: 40,000,000 steps)")
    print("=" * 80 + "\n")

    # ---------------------------------------------------------
    # STAGE 1: Train all 8 Teacher Specialists (5M steps each)
    # ---------------------------------------------------------
    from phase_10_omniview_nextgen.train_teacher_specialists_v3 import train_single_teacher

    for idx, vpid in enumerate(CORE_TEACHERS, start=1):
        print(f"\n[Stage 1: Teacher {idx}/8] Viewpoint {vpid:02d} (Budget: 5,000,000 steps)...")
        train_single_teacher(vpid, vp_dict, total_timesteps=5_000_000)

    print("\n" + "=" * 80)
    print(" ALL 8 TEACHER SPECIALISTS COMPLETED 5,000,000 STEPS EACH!")
    print(" LAUNCHING STAGE 2: CONTINUOUS VIEWPOINT JITTER DISTILLATION...")
    print("=" * 80 + "\n")

    # ---------------------------------------------------------
    # STAGE 2: Multi-Teacher Distillation into OmniView Student
    # ---------------------------------------------------------
    from phase_10_omniview_nextgen.distill_omniview_student import train_distillation
    train_distillation(num_epochs=50, steps_per_epoch=40, batch_envs=32, lr=3e-4)

    print("\n" + "=" * 80)
    print(" DISTILLATION COMPLETE! LAUNCHING STAGE 3: 1,000-TRIAL BENCHMARK...")
    print("=" * 80 + "\n")

    # ---------------------------------------------------------
    # STAGE 3: 1,000-Trial Randomized Viewpoint Benchmark
    # ---------------------------------------------------------
    from phase_10_omniview_nextgen.evaluate_omniview_1000_tests import run_1000_benchmark
    overall_acc, viable_acc, per_vp_results = run_1000_benchmark(num_tests=1000)

    total_elapsed = time.time() - t_start
    print("\n" + "=" * 80)
    print(" MASTER PHASE 10 OMNIVIEW PIPELINE FULLY COMPLETED!")
    print(f" Total Runtime: {total_elapsed / 3600:.2f} hours")
    print(f" Final Benchmark Overall Accuracy:      {overall_acc:.2f}%")
    print(f" Final Benchmark Viable Views Accuracy: {viable_acc:.2f}%")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_full_phase10_pipeline()
