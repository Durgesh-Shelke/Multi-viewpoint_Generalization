#!/usr/bin/env bash
# ==============================================================================
# Master Launcher for Overnight Unbiased Resolution & Asymmetric Fusion Benchmark
# Candidates:
#   A: Symmetric 128x128 RGB-D (Baseline)
#   B: Symmetric 256x256 RGB-D
#   C: Asymmetric Decoupled (Low-Res RGB 64x64 + High-Res Depth 256x256)
#   D: Pure Depth 256x256 (RGB Discarded)
#   E: Pure RGB 256x256 (Depth Discarded)
#
# Total Steps per Candidate: 10,000,000 steps (default, configurable)
# Parallel GPU Instances:    128 envs (Train) | 100 envs (Battery Eval)
# Estimated Total Walltime:  ~11.0 hours (10M) or ~8.8 hours (8M)
# ==============================================================================

set -e

echo "=========================================================================="
echo " STARTING OVERNIGHT RESOLUTION & SENSOR FUSION BENCHMARK STUDY"
echo " Timestamp: $(date)"
echo " Working Dir: /home/maniskill/phase_5"
echo " Candidates:  A B C D E"
echo "=========================================================================="

python3 /home/maniskill/phase_5/run_master_unbiased_benchmark.py \
  --total-timesteps 10000000 \
  --candidates A B C D E \
  --num-envs 128 \
  --eval-num-envs 100 \
  --eval-target-episodes 1000 \
  --eval-freq 25 "$@"
