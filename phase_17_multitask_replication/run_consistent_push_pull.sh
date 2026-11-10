#!/bin/bash
set -e

# Increase file descriptor limit for GPU parallelized Vulkan/EGL rendering
ulimit -n 65535 || true

REPO_DIR="/home/maniskill/Multi-viewpoint_Generalization"
PHASE17_DIR="$REPO_DIR/phase_17_multitask_replication"
LOG_DIR="$PHASE17_DIR/logs"
mkdir -p "$LOG_DIR"

LOG_FILE="$LOG_DIR/consistent_push_pull_$(date +%Y%m%d_%H%M%S).log"
LATEST_LOG="$LOG_DIR/latest_consistent_run.log"

echo "========================================================================" | tee -a "$LOG_FILE"
echo " STARTING CONSISTENT MULTI-TASK VIEWPOINT TRAINING PIPELINE" | tee -a "$LOG_FILE"
echo " Tasks: PushCube-v1 & PullCube-v1" | tee -a "$LOG_FILE"
echo " Regime: 1,920 Episodes (96,000 Transitions) | 25 Epochs (Matching PickCube)" | tee -a "$LOG_FILE"
echo " Teachers: 100% Verified PPO Oracles (pd_ee_delta_pose, 7-dim)" | tee -a "$LOG_FILE"
echo " Time: $(date)" | tee -a "$LOG_FILE"
echo " Log file: $LOG_FILE" | tee -a "$LOG_FILE"
echo "========================================================================" | tee -a "$LOG_FILE"

cp "$LOG_FILE" "$LATEST_LOG" || true

# Step 1: Run PushCube-v1
echo -e "\n>>> [1/2] TRAINING & BENCHMARKING PushCube-v1 (96K TRANSITIONS) ..." | tee -a "$LOG_FILE"
python3 -u "$PHASE17_DIR/train_push_pull_consistent.py" --task PushCube-v1 --episodes 1920 --epochs 25 2>&1 | tee -a "$LOG_FILE"

# Step 2: Run PullCube-v1
echo -e "\n>>> [2/2] TRAINING & BENCHMARKING PullCube-v1 (96K TRANSITIONS) ..." | tee -a "$LOG_FILE"
python3 -u "$PHASE17_DIR/train_push_pull_consistent.py" --task PullCube-v1 --episodes 1920 --epochs 25 2>&1 | tee -a "$LOG_FILE"

# Step 3: Synthesize master comparison plot across all tasks
echo -e "\n>>> SYNTHESIZING MASTER MULTI-TASK BENCHMARK PLOT ..." | tee -a "$LOG_FILE"
python3 -u "$PHASE17_DIR/train_push_pull_consistent.py" --task synthesize 2>&1 | tee -a "$LOG_FILE"

# Permissions
chmod -R 777 "$REPO_DIR" /home/maniskill/AGENT_HANDOFF.md /home/maniskill/TODO.md || true

echo "========================================================================" | tee -a "$LOG_FILE"
echo " CONSISTENT MULTI-TASK TRAINING COMPLETED AT $(date)" | tee -a "$LOG_FILE"
echo "========================================================================" | tee -a "$LOG_FILE"
