#!/bin/bash
set -e

# Increase file descriptor limit for GPU parallelized Vulkan/EGL rendering
ulimit -n 65535 || true

REPO_DIR="/home/maniskill/Multi-viewpoint_Generalization"
PHASE17_DIR="$REPO_DIR/phase_17_multitask_replication"
LOG_FILE="$PHASE17_DIR/logs/training_$(date +%Y%m%d_%H%M%S).log"
LATEST_LOG="$PHASE17_DIR/logs/latest_training.log"
ARTIFACT_DIR="/root/.gemini/antigravity-cli/brain/789cffef-f152-46d8-aadb-5a046942c35f"

echo "========================================================================"
echo " STARTING AUTOMATED MULTI-TASK TRAINING RUNNER"
echo " Time: $(date)"
echo " Log file: $LOG_FILE"
echo " File descriptor limit: $(ulimit -n)"
echo "========================================================================"

mkdir -p "$PHASE17_DIR/logs"

# Check if PushCube is already trained & evaluated
if [ ! -f "$PHASE17_DIR/eval_results/pushcube-v1_viewpoint_benchmark.json" ]; then
    echo "--- Step 1: Training & Benchmarking PushCube-v1 ---"
    python3 -u "$PHASE17_DIR/train_multitask_viewpoint_models.py" --task PushCube-v1 2>&1 | tee -a "$LOG_FILE"
else
    echo "--- Step 1: PushCube-v1 already trained and verified! [SKIPPING] ---"
fi

# Step 2: Run PullCube-v1
echo "--- Step 2: Training & Benchmarking PullCube-v1 ---"
python3 -u "$PHASE17_DIR/train_multitask_viewpoint_models.py" --task PullCube-v1 2>&1 | tee -a "$LOG_FILE"

# Step 3: Synthesize master comparison plot across all tasks
echo "--- Step 3: Generating Multi-Task Master Synthesis Plot ---"
python3 -u "$PHASE17_DIR/train_multitask_viewpoint_models.py" --synthesize 2>&1 | tee -a "$LOG_FILE"

echo "========================================================================"
echo " POST-TRAINING SYNCHRONIZATION & ASSET DISPATCH"
echo "========================================================================"

# Copy generated master plots to Artifact Directory
if [ -d "$ARTIFACT_DIR" ]; then
    cp -v "$PHASE17_DIR"/plots/*.png "$ARTIFACT_DIR/" || true
fi

# Ensure full permissions
chmod -R 777 "$REPO_DIR" /home/maniskill/AGENT_HANDOFF.md /home/maniskill/TODO.md || true

# Git commit and push results
cd "$REPO_DIR"
git add .
git commit -m "feat(phase_17): completed multi-task replication on PushCube and PullCube with master synthesis plot" || true
git push origin main || true

echo "========================================================================"
echo " AUTOMATED TRAINING & BENCHMARKING COMPLETE AT $(date)"
echo "========================================================================"
