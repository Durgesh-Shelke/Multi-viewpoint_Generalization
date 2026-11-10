#!/bin/bash
set -e

REPO_DIR="/home/maniskill/Multi-viewpoint_Generalization"
PHASE17_DIR="$REPO_DIR/phase_17_multitask_replication"
LOG_FILE="$PHASE17_DIR/logs/training_$(date +%Y%m%d_%H%M%S).log"
LATEST_LOG="$PHASE17_DIR/logs/latest_training.log"
ARTIFACT_DIR="/root/.gemini/antigravity-cli/brain/789cffef-f152-46d8-aadb-5a046942c35f"

echo "========================================================================"
echo " STARTING AUTOMATED MULTI-TASK TRAINING RUNNER"
echo " Time: $(date)"
echo " Log file: $LOG_FILE"
echo "========================================================================"

mkdir -p "$PHASE17_DIR/logs"

# Run Python training and evaluation pipeline
python3 -u "$PHASE17_DIR/train_multitask_viewpoint_models.py" 2>&1 | tee "$LOG_FILE" | tee "$LATEST_LOG"

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
git commit -m "feat(phase_17): multi-task viewpoint generalization replication on PushCube and PullCube, checkpoints and master benchmarks" || true
git push origin main || true

echo "========================================================================"
echo " AUTOMATED TRAINING & BENCHMARKING COMPLETE AT $(date)"
echo "========================================================================"
