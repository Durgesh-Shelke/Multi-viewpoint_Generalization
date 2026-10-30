#!/usr/bin/env bash
# ==============================================================================
# Script to resume the benchmark (either live unpause or restart from checkpoint)
# ==============================================================================

DIR="/home/maniskill/phase_5"
PAUSE_FILE="$DIR/.pause"
LOG_FILE="$DIR/master_execution.log"

# 1. Remove pause flag if present
if [ -f "$PAUSE_FILE" ]; then
    rm -f "$PAUSE_FILE"
    echo " [RESUME] Removed pause flag: $PAUSE_FILE"
fi

# 2. Check if benchmark process is already active
RUNNING_PID=$(pgrep -f "run_master_unbiased_benchmark.py" || true)

if [ -n "$RUNNING_PID" ]; then
    echo "=========================================================================="
    echo " [RESUME] Benchmark process is currently running (PID: $RUNNING_PID)!"
    echo " Training will unpause and resume instantly."
    echo ""
    echo " To monitor status:  /home/maniskill/phase_5/status.sh"
    echo " To watch live logs: tail -f $LOG_FILE"
    echo "=========================================================================="
else
    echo "=========================================================================="
    echo " [RESUME] No active benchmark process detected."
    echo " Starting master benchmark pipeline in the background..."
    echo "   - Completed candidates will be skipped automatically."
    echo "   - Active candidate will resume from its exact iteration/step."
    echo "=========================================================================="
    nohup /home/maniskill/phase_5/run_resolution_ablation.sh > "$LOG_FILE" 2>&1 &
    NEW_PID=$!
    echo " [STARTED] Benchmark launched in background (PID: $NEW_PID)"
    echo ""
    echo " To monitor status:  /home/maniskill/phase_5/status.sh"
    echo " To watch live logs: tail -f $LOG_FILE"
    echo "=========================================================================="
fi
