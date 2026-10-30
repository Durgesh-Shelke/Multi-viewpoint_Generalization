#!/usr/bin/env bash
# ==============================================================================
# Resume script for Phase 6 Viewpoint Specialists
# ==============================================================================
DIR="/home/maniskill/phase_6_viewpoint_specialists"
PAUSE_FILE="$DIR/.pause"
LOG_FILE="$DIR/master_execution.log"

if [ -f "$PAUSE_FILE" ]; then
    echo "[RESUME] Removing pause flag at $PAUSE_FILE..."
    rm -f "$PAUSE_FILE"
fi

MASTER_PID=$(pgrep -f "run_master_phase6.py" || true)
TRAIN_PID=$(pgrep -f "train_viewpoint_specialist.py" || true)

if [ -n "$TRAIN_PID" ] || [ -n "$MASTER_PID" ]; then
    echo "[RESUME] Process is alive (Master PID: ${MASTER_PID:-N/A}, Train PID: ${TRAIN_PID:-N/A})."
    echo "[RESUME] It will immediately wake up and resume training without loss."
    echo "--------------------------------------------------------------------------"
    echo " To monitor status:  $DIR/status.sh"
    echo " Logs:               tail -f $LOG_FILE"
    echo "--------------------------------------------------------------------------"
else
    echo "[RESTART] No active training process found. Relaunching pipeline in background..."
    nohup /usr/bin/python3 "$DIR/run_master_phase6.py" > "$LOG_FILE" 2>&1 &
    NEW_PID=$!
    echo "[RESTART] Pipeline launched in background with PID: $NEW_PID."
    echo "--------------------------------------------------------------------------"
    echo " To monitor status:  $DIR/status.sh"
    echo " Logs:               tail -f $LOG_FILE"
    echo "--------------------------------------------------------------------------"
fi
