#!/usr/bin/env bash
# scripts/stop_overnight.sh
# Gracefully stops the overnight supervisor watchdog and pauses active training.
set -euo pipefail

EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STATE_DIR="${EXP_STATE_DIR:-$EXP_DIR/state}"
SESSION_NAME="maniskill_supervisor"

echo "=============================================================================="
echo " STOPPING OVERNIGHT TRAINING SUPERVISOR"
echo "=============================================================================="

# 1. Stop watchdog process gracefully if lock exists
LOCK_FILE="$STATE_DIR/watchdog.lock"
if [ -f "$LOCK_FILE" ]; then
    WATCHDOG_PID=$(python3 -c "import json; print(json.load(open('$LOCK_FILE')).get('pid', ''))" 2>/dev/null || true)
    if [ -n "$WATCHDOG_PID" ] && kill -0 "$WATCHDOG_PID" 2>/dev/null; then
        echo " Sending SIGTERM to watchdog supervisor (PID $WATCHDOG_PID)..."
        kill -TERM "$WATCHDOG_PID"
        for i in {1..20}; do
            if ! kill -0 "$WATCHDOG_PID" 2>/dev/null; then
                echo " Watchdog supervisor process (PID $WATCHDOG_PID) exited cleanly."
                break
            fi
            sleep 0.5
        done
    fi
    rm -f "$LOCK_FILE"
fi

# 2. Pause training cleanly via ctl.py pause
echo " Pausing training cleanly via python3 src/ctl.py pause..."
python3 "$EXP_DIR/src/ctl.py" pause

# 3. Terminate tmux session if open
if command -v tmux >/dev/null 2>&1 && tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo " Terminating supervisor tmux session '$SESSION_NAME'..."
    tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true
fi

echo " Overnight supervisor and training stopped cleanly."
echo "=============================================================================="
