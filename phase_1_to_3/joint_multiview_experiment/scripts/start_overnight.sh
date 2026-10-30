#!/usr/bin/env bash
# scripts/start_overnight.sh
# Starts the overnight supervisor watchdog inside a detached tmux session
# under systemd-inhibit (--what=sleep:idle) to prevent laptop sleeping.
set -euo pipefail

EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION_NAME="maniskill_supervisor"

echo "=============================================================================="
echo " STARTING OVERNIGHT TRAINING SUPERVISOR"
echo "=============================================================================="

# 1. Verify tmux is available
if ! command -v tmux >/dev/null 2>&1; then
    echo "Error: tmux is required but not installed." >&2
    exit 1
fi

# 2. Check if supervisor tmux session is already running
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "Notice: Tmux session '$SESSION_NAME' is ALREADY RUNNING."
    echo ""
    echo "  Attach to session:   tmux attach -t $SESSION_NAME"
    echo "  Check ctl status:    python3 src/ctl.py status"
    echo "  View live logs:      tail -f state/watchdog.log"
    echo "  Stop supervisor:     bash scripts/stop_overnight.sh"
    echo "=============================================================================="
    exit 0
fi

# 3. Determine systemd-inhibit command
if command -v systemd-inhibit >/dev/null 2>&1; then
    INHIBIT="systemd-inhibit --what=sleep:idle --why=\"ManiSkill Joint Training Overnight Supervision\""
    echo " Sleep inhibitor: systemd-inhibit (--what=sleep:idle)"
else
    INHIBIT=""
    echo " Sleep inhibitor: systemd-inhibit not available (running directly)"
fi

# 4. Launch supervisor inside detached tmux session
ENV_VARS=""
if [ -n "${EXP_STATE_DIR:-}" ]; then
    ENV_VARS="export EXP_STATE_DIR=\"$EXP_STATE_DIR\"; "
fi
if [ -n "${EXP_RUNS_DIR:-}" ]; then
    ENV_VARS="${ENV_VARS}export EXP_RUNS_DIR=\"$EXP_RUNS_DIR\"; "
fi
RUN_CMD="cd \"$EXP_DIR\" && ${ENV_VARS}$INHIBIT python3 scripts/watchdog.py $*"
tmux new-session -d -s "$SESSION_NAME" "$RUN_CMD"

echo " Session name:   $SESSION_NAME (detached)"
echo " Watchdog log:   state/watchdog.log"
echo " Runner log:     state/runner.log"
echo " Report file:    state/OVERNIGHT_REPORT.md"
echo ""
echo "------------------------------------------------------------------------------"
echo " SUPERVISOR CONTROLS & MONITORING:"
echo "   Attach to tmux session:   tmux attach -t $SESSION_NAME"
echo "   Detach from tmux session: Ctrl+B then D"
echo "   View live watchdog log:   tail -f state/watchdog.log"
echo "   Check experiment status:  python3 src/ctl.py status"
echo "   Generate status report:   python3 src/ctl.py report"
echo "   User manual pause:        python3 src/ctl.py pause"
echo "   User manual resume:       python3 src/ctl.py resume"
echo "   Stop supervisor & run:    bash scripts/stop_overnight.sh"
echo "=============================================================================="
