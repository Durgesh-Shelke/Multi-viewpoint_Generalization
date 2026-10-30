#!/usr/bin/env bash
set -e

DIR="/home/maniskill/phase_6_viewpoint_specialists"
LOG_FILE="$DIR/master_execution.log"

cd "$DIR"

echo "=========================================================================="
echo " STARTING PHASE 6: 8-VIEWPOINT SPECIALIST BENCHMARK"
echo " Timestamp: $(date)"
echo " Working Dir: $DIR"
echo " Viewpoints: 17 (Front-Mid), 00 (Front-Low), 28 (Front-High),"
echo "             18 (Left-90°), 27 (Right-90°),"
echo "             20 (Diag-Left), 19 (Diag-Right), 06 (Low-Flank-Right)"
echo " Budget per Model: Up to 5,000,000 steps with early stopping at >=95% eval"
echo " Logging to: $LOG_FILE"
echo "=========================================================================="

python3 "$DIR/run_master_phase6.py"
