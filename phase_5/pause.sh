#!/usr/bin/env bash
# ==============================================================================
# Script to cleanly pause the running benchmark without killing it
# ==============================================================================

DIR="/home/maniskill/phase_5"
PAUSE_FILE="$DIR/.pause"

touch "$PAUSE_FILE"

echo "=========================================================================="
echo " [PAUSE] Pause signal triggered: $PAUSE_FILE"
echo " Active training process will pause at the next iteration (~2-5 seconds),"
echo " save its complete checkpoint, and idle the GPU."
echo ""
echo " To monitor status:  /home/maniskill/phase_5/status.sh"
echo " To resume training: /home/maniskill/phase_5/resume.sh"
echo "=========================================================================="
