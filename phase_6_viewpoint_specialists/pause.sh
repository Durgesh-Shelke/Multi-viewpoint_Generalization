#!/usr/bin/env bash
# ==============================================================================
# Non-destructive pause script for Phase 6 Viewpoint Specialists
# ==============================================================================
DIR="/home/maniskill/phase_6_viewpoint_specialists"
PAUSE_FILE="$DIR/.pause"

echo "[PAUSE REQUEST] Setting pause flag at $PAUSE_FILE..."
touch "$PAUSE_FILE"

echo "Pause flag written."
echo "Active training iteration will finish, save latest checkpoint, clear GPU memory, and sleep."
echo "--------------------------------------------------------------------------"
echo " To monitor status:  $DIR/status.sh"
echo " To resume training: $DIR/resume.sh"
echo "--------------------------------------------------------------------------"
