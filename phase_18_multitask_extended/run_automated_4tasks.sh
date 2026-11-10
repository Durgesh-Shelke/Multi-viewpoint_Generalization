#!/usr/bin/env bash
# Phase 18: Automated Execution of 4-Task Suite across Continuous Viewpoints
# Tasks: StackCube-v1 -> PokeCube-v1 -> PegInsertionSide-v1 -> PlugCharger-v1 -> Master Synthesis

set -e
ulimit -n 65535

PYTHON_BIN="/opt/conda/bin/python"
SCRIPT_DIR="/home/maniskill/Multi-viewpoint_Generalization/phase_18_multitask_extended"
LOG_DIR="${SCRIPT_DIR}/logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MAIN_LOG="${LOG_DIR}/training_${TIMESTAMP}.log"

mkdir -p "${LOG_DIR}"

echo "================================================================================" | tee -a "${MAIN_LOG}"
echo " STARTING AUTOMATED 4-TASK EXTENDED VIEWPOINT GENERALIZATION SUITE" | tee -a "${MAIN_LOG}"
echo " Timestamp: ${TIMESTAMP}" | tee -a "${MAIN_LOG}"
echo " Tasks: 1. StackCube-v1 | 2. PokeCube-v1 | 3. PegInsertionSide-v1 | 4. PlugCharger-v1" | tee -a "${MAIN_LOG}"
echo "================================================================================" | tee -a "${MAIN_LOG}"

# Task 1: StackCube-v1
echo -e "\n[$(date)] >>> STARTING TASK 1/4: StackCube-v1 ..." | tee -a "${MAIN_LOG}"
${PYTHON_BIN} ${SCRIPT_DIR}/train_4tasks_suite.py --task StackCube-v1 2>&1 | tee -a "${MAIN_LOG}"
echo "[$(date)] <<< TASK 1/4: StackCube-v1 COMPLETED!" | tee -a "${MAIN_LOG}"

# Task 2: PokeCube-v1
echo -e "\n[$(date)] >>> STARTING TASK 2/4: PokeCube-v1 ..." | tee -a "${MAIN_LOG}"
${PYTHON_BIN} ${SCRIPT_DIR}/train_4tasks_suite.py --task PokeCube-v1 2>&1 | tee -a "${MAIN_LOG}"
echo "[$(date)] <<< TASK 2/4: PokeCube-v1 COMPLETED!" | tee -a "${MAIN_LOG}"

# Task 3: PegInsertionSide-v1
echo -e "\n[$(date)] >>> STARTING TASK 3/4: PegInsertionSide-v1 ..." | tee -a "${MAIN_LOG}"
${PYTHON_BIN} ${SCRIPT_DIR}/train_4tasks_suite.py --task PegInsertionSide-v1 2>&1 | tee -a "${MAIN_LOG}"
echo "[$(date)] <<< TASK 3/4: PegInsertionSide-v1 COMPLETED!" | tee -a "${MAIN_LOG}"

# Task 4: PlugCharger-v1
echo -e "\n[$(date)] >>> STARTING TASK 4/4: PlugCharger-v1 ..." | tee -a "${MAIN_LOG}"
${PYTHON_BIN} ${SCRIPT_DIR}/train_4tasks_suite.py --task PlugCharger-v1 2>&1 | tee -a "${MAIN_LOG}"
echo "[$(date)] <<< TASK 4/4: PlugCharger-v1 COMPLETED!" | tee -a "${MAIN_LOG}"

# Final Master 7-Task Synthesis
echo -e "\n[$(date)] >>> SYNTHESIZING ALL 7 TASKS INTO MASTER COMPARISON PLOT ..." | tee -a "${MAIN_LOG}"
${PYTHON_BIN} ${SCRIPT_DIR}/train_4tasks_suite.py --task synthesize 2>&1 | tee -a "${MAIN_LOG}"
echo "[$(date)] <<< MASTER SYNTHESIS COMPLETED!" | tee -a "${MAIN_LOG}"

echo "================================================================================" | tee -a "${MAIN_LOG}"
echo " ALL 4 EXTENDED TASKS & MASTER 7-TASK SYNTHESIS COMPLETED SUCCESSFULLY!" | tee -a "${MAIN_LOG}"
echo "================================================================================" | tee -a "${MAIN_LOG}"
