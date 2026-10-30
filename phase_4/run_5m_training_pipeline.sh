#!/usr/bin/env bash
set -e

export PYTHONUNBUFFERED=1
cd /home/maniskill

echo "=============================================================================="
echo " LAUNCHING 5,000,000-STEP BASELINE RETRAINING CAMPAIGN"
echo " Modality 1: Pure RGB Baseline (5,000,000 steps)"
echo " Modality 2: 4-Channel RGB-D Baseline (5,000,000 steps)"
echo " Both trained from scratch on canonical Front View: [0.3, 0.0, 0.6]"
echo "=============================================================================="

# 1. Train Pure RGB Baseline for 5M steps
echo ""
echo "[PIPELINE] >>> Starting Phase 1: Pure RGB Baseline (5M steps)..."
python3 /home/maniskill/examples/baselines/ppo/ppo_rgb.py \
  --exp-name sim2real_45mm_block_rgb_5m \
  --total-timesteps 5000000 \
  --num-envs 256 \
  --num-eval-envs 8 \
  --num-minibatches 16 \
  --eval-freq 25 \
  --control-mode pd_ee_delta_pos \
  --camera-eye="0.3,0.0,0.6" \
  --camera-target="-0.1,0.0,0.1"

echo ""
echo "[PIPELINE] >>> Phase 1 Complete! Pure RGB 5M model saved to /home/maniskill/phase_4/final_models/pure_rgb_5m_front_view.pt"
echo ""

# 2. Train 4-Channel RGB-D Baseline for 5M steps
echo "[PIPELINE] >>> Starting Phase 2: 4-Channel RGB-D Baseline (5M steps)..."
python3 /home/maniskill/examples/baselines/ppo/ppo_rgbd.py \
  --exp-name sim2real_45mm_block_rgbd_5m \
  --total-timesteps 5000000 \
  --num-envs 256 \
  --num-eval-envs 8 \
  --num-minibatches 16 \
  --eval-freq 25 \
  --target-success-rate 1.01 \
  --control-mode pd_ee_delta_pos \
  --camera-eye="0.3,0.0,0.6" \
  --camera-target="-0.1,0.0,0.1"

echo ""
echo "[PIPELINE] >>> Phase 2 Complete! 4-Channel RGB-D 5M model saved to /home/maniskill/phase_4/final_models/clean_rgbd_5m_front_view.pt"
echo ""
echo "=============================================================================="
echo " 5M-STEP BASELINE RETRAINING CAMPAIGN FINISHED SUCCESSFULLY!"
echo "=============================================================================="
