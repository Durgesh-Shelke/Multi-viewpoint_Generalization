#!/usr/bin/env bash
# ==============================================================================
# Live status monitor for Phase 6 Viewpoint Specialists Benchmark
# ==============================================================================

DIR="/home/maniskill/phase_6_viewpoint_specialists"
PAUSE_FILE="$DIR/.pause"
MASTER_PID=$(pgrep -f "run_master_phase6.py" || true)
TRAIN_PID=$(pgrep -f "train_viewpoint_specialist.py" || true)

echo "=========================================================================="
echo " PHASE 6: 8-VIEWPOINT SPECIALISTS BENCHMARK STATUS"
echo " Timestamp: $(date)"
echo "=========================================================================="

if [ -f "$PAUSE_FILE" ]; then
    echo " PIPELINE STATE: [PAUSED] (Pause flag active: $PAUSE_FILE)"
    if [ -n "$TRAIN_PID" ]; then
        echo "                 Process is alive (PID: $TRAIN_PID) in sleep loop (GPU idle)"
    fi
elif [ -n "$MASTER_PID" ] || [ -n "$TRAIN_PID" ]; then
    echo " PIPELINE STATE: [RUNNING] (Master PID: ${MASTER_PID:-N/A}, Training PID: ${TRAIN_PID:-N/A})"
else
    echo " PIPELINE STATE: [STOPPED / IDLE]"
fi

echo "--------------------------------------------------------------------------"
echo " 8 SPECIALIST VIEWPOINTS PROGRESS:"
python3 -c "
import os, json, torch

DIR = '/home/maniskill/phase_6_viewpoint_specialists'
CORE_VIPS = [17, 0, 28, 18, 27, 20, 19, 6]
NAMES = {
    17: 'Front-Mid (Baseline)  [-4.7°, 31.8°]',
    0:  'Front-Low (Graze)     [ 0.0°, 12.3°]',
    28: 'Front-High (Tilt)     [+2.9°, 46.2°]',
    18: 'Left-Flank 90°       [-89.7°, 33.1°]',
    27: 'Right-Flank 90°      [+87.9°, 44.6°]',
    20: 'Diag Front-Left      [-37.2°, 35.0°]',
    19: 'Diag Front-Right     [+47.8°, 33.7°]',
    6:  'Low Flank Right      [+72.6°, 18.3°]',
}

for vp in CORE_VIPS:
    latest_pt = os.path.join(DIR, 'checkpoints', f'specialist_vp_{vp:02d}_latest.pt')
    metrics_json = os.path.join(DIR, 'checkpoints', f'specialist_vp_{vp:02d}_train_metrics.json')
    best_pt = os.path.join(DIR, 'checkpoints', f'specialist_vp_{vp:02d}_best.pt')
    
    status = 'PENDING'
    info = ''
    
    if os.path.exists(metrics_json) and os.path.exists(best_pt):
        try:
            with open(metrics_json) as f: m = json.load(f)
            succ = m.get('best_eval_success', 0.0)
            steps = m.get('total_timesteps', 0)
            sps = m.get('avg_sps', 0)
            mins = m.get('total_walltime_seconds', 0) / 60
            if succ >= 95.0 or steps >= 5000000:
                status = 'COMPLETED'
                early = ' [Ceiling Mastered]' if succ >= 95.0 else ' [Max Steps]'
            else:
                status = 'IN PROGRESS'
                early = ' [Continuing to 5M]'
            info = f'Steps: {steps:,} / 5,000,000 | Best Eval: {succ:5.1f}% | Avg SPS: {sps} | Time: {mins:4.1f}m{early}'
        except Exception:
            pass
    elif os.path.exists(latest_pt):
        try:
            ckpt = torch.load(latest_pt, map_location='cpu')
            status = 'IN PROGRESS'
            cur_step = ckpt.get('global_step', 0)
            cur_succ = ckpt.get('best_eval_success', 0.0)
            info = f'Step: {cur_step:,} / 5,000,000 | Best Eval: {cur_succ:5.1f}%'
        except Exception:
            status = 'IN PROGRESS (writing ckpt)'
            
    print(f'  VP {vp:02d} ({NAMES[vp]}): [{status:11s}] {info}')
"

echo "--------------------------------------------------------------------------"
echo " GPU HARDWARE TELEMETRY:"
nvidia-smi --query-gpu=name,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv,noheader
echo "=========================================================================="
echo " Controls:  Pause: $DIR/pause.sh | Resume: $DIR/resume.sh"
echo " Logs:      tail -f $DIR/master_execution.log"
echo "=========================================================================="
