#!/usr/bin/env bash
# ==============================================================================
# Live status monitor for the resolution ablation study
# ==============================================================================

DIR="/home/maniskill/phase_5"
PAUSE_FILE="$DIR/.pause"
MASTER_PID=$(pgrep -f "run_master_unbiased_benchmark.py" || true)
TRAIN_PID=$(pgrep -f "train_candidate.py" || true)

echo "=========================================================================="
echo " RESOLUTION & SENSOR FUSION BENCHMARK STATUS"
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
echo " CANDIDATES PROGRESS:"
python3 -c "
import os, json, torch

DIR = '/home/maniskill/phase_5'
for cand in ['A', 'B', 'C', 'D', 'E']:
    latest_pt = os.path.join(DIR, 'checkpoints', f'candidate_{cand}_latest.pt')
    metrics_json = os.path.join(DIR, 'checkpoints', f'candidate_{cand}_train_metrics.json')
    eval_json = os.path.join(DIR, 'eval_results', f'candidate_{cand}_results.json')
    
    status = 'PENDING'
    info = ''
    
    if os.path.exists(metrics_json) and os.path.exists(eval_json):
        try:
            with open(metrics_json) as f: m = json.load(f)
            with open(eval_json) as f: e = json.load(f)
            t1 = e.get('test1_front_view_1000', {}).get('success_rate', 0.0)
            status = 'COMPLETED'
            info = f'Steps: {m.get(\"total_timesteps\", 0):,} | In-Domain Eval: {t1:.1f}% | Avg SPS: {m.get(\"avg_sps\", 0)}'
        except Exception:
            pass
    elif os.path.exists(latest_pt):
        try:
            ckpt = torch.load(latest_pt, map_location='cpu')
            status = 'IN PROGRESS'
            info = f'Step: {ckpt.get(\"global_step\", 0):,} (Iter {ckpt.get(\"iteration\", 0)}) | Best Eval: {ckpt.get(\"best_eval_success\", 0.0):.1f}%'
        except Exception:
            status = 'IN PROGRESS (writing ckpt)'
            
    cand_names = {
        'A': 'Sym 128x128 RGB-D',
        'B': 'Sym 256x256 RGB-D',
        'C': 'Asym (64 RGB+256 D)',
        'D': 'Pure Depth 256x256',
        'E': 'Pure RGB 256x256',
    }
    print(f'  Candidate {cand} ({cand_names[cand]:20s}): [{status:11s}] {info}')
"

echo "--------------------------------------------------------------------------"
echo " GPU HARDWARE TELEMETRY:"
nvidia-smi --query-gpu=name,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv,noheader
echo "=========================================================================="
echo " Controls:  Pause: /home/maniskill/phase_5/pause.sh | Resume: /home/maniskill/phase_5/resume.sh"
echo " Logs:      tail -f /home/maniskill/phase_5/master_execution.log"
echo "=========================================================================="
