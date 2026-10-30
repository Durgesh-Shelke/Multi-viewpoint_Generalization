# Resolution & Asymmetric Sensor Fusion Study

This directory contains the autonomous benchmark suite comparing spatial resolution and sensory decoupling for visual reinforcement learning in robotic manipulation (`PickCube-v1`).

---

## 1. Candidate Architectures Under Test

| Candidate | Name | Modality | Visual Resolution | Input Channels | Latent Size | Rationale |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **A** | **Symmetric 128x128** | RGB-D | $128 \times 128$ | 4 | 512 | Our established baseline control. Fast throughput (~1,850 SPS). |
| **B** | **Symmetric 256x256** | RGB-D | $256 \times 256$ | 4 | 512 | Tests if $4\times$ pixel resolution improves millimeter contact or causes compute/overfitting penalties. |
| **C** | **Asymmetric Decoupled** | Dual-Stream | RGB: $64 \times 64$<br>Depth: $256 \times 256$ | 3 (RGB) + 1 (Depth) | 640 | Downsamples RGB to extract coarse semantics while preserving sharp $256 \times 256$ depth for contact. |
| **D** | **Pure Depth 256x256** | Depth Only | $256 \times 256$ | 1 (Depth only) | 512 | Ablation control: tests if RGB is redundant for physical object manipulation when metric depth is present. |
| **E** | **Pure RGB 256x256** | RGB Only | $256 \times 256$ | 3 (RGB only) | 512 | High-res color/texture control: tests policy precision when metric depth is completely absent. |

---

## 2. Unbiased 4-Tier Evaluation Battery

Each candidate is subjected to:
1. **Test 1: In-Domain Grasp Precision (1,000 runs, canonical front view):** Success rate (%), mean episode length (steps to grasp), trajectory return.
2. **Test 2: Angular Jitter Tolerance (3,000 runs total: 75°, 90°, 105° x 1,000 runs):** Measures tolerance to $\pm 15^\circ$ physical mounting wobble.
3. **Test 3: Photometric Domain Gap / Lighting Perturbation (2,000 runs total):** Tests resilience against ambient lighting changes (darkened 0.7x and brightened 1.3x).
4. **Test 4: Hardware & Latency Telemetry:** Parameter count, single-step inference latency (ms), training SPS, and peak VRAM.

---

## 3. Execution Commands

### To Run the Full Overnight Benchmark (All 5 Candidates A, B, C, D, E):
```bash
# Default 10M steps each (~11.0 hours)
/home/maniskill/phase_5/run_resolution_ablation.sh

# Or 8M steps each (~8.8 hours, fits within an 8-9 hour window)
/home/maniskill/phase_5/run_resolution_ablation.sh --total-timesteps 8000000
```
Or directly via python:
```bash
python3 /home/maniskill/phase_5/run_master_unbiased_benchmark.py --candidates A B C D E --total-timesteps 10000000
```

### Generated Artifacts & Reports:
- **Master JSON Summary:** `master_benchmark_summary.json`
- **Growth Curves & Plateau Analysis Chart:** `growth_curves_comparison.png`
- **Plateau Metrics Summary:** `eval_results/plateau_summary.json`
- **Formal Research Report:** [`/home/maniskill/phase_5/reports/14_resolution_and_asymmetric_depth_ablation_report.md`](file:///home/maniskill/phase_5/reports/14_resolution_and_asymmetric_depth_ablation_report.md)
- **Execution Log:** `master_execution.log`
- **Saved Model Checkpoints:** `checkpoints/candidate_{A,B,C,D,E}.pt`

---

## 4. Live Control & Checkpoint Resumption

The benchmark suite features zero-loss pause and resume functionality:

| Command | Action | Mechanism |
| :--- | :--- | :--- |
| **`/home/maniskill/phase_5/pause.sh`** | **Pause Execution** | Checkpoints active model state, frees GPU VRAM cache, enters low-CPU sleep. |
| **`/home/maniskill/phase_5/resume.sh`** | **Resume Execution** | If running: unpauses instantly. If stopped: restarts pipeline, skipping completed candidates and resuming from exact step. |
| **`/home/maniskill/phase_5/status.sh`** | **Live Status Dashboard** | Displays current pipeline state, steps reached per candidate, and GPU utilization/VRAM. |

