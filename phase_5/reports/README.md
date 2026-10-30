# Research Reports Compendium: Visual RL Robotic Manipulation

This directory archives all formal benchmark reports, empirical sensitivity sweeps, catastrophic forgetting studies, and comparative evaluations across the project.

---

## Index of Reports

| # | Report Document | Primary Subject | Core Metric / Finding |
| :-: | :--- | :--- | :--- |
| **01** | [`01_sim2real_1000_run_benchmark.md`](file:///home/maniskill/research_reports/01_sim2real_1000_run_benchmark.md) | Sim-to-Real 1,000-Run Benchmark (Front View) | **97.10% success** (971 / 1,000) on 4.5 cm solid block. |
| **02** | [`02_side_view_evaluation_report.md`](file:///home/maniskill/research_reports/02_side_view_evaluation_report.md) | Shifted Side-View Camera (90° Rotation) | Performance collapses from 97.10% to **0.40%** due to coordinate frame axis inversion. |
| **03** | [`03_live_realtime_evaluation.md`](file:///home/maniskill/research_reports/03_live_realtime_evaluation.md) | 1:1 Physical Clock Sync Telemetry (20 Hz) | Time to grasp: **0.65s (Step 13)**; 1.01x real-time speed ratio. |
| **04** | [`04_camera_viewpoint_sweep_report.md`](file:///home/maniskill/research_reports/04_camera_viewpoint_sweep_report.md) | Semicircular Horizontal Sweep (13 Angles, 13,000 runs) | $\pm 15^\circ$ deviation causes $>85\%$ collapse (drops to 4.2%–11.2%). |
| **05** | [`05_fibonacci_hemisphere_20vp_report.md`](file:///home/maniskill/research_reports/05_fibonacci_hemisphere_20vp_report.md) | 20-Viewpoint Fibonacci Equal-Area Hemisphere | **4.03% mean success** across 3D dome. Peak at VP 10 (45.0%). |
| **06** | [`06_fibonacci_hemisphere_30vp_report.md`](file:///home/maniskill/research_reports/06_fibonacci_hemisphere_30vp_report.md) | 30-Viewpoint Fibonacci Equal-Area Hemisphere | **5.65% mean success**; peak 99.0% at VP 17 (close to training pose). |
| **07** | [`07_sequential_training_architecture.md`](file:///home/maniskill/research_reports/07_sequential_training_architecture.md) | Sequential Continual Learning Pipeline Setup | Resumable state tracker and model preservation architecture. |
| **08** | [`08_catastrophic_forgetting_midway_report.md`](file:///home/maniskill/research_reports/08_catastrophic_forgetting_midway_report.md) | Midway Evaluation (Viewpoints 00–06) | VP 00 collapsed from 100% to **1.0%**; VP 17 collapsed from 99% to **0.0%**. |
| **09** | [`09_final_sequential_training_benchmark_report.md`](file:///home/maniskill/research_reports/09_final_sequential_training_benchmark_report.md) | Final Sequential Training Benchmark (All 30 Viewpoints) | Conclusive proof of **Catastrophic Forgetting**: 0.50% dome retention; VP 00 at **0.0%**. |
| **10** | [`10_joint_multiview_training_setup.md`](file:///home/maniskill/research_reports/10_joint_multiview_training_setup.md) | Joint Multi-View Domain Randomization Architecture | Concurrently distributes all 30 viewpoints across 128 parallel environments. |
| **11** | [`11_rgbd_baseline_benchmark_report.md`](file:///home/maniskill/research_reports/11_rgbd_baseline_benchmark_report.md) | 4-Channel RGB-D Baseline Benchmark | **90.60% 1,000-run success**, **100.0% eval peak**, and **16.65 steps mean length** on canonical front view. |
| **12** | [`12_rgb_vs_rgbd_comparative_research_report.md`](file:///home/maniskill/research_reports/12_rgb_vs_rgbd_comparative_research_report.md) | Comprehensive RGB vs. RGB-D Head-to-Head Comparison | True apples-to-apples comparison (same front viewpoint, 256 envs, 16 minibatches, 34,000 total runs). |
| **13** | [`13_definitive_5m_rgb_vs_rgbd_comparative_report.md`](file:///home/maniskill/research_reports/13_definitive_5m_rgb_vs_rgbd_comparative_report.md) | Definitive 5M Fixed-Budget Parity Study | Fixed 5,000,000 steps each on front view; 1k front-view, 13k sweep, 3k Fibonacci dome benchmarks. |
| **Master** | [`consolidated_research_reports_master.md`](file:///home/maniskill/research_reports/consolidated_research_reports_master.md) | Complete Consolidated Research Compendium | Full standalone multi-chapter compilation of all findings. |
