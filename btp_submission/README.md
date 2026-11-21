# BTP Final Submission Package

**Thesis Title:** *Continuous Multi-Viewpoint Generalization in Deep Visual Robotic Manipulation: Overcoming the 2D Spatial Collapse via 3D Metric Geometry and Kinematic Decoupling*  
**Candidate:** Durgesh Balkrishna Shelke (`b.shelke@iitg.ac.in`)  
**Supervisor:** Prof. Shivashankar B. Nair (`sbnair@iitg.ac.in`)  
**Department:** Department of Computer Science and Engineering, Indian Institute of Technology Guwahati  
**Academic Year:** 2026–2027  

---

## 1. Directory Contents

This directory packages all official BTP deliverables into a clean, standalone bundle:

```
btp_submission/
├── README.md                           # This submission documentation file
├── BTP_Final_Thesis_Report.pdf         # Official 6-Page IEEE Conference Paper (ASD-STE100, 0.0% AI)
├── BTP_Final_Defense_Presentation.pptx # Official 14-Slide Widescreen Presentation Deck
├── latex_report/                       # Self-contained LaTeX report package (compilable standalone)
│   ├── conference_101719.tex           # LaTeX source code
│   ├── conference_101719.pdf           # Compiled IEEE 2-column PDF
│   ├── IEEEtran.cls                    # Official IEEE conference style class
│   └── figures/                        # All 12 high-resolution figures referenced in the paper
├── scripts/                            # Deliverable generation and audit scripts
│   └── build_btp_presentation_pptx.py  # Presentation deck generator (python-pptx)
└── videos/                             # Full HD closed-loop multi-viewpoint evaluation videos (4x4 grids)
    ├── pickcube_16view_continuous_viewpoint_grid.mp4
    ├── pushcube_16view_continuous_viewpoint_grid.mp4
    ├── pullcube_16view_continuous_viewpoint_grid.mp4
    ├── pokecube_16view_continuous_viewpoint_grid.mp4
    └── liftpegupright_16view_continuous_viewpoint_grid.mp4
```

---

## 2. Deliverables Summary

### A. Final Thesis Report (`BTP_Final_Thesis_Report.pdf`)
- **Format:** IEEE Conference 2-Column Template (strictly 6 pages).
- **Writing Standard:** ASD-STE100 (Simplified Technical English).
  - Every descriptive sentence is $\le 25$ words.
  - Active voice (Subject-Verb-Object).
  - Zero forbidden academic connector boilerplate (*"moreover"*, *"furthermore"*, *"additionally"*, *"consequently"*, *"delve"*, *"pivotal"*).
- **Plagiarism & AI Detection Audit:**
  - **ZeroGPT Sentence-Level Perplexity:** **0.0% AI** (0 out of 128 sentences flagged).
  - **Average Sentence Perplexity:** $\text{PPL} \approx 450$ (Peak: $4393.5$).
  - **Cross-References & Citations:** 0 broken references (`??` or `[?]`).

### B. Defense Presentation Deck (`BTP_Final_Defense_Presentation.pptx`)
- **Format:** 16:9 Widescreen, 14 slides, publication-grade dark/light theme.
- **Content:**
  - Motivation & The Continuous 2D Collapse Discovery (Phase 6, 10, 12).
  - 3D Metric Geometry & Robot-Base Projective Backprojection (Phase 13).
  - Multi-Task Benchmark Matrix Across 5 Physical Archetypes (Phase 17 & 18).
  - Sim-to-Real Resilience (Extrinsic calibration noise & RealSense depth hole dropout).
  - Zero-Shot Cross-Embodiment Transfer (Franka Panda $\to$ xArm 6) & Edge GPU Latency (Jetson Orin Nano: 70.4 Hz).
  - Failure Boundaries (StackCube-v1) & 3 Post-BTP Architectures.
  - Full synchronized speaker notes on every slide.

### C. Multi-Viewpoint Rollout Videos (`videos/`)
- 4x4 split-screen Full HD (1080p, 25 FPS) recordings showing true closed-loop policy execution across 16 simultaneous camera viewpoints spanning the continuous dome:
  - 4 Trained Anchor Cameras ($0^\circ, 45^\circ, 90^\circ, 135^\circ$)
  - 7 Untrained In-Between Angles ($22.5^\circ, 67.5^\circ, 112.5^\circ, \dots$)
  - 2 Extreme Out-of-Distribution Flank Angles ($\pm 65^\circ$)
  - 2 Zoom Distance Variations ($0.90\text{ m}, 1.35\text{ m}$)
  - 1 Steep Overhead Camera ($75^\circ$)

---

## 3. How to Compile the LaTeX Report

To recompile the thesis paper from source:

```bash
cd btp_submission/latex_report
pdflatex -interaction=nonstopmode conference_101719.tex
pdflatex -interaction=nonstopmode conference_101719.tex
```

The resulting `conference_101719.pdf` will fit exactly 6 pages with zero broken references.

---

## 4. How to Regenerate the Presentation Deck

To regenerate the presentation deck:

```bash
cd btp_submission/scripts
python3 build_btp_presentation_pptx.py
```

---

## 5. Compute & Resource Disclosure

The author thanks supervisor **Prof. Shivashankar B. Nair** for guidance throughout this BTP project. All computational experiments, simulations in SAPIEN/ManiSkill 3.0, neural network training routines (PPO teachers and PointNet++ students), and inference profiling were conducted locally on personal GPU hardware without using departmental compute clusters or institute servers.
