# 30-Viewpoint Fibonacci Equal-Area Hemispherical Benchmark (6,000 Runs)

## Executive Summary

Using **128 parallel visual environments**, the policy was benchmarked across **30 equal-area Fibonacci viewpoints** uniformly covering the upper-frontal viewing hemisphere ($R = 1.0942\text{ m}$ centered at the robot base `[-0.615, 0.0, 0.0]`).

- **Total Evaluations:** 6,000 runs (200 episodes per viewpoint).
- **Execution Wall-Time:** **2.72 minutes** (~37 episodes per second).
- **Peak Viewpoint:** **Viewpoint 17** achieved **99.00% pickup success** (198 / 200 picks).
- **Overall Mean Hemispherical Generalization:** **5.65%**.

---

## 1. 30-Viewpoint Benchmark Table

| ID | Azimuth $\theta$ | Elevation $\phi$ | Camera Position $[X, Y, Z]$ | Success Rate (200 Runs) | Successful Picks | Mean Return | Viewpoint Region |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **00** | $+0.0^\circ$ | $12.3^\circ$ | `[ 0.454,  0.000, 0.233]` | **1.00%** | 2 / 200 | 6.10 | Low frontal horizon |
| **01** | $-85.0^\circ$ | $13.4^\circ$ | `[-0.522, -1.061, 0.253]` | **0.00%** | 0 / 200 | 3.21 | Low right flank |
| **02** | $+52.5^\circ$ | $13.9^\circ$ | `[ 0.031,  0.843, 0.263]` | **1.50%** | 3 / 200 | 3.69 | Low left diagonal |
| **03** | $-32.5^\circ$ | $15.0^\circ$ | `[ 0.277, -0.567, 0.283]` | **2.00%** | 4 / 200 | 5.24 | Low right diagonal |
| **04** | $+20.1^\circ$ | $16.6^\circ$ | `[ 0.370,  0.360, 0.314]` | **0.00%** | 0 / 200 | 5.15 | Low left diagonal |
| **05** | $-64.9^\circ$ | $17.8^\circ$ | `[-0.173, -0.944, 0.334]` | **0.00%** | 0 / 200 | 3.13 | Low right flank |
| **06** | $+72.6^\circ$ | $18.3^\circ$ | `[-0.304,  0.991, 0.344]` | **1.00%** | 2 / 200 | 3.58 | Low left flank |
| **07** | **$-12.4^\circ$** | **$19.4^\circ$** | `[ 0.393, -0.222, 0.364]` | **10.00%** | 20 / 200 | **7.85** | Mid-low right cone |
| **08** | $+40.1^\circ$ | $21.1^\circ$ | `[ 0.165,  0.658, 0.395]` | **1.50%** | 3 / 200 | 4.19 | Mid-low left diagonal |
| **09** | $-44.9^\circ$ | $22.3^\circ$ | `[ 0.103, -0.714, 0.415]` | **1.00%** | 2 / 200 | 3.97 | Mid-low right diagonal |
| **10** | **$+7.7^\circ$** | **$24.0^\circ$** | `[ 0.376,  0.133, 0.445]` | **10.50%** | 21 / 200 | **6.51** | Mid-low left cone |
| **11** | $-77.3^\circ$ | $25.2^\circ$ | `[-0.398, -0.966, 0.465]` | **0.00%** | 0 / 200 | 2.86 | Mid right flank |
| **12** | $+60.2^\circ$ | $25.8^\circ$ | `[-0.125,  0.855, 0.475]` | **0.50%** | 1 / 200 | 3.87 | Mid left flank |
| **13** | **$-24.8^\circ$** | **$26.9^\circ$** | `[ 0.271, -0.409, 0.496]` | **32.00%** | 64 / 200 | **7.13** | **Approach corridor** |
| **14** | $+27.7^\circ$ | $28.7^\circ$ | `[ 0.234,  0.446, 0.526]` | **1.50%** | 3 / 200 | 5.50 | Mid left diagonal |
| **15** | $-57.3^\circ$ | $29.9^\circ$ | `[-0.102, -0.798, 0.546]` | **1.00%** | 2 / 200 | 3.25 | Mid right diagonal |
| **16** | $+80.2^\circ$ | $30.6^\circ$ | `[-0.455,  0.929, 0.556]` | **0.50%** | 1 / 200 | 3.52 | Mid left flank |
| **17** | **$-4.7^\circ$** | **$31.8^\circ$** | `[ 0.312, -0.077, 0.577]` | **99.00%** | **198 / 200** | **5.81** | **Peak Generalizer (15.3 steps)** |
| **18** | $-89.7^\circ$ | $33.1^\circ$ | `[-0.611, -0.917, 0.597]` | **1.00%** | 2 / 200 | 3.75 | Pure right side view |
| **19** | $+47.8^\circ$ | $33.7^\circ$ | `[-0.003,  0.674, 0.607]` | **0.50%** | 1 / 200 | 3.55 | Upper left diagonal |
| **20** | $-37.2^\circ$ | $35.0^\circ$ | `[ 0.099, -0.542, 0.627]` | **1.00%** | 2 / 200 | 5.13 | Upper right diagonal |
| **21** | $+15.3^\circ$ | $36.9^\circ$ | `[ 0.228,  0.231, 0.658]` | **1.00%** | 2 / 200 | 4.94 | Upper left cone |
| **22** | $-69.7^\circ$ | $38.3^\circ$ | `[-0.316, -0.805, 0.678]` | **0.00%** | 0 / 200 | 3.44 | Upper right flank |
| **23** | $+67.8^\circ$ | $39.0^\circ$ | `[-0.294,  0.788, 0.688]` | **0.00%** | 0 / 200 | 3.11 | Upper left flank |
| **24** | $-17.1^\circ$ | $40.3^\circ$ | `[ 0.182, -0.246, 0.708]` | **2.00%** | 4 / 200 | 6.30 | High right corridor |
| **25** | $+35.4^\circ$ | $42.5^\circ$ | `[ 0.043,  0.468, 0.739]` | **0.50%** | 1 / 200 | 4.65 | High left diagonal |
| **26** | $-49.6^\circ$ | $43.9^\circ$ | `[-0.104, -0.600, 0.759]` | **0.50%** | 1 / 200 | 3.65 | High right diagonal |
| **27** | $+87.9^\circ$ | $44.6^\circ$ | `[-0.587,  0.778, 0.769]` | **0.00%** | 0 / 200 | 3.22 | High left flank |
| **28** | $+2.9^\circ$ | $46.2^\circ$ | `[ 0.142,  0.039, 0.789]` | **0.00%** | 0 / 200 | 6.28 | Near zenith frontal |
| **29** | $-82.1^\circ$ | $47.7^\circ$ | `[-0.513, -0.729, 0.809]` | **0.00%** | 0 / 200 | 3.39 | Near zenith right flank |

---

## 2. Multi-Perspective Demonstration Videos

````carousel
![Viewpoint 00: Low Elevation (12deg)](/root/.gemini/antigravity-cli/brain/82673f33-5c96-4154-bb42-cf6de559c7b7/viewpoint_00_az+0_el12.mp4)
<!-- slide -->
![Viewpoint 18: Pure Right Flank (Az -90deg, El 33deg)](/root/.gemini/antigravity-cli/brain/82673f33-5c96-4154-bb42-cf6de559c7b7/viewpoint_18_az-90_el33.mp4)
<!-- slide -->
![Viewpoint 27: Pure Left Flank (Az +88deg, El 45deg)](/root/.gemini/antigravity-cli/brain/82673f33-5c96-4154-bb42-cf6de559c7b7/viewpoint_27_az+88_el45.mp4)
<!-- slide -->
![Viewpoint 29: Near-Zenith Steep Angle (Az -82deg, El 48deg)](/root/.gemini/antigravity-cli/brain/82673f33-5c96-4154-bb42-cf6de559c7b7/viewpoint_29_az-82_el48.mp4)
````

---

## 3. Scientific Insights: The 3D Visual Basin of Attraction

1. **Spot-On Alignment at Viewpoint 17:**
   - Training camera eye: `[0.300, 0.000, 0.600]`
   - Viewpoint 17 eye: `[0.312, -0.077, 0.577]`
   - Offset: $\Delta X = +1.2\text{ cm}$, $\Delta Y = -7.7\text{ cm}$, $\Delta Z = -2.3\text{ cm}$.
   - Result: **99.00% success** (198/200), demonstrating that small positional shifts within $\approx 8\text{ cm}$ maintain peak grasp accuracy.
2. **Asymmetric Tolerance Corridor (Right Flank / Viewpoint 13):**
   - Viewpoint 13 ($\theta = -24.8^\circ, \phi = 26.9^\circ$) maintains **32.00% success** because looking from the right side keeps the Franka Panda forearm aligned with the gripper approach vector, minimizing self-occlusion.
3. **Steep Degradation Across the Remaining Dome:**
   - At 26 out of the 30 uniform viewpoints (87% of the hemisphere), pickup success is $\le 2\%$, confirming that single-view 2D CNN visual policies possess a localized 3D directional cone.
