# Catastrophic Forgetting in Sequential Viewpoint RL: Midway Benchmark Report

## Executive Summary

We evaluated the latest model (`latest_in_progress.pt`, trained sequentially across Viewpoints 00 through 06) on our **30-viewpoint Fibonacci Equal-Area Hemispherical Benchmark** across **3,000 total test runs**.

The empirical results reveal a textbook machine learning phenomenon: **Catastrophic Forgetting**.

---

## 1. The Core Discovery: Peak Mastery vs. Retention

When each viewpoint was trained individually, the policy adapted and reached mastery on that specific camera view:
- **Viewpoint 00:** Reached **100.0%** accuracy
- **Viewpoint 03:** Reached **96.0%** accuracy
- **Viewpoint 04:** Reached **98.0%** accuracy
- **Viewpoint 05:** Reached **96.0%** accuracy

However, when evaluating the **latest model** across all 30 viewpoints simultaneously:

| Viewpoint ID | Perspective | Accuracy When Finished Training | Accuracy Now (in Latest Model) | Status |
| :---: | :--- | :---: | :---: | :---: |
| **VP 00** | Low Frontal ($0^\circ, 12^\circ$) | **100.0%** | **1.0%** | **Forgotten** |
| **VP 01** | Low Right Flank ($-85^\circ, 13^\circ$) | 52.0% | 0.0% | Forgotten |
| **VP 02** | Low Left Diagonal ($+53^\circ, 14^\circ$) | 74.0% | 1.0% | Forgotten |
| **VP 03** | Mid Right Diagonal ($-32^\circ, 15^\circ$) | **96.0%** | **2.0%** | **Forgotten** |
| **VP 04** | Mid Left Diagonal ($+20^\circ, 17^\circ$) | **98.0%** | **0.0%** | **Forgotten** |
| **VP 05** | Far Right Flank ($-65^\circ, 18^\circ$) | **96.0%** | **1.0%** | **Forgotten** |
| **VP 06** | Far Left Flank ($+73^\circ, 18^\circ$) | 46.0% | 1.0% | Forgotten |
| **VP 17** | **Original Training Front View** | **99.0%** | **0.0%** | **Overwritten** |

- **Overall Mean Hemispherical Success Rate:** **0.77%** (compared to 5.65% baseline).
- **Highest Viewpoint:** VP 07 (which it was actively training on before pause) at 3.0%.

---

## 2. Why Did Sequential Training Cause This?

1. **No Memory Retention Constraint:**
   - In standard gradient updates (Adam/PPO), the objective function $\mathcal{L}_{\text{VP\_06}}(\theta)$ optimizes *only* for the current camera's observation stream.
   - Gradients aggressively shift the convolutional weights to minimize loss on the current viewpoint, with zero penalty for altering weights that solved previous viewpoints.
2. **Re-Representing the 2D Coordinate Frame:**
   - Because a 2D CNN extracts spatial coordinates relative to the image borders, learning a $+73^\circ$ perspective requires flipping the internal feature map representation.
   - When the network learns to map horizontal pixel translation to vertical arm delta commands, it wipes out the representation that previously mapped vertical pixel changes to forward arm motion.

---

## 3. The Solution: Multi-View Domain Randomization (Joint Training)

To achieve true viewpoint invariance where the model solves **all 30 viewpoints at the same time**:

Instead of sequential training ($V_1 \to V_2 \to V_3$), the standard solution in robotics is **Joint Multi-View Domain Randomization**:
- During rollout collection across the 128 parallel environments:
  - Environment 0 uses Viewpoint 0
  - Environment 1 uses Viewpoint 1
  - Environment 2 uses Viewpoint 2
  - ... (randomly sampling from the 30 viewpoints every episode reset)
- **Why this works:** The network's gradient updates are computed on a minibatch containing observations from **all viewpoints simultaneously**. The CNN is forced to discover features that work for *every* angle simultaneously, preventing forgetting and producing true 3D spatial generalization.
