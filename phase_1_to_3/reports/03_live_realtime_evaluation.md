# Live Real-Time Evaluation: 4.5 cm Solid Block Pick

## Evaluation Details

The trained visual manipulation policy was evaluated for a single real-time run at strict **1:1 physical clock synchronization** ($\text{sim time} = \text{real time}$, 20 Hz / 50 ms per control step, with zero simulation acceleration).

- **Live GUI Window:** Rendered live to SAPIEN 3D Viewer on `DISPLAY=:0`.
- **Policy Checkpoint:** [`/home/mahoraga/maniskill/best_sim2real_policy.pt`](file:///home/mahoraga/maniskill/best_sim2real_policy.pt)
- **Object:** 4.5 cm solid block (Option A for physical parallel-jaw gripper transfer).
- **Control Rate:** 20 Hz ($\Delta t = 0.05\text{ s}$).

---

## 1:1 Real-Time Execution Video

The video below is rendered and encoded at the native 20 FPS control rate, showing the true physical speed of the arm during reach, grasp, and lift:

![Live Real-Time Pickup (1.0x Speed)](/root/.gemini/antigravity-cli/brain/82673f33-5c96-4154-bb42-cf6de559c7b7/live_realtime_pickup.mp4)

---

## Execution Telemetry

| Parameter | Value |
| :--- | :--- |
| **Simulated Duration** | 2.10 seconds (42 steps) |
| **Real Wall-Clock Duration** | 2.11 seconds |
| **Speed Ratio** | **1.01x (Pure Real-Time / No Acceleration)** |
| **Time to Grasp & Lift** | **0.65 seconds (Step 13)** |
| **Post-Grasp Hold** | 1.45 seconds stable vertical hold |
| **Evaluation Result** | **SUCCESSFUL PICK** |

### Execution Timeline

```
t = 0.00s : Episode initialized, initial RGB camera capture.
t = 0.25s : End-effector descends towards block coordinates.
t = 0.50s : Parallel jaws align around block contours.
t = 0.65s : Gripper closes; block lifted off the tabletop (SUCCESS).
t = 0.65s - 2.10s : Robot holds the block steadily aloft in operational space.
```
