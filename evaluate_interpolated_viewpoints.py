"""
Phase 10: Spatial Generalization and Interpolated Viewpoint Evaluation Harness.
Evaluates the Phase 10 OmniView Student Policy on camera positions strictly in-between
and not among the ones it was trained on.

Sections evaluated:
1. Untrained Catalog Intermediate Viewpoints (VP 10, VP 13, VP 14, VP 04, VP 07, VP 21, VP 24, VP 12, VP 15)
2. Continuous Spatial Generalization Basin (0cm, 2cm, 5cm, 10cm, 15cm, 25cm, 35cm offset from VP 17)
3. Radial Distance / Zoom Generalization (0.90m, 1.00m, 1.04m, 1.09m, 1.15m, 1.25m)
"""

import os
import sys
sys.path.append("/home/maniskill")
import json
import time
import numpy as np
import torch
import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mani_skill.envs
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper, FlattenActionSpaceWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from phase_10_omniview_nextgen.train_teacher_specialists_v3 import Phase10TeacherAgent, append_extrinsics

STUDENT_CKPT = "/home/maniskill/phase_10_omniview_nextgen/student/omniview_student_best.pt"
VIEWPOINTS_JSON = "/home/maniskill/phase_1_to_3/sequential_viewpoint_training/viewpoints.json"
OUTPUT_DIR = "/home/maniskill/phase_10_omniview_nextgen/eval_results"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def evaluate_camera_position(student, cam_eye, num_episodes=50, batch_envs=50, seed=42):
    cam_eye_t = torch.tensor(cam_eye, dtype=torch.float32, device=DEVICE)

    env_kwargs = dict(
        obs_mode="rgbd",
        render_mode="all",
        sim_backend="physx_cuda",
        sensor_cam_eye_pos=cam_eye,
        sensor_cam_target_pos=[-0.1, 0.0, 0.1],
        control_mode="pd_ee_delta_pos",
        sensor_configs=dict(width=128, height=128),
    )
    env = gym.make("PickCube-v1", num_envs=batch_envs, reconfiguration_freq=None, **env_kwargs)
    env = FlattenRGBDObservationWrapper(env, rgb=True, depth=True, state=True, sep_depth=True)
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    eval_envs = ManiSkillVectorEnv(env, batch_envs, ignore_terminations=True, record_metrics=True)

    evaluated = 0
    total_succ = 0
    total_grasp = 0
    total_rewards = []

    while evaluated < num_episodes:
        cur_batch = min(batch_envs, num_episodes - evaluated)
        obs, _ = eval_envs.reset(seed=int(seed + evaluated * 17))
        grasped_any = torch.zeros(cur_batch, dtype=torch.bool, device=DEVICE)
        ep_rewards = torch.zeros(cur_batch, device=DEVICE)

        for step in range(50):
            with torch.no_grad():
                s33 = append_extrinsics(obs["state"][:cur_batch], cam_eye_t, noise_std=0.0, device=DEVICE)
                act = student.get_action(obs["rgb"][:cur_batch], obs["depth"][:cur_batch], s33, deterministic=True)
            obs, rew, term, trunc, infos = eval_envs.step(act)
            ep_rewards += rew[:cur_batch]
            if "is_grasped" in infos:
                grasped_any |= infos["is_grasped"][:cur_batch]

        succ = infos["final_info"]["episode"]["success_once"][:cur_batch]
        total_succ += succ.sum().item()
        total_grasp += grasped_any.sum().item()
        total_rewards.extend(ep_rewards.cpu().tolist())
        evaluated += cur_batch

    eval_envs.close()
    succ_rate = (total_succ / num_episodes) * 100.0
    grasp_rate = (total_grasp / num_episodes) * 100.0
    mean_return = float(np.mean(total_rewards))

    return {
        "trials": num_episodes,
        "successes": total_succ,
        "success_rate": round(succ_rate, 2),
        "grasps": total_grasp,
        "grasp_rate": round(grasp_rate, 2),
        "mean_return": round(mean_return, 2),
    }


def main():
    print("=" * 80)
    print(" PHASE 10: IN-BETWEEN AND NOVEL VIEWPOINT EVALUATION")
    print(f" Student Checkpoint: {STUDENT_CKPT}")
    print("=" * 80)

    # Load Student Model
    student = Phase10TeacherAgent(action_dim=4, state_dim=33).to(DEVICE)
    ckpt = torch.load(STUDENT_CKPT, map_location=DEVICE)
    student.load_state_dict(ckpt.get("model_state", ckpt))
    student.eval()

    with open(VIEWPOINTS_JSON) as f:
        vps = json.load(f)
    vp_dict = {v["id"]: v for v in vps}

    # ---------------------------------------------------------
    # EXPERIMENT 1: Untrained Catalog Intermediate Viewpoints
    # ---------------------------------------------------------
    print("\n[PART 1] Testing Untrained In-Between Catalog Viewpoints vs Trained Anchors...")
    catalog_configs = [
        # Trained Anchors (for reference)
        {"id": 17, "label": "VP 17 (Front-Mid Anchor)", "type": "trained", "cam_eye": vp_dict[17]["world_eye"], "az": vp_dict[17]["azimuth_deg"], "el": vp_dict[17]["elevation_deg"]},
        {"id": 0,  "label": "VP 00 (Front-Low Anchor)", "type": "trained", "cam_eye": vp_dict[0]["world_eye"], "az": vp_dict[0]["azimuth_deg"], "el": vp_dict[0]["elevation_deg"]},
        {"id": 20, "label": "VP 20 (Diag-Left Anchor)", "type": "trained", "cam_eye": vp_dict[20]["world_eye"], "az": vp_dict[20]["azimuth_deg"], "el": vp_dict[20]["elevation_deg"]},
        {"id": 19, "label": "VP 19 (Diag-Right Anchor)", "type": "trained", "cam_eye": vp_dict[19]["world_eye"], "az": vp_dict[19]["azimuth_deg"], "el": vp_dict[19]["elevation_deg"]},

        # Untrained In-Between Viewpoints
        {"id": 10, "label": "VP 10 (In-Between El: VP00-VP17)", "type": "untrained_interp", "cam_eye": vp_dict[10]["world_eye"], "az": vp_dict[10]["azimuth_deg"], "el": vp_dict[10]["elevation_deg"]},
        {"id": 13, "label": "VP 13 (In-Between Az: VP17-VP20)", "type": "untrained_interp", "cam_eye": vp_dict[13]["world_eye"], "az": vp_dict[13]["azimuth_deg"], "el": vp_dict[13]["elevation_deg"]},
        {"id": 14, "label": "VP 14 (In-Between Az: VP17-VP19)", "type": "untrained_interp", "cam_eye": vp_dict[14]["world_eye"], "az": vp_dict[14]["azimuth_deg"], "el": vp_dict[14]["elevation_deg"]},
        {"id": 4,  "label": "VP 04 (In-Between Low-Right)",   "type": "untrained_interp", "cam_eye": vp_dict[4]["world_eye"], "az": vp_dict[4]["azimuth_deg"], "el": vp_dict[4]["elevation_deg"]},
        {"id": 7,  "label": "VP 07 (In-Between Low-Left)",    "type": "untrained_interp", "cam_eye": vp_dict[7]["world_eye"], "az": vp_dict[7]["azimuth_deg"], "el": vp_dict[7]["elevation_deg"]},
        {"id": 21, "label": "VP 21 (In-Between Mid-Right)",   "type": "untrained_interp", "cam_eye": vp_dict[21]["world_eye"], "az": vp_dict[21]["azimuth_deg"], "el": vp_dict[21]["elevation_deg"]},
        {"id": 24, "label": "VP 24 (In-Between High-Mid)",    "type": "untrained_interp", "cam_eye": vp_dict[24]["world_eye"], "az": vp_dict[24]["azimuth_deg"], "el": vp_dict[24]["elevation_deg"]},
        {"id": 12, "label": "VP 12 (Flank In-Between R)",     "type": "untrained_interp", "cam_eye": vp_dict[12]["world_eye"], "az": vp_dict[12]["azimuth_deg"], "el": vp_dict[12]["elevation_deg"]},
        {"id": 15, "label": "VP 15 (Flank In-Between L)",     "type": "untrained_interp", "cam_eye": vp_dict[15]["world_eye"], "az": vp_dict[15]["azimuth_deg"], "el": vp_dict[15]["elevation_deg"]},
    ]

    catalog_results = []
    for cfg in catalog_configs:
        res = evaluate_camera_position(student, cfg["cam_eye"], num_episodes=50, batch_envs=50, seed=100 + cfg["id"])
        res.update(cfg)
        catalog_results.append(res)
        tag = "[TRAINED]" if cfg["type"] == "trained" else "[UNTRAINED]"
        print(f"  {tag:<11} {cfg['label']:<34} (Az={cfg['az']:5.1f}°, El={cfg['el']:4.1f}°): Succ={res['success_rate']:5.1f}% | Grasp={res['grasp_rate']:5.1f}% | Ret={res['mean_return']:5.2f}")

    # ---------------------------------------------------------
    # EXPERIMENT 2: Continuous Spatial Generalization Basin around VP 17
    # ---------------------------------------------------------
    print("\n[PART 2] Testing Continuous Spatial Basin around VP 17...")
    vp17_eye = np.array(vp_dict[17]["world_eye"])
    basin_offsets = [
        {"name": "0 cm (Exact Anchor)", "offset_m": 0.00, "deg": 0.0,  "shift": np.array([0.0, 0.00, 0.0])},
        {"name": "+2 cm lateral",       "offset_m": 0.02, "deg": 1.0,  "shift": np.array([0.0, 0.02, 0.0])},
        {"name": "+5 cm lateral",       "offset_m": 0.05, "deg": 2.6,  "shift": np.array([0.0, 0.05, 0.0])},
        {"name": "+10 cm lateral",      "offset_m": 0.10, "deg": 5.2,  "shift": np.array([0.0, 0.10, 0.0])},
        {"name": "+15 cm lateral",      "offset_m": 0.15, "deg": 7.8,  "shift": np.array([0.0, 0.15, 0.0])},
        {"name": "+20 cm lateral",      "offset_m": 0.20, "deg": 10.4, "shift": np.array([0.0, 0.20, 0.0])},
        {"name": "+30 cm lateral",      "offset_m": 0.30, "deg": 15.6, "shift": np.array([0.0, 0.30, 0.0])},
    ]

    basin_results = []
    for cfg in basin_offsets:
        eye = (vp17_eye + cfg["shift"]).tolist()
        res = evaluate_camera_position(student, eye, num_episodes=50, batch_envs=50, seed=200 + int(cfg["offset_m"] * 100))
        res.update(cfg)
        basin_results.append(res)
        print(f"  Offset {cfg['name']:<22} (~{cfg['deg']:4.1f}°): Succ={res['success_rate']:5.1f}% | Grasp={res['grasp_rate']:5.1f}% | Ret={res['mean_return']:5.2f}")

    # ---------------------------------------------------------
    # EXPERIMENT 3: Radial Distance (Zoom In / Zoom Out) Generalization
    # ---------------------------------------------------------
    print("\n[PART 3] Testing Radial Distance (Zoom In / Zoom Out) Generalization...")
    base_pos = np.array([-0.615, 0.0, 0.0])
    rel_vec = vp17_eye - base_pos
    orig_dist = float(np.linalg.norm(rel_vec))
    unit_dir = rel_vec / orig_dist

    zoom_configs = [
        {"name": "0.90 m (-19.4cm Zoom-In)",  "dist": 0.90, "delta_cm": -19.4},
        {"name": "0.95 m (-14.4cm Zoom-In)",  "dist": 0.95, "delta_cm": -14.4},
        {"name": "1.00 m (-9.4cm Zoom-In)",   "dist": 1.00, "delta_cm": -9.4},
        {"name": "1.05 m (-4.4cm Zoom-In)",   "dist": 1.05, "delta_cm": -4.4},
        {"name": "1.09 m (Exact Anchor Dist)","dist": orig_dist, "delta_cm": 0.0},
        {"name": "1.15 m (+5.6cm Zoom-Out)",  "dist": 1.15, "delta_cm": +5.6},
        {"name": "1.20 m (+10.6cm Zoom-Out)", "dist": 1.20, "delta_cm": +10.6},
        {"name": "1.25 m (+15.6cm Zoom-Out)", "dist": 1.25, "delta_cm": +15.6},
    ]

    zoom_results = []
    for cfg in zoom_configs:
        eye = (base_pos + unit_dir * cfg["dist"]).tolist()
        res = evaluate_camera_position(student, eye, num_episodes=50, batch_envs=50, seed=300 + int(cfg["dist"] * 100))
        res.update(cfg)
        zoom_results.append(res)
        print(f"  Distance {cfg['name']:<28}: Succ={res['success_rate']:5.1f}% | Grasp={res['grasp_rate']:5.1f}% | Ret={res['mean_return']:5.2f}")

    # ---------------------------------------------------------
    # Save Full JSON Logs
    # ---------------------------------------------------------
    output_data = {
        "student_checkpoint": STUDENT_CKPT,
        "catalog_results": catalog_results,
        "basin_results": basin_results,
        "zoom_results": zoom_results,
    }
    json_path = os.path.join(OUTPUT_DIR, "interpolated_viewpoints_results.json")
    with open(json_path, "w") as f:
        json.dump(output_data, f, indent=2, default=lambda x: x.tolist() if isinstance(x, (np.ndarray, np.generic)) else str(x))
    print(f"\n[OK] Saved complete evaluation results to {json_path}")

    # ---------------------------------------------------------
    # Generate Publication-Quality Multi-Panel Figure
    # ---------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(21, 6.5))

    # Panel 1: Catalog Viewpoints (Trained vs Untrained In-Between)
    ax1 = axes[0]
    names = [c["label"].replace(" (In-Between", "\n(In-Between").replace(" Anchor)", "\n(Anchor)") for c in catalog_results]
    succs = [c["success_rate"] for c in catalog_results]
    grasps = [c["grasp_rate"] for c in catalog_results]
    colors = ["#1b5e20" if c["type"] == "trained" else "#d32f2f" for c in catalog_results]
    x = np.arange(len(names))
    width = 0.35

    ax1.bar(x - width/2, grasps, width, label="Grasp Initiation Rate (%)", color="#90caf9", edgecolor="black", alpha=0.85)
    bars = ax1.bar(x + width/2, succs, width, label="Success Rate (%)", color=colors, edgecolor="black")
    ax1.set_ylabel("Rate (%)", fontsize=11, fontweight="bold")
    ax1.set_title("A. Trained Anchors vs Untrained In-Between Viewpoints", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"VP{c['id']:02d}" for c in catalog_results], fontsize=9, fontweight="bold")
    ax1.set_ylim(0, 105)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right", fontsize=9)

    # Panel 2: Continuous Lateral Basin Curve
    ax2 = axes[1]
    b_offsets = [c["offset_m"] * 100 for c in basin_results]
    b_succs = [c["success_rate"] for c in basin_results]
    b_grasps = [c["grasp_rate"] for c in basin_results]

    ax2.plot(b_offsets, b_grasps, "o--", color="#1976d2", linewidth=2.5, markersize=7, label="Grasp Rate (%)")
    ax2.plot(b_offsets, b_succs, "s-", color="#d32f2f", linewidth=2.5, markersize=8, label="Success Rate (%)")
    ax2.axvspan(0, 3, color="#c8e6c9", alpha=0.5, label="Training Jitter Zone (±3cm)")
    ax2.set_xlabel("Lateral Camera Shift from VP 17 Anchor (cm)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Rate (%)", fontsize=11, fontweight="bold")
    ax2.set_title("B. Spatial Generalization Basin around Anchor", fontsize=12, fontweight="bold")
    ax2.set_ylim(-2, 105)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right", fontsize=9)

    # Panel 3: Radial Distance (Zoom) Curve
    ax3 = axes[2]
    z_dists = [c["dist"] for c in zoom_results]
    z_succs = [c["success_rate"] for c in zoom_results]
    z_grasps = [c["grasp_rate"] for c in zoom_results]

    ax3.plot(z_dists, z_grasps, "o--", color="#0288d1", linewidth=2.5, markersize=7, label="Grasp Rate (%)")
    ax3.plot(z_dists, z_succs, "s-", color="#e65100", linewidth=2.5, markersize=8, label="Success Rate (%)")
    ax3.axvline(orig_dist, color="#2e7d32", linestyle=":", linewidth=2, label=f"Anchor Dist ({orig_dist:.2f}m)")
    ax3.axvspan(orig_dist - 0.03, orig_dist + 0.03, color="#c8e6c9", alpha=0.5, label="Training Jitter (±3cm)")
    ax3.set_xlabel("Camera Distance to Base (m)", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Rate (%)", fontsize=11, fontweight="bold")
    ax3.set_title("C. Radial Distance / Zoom Generalization", fontsize=12, fontweight="bold")
    ax3.set_ylim(-2, 105)
    ax3.grid(True, linestyle="--", alpha=0.5)
    ax3.legend(loc="upper right", fontsize=9)

    plt.suptitle("Phase 10 OmniView Student Policy: Out-of-Distribution & Interpolated Viewpoint Generalization", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    fig_path = os.path.join(OUTPUT_DIR, "phase10_interpolated_viewpoints_benchmark.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved summary plot to {fig_path}")


if __name__ == "__main__":
    main()
