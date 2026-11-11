"""
Phase 10 Next-Gen Multi-Teacher Policy Distillation.
Distills the 8 temporal-extrinsic teachers into a single unified OmniView student policy.
Key Features:
1. Continuous Camera Viewpoint Jitter across the entire hemisphere (+-5cm in 3D).
2. Camera Extrinsics Conditioning with Gaussian noise injection (sigma = 0.02m).
3. 3-Frame Temporal Stack (12 channels) for working memory under occlusion.
4. Expert-guided DAgger interactive distillation: Environment is driven by the converged
   teachers (80% teacher / 20% student) ensuring dense successful reach, grasp, and lift
   demonstrations across all viewpoints.
"""

import os
import sys
sys.path.append("/home/maniskill")
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym
import mani_skill.envs
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper, FlattenActionSpaceWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from phase_10_omniview_nextgen.models_nextgen import OmniViewAgent
from phase_10_omniview_nextgen.train_teacher_specialists_v3 import Phase10TeacherAgent
from phase_10_omniview_nextgen.wrappers import TemporalExtrinsicWrapper

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VIEWPOINTS_JSON = "/home/maniskill/Multi-viewpoint_Generalization/phase_1_to_3/sequential_viewpoint_training/viewpoints.json"
TEACHERS_DIR = "/home/maniskill/Multi-viewpoint_Generalization/phase_10_omniview_nextgen/teachers"
STUDENT_DIR = "/home/maniskill/Multi-viewpoint_Generalization/phase_10_omniview_nextgen/student"

CORE_TEACHERS = [17, 0, 28, 18, 27, 20, 19, 6]


def load_all_teachers(vp_dict: dict):
    teachers = {}
    for vpid in CORE_TEACHERS:
        ckpt_path = os.path.join(TEACHERS_DIR, f"specialist_vp_{vpid:02d}_best.pt")
        if not os.path.exists(ckpt_path):
            ckpt_path = os.path.join(TEACHERS_DIR, f"specialist_vp_{vpid:02d}_final.pt")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Teacher checkpoint not found for VP {vpid:02d}")
        teacher = Phase10TeacherAgent(action_dim=4, state_dim=33).to(DEVICE)
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        teacher.load_state_dict(ckpt.get("model_state", ckpt))
        teacher.eval()
        teachers[vpid] = teacher
    print(f"Successfully loaded {len(teachers)} Phase 10 temporal-extrinsic teachers.")
    return teachers


def find_nearest_teacher(cam_eye: list, vp_dict: dict):
    target = np.array(cam_eye)
    best_vpid = CORE_TEACHERS[0]
    best_dist = float("inf")
    for vpid in CORE_TEACHERS:
        t_eye = np.array(vp_dict[vpid]["world_eye"])
        d = np.linalg.norm(target - t_eye)
        if d < best_dist:
            best_dist = d
            best_vpid = vpid
    return best_vpid


def evaluate_student(student: nn.Module, vp_dict: dict, num_episodes: int = 50):
    student.eval()
    results = {}
    for vpid in CORE_TEACHERS:
        meta = vp_dict[vpid]
        cam_eye = meta["world_eye"]
        cam_target = [-0.1, 0.0, 0.1]

        env_kwargs = dict(
            obs_mode="rgbd",
            render_mode="all",
            sim_backend="physx_cuda",
            sensor_cam_eye_pos=cam_eye,
            sensor_cam_target_pos=cam_target,
            control_mode="pd_ee_delta_pos",
            sensor_configs=dict(width=128, height=128),
        )
        env = gym.make("PickCube-v1", num_envs=50, reconfiguration_freq=None, **env_kwargs)
        env = FlattenRGBDObservationWrapper(env, rgb=True, depth=True, state=True, sep_depth=True)
        if isinstance(env.action_space, gym.spaces.Dict):
            env = FlattenActionSpaceWrapper(env)
        env = TemporalExtrinsicWrapper(env, cam_eye=cam_eye, frame_stack=3, noise_std=0.0, device="cuda")
        eval_envs = ManiSkillVectorEnv(env, 50, ignore_terminations=True, record_metrics=True)

        obs, _ = eval_envs.reset()
        collected = 0
        successes = 0

        with torch.no_grad():
            for _ in range(50):
                if collected >= num_episodes:
                    break
                act = student.get_action(obs["rgbd"], obs["state"], deterministic=True)
                obs, _, _, _, infos = eval_envs.step(act)

                if "final_info" in infos and "_final_info" in infos:
                    mask = infos["_final_info"]
                    ep_info = infos["final_info"].get("episode", {})
                    succ_key = "success_once" if "success_once" in ep_info else "success"
                    for env_i in range(len(mask)):
                        if collected >= num_episodes:
                            break
                        if mask[env_i]:
                            val = ep_info[succ_key][env_i]
                            succ = bool(val.item() if torch.is_tensor(val) else val)
                            if succ:
                                successes += 1
                            collected += 1

        eval_envs.close()
        acc = (successes / max(collected, 1)) * 100.0
        results[vpid] = acc

    return results


def train_distillation(
    num_epochs: int = 80,
    steps_per_epoch: int = 50,
    batch_envs: int = 64,
    lr: float = 5e-4,
):
    os.makedirs(STUDENT_DIR, exist_ok=True)
    with open(VIEWPOINTS_JSON) as f:
        vp_dict = {v["id"]: v for v in json.load(f)}

    teachers = load_all_teachers(vp_dict)

    # Initialize Student Policy
    student = OmniViewAgent(action_dim=4, state_dim=33).to(DEVICE)
    # Warm-start student from the canonical teacher (VP 17)
    student.warm_start_from_single_frame(teachers[17].state_dict())

    optimizer = optim.Adam(student.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)
    best_mean_acc = 0.0
    best_student_path = os.path.join(STUDENT_DIR, "omniview_student_best.pt")

    print("\n" + "=" * 80)
    print(" STARTING PHASE 10 EXPERT-GUIDED MULTI-TEACHER CONTINUOUS DISTILLATION")
    print(f" Teachers: {CORE_TEACHERS}")
    print(" Continuous Camera Jitter (+-5cm) + Extrinsics Gaussian Noise (sigma = 0.02m)")
    print(f" Envs: {batch_envs} | Steps/Epoch: {steps_per_epoch} | Total Epochs: {num_epochs}")
    print("=" * 80)

    t0 = time.time()
    for epoch in range(1, num_epochs + 1):
        student.train()
        epoch_loss = 0.0

        # Uniformly sample an anchor teacher to emulate in this training epoch
        base_vp = np.random.choice(CORE_TEACHERS)
        base_eye = np.array(vp_dict[base_vp]["world_eye"])
        # Add continuous camera position jitter (+-5cm in 3D space)
        jitter = np.random.normal(0, 0.03, size=3)
        cam_eye = (base_eye + jitter).tolist()

        nearest_teacher_id = find_nearest_teacher(cam_eye, vp_dict)
        teacher = teachers[nearest_teacher_id]

        # Create training environment at jittered viewpoint
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
        env = TemporalExtrinsicWrapper(env, cam_eye=cam_eye, frame_stack=3, noise_std=0.02, device="cuda")
        train_envs = ManiSkillVectorEnv(env, batch_envs, ignore_terminations=False, record_metrics=True)

        obs, _ = train_envs.reset()

        for step in range(steps_per_epoch):
            # Query teacher on newest 4-channel frame
            latest_rgbd = obs["rgbd"][:, 8:12, :, :]
            with torch.no_grad():
                vis_feat_t = teacher.feature_net.cnn(latest_rgbd)
                state_feat_t = teacher.feature_net.state_fc(obs["state"])
                feat_t = torch.cat([vis_feat_t, state_feat_t], dim=1)
                t_mean = teacher.actor_mean(feat_t)

            # Student forward pass on full 12-channel 3-frame stack
            feat_s = student.get_features(obs["rgbd"], obs["state"])
            s_mean = student.actor_mean(feat_s)

            # Distillation Loss: Action MSE regression to teacher actions
            loss = F.mse_loss(s_mean, t_mean)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()

            # DAgger / Expert Trajectory Guidance:
            # 85% teacher action (ensures the robot executes dense successful grasps/lifts),
            # 15% student action (allows student to explore recovery near the trajectory).
            with torch.no_grad():
                blend_mask = (torch.rand(batch_envs, 1, device=DEVICE) < 0.85).float()
                action = blend_mask * t_mean + (1.0 - blend_mask) * s_mean
            obs, _, _, _, _ = train_envs.step(action)

        train_envs.close()
        scheduler.step()
        avg_loss = epoch_loss / steps_per_epoch
        elapsed = time.time() - t0

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:02d}/{num_epochs:02d} | Anchor: VP {base_vp:02d} -> Teacher VP {nearest_teacher_id:02d} | MSE Loss: {avg_loss:.4f} | Time: {elapsed:.1f}s")

        # Evaluate student policy periodically
        if epoch % 10 == 0 or epoch == num_epochs:
            eval_results = evaluate_student(student, vp_dict, num_episodes=50)
            mean_acc = float(np.mean(list(eval_results.values())))
            viable_acc = float(np.mean([eval_results[v] for v in CORE_TEACHERS if v != 28]))
            print(f"  >>> [EVALUATION Epoch {epoch:02d}] Mean: {mean_acc:.1f}% | Viable: {viable_acc:.1f}% | VP 28: {eval_results[28]:.1f}% | VP 17: {eval_results[17]:.1f}%")

            if mean_acc >= best_mean_acc:
                best_mean_acc = mean_acc
                torch.save({
                    "student_state": student.state_dict(),
                    "eval_results": eval_results,
                    "mean_acc": mean_acc,
                    "viable_acc": viable_acc,
                    "epoch": epoch
                }, best_student_path)
                print(f"  >>> [NEW BEST STUDENT SAVED] Mean: {best_mean_acc:.1f}% -> {best_student_path}")

    print("\n" + "=" * 80)
    print(f" DISTILLATION COMPLETE! Best Student Accuracy: {best_mean_acc:.1f}%")
    print("=" * 80)
    return best_mean_acc


if __name__ == "__main__":
    train_distillation(num_epochs=80, steps_per_epoch=50, batch_envs=64)
