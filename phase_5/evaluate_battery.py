"""
4-Tier Unbiased Evaluation Battery for Candidate Policies.
Evaluates:
  - Test 1: In-Domain Grasp Precision (1,000 runs on canonical front view)
  - Test 2: Angular Jitter Tolerance (75°, 90°, 105° x 1,000 runs = 3,000 runs)
  - Test 3: Photometric Domain Gap / Lighting Perturbation (1,000 runs with ±30% brightness shift)
  - Test 4: Inference Latency & Parameter Count
"""

import argparse
import gc
import json
import os
import sys
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, "/home/maniskill")

import mani_skill.envs
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper, FlattenRGBDObservationWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from models import UnifiedAblationAgent

BASE_POS = np.array([-0.615, 0.0, 0.0])
TARGET_POS = np.array([-0.1, 0.0, 0.1])
R_HORIZ = 0.915
Z_CAM = 0.600


def get_camera_eye(deg: float) -> list:
    rad = np.deg2rad(deg)
    x_rel = R_HORIZ * np.sin(rad)
    y_rel = R_HORIZ * np.cos(rad)
    return [float(BASE_POS[0] + x_rel), float(BASE_POS[1] + y_rel), float(Z_CAM)]


def run_eval_rollouts(
    agent: UnifiedAblationAgent,
    num_envs: int = 100,
    target_episodes: int = 1000,
    camera_eye: list = None,
    resolution: int = 128,
    lighting_shift_factor: float = 1.0,
    seed: int = 12345,
    device: str = "cuda",
) -> dict:
    if camera_eye is None:
        camera_eye = [0.3, 0.0, 0.6]

    env_kwargs = dict(
        obs_mode="rgbd",
        control_mode="pd_ee_delta_pos",
        sim_backend="physx_cuda",
        sensor_cam_eye_pos=camera_eye,
        sensor_cam_target_pos=[-0.1, 0.0, 0.1],
        sensor_configs=dict(width=resolution, height=resolution),
    )

    env = gym.make("PickCube-v1", num_envs=num_envs, **env_kwargs)
    env = FlattenRGBDObservationWrapper(env, rgb=True, depth=True, state=True, sep_depth=True)
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    env = ManiSkillVectorEnv(env, num_envs, ignore_terminations=False, record_metrics=True)

    obs, _ = env.reset(seed=seed)
    completed = 0
    successes = 0
    returns = []
    lens = []
    step_latencies = []

    t0 = time.time()

    with torch.no_grad():
        while completed < target_episodes:
            # Apply lighting shift if testing photometric robustness
            if lighting_shift_factor != 1.0 and "rgb" in obs:
                obs["rgb"] = torch.clamp(obs["rgb"].float() * lighting_shift_factor, 0, 255).to(torch.uint8)

            t_step0 = time.time()
            action = agent.get_action(obs, deterministic=True)
            step_latencies.append((time.time() - t_step0) * 1000.0)

            obs, rew, term, trunc, infos = env.step(action)

            if "final_info" in infos:
                mask = infos["_final_info"]
                for i in range(len(mask)):
                    if mask[i]:
                        ep_info = infos["final_info"]["episode"]
                        succ_key = "success_once" if "success_once" in ep_info else "success"
                        succ = bool(ep_info[succ_key][i].item() if torch.is_tensor(ep_info[succ_key]) else ep_info[succ_key])
                        if succ:
                            successes += 1
                        completed += 1
                        returns.append(float(ep_info["return"][i].item() if torch.is_tensor(ep_info["return"]) else ep_info["return"]))
                        lens.append(float(ep_info["episode_len"][i].item() if torch.is_tensor(ep_info["episode_len"]) else ep_info["episode_len"]))
                        if completed >= target_episodes:
                            break
            if completed > 0 and completed % 200 == 0:
                print(f"       -> Evaluated {completed}/{target_episodes} rollouts ({successes} successes)...", flush=True)

    env.close()
    elapsed = time.time() - t0
    rate = (successes / completed) * 100.0 if completed > 0 else 0.0

    return {
        "episodes": completed,
        "success_count": successes,
        "success_rate": round(rate, 2),
        "mean_return": round(float(np.mean(returns)), 4) if returns else 0.0,
        "mean_length": round(float(np.mean(lens)), 2) if lens else 0.0,
        "elapsed_seconds": round(elapsed, 2),
        "mean_latency_ms": round(float(np.mean(step_latencies)), 2) if step_latencies else 0.0,
    }


def evaluate_candidate_battery(
    candidate: str,
    checkpoint_path: str,
    num_envs: int = 100,
    target_episodes: int = 1000,
    output_dir: str = "/home/maniskill/phase_5/eval_results",
    device_str: str = "cuda",
) -> dict:
    candidate = candidate.upper()
    res = 128 if candidate == "A" else 256
    device = torch.device(device_str)

    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 78)
    print(f" EVALUATING CANDIDATE {candidate} ({res}x{res}) ACROSS 4-TIER BATTERY")
    print(f" Checkpoint: {checkpoint_path}")
    print("=" * 78 + "\n")

    # Instantiate Agent
    dummy_env = gym.make("PickCube-v1", num_envs=1, obs_mode="rgbd", control_mode="pd_ee_delta_pos", sim_backend="physx_cuda", sensor_configs=dict(width=res, height=res))
    dummy_env = FlattenRGBDObservationWrapper(dummy_env, rgb=True, depth=True, state=True, sep_depth=True)
    if isinstance(dummy_env.action_space, gym.spaces.Dict):
        dummy_env = FlattenActionSpaceWrapper(dummy_env)
    dummy_env = ManiSkillVectorEnv(dummy_env, 1, ignore_terminations=False)
    sample_obs, _ = dummy_env.reset()

    agent = UnifiedAblationAgent(dummy_env, sample_obs=sample_obs, candidate=candidate).to(device)
    agent.load_state_dict(torch.load(checkpoint_path, map_location=device))
    agent.eval()
    dummy_env.close()

    total_params = sum(p.numel() for p in agent.parameters())

    # --- TEST 1: In-Domain Canonical Front-View Benchmark ---
    print(f"\n[Test 1/4] Running {target_episodes}-Episode Canonical Front-View Benchmark...")
    t1_results = run_eval_rollouts(agent, num_envs=num_envs, target_episodes=target_episodes, camera_eye=[0.3, 0.0, 0.6], resolution=res, device=device)
    print(f"  -> In-Domain Success: {t1_results['success_count']}/{target_episodes} ({t1_results['success_rate']}%) | Mean Len: {t1_results['mean_length']} | Mean Ret: {t1_results['mean_return']}")

    # --- TEST 2: Angular Jitter Tolerance (75°, 90°, 105°) ---
    print(f"\n[Test 2/4] Running Angular Jitter Sweep (75°, 90°, 105° x {target_episodes} runs)...")
    jitter_angles = [75.0, 90.0, 105.0]
    t2_results = {}
    for ang in jitter_angles:
        eye = get_camera_eye(ang)
        res_ang = run_eval_rollouts(agent, num_envs=num_envs, target_episodes=target_episodes, camera_eye=eye, resolution=res, device=device)
        t2_results[f"angle_{int(ang)}deg"] = res_ang
        print(f"  -> Angle {ang:5.1f}°: {res_ang['success_count']}/{target_episodes} ({res_ang['success_rate']}%)")
    mean_jitter_success = round(float(np.mean([t2_results[k]["success_rate"] for k in t2_results])), 2)

    # --- TEST 3: Photometric Domain Gap / Lighting Perturbation ---
    print(f"\n[Test 3/4] Running Photometric Shift Benchmark (Darkened 0.7x Lighting, {target_episodes} runs)...")
    t3_dark = run_eval_rollouts(agent, num_envs=num_envs, target_episodes=target_episodes, camera_eye=[0.3, 0.0, 0.6], resolution=res, lighting_shift_factor=0.7, device=device)
    print(f"  -> 0.7x Lighting Shift: {t3_dark['success_count']}/{target_episodes} ({t3_dark['success_rate']}%) | Retention: {(t3_dark['success_rate'] / max(t1_results['success_rate'], 1e-5))*100:.1f}%")

    print(f"\n[Test 3/4] Running Photometric Shift Benchmark (Brightened 1.3x Lighting, {target_episodes} runs)...")
    t3_bright = run_eval_rollouts(agent, num_envs=num_envs, target_episodes=target_episodes, camera_eye=[0.3, 0.0, 0.6], resolution=res, lighting_shift_factor=1.3, device=device)
    print(f"  -> 1.3x Lighting Shift: {t3_bright['success_count']}/{target_episodes} ({t3_bright['success_rate']}%) | Retention: {(t3_bright['success_rate'] / max(t1_results['success_rate'], 1e-5))*100:.1f}%")

    # --- TEST 4: Hardware & Latency Telemetry ---
    hardware_telemetry = {
        "parameter_count": total_params,
        "mean_inference_latency_ms": t1_results["mean_latency_ms"],
    }

    battery_summary = {
        "candidate": candidate,
        "resolution": res,
        "checkpoint": checkpoint_path,
        "test1_front_view_1000": t1_results,
        "test2_angular_jitter": {
            "mean_success_rate": mean_jitter_success,
            "angles": t2_results,
        },
        "test3_photometric_shift": {
            "darkened_0.7x": t3_dark,
            "brightened_1.3x": t3_bright,
            "mean_perturbed_success": round((t3_dark["success_rate"] + t3_bright["success_rate"]) / 2.0, 2),
        },
        "test4_hardware": hardware_telemetry,
    }

    out_file = os.path.join(output_dir, f"candidate_{candidate}_results.json")
    with open(out_file, "w") as f:
        json.dump(battery_summary, f, indent=2)

    print(f"\n[Saved] Candidate {candidate} evaluation saved to: {out_file}\n")

    del agent
    gc.collect()
    torch.cuda.empty_cache()

    return battery_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=str, required=True, choices=["A", "B", "C", "D", "E"])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num-envs", type=int, default=100)
    parser.add_argument("--target-episodes", type=int, default=1000)
    parser.add_argument("--output-dir", type=str, default="/home/maniskill/phase_5/eval_results")
    args = parser.parse_args()

    evaluate_candidate_battery(
        candidate=args.candidate,
        checkpoint_path=args.checkpoint,
        num_envs=args.num_envs,
        target_episodes=args.target_episodes,
        output_dir=args.output_dir,
    )
