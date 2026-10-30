"""
Fibonacci Equal-Area Hemispherical Viewpoint Generalization Benchmark.

Distributes N camera viewpoints uniformly (equal surface area) across the upper-frontal
hemisphere centered at the robot base [-0.615, 0.0, 0.0] with sphere radius R = 1.0942m.
Evaluates visual manipulation policy robustness across azimuth (-90 to +90 deg) and elevation (12 to 80 deg).
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import gymnasium as gym
import imageio
import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, "/home/maniskill")
sys.path.insert(0, "/home/maniskill/examples/baselines/ppo")

import mani_skill.envs
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper, FlattenRGBDObservationWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from ppo_rgb import Agent

BASE_POS = np.array([-0.615, 0.0, 0.0])
TARGET_POS = np.array([-0.1, 0.0, 0.1])
SPHERE_RADIUS = 1.0942  # meters (calibrated to match training distance)


def generate_fibonacci_hemisphere_points(
    num_points: int = 25,
    radius: float = SPHERE_RADIUS,
    min_elev_deg: float = 12.0,
    max_elev_deg: float = 80.0,
) -> list:
    """
    Generates N equal-area viewpoints over the front-upper hemisphere (x >= 0, z in [z_min, z_max]).
    Uses Fibonacci spiral lattice for uniform spherical area coverage.
    """
    golden_ratio = (1.0 + np.sqrt(5.0)) / 2.0
    golden_angle = 2.0 * np.pi * (1.0 - 1.0 / golden_ratio)

    z_min = np.sin(np.deg2rad(min_elev_deg))
    z_max = np.sin(np.deg2rad(max_elev_deg))

    points = []
    total_samples = int(num_points * 2.8)

    for i in range(total_samples):
        # Uniform z sampling for equal spherical area
        z_norm = z_min + (z_max - z_min) * (i + 0.5) / total_samples
        r_xy = np.sqrt(max(0.0, 1.0 - z_norm * z_norm))
        theta = i * golden_angle
        x_norm = r_xy * np.cos(theta)
        y_norm = r_xy * np.sin(theta)

        # Restrict to frontal hemisphere (x >= 0, in front of the robot)
        if x_norm >= 0.0:
            x_rel = float(x_norm * radius)
            y_rel = float(y_norm * radius)
            z_rel = float(z_norm * radius)

            # World coordinates
            world_eye = [
                float(BASE_POS[0] + x_rel),
                float(BASE_POS[1] + y_rel),
                float(BASE_POS[2] + z_rel),
            ]

            azimuth_deg = float(np.rad2deg(np.arctan2(y_rel, x_rel)))
            elev_deg = float(np.rad2deg(np.arcsin(z_norm)))

            points.append({
                "id": len(points),
                "rel_xyz": [round(x_rel, 4), round(y_rel, 4), round(z_rel, 4)],
                "world_eye": [round(c, 4) for c in world_eye],
                "azimuth_deg": round(azimuth_deg, 2),
                "elevation_deg": round(elev_deg, 2),
                "distance_to_base": round(float(radius), 4),
            })

            if len(points) == num_points:
                break

    return points


def evaluate_viewpoint(
    agent: Agent,
    viewpoint: dict,
    episodes: int = 200,
    num_envs: int = 48,
    device: str = "cuda",
) -> dict:
    eye = viewpoint["world_eye"]
    target = TARGET_POS.tolist()
    vp_id = viewpoint["id"]
    az = viewpoint["azimuth_deg"]
    el = viewpoint["elevation_deg"]

    print(f"\n[{vp_id+1:02d}] Viewpoint {vp_id:02d}: Azimuth={az:+6.1f}°, Elevation={el:4.1f}° | Eye={eye}")

    env = gym.make(
        "PickCube-v1",
        num_envs=num_envs,
        obs_mode="rgb",
        control_mode="pd_ee_delta_pos",
        render_mode=None,
        sim_backend="physx_cuda",
        sensor_cam_eye_pos=eye,
        sensor_cam_target_pos=target,
    )
    env = FlattenRGBDObservationWrapper(env, rgb=True, depth=False, state=True)
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    env = ManiSkillVectorEnv(env, num_envs, ignore_terminations=False, record_metrics=True)

    obs, _ = env.reset(seed=int(vp_id * 1000 + 42))

    episodes_completed = 0
    success_count = 0
    returns = []
    episode_lens = []

    start_time = time.time()

    with torch.no_grad():
        while episodes_completed < episodes:
            action = agent.get_action(obs, deterministic=True)
            obs, rew, term, trunc, infos = env.step(action)

            if "final_info" in infos:
                mask = infos["_final_info"]
                for i in range(len(mask)):
                    if mask[i]:
                        ep_info = infos["final_info"]["episode"]
                        succ_key = (
                            "success_once"
                            if "success_once" in ep_info
                            else ("success" if "success" in ep_info else "success_at_end")
                        )
                        succ = bool(ep_info[succ_key][i].item() if torch.is_tensor(ep_info[succ_key]) else ep_info[succ_key])
                        ret = float(ep_info["return"][i].item() if torch.is_tensor(ep_info["return"]) else ep_info["return"])
                        ep_len = float(ep_info["episode_len"][i].item() if torch.is_tensor(ep_info["episode_len"]) else ep_info["episode_len"])

                        if succ:
                            success_count += 1
                        returns.append(ret)
                        episode_lens.append(ep_len)
                        episodes_completed += 1

                        if episodes_completed >= episodes:
                            break

    elapsed = time.time() - start_time
    env.close()

    success_rate = (success_count / episodes) * 100.0
    mean_ret = float(np.mean(returns)) if returns else 0.0
    mean_len = float(np.mean(episode_lens)) if episode_lens else 0.0

    print(f"     -> Result: {success_count}/{episodes} ({success_rate:5.1f}%) in {elapsed:.1f}s | Return: {mean_ret:.2f}")

    res = dict(viewpoint)
    res.update({
        "episodes": episodes,
        "success_count": success_count,
        "success_rate": round(success_rate, 2),
        "mean_return": round(mean_ret, 2),
        "mean_length": round(mean_len, 2),
        "elapsed_seconds": round(elapsed, 2),
    })
    return res


def record_viewpoint_video(
    agent: Agent,
    viewpoint: dict,
    output_path: str,
    seed: int = 42,
    max_steps: int = 50,
):
    eye = viewpoint["world_eye"]
    target = TARGET_POS.tolist()
    az = viewpoint["azimuth_deg"]
    el = viewpoint["elevation_deg"]

    env = gym.make(
        "PickCube-v1",
        num_envs=1,
        obs_mode="rgb",
        control_mode="pd_ee_delta_pos",
        render_mode="rgb_array",
        sim_backend="physx_cuda",
        sensor_cam_eye_pos=eye,
        sensor_cam_target_pos=target,
    )
    env = FlattenRGBDObservationWrapper(env, rgb=True, depth=False, state=True)
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    env = ManiSkillVectorEnv(env, 1, ignore_terminations=False, record_metrics=True)

    obs, _ = env.reset(seed=seed)
    frames = []

    with torch.no_grad():
        for step in range(max_steps):
            render_frame = env.render()
            if isinstance(render_frame, torch.Tensor):
                render_frame = render_frame.cpu().numpy()
            if render_frame.ndim == 4:
                render_frame = render_frame[0]

            sensor_frame = obs["rgb"][0].cpu().numpy()
            sensor_large = cv2.resize(sensor_frame, (512, 512), interpolation=cv2.INTER_NEAREST)

            cv2.putText(
                render_frame,
                "Workspace (External)",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                sensor_large,
                f"Camera: Az={az:+.0f}deg, El={el:.0f}deg",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )

            composite = np.concatenate([sensor_large, render_frame], axis=1)
            frames.append(composite)

            action = agent.get_action(obs, deterministic=True)
            obs, rew, term, trunc, info = env.step(action)
            if term.any() or trunc.any():
                for _ in range(8):
                    frames.append(composite)
                break

    env.close()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    imageio.mimsave(output_path, frames, fps=20)
    print(f"     -> Recorded video: {output_path}")


def run_fibonacci_benchmark(
    checkpoint_path: str = "/home/maniskill/phase_1_to_3/visual_manipulation/best_sim2real_policy.pt",
    num_viewpoints: int = 30,
    episodes_per_viewpoint: int = 100,
    num_envs: int = 50,
    output_dir: str = "/home/maniskill/phase_1_to_3/fibonacci_hemisphere_30vp_results",
    viewpoints_file: str = "/home/maniskill/phase_1_to_3/sequential_viewpoint_training/viewpoints.json",
    device: str = "cuda",
):
    os.makedirs(output_dir, exist_ok=True)
    if viewpoints_file and os.path.exists(viewpoints_file):
        with open(viewpoints_file) as f:
            viewpoints = json.load(f)
        num_viewpoints = len(viewpoints)
    else:
        viewpoints = generate_fibonacci_hemisphere_points(num_points=num_viewpoints)

    print("\n" + "=" * 78)
    print(f" FIBONACCI EQUAL-AREA HEMISPHERE VIEWPOINT BENCHMARK")
    print(f" Checkpoint:               {checkpoint_path}")
    print(f" Total Uniform Viewpoints: {num_viewpoints}")
    print(f" Episodes per Viewpoint:   {episodes_per_viewpoint:,}")
    print(f" Total Evaluations:        {num_viewpoints * episodes_per_viewpoint:,} runs")
    print(f" Sphere Radius:            {SPHERE_RADIUS} m (Centered at Robot Base)")
    print(f" Output Directory:         {output_dir}")
    print("=" * 78)

    # Initialize dummy env to instantiate agent
    dummy_env = gym.make("PickCube-v1", num_envs=1, obs_mode="rgb", control_mode="pd_ee_delta_pos", sim_backend="physx_cuda")
    dummy_env = FlattenRGBDObservationWrapper(dummy_env, rgb=True, depth=False, state=True)
    if isinstance(dummy_env.action_space, gym.spaces.Dict):
        dummy_env = FlattenActionSpaceWrapper(dummy_env)
    dummy_env = ManiSkillVectorEnv(dummy_env, 1, ignore_terminations=False, record_metrics=True)
    sample_obs, _ = dummy_env.reset()

    agent = Agent(dummy_env, sample_obs=sample_obs).to(device)
    agent.load_state_dict(torch.load(checkpoint_path, map_location=device))
    agent.eval()
    dummy_env.close()

    results = []
    t_start = time.time()

    # Identify representative viewpoint IDs for video recording:
    # 1 lowest elevation, 1 highest elevation, 1 extreme left, 1 extreme right, and 1 center/front
    azimuths = [vp["azimuth_deg"] for vp in viewpoints]
    elevations = [vp["elevation_deg"] for vp in viewpoints]
    video_vp_ids = {
        int(np.argmin(elevations)),  # lowest elevation
        int(np.argmax(elevations)),  # highest elevation
        int(np.argmin(azimuths)),    # extreme right
        int(np.argmax(azimuths)),    # extreme left
        int(np.argmin(np.abs(azimuths))),  # closest to center
    }

    for vp in viewpoints:
        res = evaluate_viewpoint(
            agent=agent,
            viewpoint=vp,
            episodes=episodes_per_viewpoint,
            num_envs=num_envs,
            device=device,
        )
        results.append(res)

        if vp["id"] in video_vp_ids:
            vid_path = os.path.join(output_dir, f"viewpoint_{vp['id']:02d}_az{vp['azimuth_deg']:+.0f}_el{vp['elevation_deg']:.0f}.mp4")
            record_viewpoint_video(agent=agent, viewpoint=vp, output_path=vid_path)

    total_time = time.time() - t_start

    # Summary metrics
    success_rates = [r["success_rate"] for r in results]
    mean_generalization = float(np.mean(success_rates))
    max_success = float(np.max(success_rates))
    best_vp = results[int(np.argmax(success_rates))]

    # Save JSON report
    summary = {
        "checkpoint": checkpoint_path,
        "num_viewpoints": num_viewpoints,
        "episodes_per_viewpoint": episodes_per_viewpoint,
        "total_episodes": num_viewpoints * episodes_per_viewpoint,
        "total_walltime_seconds": round(total_time, 2),
        "mean_hemispherical_success_rate": round(mean_generalization, 2),
        "max_success_rate": round(max_success, 2),
        "best_viewpoint": best_vp,
        "viewpoints": results,
    }

    json_path = os.path.join(output_dir, "fibonacci_hemisphere_results.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary table
    print("\n" + "=" * 78)
    print(" FIBONACCI HEMISPHERE BENCHMARK RESULTS")
    print("=" * 78)
    print(f"{'ID':>3} | {'Azimuth':>9} | {'Elevation':>9} | {'Success Rate':>12} | {'Mean Ret':>8} | {'Mean Len':>8}")
    print("-" * 78)
    for r in results:
        print(
            f"{r['id']:3d} | {r['azimuth_deg']:+8.1f}° | {r['elevation_deg']:8.1f}° | {r['success_rate']:11.2f}% | {r['mean_return']:8.2f} | {r['mean_length']:8.1f}"
        )
    print("=" * 78)
    print(f"Overall Mean Hemispherical Generalization: {mean_generalization:.2f}%")
    print(f"Peak Success Viewpoint: Viewpoint {best_vp['id']} (Az={best_vp['azimuth_deg']}°, El={best_vp['elevation_deg']}°) with {max_success:.2f}%")
    print(f"Total Benchmark Time: {total_time/60:.2f} minutes")
    print(f"Results saved to: {json_path}\n")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fibonacci Equal-Area Hemisphere Viewpoint Evaluation")
    parser.add_argument("--checkpoint", type=str, default="/home/maniskill/phase_1_to_3/visual_manipulation/best_sim2real_policy.pt")
    parser.add_argument("--num_viewpoints", type=int, default=25, help="Number of equal-area viewpoints")
    parser.add_argument("--episodes", type=int, default=200, help="Episodes per viewpoint")
    parser.add_argument("--num_envs", type=int, default=48, help="Parallel environments")
    parser.add_argument("--output_dir", type=str, default="/home/maniskill/phase_1_to_3/fibonacci_hemisphere_30vp_results")
    parser.add_argument("--viewpoints_file", type=str, default="/home/maniskill/phase_1_to_3/sequential_viewpoint_training/viewpoints.json")
    args = parser.parse_args()

    run_fibonacci_benchmark(
        checkpoint_path=args.checkpoint,
        num_viewpoints=args.num_viewpoints,
        episodes_per_viewpoint=args.episodes,
        num_envs=args.num_envs,
        output_dir=args.output_dir,
        viewpoints_file=args.viewpoints_file,
    )
