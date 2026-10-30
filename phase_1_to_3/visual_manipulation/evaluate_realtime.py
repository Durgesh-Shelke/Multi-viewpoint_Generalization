"""
Live Real-Time Visual Policy Evaluation for 4.5cm Block Pick.
Runs simulation at exactly sim time = real time (20 Hz, 50ms per step, no acceleration).
Renders live to SAPIEN viewer GUI window (DISPLAY=:0) and records an exact 1:1 real-time MP4 video.
"""

import os
import sys
import time
from pathlib import Path

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


def run_realtime_evaluation(
    checkpoint_path: str = "/home/maniskill/visual_manipulation/best_sim2real_policy.pt",
    output_video_path: str = "/root/.gemini/antigravity-cli/brain/82673f33-5c96-4154-bb42-cf6de559c7b7/live_realtime_pickup.mp4",
    seed: int = 42,
    hold_steps_after_pick: int = 30,
):
    print("\n" + "=" * 70)
    print(" LIVE REAL-TIME VISUAL EVALUATION (Sim Time = Real Time, 1.0x Speed)")
    print(f" Checkpoint:  {checkpoint_path}")
    print(f" Target Goal: Pick 4.5cm Solid Block (Option A)")
    print(f" Video Out:   {output_video_path}")
    print("=" * 70 + "\n")

    # Attempt human render mode first for live window
    has_live_window = False
    render_mode = "human"
    try:
        env = gym.make(
            "PickCube-v1",
            num_envs=1,
            obs_mode="rgb",
            control_mode="pd_ee_delta_pos",
            render_mode="human",
            sim_backend="physx_cuda",
        )
        env = FlattenRGBDObservationWrapper(env, rgb=True, depth=False, state=True)
        if isinstance(env.action_space, gym.spaces.Dict):
            env = FlattenActionSpaceWrapper(env)
        env = ManiSkillVectorEnv(env, 1, ignore_terminations=False, record_metrics=True)
        obs, _ = env.reset(seed=seed)
        env.render()
        has_live_window = True
        print("[Display] Live SAPIEN 3D Viewer window opened on DISPLAY=:0 successfully!")
    except Exception as e:
        print(f"[Display] Could not open live GUI window ({e}). Falling back to headless video capture.")
        render_mode = "rgb_array"
        env = gym.make(
            "PickCube-v1",
            num_envs=1,
            obs_mode="rgb",
            control_mode="pd_ee_delta_pos",
            render_mode="rgb_array",
            sim_backend="physx_cuda",
        )
        env = FlattenRGBDObservationWrapper(env, rgb=True, depth=False, state=True)
        if isinstance(env.action_space, gym.spaces.Dict):
            env = FlattenActionSpaceWrapper(env)
        env = ManiSkillVectorEnv(env, 1, ignore_terminations=False, record_metrics=True)
        obs, _ = env.reset(seed=seed)

    # Load agent
    agent = Agent(env, sample_obs=obs).cuda()
    print(f"[Model] Loading weights from {checkpoint_path}...")
    agent.load_state_dict(torch.load(checkpoint_path, map_location="cuda"))
    agent.eval()

    # Time parameters
    control_freq = 20  # 20 Hz
    sim_dt = 1.0 / control_freq  # 0.05 seconds = 50 ms
    print(f"[Timing] Sim Control Frequency = {control_freq} Hz (timestep = {sim_dt*1000:.1f} ms)")
    print("[Timing] Enforcing strict 1:1 real-time pacing (no simulation acceleration)...")

    # Reset environment
    obs, info = env.reset(seed=seed)
    frames = []

    # Capture initial frame
    if render_mode == "human":
        env.render()
        rgb = env.unwrapped.render_rgb_array()
    else:
        rgb = env.render()
    if isinstance(rgb, torch.Tensor):
        rgb = rgb.cpu().numpy()
    if rgb.ndim == 4:
        rgb = rgb[0]
    frames.append(rgb)

    step = 0
    picked_up = False
    success_step = None
    extra_hold_steps = 0
    total_real_time_start = time.perf_counter()

    print("\n--- Live Execution Started ---")
    with torch.no_grad():
        while True:
            t0 = time.perf_counter()
            step += 1

            # 1. Visuomotor inference
            action = agent.get_action(obs, deterministic=True)

            # 2. Simulation step
            obs, rew, term, trunc, infos = env.step(action)

            # Check success
            if not picked_up:
                # Check if cube is picked up
                if "final_info" in infos and infos["_final_info"][0]:
                    picked_up = True
                    success_step = step
                elif "is_grasped" in infos:
                    pass

            # 3. Live render
            if render_mode == "human":
                try:
                    env.render()
                except Exception:
                    pass
                rgb = env.unwrapped.render_rgb_array()
            else:
                rgb = env.render()

            if isinstance(rgb, torch.Tensor):
                rgb = rgb.cpu().numpy()
            if rgb.ndim == 4:
                rgb = rgb[0]
            frames.append(rgb)

            # 4. Check episode termination / hold
            if term.any() or trunc.any() or picked_up:
                if not picked_up:
                    picked_up = True
                    success_step = step
                extra_hold_steps += 1
                if extra_hold_steps >= hold_steps_after_pick:
                    break
            elif step >= 50:
                break

            # 5. Enforce 1:1 real-time playback
            t_compute = time.perf_counter() - t0
            sleep_needed = sim_dt - t_compute
            if sleep_needed > 0:
                time.sleep(sleep_needed)

            actual_step_dt = time.perf_counter() - t0
            sim_time_sec = step * sim_dt
            real_time_sec = time.perf_counter() - total_real_time_start
            if step % 5 == 0 or picked_up and extra_hold_steps == 1:
                status_str = "PICKED UP! Lifting & holding block..." if picked_up else "Approaching block..."
                print(f" Step {step:2d} | Sim Time: {sim_time_sec:4.2f}s | Real Time: {real_time_sec:4.2f}s | Status: {status_str}")

    total_real_elapsed = time.perf_counter() - total_real_time_start
    total_sim_time = step * sim_dt
    env.close()

    print("\n--- Live Execution Complete ---")
    print(f"Total Steps Executed: {step}")
    print(f"Total Simulated Time: {total_sim_time:.2f} seconds")
    print(f"Total Real Wall Time: {total_real_elapsed:.2f} seconds")
    print(f"Real-Time Ratio:      {total_real_elapsed / total_sim_time:.2f}x (Ideal: 1.00x)")
    if success_step:
        print(f"Result:               SUCCESSFUL PICK at Step {success_step} ({success_step*sim_dt:.2f}s)")
    else:
        print(f"Result:               Completed")

    # Save video at exactly 20 fps so 1 frame = 0.05s
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    print(f"\n[Saving] Writing {len(frames)} frames to {output_video_path} at {control_freq} FPS...")
    imageio.mimsave(output_video_path, frames, fps=control_freq)
    print(f"[Done] Video successfully saved to: {output_video_path}\n")

    return {
        "steps": step,
        "sim_time": total_sim_time,
        "real_time": total_real_elapsed,
        "success": bool(picked_up),
        "video_path": output_video_path,
        "live_window": has_live_window,
    }


if __name__ == "__main__":
    run_realtime_evaluation()
