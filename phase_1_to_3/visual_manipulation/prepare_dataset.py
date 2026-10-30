"""
Utility to download PickCube demonstrations and replay them with RGB camera observations.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd: str):
    print(f"\n[Running] {cmd}\n")
    ret = subprocess.run(cmd, shell=True, check=True)
    return ret


def main():
    parser = argparse.ArgumentParser(description="Download and replay ManiSkill demonstrations with RGB cameras")
    parser.add_argument("--env_id", type=str, default="PickCube-v1", help="ManiSkill environment ID")
    parser.add_argument("--count", type=int, default=100, help="Number of demonstrations to replay")
    parser.add_argument("--control_mode", type=str, default="pd_ee_delta_pos", help="Target controller mode")
    parser.add_argument("--num_procs", type=int, default=10, help="Number of CPU workers for parallel replay")
    args = parser.parse_args()

    # Step 1: Download demonstrations
    print(f"=== Step 1: Downloading demonstrations for {args.env_id} ===")
    run_cmd(f"python3 -m mani_skill.utils.download_demo {args.env_id}")

    demo_root = Path.home() / ".maniskill" / "demos" / args.env_id / "motionplanning"
    input_traj = demo_root / "trajectory.h5"
    if not input_traj.exists():
        print(f"Error: {input_traj} not found after download.")
        sys.exit(1)

    # Step 2: Replay with RGB observations
    print(f"=== Step 2: Replaying demonstrations with RGB observations ({args.count} episodes) ===")
    replay_cmd = (
        f"python3 -m mani_skill.trajectory.replay_trajectory "
        f"--traj-path {input_traj} "
        f"--use-first-env-state -c {args.control_mode} -o rgb "
        f"--save-traj --num-envs {args.num_procs} -b physx_cpu --count {args.count}"
    )
    run_cmd(replay_cmd)

    output_h5 = demo_root / f"trajectory.rgb.{args.control_mode}.physx_cpu.h5"
    print(f"\n[Success] Dataset generated at: {output_h5}\n")


if __name__ == "__main__":
    main()
