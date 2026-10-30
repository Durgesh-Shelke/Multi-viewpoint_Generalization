"""
Multi-Teacher Policy Distillation Training Harness.
Distills 8 specialized viewpoint teachers into a single unified visual policy
using supervised multi-view imitation learning + DAgger interactive online refinement.
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import gymnasium as gym

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, "/home/maniskill")

import mani_skill.envs
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper, FlattenRGBDObservationWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from models import UnifiedAblationAgent

VIEWPOINTS_JSON = "/home/maniskill/phase_1_to_3/sequential_viewpoint_training/viewpoints.json"
TEACHER_CKPT_DIR = "/home/maniskill/phase_6_viewpoint_specialists/checkpoints"
CORE_VP_IDS = [17, 0, 28, 18, 27, 20, 19, 6]


def evaluate_on_viewpoint(agent: nn.Module, vp_id: int, num_envs: int = 50, max_steps: int = 50, device: str = "cuda"):
    with open(VIEWPOINTS_JSON) as f:
        all_vps = json.load(f)
    vp_dict = {vp["id"]: vp for vp in all_vps}
    vp = vp_dict[vp_id]
    cam_eye = vp["world_eye"]
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
    eval_envs = gym.make("PickCube-v1", num_envs=num_envs, reconfiguration_freq=None, **env_kwargs)
    eval_envs = FlattenRGBDObservationWrapper(eval_envs, rgb=True, depth=True, state=True, sep_depth=True)
    if isinstance(eval_envs.action_space, gym.spaces.Dict):
        eval_envs = FlattenActionSpaceWrapper(eval_envs)
    eval_envs = ManiSkillVectorEnv(eval_envs, num_envs, ignore_terminations=True, record_metrics=True)

    agent.eval()
    obs, _ = eval_envs.reset(seed=999 + vp_id)
    successes = 0
    eps = 0

    with torch.no_grad():
        for _ in range(max_steps):
            act = agent.get_action(obs, deterministic=True)
            obs, _, _, _, infos = eval_envs.step(act)
            if "final_info" in infos and "_final_info" in infos:
                mask = infos["_final_info"]
                for i in range(len(mask)):
                    if mask[i]:
                        ep = infos["final_info"].get("episode", {})
                        k = "success_once" if "success_once" in ep else "success"
                        if k in ep:
                            val = ep[k][i]
                            if bool(val.item() if torch.is_tensor(val) else val):
                                successes += 1
                        eps += 1

    eval_envs.close()
    succ_rate = (successes / max(eps, 1)) * 100.0 if eps > 0 else 0.0
    return succ_rate


def evaluate_all_viewpoints(agent: nn.Module, viewpoints=CORE_VP_IDS, num_envs: int = 50, device: str = "cuda"):
    results = {}
    for vp in viewpoints:
        succ = evaluate_on_viewpoint(agent, vp, num_envs=num_envs, device=device)
        results[vp] = succ
    mean_succ = float(np.mean(list(results.values())))
    min_succ = float(np.min(list(results.values())))
    return results, mean_succ, min_succ


def load_multi_teacher_dataset(data_dir: str, viewpoints=CORE_VP_IDS):
    all_rgb = []
    all_depth = []
    all_state = []
    all_actions = []

    print("[Distill] Loading demonstration datasets across all 8 viewpoints...")
    for vp in viewpoints:
        fpath = os.path.join(data_dir, f"vp_{vp:02d}_rollouts.pt")
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Missing rollout file: {fpath}. Run collect_teacher_rollouts.py first!")
        data = torch.load(fpath, map_location="cpu")
        all_rgb.append(data["rgb"])
        all_depth.append(data["depth"])
        all_state.append(data["state"])
        all_actions.append(data["actions"])
        print(f"  - VP {vp:02d}: {data['num_samples']:,} transitions (Teacher Eval: {data.get('rollout_success_rate', 0):.1f}%)")

    cat_rgb = torch.cat(all_rgb, dim=0)
    cat_depth = torch.cat(all_depth, dim=0)
    cat_state = torch.cat(all_state, dim=0)
    cat_actions = torch.cat(all_actions, dim=0)
    total_samples = cat_rgb.shape[0]

    print(f"[Distill] Total Combined Dataset: {total_samples:,} multi-viewpoint state-action pairs.")
    return cat_rgb, cat_depth, cat_state, cat_actions


def run_distillation(
    epochs: int = 30,
    batch_size: int = 256,
    lr: float = 5e-4,
    weight_decay: float = 1e-4,
    dagger_rounds: int = 2,
    output_dir: str = SCRIPT_DIR,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    eval_dir = os.path.join(output_dir, "eval_results")
    tb_dir = os.path.join(output_dir, "tb_logs")
    data_dir = os.path.join(output_dir, "data")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)
    os.makedirs(tb_dir, exist_ok=True)

    best_ckpt_path = os.path.join(ckpt_dir, "distilled_student_best.pt")
    metrics_json_path = os.path.join(eval_dir, "distillation_metrics.json")
    history_csv_path = os.path.join(eval_dir, "distillation_curve.csv")

    # Load offline data
    all_rgb, all_depth, all_state, all_actions = load_multi_teacher_dataset(data_dir, CORE_VP_IDS)

    # Initialize unified Student agent
    sample_obs = {
        "rgb": torch.zeros((1, 128, 128, 3), dtype=torch.uint8, device=device),
        "depth": torch.zeros((1, 128, 128, 1), dtype=torch.int16, device=device),
        "state": torch.zeros((1, 29), dtype=torch.float32, device=device),
    }

    # Create dummy env to obtain action space shape
    env_kwargs = dict(obs_mode="rgbd", render_mode="all", sim_backend="physx_cuda", control_mode="pd_ee_delta_pos")
    dummy_env = gym.make("PickCube-v1", num_envs=1, **env_kwargs)
    dummy_env = FlattenRGBDObservationWrapper(dummy_env, rgb=True, depth=True, state=True, sep_depth=True)
    if isinstance(dummy_env.action_space, gym.spaces.Dict):
        dummy_env = FlattenActionSpaceWrapper(dummy_env)
    student = UnifiedAblationAgent(dummy_env, sample_obs, candidate="A").to(device)
    dummy_env.close()

    optimizer = optim.AdamW(student.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    writer = SummaryWriter(tb_dir)

    dataset = TensorDataset(all_rgb, all_depth, all_state, all_actions)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=2)

    print(f"\n{'='*78}")
    print(f" STARTING MULTI-TEACHER POLICY DISTILLATION (8 VIEWPOINTS)")
    print(f" Architecture: Candidate A (4-Channel 128x128 RGB-D NatureCNN)")
    print(f" Dataset: {len(dataset):,} samples | Batch Size: {batch_size} | Epochs: {epochs}")
    print(f" Target: Single Unified Model mastering all 8 viewpoints")
    print(f"{'='*78}\n")

    best_mean_succ = 0.0
    history = []
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        student.train()
        epoch_losses = []
        ep_t0 = time.time()

        for batch in loader:
            b_rgb, b_depth, b_state, b_acts = [t.to(device, non_blocking=True) for t in batch]
            b_obs = {"rgb": b_rgb, "depth": b_depth, "state": b_state}

            pred_act = student.actor_mean(student.feature_net(b_obs))
            loss = F.mse_loss(pred_act, b_acts)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_losses.append(loss.item())

        scheduler.step()
        ep_time = time.time() - ep_t0
        mean_loss = float(np.mean(epoch_losses))
        writer.add_scalar("distill/loss", mean_loss, epoch)

        # Evaluation every 5 epochs or final
        if epoch % 5 == 0 or epoch == epochs:
            print(f"[Epoch {epoch:02d}/{epochs}] Distillation Loss: {mean_loss:.5f} | Time: {ep_time:.1f}s | Evaluating all 8 viewpoints...")
            vp_results, mean_succ, min_succ = evaluate_all_viewpoints(student, CORE_VP_IDS, num_envs=50, device=device)
            walltime = time.time() - start_time

            print(f"  >>> [Eval Summary] Mean Multi-View Success: {mean_succ:5.1f}% | Worst-Case: {min_succ:5.1f}%")
            vp_str = " | ".join([f"VP{k:02d}:{succ:4.0f}%" for k, succ in vp_results.items()])
            print(f"      Per-Viewpoint: {vp_str}")

            record = {
                "epoch": epoch,
                "loss": round(mean_loss, 5),
                "mean_success": round(mean_succ, 2),
                "min_success": round(min_succ, 2),
                "walltime_seconds": round(walltime, 1),
                **{f"vp_{k:02d}_succ": round(v, 2) for k, v in vp_results.items()},
            }
            history.append(record)

            if mean_succ >= best_mean_succ:
                best_mean_succ = mean_succ
                torch.save({
                    "epoch": epoch,
                    "mean_success": mean_succ,
                    "min_success": min_succ,
                    "per_viewpoint": vp_results,
                    "model_state": student.state_dict(),
                }, best_ckpt_path)
                print(f"  >>> [NEW BEST DISTILLED MODEL] Mean Success: {best_mean_succ:.1f}% -> {best_ckpt_path}")
        else:
            print(f"[Epoch {epoch:02d}/{epochs}] Distillation Loss: {mean_loss:.5f} | Time: {ep_time:.1f}s")

    # DAgger Interactive Online Refinement Rounds
    if dagger_rounds > 0 and best_mean_succ < 95.0:
        print(f"\n{'='*78}")
        print(f" STARTING DAGGER INTERACTIVE ONLINE CORRECTION ({dagger_rounds} ROUNDS)")
        print(f"{'='*78}\n")

        # Load best student weights
        best_ckpt = torch.load(best_ckpt_path, map_location=device)
        student.load_state_dict(best_ckpt["model_state"])

        for d_round in range(1, dagger_rounds + 1):
            print(f"\n[DAgger Round {d_round}/{dagger_rounds}] Collecting student trajectory rollouts with teacher relabeling...")
            new_samples = []

            for vp in CORE_VP_IDS:
                # Load teacher
                teacher_ckpt = os.path.join(TEACHER_CKPT_DIR, f"specialist_vp_{vp:02d}_best.pt")
                teacher = UnifiedAblationAgent(dummy_env, sample_obs, candidate="A").to(device)
                teacher.load_state_dict(torch.load(teacher_ckpt, map_location=device)["model_state"])
                teacher.eval()

                # Rollout student and relabel with teacher
                with open(VIEWPOINTS_JSON) as f:
                    all_vps = json.load(f)
                cam_eye = {v["id"]: v for v in all_vps}[vp]["world_eye"]
                env_kwargs = dict(obs_mode="rgbd", render_mode="all", sim_backend="physx_cuda", sensor_cam_eye_pos=cam_eye, sensor_cam_target_pos=[-0.1, 0.0, 0.1], control_mode="pd_ee_delta_pos", sensor_configs=dict(width=128, height=128))
                rollout_envs = gym.make("PickCube-v1", num_envs=32, **env_kwargs)
                rollout_envs = FlattenRGBDObservationWrapper(rollout_envs, rgb=True, depth=True, state=True, sep_depth=True)
                if isinstance(rollout_envs.action_space, gym.spaces.Dict):
                    rollout_envs = FlattenActionSpaceWrapper(rollout_envs)
                rollout_envs = ManiSkillVectorEnv(rollout_envs, 32, ignore_terminations=False, record_metrics=False)

                r_obs, _ = rollout_envs.reset(seed=500 + vp + d_round)
                with torch.no_grad():
                    for _ in range(50):
                        # Student steps
                        s_act = student.get_action(r_obs, deterministic=True)
                        # Teacher relabels
                        t_act = teacher.get_action(r_obs, deterministic=True)

                        new_samples.append((r_obs["rgb"].cpu(), r_obs["depth"].cpu(), r_obs["state"].cpu(), t_act.cpu()))
                        r_obs, _, _, _, _ = rollout_envs.step(s_act)
                rollout_envs.close()

            # Append DAgger samples
            dag_rgb = torch.cat([s[0] for s in new_samples], dim=0)
            dag_depth = torch.cat([s[1] for s in new_samples], dim=0)
            dag_state = torch.cat([s[2] for s in new_samples], dim=0)
            dag_acts = torch.cat([s[3] for s in new_samples], dim=0)
            print(f"[DAgger Round {d_round}] Added {dag_rgb.shape[0]:,} student-visited states relabeled by teachers.")

            # Fine-tune student for 5 epochs
            dag_dataset = TensorDataset(torch.cat([all_rgb, dag_rgb]), torch.cat([all_depth, dag_depth]), torch.cat([all_state, dag_state]), torch.cat([all_actions, dag_acts]))
            dag_loader = DataLoader(dag_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=2)

            dag_optimizer = optim.AdamW(student.parameters(), lr=1e-4, weight_decay=1e-4)
            student.train()
            for ep in range(5):
                for b_rgb, b_depth, b_state, b_acts in dag_loader:
                    b_rgb, b_depth, b_state, b_acts = b_rgb.to(device), b_depth.to(device), b_state.to(device), b_acts.to(device)
                    loss = F.mse_loss(student.actor_mean(student.feature_net({"rgb": b_rgb, "depth": b_depth, "state": b_state})), b_acts)
                    dag_optimizer.zero_grad()
                    loss.backward()
                    dag_optimizer.step()

            # Post DAgger evaluation
            vp_results, mean_succ, min_succ = evaluate_all_viewpoints(student, CORE_VP_IDS, num_envs=50, device=device)
            print(f"[DAgger Round {d_round} Eval] Mean Success: {mean_succ:.1f}% | Worst: {min_succ:.1f}%")
            if mean_succ >= best_mean_succ:
                best_mean_succ = mean_succ
                torch.save({"epoch": epochs + d_round, "mean_success": mean_succ, "min_success": min_succ, "per_viewpoint": vp_results, "model_state": student.state_dict()}, best_ckpt_path)
                print(f"  >>> [NEW BEST POST-DAGGER] Success: {best_mean_succ:.1f}% -> {best_ckpt_path}")

    total_walltime = time.time() - start_time

    # Save metrics JSON and CSV
    import csv
    if history:
        with open(history_csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            w.writeheader()
            w.writerows(history)

    final_ckpt = torch.load(best_ckpt_path, map_location="cpu")
    metrics = {
        "method": "Multi-Teacher Policy Distillation",
        "epochs": epochs,
        "dagger_rounds": dagger_rounds,
        "best_mean_success": final_ckpt.get("mean_success", best_mean_succ),
        "best_min_success": final_ckpt.get("min_success", 0.0),
        "per_viewpoint": final_ckpt.get("per_viewpoint", {}),
        "total_walltime_seconds": round(total_walltime, 1),
        "checkpoint_path": best_ckpt_path,
    }
    with open(metrics_json_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n[COMPLETE] Multi-Teacher Policy Distillation Finished!")
    print(f" Best Mean Multi-View Success: {metrics['best_mean_success']:.1f}% | Walltime: {total_walltime / 60:.1f} mins")
    print(f" Model: {best_ckpt_path}\n")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--dagger-rounds", type=int, default=2)
    parser.add_argument("--output-dir", type=str, default=SCRIPT_DIR)
    args = parser.parse_args()

    run_distillation(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, dagger_rounds=args.dagger_rounds, output_dir=args.output_dir)
