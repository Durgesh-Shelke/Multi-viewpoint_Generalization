"""
Federated Learning Policy Aggregation: FedAvg & FedProx for Multi-Viewpoint RL.
Implements iterative communication rounds where client policies train locally on individual viewpoints
with optional FedProx proximal regularization, and a central server aggregates weights across rounds.
"""

import os
import sys
import json
import time
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, "/home/maniskill")

import mani_skill.envs
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper, FlattenRGBDObservationWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from models import UnifiedAblationAgent

VIEWPOINTS_JSON = "/home/maniskill/phase_1_to_3/sequential_viewpoint_training/viewpoints.json"
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
    obs, _ = eval_envs.reset(seed=1234 + vp_id)
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
    std_succ = float(np.std(list(results.values())))
    return results, mean_succ, min_succ, std_succ


def local_client_update(
    global_state_dict: dict,
    vp_id: int,
    local_iterations: int = 5,
    num_envs: int = 64,
    num_steps: int = 50,
    mu: float = 0.0,  # FedProx proximal weight (mu=0 for standard FedAvg)
    lr: float = 3e-4,
    device: str = "cuda",
):
    with open(VIEWPOINTS_JSON) as f:
        all_vps = json.load(f)
    vp_dict = {vp["id"]: vp for vp in all_vps}
    vp = vp_dict[vp_id]
    cam_eye = vp["world_eye"]

    env_kwargs = dict(
        obs_mode="rgbd",
        render_mode="all",
        sim_backend="physx_cuda",
        sensor_cam_eye_pos=cam_eye,
        sensor_cam_target_pos=[-0.1, 0.0, 0.1],
        control_mode="pd_ee_delta_pos",
        sensor_configs=dict(width=128, height=128),
    )
    envs = gym.make("PickCube-v1", num_envs=num_envs, reconfiguration_freq=None, **env_kwargs)
    envs = FlattenRGBDObservationWrapper(envs, rgb=True, depth=True, state=True, sep_depth=True)
    if isinstance(envs.action_space, gym.spaces.Dict):
        envs = FlattenActionSpaceWrapper(envs)
    envs = ManiSkillVectorEnv(envs, num_envs, ignore_terminations=False, record_metrics=True)

    obs, _ = envs.reset(seed=42 + vp_id)
    client_agent = UnifiedAblationAgent(envs, obs, candidate="A").to(device)
    client_agent.load_state_dict(global_state_dict)

    # Reference copy of global weights for proximal regularization
    global_weights = {k: v.clone().detach() for k, v in global_state_dict.items()}

    optimizer = optim.Adam(client_agent.parameters(), lr=lr, eps=1e-5)
    batch_size = num_envs * num_steps
    minibatch_size = batch_size // 4

    obs_rgb = torch.zeros((num_steps, num_envs, 128, 128, 3), dtype=torch.uint8, device=device)
    obs_depth = torch.zeros((num_steps, num_envs, 128, 128, 1), dtype=torch.int16, device=device)
    obs_state = torch.zeros((num_steps, num_envs, 29), dtype=torch.float32, device=device)
    actions = torch.zeros((num_steps, num_envs) + envs.single_action_space.shape, device=device)
    logprobs = torch.zeros((num_steps, num_envs), device=device)
    rewards = torch.zeros((num_steps, num_envs), device=device)
    dones = torch.zeros((num_steps, num_envs), device=device)
    values = torch.zeros((num_steps, num_envs), device=device)

    next_obs = obs
    next_done = torch.zeros(num_envs, device=device)

    client_agent.train()
    for _ in range(local_iterations):
        for step in range(num_steps):
            obs_rgb[step] = next_obs["rgb"]
            obs_depth[step] = next_obs["depth"]
            obs_state[step] = next_obs["state"]
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = client_agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            next_obs, reward, terminations, truncations, _ = envs.step(action)
            next_done = torch.logical_or(terminations, truncations).to(torch.float32)
            rewards[step] = reward.view(-1)

        # GAE
        with torch.no_grad():
            next_value = client_agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(num_steps)):
                nextnonterminal = 1.0 - next_done if t == num_steps - 1 else 1.0 - dones[t + 1]
                nextvalues = next_value if t == num_steps - 1 else values[t + 1]
                delta = rewards[t] + 0.8 * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + 0.8 * 0.9 * nextnonterminal * lastgaelam
            returns = advantages + values

        b_obs = {
            "rgb": obs_rgb.reshape((-1, 128, 128, 3)),
            "depth": obs_depth.reshape((-1, 128, 128, 1)),
            "state": obs_state.reshape((-1, 29)),
        }
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_advantages = (advantages.reshape(-1) - advantages.mean()) / (advantages.std() + 1e-8)
        b_returns = returns.reshape(-1)

        # Update
        b_inds = np.arange(batch_size)
        for epoch in range(4):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                mb_obs = {k: v[mb_inds] for k, v in b_obs.items()}
                _, newlogprob, entropy, newvalue = client_agent.get_action_and_value(mb_obs, b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                pg_loss1 = -b_advantages[mb_inds] * ratio
                pg_loss2 = -b_advantages[mb_inds] * torch.clamp(ratio, 0.8, 1.2)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                v_loss = 0.5 * ((newvalue.view(-1) - b_returns[mb_inds]) ** 2).mean()
                loss = pg_loss + 0.5 * v_loss

                # FedProx Proximal Regularization Term: (mu / 2) * || w_k - w_global ||^2
                if mu > 0.0:
                    prox_loss = 0.0
                    for name, param in client_agent.named_parameters():
                        if name in global_weights:
                            prox_loss += ((param - global_weights[name]) ** 2).sum()
                    loss += (mu / 2.0) * prox_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(client_agent.parameters(), 0.5)
                optimizer.step()

    envs.close()
    return client_agent.state_dict()


def run_federated_pipeline(
    method: str = "fedprox",  # "fedavg" or "fedprox"
    rounds: int = 15,
    local_iterations: int = 5,
    mu: float = 0.01,
    num_envs: int = 64,
    output_dir: str = SCRIPT_DIR,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if method.lower() == "fedavg":
        mu = 0.0

    exp_dir = os.path.join(output_dir, f"results_{method.lower()}")
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    metrics_json_path = os.path.join(exp_dir, f"{method.lower()}_metrics.json")
    history_csv_path = os.path.join(exp_dir, f"{method.lower()}_rounds_history.csv")
    best_model_path = os.path.join(ckpt_dir, f"{method.lower()}_global_best.pt")

    print(f"\n{'='*78}")
    print(f" STARTING FEDERATED EXPERIMENT: {method.upper()}")
    print(f" Proximal Weight (mu): {mu} | Communication Rounds: {rounds}")
    print(f" Local Iterations per Client: {local_iterations} (~{local_iterations * num_envs * 50:,} steps/round)")
    print(f" Participating Viewpoints: {CORE_VP_IDS}")
    print(f"{'='*78}\n")

    # Initialize Global Agent
    env_kwargs = dict(obs_mode="rgbd", render_mode="all", sim_backend="physx_cuda", control_mode="pd_ee_delta_pos")
    dummy_env = gym.make("PickCube-v1", num_envs=1, **env_kwargs)
    dummy_env = FlattenRGBDObservationWrapper(dummy_env, rgb=True, depth=True, state=True, sep_depth=True)
    if isinstance(dummy_env.action_space, gym.spaces.Dict):
        dummy_env = FlattenActionSpaceWrapper(dummy_env)
    sample_obs = {
        "rgb": torch.zeros((1, 128, 128, 3), dtype=torch.uint8, device=device),
        "depth": torch.zeros((1, 128, 128, 1), dtype=torch.int16, device=device),
        "state": torch.zeros((1, 29), dtype=torch.float32, device=device),
    }
    global_agent = UnifiedAblationAgent(dummy_env, sample_obs, candidate="A").to(device)
    dummy_env.close()

    global_weights = copy.deepcopy(global_agent.state_dict())
    best_mean_succ = 0.0
    history = []
    t0 = time.time()

    for r in range(1, rounds + 1):
        round_t0 = time.time()
        print(f"\n--- [Round {r:02d}/{rounds}] Distributing Global Model to 8 Viewpoint Clients ---")
        client_weights = []

        for idx, vp in enumerate(CORE_VP_IDS, 1):
            w_k = local_client_update(
                global_state_dict=global_weights,
                vp_id=vp,
                local_iterations=local_iterations,
                num_envs=num_envs,
                mu=mu,
                device=device,
            )
            client_weights.append(w_k)
            print(f"  [{idx}/8] Local update complete for VP {vp:02d}.")

        # Server Aggregation: W_global = (1 / K) * sum(W_k)
        print(f"[Round {r:02d}] Aggregating parameters across {len(client_weights)} clients...")
        aggregated = {}
        for key in global_weights.keys():
            aggregated[key] = torch.stack([w[key].float() for w in client_weights], dim=0).mean(dim=0)
            if global_weights[key].dtype == torch.int64 or global_weights[key].dtype == torch.bool:
                aggregated[key] = aggregated[key].to(global_weights[key].dtype)
            else:
                aggregated[key] = aggregated[key].to(global_weights[key].dtype)
        global_weights = aggregated
        global_agent.load_state_dict(global_weights)

        # Global Evaluation across all 8 viewpoints
        print(f"[Round {r:02d}] Evaluating aggregated global model on all 8 viewpoints...")
        vp_res, mean_s, min_s, std_s = evaluate_all_viewpoints(global_agent, CORE_VP_IDS, num_envs=50, device=device)
        round_time = time.time() - round_t0
        cum_time = time.time() - t0

        print(f"  >>> [Round {r:02d} Results] Mean: {mean_s:5.1f}% | Worst: {min_s:5.1f}% | Std: {std_s:4.1f}% | Time: {round_time:.1f}s")
        vp_str = " | ".join([f"VP{k:02d}:{s:4.0f}%" for k, s in vp_res.items()])
        print(f"      Viewpoints: {vp_str}")

        record = {
            "round": r,
            "method": method.upper(),
            "mean_success": round(mean_s, 2),
            "min_success": round(min_s, 2),
            "std_success": round(std_s, 2),
            "round_seconds": round(round_time, 1),
            "cumulative_seconds": round(cum_time, 1),
            **{f"vp_{k:02d}_succ": round(v, 2) for k, v in vp_res.items()},
        }
        history.append(record)

        if mean_s >= best_mean_succ:
            best_mean_succ = mean_s
            torch.save({
                "round": r,
                "method": method.upper(),
                "mean_success": mean_s,
                "min_success": min_s,
                "per_viewpoint": vp_res,
                "model_state": global_weights,
            }, best_model_path)
            print(f"  >>> [NEW BEST GLOBAL MODEL] Mean Success: {best_mean_succ:.1f}% -> {best_model_path}")

    # Save summary
    import csv
    if history:
        with open(history_csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            w.writeheader()
            w.writerows(history)

    final_metrics = {
        "method": method.upper(),
        "mu": mu,
        "rounds": rounds,
        "local_iterations": local_iterations,
        "best_mean_success": best_mean_succ,
        "best_min_success": history[-1]["min_success"] if history else 0.0,
        "total_walltime_seconds": round(time.time() - t0, 1),
        "best_model_path": best_model_path,
    }
    with open(metrics_json_path, "w") as f:
        json.dump(final_metrics, f, indent=2)

    print(f"\n[COMPLETED] Federated {method.upper()} Experiment Finished!")
    print(f" Best Mean Success: {best_mean_succ:.1f}% | Walltime: {final_metrics['total_walltime_seconds'] / 60:.1f} mins\n")
    return final_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="fedprox", choices=["fedavg", "fedprox"])
    parser.add_argument("--rounds", type=int, default=15)
    parser.add_argument("--local-iterations", type=int, default=5)
    parser.add_argument("--mu", type=float, default=0.01)
    parser.add_argument("--output-dir", type=str, default=SCRIPT_DIR)
    args = parser.parse_args()

    run_federated_pipeline(method=args.method, rounds=args.rounds, local_iterations=args.local_iterations, mu=args.mu, output_dir=args.output_dir)
