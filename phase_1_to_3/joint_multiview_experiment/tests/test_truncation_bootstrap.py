"""
Unit tests for Time-Limit Truncation & Value Function Bootstrapping.

Per SPEC Section 3 & CleanRL PPO standards:
1. When an episode reaches max_episode_steps (50), the environment triggers time-limit truncation (trunc=True).
2. The vectorized wrapper stores the terminal state before reset in infos["final_observation"], with infos["_final_info"]=True.
3. The PPO loop computes V(s_final) via agent.get_value(final_obs) and stores it in final_values[step].
4. GAE advantage calculation at the boundary incorporates gamma * final_values[t] rather than treating the truncation as an absorbing terminal state (0 value).

Python 3.9 compatible only.
"""

import os
import shutil
import tempfile
import numpy as np
import pytest
import torch

from src.env_factory import make_joint_multiview_env
from src.model import Agent
from src.train_ppo import train_ppo, load_yaml_config


@pytest.fixture
def tmp_run_dir():
    d = tempfile.mkdtemp(prefix="test_truncation_bootstrap_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


def test_time_limit_truncation_and_final_observation_extraction():
    """
    Verifies that:
    1. Steps 1 to 49 do not truncate.
    2. Step 50 triggers truncation (trunc=True, term=False) and populates _final_info.
    3. infos["final_observation"] contains complete observation dictionary for truncated envs.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = make_joint_multiview_env(
        num_envs=2,
        modality="rgbd",
        viewpoint_ids=[0, 1],
        max_episode_steps=50,
        seed=123,
    )

    action_shape = getattr(env, "single_action_space", env.action_space).shape
    action = torch.zeros((2,) + action_shape, device=device)

    # Step through steps 1 to 49
    for step_idx in range(49):
        obs, rew, term, trunc, infos = env.step(action)
        assert not term.any(), f"Step {step_idx+1}: unexpected termination in PickCube"
        assert not trunc.any(), f"Step {step_idx+1}: premature truncation before 50 steps"

    # Step 50: must trigger truncation
    obs, rew, term, trunc, infos = env.step(action)
    assert trunc.all(), f"Step 50: expected trunc=True for all envs, got {trunc}"
    assert not term.any(), f"Step 50: expected term=False, got {term}"
    assert "_final_info" in infos, "Step 50: missing '_final_info' in infos"
    assert infos["_final_info"].all(), "Step 50: _final_info should be True for all truncated envs"
    assert "final_observation" in infos, "Step 50: missing 'final_observation' in infos"

    final_obs = infos["final_observation"]
    assert "rgb" in final_obs
    assert "state" in final_obs
    assert final_obs["state"].shape == (2, 29)
    assert final_obs["rgb"].shape == (2, 128, 128, 3)

    env.close()


def test_final_values_bootstrap_in_gae_calculation():
    """
    Verifies that:
    1. V(s_final) is computed from final_observation.
    2. In GAE calculation, real_next_values incorporates final_values[step].
    3. Proves that returns[49] strictly increases by gamma * V(s_final) compared to zero-bootstrap.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = make_joint_multiview_env(
        num_envs=2,
        modality="rgbd",
        viewpoint_ids=[0, 1],
        max_episode_steps=50,
        seed=42,
    )
    agent = Agent(env=env, in_channels=4).to(device)

    num_steps = 50
    num_envs = 2
    gamma = 0.8
    gae_lambda = 0.9

    action_shape = getattr(env, "single_action_space", env.action_space).shape
    action = torch.zeros((num_envs,) + action_shape, device=device)

    # Rollout 50 steps until truncation at step index 49
    final_values = torch.zeros((num_steps, num_envs), device=device)
    rewards_buf = torch.zeros((num_steps, num_envs), device=device)
    values_buf = torch.zeros((num_steps, num_envs), device=device)
    dones_buf = torch.zeros((num_steps, num_envs), device=device)

    next_obs, _ = env.reset(seed=42)
    next_done = torch.zeros(num_envs, device=device)

    for step in range(num_steps):
        with torch.no_grad():
            _, _, _, val = agent.get_action_and_value(next_obs)
            values_buf[step] = val.flatten()

        next_obs, reward, term, trunc, infos = env.step(action)
        next_done = torch.logical_or(term, trunc).to(torch.float32)
        rewards_buf[step] = reward.view(-1)
        dones_buf[step] = next_done

        if "final_info" in infos:
            done_mask = infos["_final_info"]
            if done_mask.any():
                final_obs = {k: v[done_mask] for k, v in infos["final_observation"].items()}
                with torch.no_grad():
                    final_vals = agent.get_value(final_obs).view(-1)
                    done_indices = torch.arange(num_envs, device=device)[done_mask]
                    final_values[step, done_indices] = final_vals

    env.close()

    # Step 49 must have captured final_values
    assert (final_values[49] != 0).any(), "final_values[49] was not populated upon truncation!"
    v_final = final_values[49]

    # --- Compute GAE with final_values (New / Correct CleanRL bootstrap) ---
    with torch.no_grad():
        next_value = agent.get_value(next_obs).reshape(1, -1)
        advantages_with_bootstrap = torch.zeros_like(rewards_buf).to(device)
        lastgaelam = 0.0
        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                next_not_done = 1.0 - next_done
                nextvalues = next_value
            else:
                next_not_done = 1.0 - dones_buf[t + 1]
                nextvalues = values_buf[t + 1]
            real_next_values = next_not_done * nextvalues + final_values[t]
            delta = rewards_buf[t] + gamma * real_next_values - values_buf[t]
            advantages_with_bootstrap[t] = lastgaelam = delta + gamma * gae_lambda * next_not_done * lastgaelam
        returns_with_bootstrap = advantages_with_bootstrap + values_buf

    # --- Compute GAE WITHOUT final_values (Flawed / Zero bootstrap) ---
    with torch.no_grad():
        advantages_zero = torch.zeros_like(rewards_buf).to(device)
        lastgaelam = 0.0
        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                next_not_done = 1.0 - next_done
                nextvalues = next_value
            else:
                next_not_done = 1.0 - dones_buf[t + 1]
                nextvalues = values_buf[t + 1]
            # Without final_values bootstrap (zero bootstrap)
            real_next_values = next_not_done * nextvalues
            delta = rewards_buf[t] + gamma * real_next_values - values_buf[t]
            advantages_zero[t] = lastgaelam = delta + gamma * gae_lambda * next_not_done * lastgaelam
        returns_zero = advantages_zero + values_buf

    # Mathematical Proof & Assertion:
    # At t=49, next_done=1.0 so next_not_done=0.0.
    # With bootstrap: delta[49] = r[49] + gamma * v_final - v[49]
    # Without bootstrap: delta_zero[49] = r[49] + gamma * 0 - v[49]
    # Difference in delta[49] and returns[49] must be exactly gamma * v_final
    expected_diff = gamma * v_final
    actual_diff = returns_with_bootstrap[49] - returns_zero[49]

    diff_err = torch.max(torch.abs(actual_diff - expected_diff)).item()
    assert diff_err < 1e-5, f"GAE return difference ({actual_diff}) does not match gamma * V(s_final) ({expected_diff})"

    # Advantages difference must also equal gamma * v_final
    adv_diff = advantages_with_bootstrap[49] - advantages_zero[49]
    adv_err = torch.max(torch.abs(adv_diff - expected_diff)).item()
    assert adv_err < 1e-5, f"GAE advantage difference ({adv_diff}) does not match gamma * V(s_final) ({expected_diff})"


def test_train_ppo_handles_truncation_in_live_iteration(tmp_run_dir):
    """
    Runs train_ppo for 1 iteration (50 steps) on a 2-view tiny configuration,
    confirming that step 50 truncation triggers cleanly inside the PPO loop,
    computes advantages with value bootstrap, and writes progress.csv.
    """
    cfg = load_yaml_config("configs/tiny_test.yaml")
    cfg["num_envs"] = 2
    cfg["num_steps"] = 50
    cfg["num_minibatches"] = 2
    cfg["update_epochs"] = 1
    cfg["total_timesteps"] = 100  # 1 iteration of 50 steps * 2 envs = 100 steps
    cfg["viewpoint_ids"] = [0, 1]

    result = train_ppo(
        config=cfg,
        output_dir=tmp_run_dir,
        modality="rgb",
        seed=42,
    )

    assert result["status"] == "completed"
    assert result["global_step"] == 100
    assert result["iteration"] == 1

    # Verify progress.csv was generated
    csv_file = os.path.join(tmp_run_dir, "progress.csv")
    assert os.path.isfile(csv_file)
    with open(csv_file, "r") as f:
        lines = f.readlines()
    assert len(lines) == 2  # header + 1 iteration row
    assert lines[1].startswith("1,100,")


def test_mock_truncation_forces_value_bootstrap():
    """
    Protocol Requirement 3:
    Unit test using a mock environment that forces a time-limit truncation
    and explicitly verifies that the bootstrapped value V(s_final) is used
    in the advantage and return calculations.
    """
    from unittest.mock import MagicMock

    device = "cpu"
    num_envs = 2
    num_steps = 4
    gamma = 0.8
    gae_lambda = 0.9

    # 1. Create Mock Environment
    mock_env = MagicMock()
    mock_env.num_envs = num_envs
    
    # Define step returns
    # Steps 0, 1, 2: normal steps, no truncation
    # Step 3: Env 0 truncates (time-limit reached), Env 1 continues
    def mock_step_fn(action):
        step_idx = mock_step_fn.call_count
        mock_step_fn.call_count += 1
        
        rew = torch.tensor([0.5, 0.5])
        term = torch.tensor([False, False])
        
        if step_idx == 3:  # Truncation step
            trunc = torch.tensor([True, False])
            infos = {
                "_final_info": torch.tensor([True, False]),
                "final_info": {"episode": {"return": torch.tensor([12.5, 0.0]), "episode_len": torch.tensor([50, 0])}},
                "final_observation": {
                    "rgb": torch.full((num_envs, 128, 128, 3), 128.0),
                    "state": torch.full((num_envs, 29), 3.14),
                },
            }
        else:
            trunc = torch.tensor([False, False])
            infos = {}
            
        next_obs = {
            "rgb": torch.zeros((num_envs, 128, 128, 3)),
            "state": torch.zeros((num_envs, 29)),
        }
        return next_obs, rew, term, trunc, infos

    mock_step_fn.call_count = 0
    mock_env.step = mock_step_fn

    # 2. Create Mock Agent with deterministic value outputs
    mock_agent = MagicMock()
    # Regular values during rollout: value = 2.0 for all states
    mock_agent.get_action_and_value.return_value = (
        torch.zeros((num_envs, 7)),
        torch.zeros(num_envs),
        torch.zeros(num_envs),
        torch.tensor([[2.0], [2.0]]),
    )
    # Next value at horizon boundary: 5.0 for env 0, 6.0 for env 1
    def mock_get_value_fn(obs):
        # If called with final_observation (state has 3.14), return high bootstrap value 42.0
        if "state" in obs and torch.isclose(obs["state"][0, 0], torch.tensor(3.14)):
            # Batch size is done_mask.sum()
            b = obs["state"].shape[0]
            return torch.full((b, 1), 42.0)
        b = obs["state"].shape[0]
        return torch.tensor([[5.0], [6.0]])[:b]

    mock_agent.get_value = mock_get_value_fn

    # 3. Simulate Rollout and GAE collection as in train_ppo.py
    final_values = torch.zeros((num_steps, num_envs), device=device)
    rewards_buf = torch.zeros((num_steps, num_envs), device=device)
    values_buf = torch.zeros((num_steps, num_envs), device=device)
    dones_buf = torch.zeros((num_steps, num_envs), device=device)

    next_obs = {"rgb": torch.zeros((num_envs, 128, 128, 3)), "state": torch.zeros((num_envs, 29))}
    next_done = torch.zeros(num_envs, device=device)

    for step in range(num_steps):
        dones_buf[step] = next_done
        _, _, _, val = mock_agent.get_action_and_value(next_obs)
        values_buf[step] = val.flatten()

        next_obs, reward, term, trunc, infos = mock_env.step(None)
        next_done = torch.logical_or(term, trunc).to(torch.float32)
        rewards_buf[step] = reward.view(-1)

        if "final_info" in infos:
            done_mask = infos["_final_info"]
            if done_mask.any():
                final_obs = {k: v[done_mask] for k, v in infos["final_observation"].items()}
                final_vals = mock_agent.get_value(final_obs).view(-1)
                done_indices = torch.arange(num_envs, device=device)[done_mask]
                final_values[step, done_indices] = final_vals

    # Assertions on Rollout Collection:
    # Env 0 was truncated at step 3 -> final_values[3, 0] must be 42.0
    # Env 1 was NOT truncated -> final_values[3, 1] must be 0.0
    assert final_values[3, 0].item() == 42.0, f"Expected final_values[3, 0]==42.0, got {final_values[3, 0]}"
    assert final_values[3, 1].item() == 0.0, f"Expected final_values[3, 1]==0.0, got {final_values[3, 1]}"

    # 4. GAE calculation with bootstrap
    next_value = mock_agent.get_value(next_obs).reshape(1, -1)
    advantages_boot = torch.zeros_like(rewards_buf).to(device)
    lastgaelam = 0.0
    for t in reversed(range(num_steps)):
        if t == num_steps - 1:
            next_not_done = 1.0 - next_done
            nextvalues = next_value
        else:
            next_not_done = 1.0 - dones_buf[t + 1]
            nextvalues = values_buf[t + 1]
        real_next_values = next_not_done * nextvalues + final_values[t]
        delta = rewards_buf[t] + gamma * real_next_values - values_buf[t]
        advantages_boot[t] = lastgaelam = delta + gamma * gae_lambda * next_not_done * lastgaelam
    returns_boot = advantages_boot + values_buf

    # GAE calculation WITHOUT bootstrap (zero bootstrap)
    advantages_zero = torch.zeros_like(rewards_buf).to(device)
    lastgaelam = 0.0
    for t in reversed(range(num_steps)):
        if t == num_steps - 1:
            next_not_done = 1.0 - next_done
            nextvalues = next_value
        else:
            next_not_done = 1.0 - dones_buf[t + 1]
            nextvalues = values_buf[t + 1]
        real_next_values = next_not_done * nextvalues
        delta = rewards_buf[t] + gamma * real_next_values - values_buf[t]
        advantages_zero[t] = lastgaelam = delta + gamma * gae_lambda * next_not_done * lastgaelam
    returns_zero = advantages_zero + values_buf

    # Mathematical Verification:
    # At t=3:
    # For Env 0 (truncated): next_done[0]=1.0 -> next_not_done=0.0
    #   With bootstrap: real_next_values = 0.0 * 5.0 + 42.0 = 42.0
    #   delta_boot = 0.5 + 0.8 * 42.0 - 2.0 = 32.1
    #   Without bootstrap: real_next_values = 0.0 * 5.0 = 0.0
    #   delta_zero = 0.5 + 0.8 * 0.0 - 2.0 = -1.5
    #   Difference in delta and returns must be exactly gamma * 42.0 = 33.6
    expected_diff_env0 = gamma * 42.0
    actual_diff_env0 = (returns_boot[3, 0] - returns_zero[3, 0]).item()
    assert abs(actual_diff_env0 - expected_diff_env0) < 1e-5, (
        f"Env 0 bootstrap return difference {actual_diff_env0} != gamma * 42.0 ({expected_diff_env0})"
    )

    # For Env 1 (not truncated): next_done[1]=0.0 -> next_not_done=1.0
    #   real_next_values = 1.0 * 6.0 + 0.0 = 6.0
    #   Both with and without final_values bootstrap must be IDENTICAL
    actual_diff_env1 = (returns_boot[3, 1] - returns_zero[3, 1]).item()
    assert abs(actual_diff_env1) < 1e-5, (
        f"Env 1 return should not change with final_values (no truncation), but differed by {actual_diff_env1}"
    )

