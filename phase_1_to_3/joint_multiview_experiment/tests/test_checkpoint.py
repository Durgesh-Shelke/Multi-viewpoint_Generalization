"""
Unit tests for src/checkpoint.py per SPEC Section 2.2 and Section 2.3.

Covers:
1. Atomic checkpoint save, companion .sha256 file, and cleanup of temporary files.
2. Free disk space checks (enforcing LowDiskSpaceError when below safety margin).
3. Rolling checkpoint retention (keeping exactly last 2 rolling checkpoints).
4. Milestone checkpoints (every 5M steps) and final checkpoints are never pruned.
5. Truncated checkpoint fallback: when the newest checkpoint is truncated, falls back to previous.
6. Checksum mismatch fallback: when checkpoint bytes are tampered with, falls back to previous.
7. Config-hash guard: refuses resume if configuration has changed (ConfigMismatchError).
8. RNG state capture and bitwise reproduction across Python, NumPy, and PyTorch.
9. Empty directory handling and error handling when all candidates are corrupt.

Python 3.9 compatible only.
"""

import os
import random
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

# Ensure experiment root is on sys.path
EXP_ROOT = str(Path(__file__).resolve().parent.parent)
if EXP_ROOT not in sys.path:
    sys.path.insert(0, EXP_ROOT)

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from src.checkpoint import (
    ConfigMismatchError,
    CorruptCheckpointError,
    LowDiskSpaceError,
    capture_rng_states,
    check_disk_space,
    cleanup_orphan_tmp_files,
    compute_config_hash,
    compute_file_sha256,
    filter_config_for_hashing,
    get_checkpoint_candidates,
    load_checkpoint_with_fallback,
    restore_rng_states,
    save_checkpoint,
    validate_checkpoint,
    verify_config_hash,
)


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 4)

    def forward(self, x):
        return self.fc(x)


@pytest.fixture
def tmp_ckpt_dir():
    """Provides a clean temporary directory for checkpoints."""
    d = tempfile.mkdtemp(prefix="test_ckpt_")
    yield d
    if os.path.exists(d):
        shutil.rmtree(d)


@pytest.fixture
def sample_components():
    """Provides a dummy model and optimizer for checkpointing."""
    model = SimpleModel()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    config = {
        "learning_rate": 0.0002,
        "gamma": 0.8,
        "gae_lambda": 0.9,
        "num_envs": 160,
        "num_steps": 50,
    }
    return model, optimizer, config


def test_atomic_save_and_sha256(tmp_ckpt_dir, sample_components):
    """Verifies atomic save, companion .sha256 file, and latest.pt pointer."""
    model, optimizer, config = sample_components
    saved_path = save_checkpoint(
        ckpt_dir=tmp_ckpt_dir,
        model=model,
        optimizer=optimizer,
        global_step=1000,
        iteration=20,
        config=config,
        active_time_s=12.5,
    )

    assert os.path.exists(saved_path), "Checkpoint file was not created"
    assert os.path.basename(saved_path) == "ckpt_step_000001000.pt"

    # Verify sha256 companion file
    sha_file = f"{saved_path}.sha256"
    assert os.path.exists(sha_file), ".sha256 file was not created"
    with open(sha_file) as f:
        expected_sha = f.read().strip()
    assert expected_sha == compute_file_sha256(saved_path)

    # Verify latest.pt
    latest_pt = os.path.join(tmp_ckpt_dir, "latest.pt")
    assert os.path.exists(latest_pt), "latest.pt was not created"

    # Verify no temporary files left
    all_files = os.listdir(tmp_ckpt_dir)
    tmp_files = [f for f in all_files if ".tmp" in f]
    assert len(tmp_files) == 0, f"Temporary files were not cleaned up: {tmp_files}"

    # Validate checkpoint content
    is_valid, loaded = validate_checkpoint(saved_path)
    assert is_valid
    assert loaded["global_step"] == 1000
    assert loaded["iteration"] == 20
    assert loaded["active_time_s"] == 12.5
    assert loaded["config_hash"] == compute_config_hash(config)


def test_free_disk_space_check(tmp_ckpt_dir, sample_components):
    """Verifies that free disk space checks pass normally and reject when insufficient."""
    model, optimizer, config = sample_components

    # Normal free space check should succeed
    free_bytes = check_disk_space(tmp_ckpt_dir, min_free_bytes=1024 * 1024)
    assert free_bytes > 0

    # Impossibly high requirement should raise LowDiskSpaceError
    huge_requirement = 100 * 1024 * 1024 * 1024 * 1024  # 100 TB
    with pytest.raises(LowDiskSpaceError) as exc_info:
        check_disk_space(tmp_ckpt_dir, min_free_bytes=huge_requirement)
    assert "Low disk space" in str(exc_info.value)

    # save_checkpoint should also raise LowDiskSpaceError and write nothing
    with pytest.raises(LowDiskSpaceError):
        save_checkpoint(
            ckpt_dir=tmp_ckpt_dir,
            model=model,
            optimizer=optimizer,
            global_step=1000,
            config=config,
            min_free_bytes=huge_requirement,
        )

    assert len(os.listdir(tmp_ckpt_dir)) == 0


def test_rolling_retention_keeps_last_2(tmp_ckpt_dir, sample_components):
    """Verifies rolling policy keeps only the last 2 rolling checkpoints."""
    model, optimizer, config = sample_components

    # Save 5 consecutive rolling checkpoints
    for step in [1000, 2000, 3000, 4000, 5000]:
        save_checkpoint(
            ckpt_dir=tmp_ckpt_dir,
            model=model,
            optimizer=optimizer,
            global_step=step,
            iteration=step // 50,
            config=config,
            rolling_keep=2,
        )

    # Check which rolling checkpoint files remain
    rolling_files = sorted(
        [f for f in os.listdir(tmp_ckpt_dir) if f.startswith("ckpt_step_") and f.endswith(".pt")]
    )
    assert rolling_files == [
        "ckpt_step_000004000.pt",
        "ckpt_step_000005000.pt",
    ], f"Rolling retention failed, remaining files: {rolling_files}"

    # Verify sha256 files for pruned checkpoints were also cleaned up
    assert not os.path.exists(os.path.join(tmp_ckpt_dir, "ckpt_step_000001000.pt.sha256"))
    assert not os.path.exists(os.path.join(tmp_ckpt_dir, "ckpt_step_000002000.pt.sha256"))
    assert not os.path.exists(os.path.join(tmp_ckpt_dir, "ckpt_step_000003000.pt.sha256"))
    assert os.path.exists(os.path.join(tmp_ckpt_dir, "ckpt_step_000004000.pt.sha256"))
    assert os.path.exists(os.path.join(tmp_ckpt_dir, "ckpt_step_000005000.pt.sha256"))


def test_milestones_and_final_are_never_pruned(tmp_ckpt_dir, sample_components):
    """Verifies milestone checkpoints (every 5M steps) and final checkpoints are never pruned."""
    model, optimizer, config = sample_components

    # Save rolling step 1000
    save_checkpoint(tmp_ckpt_dir, model, optimizer, global_step=1000, config=config, rolling_keep=2)

    # Save milestone at step 5,000,000
    save_checkpoint(
        tmp_ckpt_dir,
        model,
        optimizer,
        global_step=5_000_000,
        config=config,
        milestone_interval=5_000_000,
        rolling_keep=2,
    )

    # Save rolling step 5,001,000
    save_checkpoint(tmp_ckpt_dir, model, optimizer, global_step=5_001_000, config=config, rolling_keep=2)

    # Save final checkpoint at step 5,002,000
    save_checkpoint(
        tmp_ckpt_dir,
        model,
        optimizer,
        global_step=5_002_000,
        config=config,
        is_final=True,
        rolling_keep=2,
    )

    # Save rolling step 5,003,000
    save_checkpoint(tmp_ckpt_dir, model, optimizer, global_step=5_003_000, config=config, rolling_keep=2)

    # Save rolling step 5,004,000
    save_checkpoint(tmp_ckpt_dir, model, optimizer, global_step=5_004_000, config=config, rolling_keep=2)

    all_files = os.listdir(tmp_ckpt_dir)

    # Milestone MUST exist
    assert "milestone_step_005000000.pt" in all_files
    # Final checkpoint MUST exist
    assert "ckpt_final_step_005002000.pt" in all_files

    # Rolling checkpoints must be exactly the last 2 (5,003,000 and 5,004,000)
    rolling = sorted([f for f in all_files if f.startswith("ckpt_step_") and f.endswith(".pt")])
    assert rolling == [
        "ckpt_step_005003000.pt",
        "ckpt_step_005004000.pt",
    ]


def test_truncated_checkpoint_fallback(tmp_ckpt_dir, sample_components):
    """
    CRITICAL SPEC TEST:
    When the newest checkpoint is truncated/corrupt, load_checkpoint_with_fallback
    must log a warning, fall back to the previous valid checkpoint, and succeed.
    """
    model, optimizer, config = sample_components

    # 1. Save checkpoint 1 at step 1000
    ckpt1_path = save_checkpoint(
        tmp_ckpt_dir, model, optimizer, global_step=1000, iteration=10, config=config
    )

    # 2. Save checkpoint 2 at step 2000
    ckpt2_path = save_checkpoint(
        tmp_ckpt_dir, model, optimizer, global_step=2000, iteration=20, config=config
    )

    # Check candidates order
    candidates = get_checkpoint_candidates(tmp_ckpt_dir)
    assert os.path.realpath(candidates[0]) == os.path.realpath(ckpt2_path)

    # 3. Truncate checkpoint 2 to 100 bytes (corrupting it)
    with open(ckpt2_path, "wb") as f:
        f.write(b"CORRUPTED_TRUNCATED_CHECKPOINT_DATA_HEADER" * 2)

    # Verify validate_checkpoint reports corrupt
    is_valid, err = validate_checkpoint(ckpt2_path)
    assert not is_valid
    assert isinstance(err, (CorruptCheckpointError, Exception))

    # 4. load_checkpoint_with_fallback should catch this and recover checkpoint 1
    loaded_data, loaded_path = load_checkpoint_with_fallback(tmp_ckpt_dir, current_config=config)

    assert loaded_data is not None
    assert os.path.realpath(loaded_path) == os.path.realpath(ckpt1_path)
    assert loaded_data["global_step"] == 1000
    assert loaded_data["iteration"] == 10


def test_checksum_mismatch_fallback(tmp_ckpt_dir, sample_components):
    """
    CRITICAL SPEC TEST:
    If a checkpoint file's content is modified (tampered byte), checksum validation
    detects corruption and falls back to previous valid checkpoint.
    """
    model, optimizer, config = sample_components

    ckpt1_path = save_checkpoint(tmp_ckpt_dir, model, optimizer, global_step=1000, config=config)
    ckpt2_path = save_checkpoint(tmp_ckpt_dir, model, optimizer, global_step=2000, config=config)

    # Tamper with checkpoint 2 without truncating
    with open(ckpt2_path, "r+b") as f:
        f.seek(15)
        byte = f.read(1)
        f.seek(15)
        f.write(bytes([byte[0] ^ 0xFF]))

    # Fallback must skip corrupted ckpt2 and load ckpt1
    loaded_data, loaded_path = load_checkpoint_with_fallback(tmp_ckpt_dir, current_config=config)
    assert loaded_data is not None
    assert os.path.realpath(loaded_path) == os.path.realpath(ckpt1_path)
    assert loaded_data["global_step"] == 1000


def test_config_hash_mismatch_refusal(tmp_ckpt_dir, sample_components):
    """
    CRITICAL SPEC TEST:
    Resuming with a modified config must be strictly refused with ConfigMismatchError.
    """
    model, optimizer, config = sample_components

    # Save checkpoint with original config
    save_checkpoint(tmp_ckpt_dir, model, optimizer, global_step=1000, config=config)

    # Loading with exact same config must succeed
    loaded_data, _ = load_checkpoint_with_fallback(tmp_ckpt_dir, current_config=config)
    assert loaded_data is not None

    # Modify config (e.g. changed learning rate)
    modified_config = dict(config)
    modified_config["learning_rate"] = 0.0005

    # Loading with modified config MUST raise ConfigMismatchError
    with pytest.raises(ConfigMismatchError) as exc_info:
        load_checkpoint_with_fallback(tmp_ckpt_dir, current_config=modified_config)

    assert "Config hash mismatch" in str(exc_info.value)
    assert "Resuming with modified config is refused" in str(exc_info.value)


def test_rng_capture_and_restore():
    """Verifies RNG state capture and bitwise reproduction across Python, NumPy, and PyTorch."""
    # Seed all generators
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # Advance generators a bit
    _ = [random.random() for _ in range(10)]
    _ = np.random.randn(10)
    _ = torch.randn(10)

    # Capture state
    saved_rng = capture_rng_states()

    # Generate sequence A
    py_seq_a = [random.random() for _ in range(5)]
    np_seq_a = np.random.randn(5).tolist()
    torch_seq_a = torch.randn(5).tolist()

    # Alter generators with random noise
    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)
    _ = [random.random() for _ in range(50)]

    # Restore saved RNG states
    restore_rng_states(saved_rng)

    # Generate sequence B
    py_seq_b = [random.random() for _ in range(5)]
    np_seq_b = np.random.randn(5).tolist()
    torch_seq_b = torch.randn(5).tolist()

    # Verify exact equality
    assert py_seq_a == py_seq_b, "Python random state was not restored"
    assert np_seq_a == np_seq_b, "NumPy random state was not restored"
    assert torch_seq_a == torch_seq_b, "PyTorch random state was not restored"


def test_all_corrupt_raises_corrupt_checkpoint_error(tmp_ckpt_dir, sample_components):
    """When all candidate checkpoints are corrupt, load_checkpoint_with_fallback raises error."""
    model, optimizer, config = sample_components
    p1 = save_checkpoint(tmp_ckpt_dir, model, optimizer, global_step=1000, config=config)
    p2 = save_checkpoint(tmp_ckpt_dir, model, optimizer, global_step=2000, config=config)

    # Corrupt both
    with open(p1, "wb") as f:
        f.write(b"bad1")
    with open(p2, "wb") as f:
        f.write(b"bad2")

    with pytest.raises(CorruptCheckpointError) as exc_info:
        load_checkpoint_with_fallback(tmp_ckpt_dir)

    assert "failed validation" in str(exc_info.value)


def test_truncated_latest_symlink_fallback(tmp_ckpt_dir, sample_components):
    """
    Verifies that when latest.pt (symlink or file) is truncated directly,
    load_checkpoint_with_fallback falls back to the previous checkpoint.
    """
    model, optimizer, config = sample_components
    ckpt1_path = save_checkpoint(tmp_ckpt_dir, model, optimizer, global_step=1000, config=config)
    _ = save_checkpoint(tmp_ckpt_dir, model, optimizer, global_step=2000, config=config)

    latest_pt = os.path.join(tmp_ckpt_dir, "latest.pt")
    assert os.path.exists(latest_pt)

    # Truncate latest.pt
    with open(latest_pt, "wb") as f:
        f.write(b"CORRUPT")

    # Load from directory: should fall back to ckpt1
    data, loaded_path = load_checkpoint_with_fallback(tmp_ckpt_dir, current_config=config)
    assert data is not None
    assert os.path.realpath(loaded_path) == os.path.realpath(ckpt1_path)
    assert data["global_step"] == 1000


def test_load_corrupt_file_path_falls_back_to_directory(tmp_ckpt_dir, sample_components):
    """
    When a specific corrupt file path is passed to load_checkpoint_with_fallback,
    it automatically searches the parent directory and falls back to previous valid checkpoint.
    """
    model, optimizer, config = sample_components
    ckpt1_path = save_checkpoint(tmp_ckpt_dir, model, optimizer, global_step=1000, config=config)
    ckpt2_path = save_checkpoint(tmp_ckpt_dir, model, optimizer, global_step=2000, config=config)

    # Truncate ckpt2
    with open(ckpt2_path, "wb") as f:
        f.write(b"TRUNCATED")

    # Pass the corrupt file path directly
    data, loaded_path = load_checkpoint_with_fallback(ckpt2_path, current_config=config)
    assert data is not None
    assert os.path.realpath(loaded_path) == os.path.realpath(ckpt1_path)
    assert data["global_step"] == 1000


def test_scheduler_and_metadata_persistence(tmp_ckpt_dir, sample_components):
    """Verifies that LR scheduler, normalization stats, and evaluation history are preserved."""
    model, optimizer, config = sample_components
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    optimizer.step()
    scheduler.step()

    norm_stats = {"mean": torch.zeros(10), "std": torch.ones(10)}
    eval_history = [{"step": 1000, "train_success": 0.45, "val_success": 0.30}]

    ckpt_path = save_checkpoint(
        ckpt_dir=tmp_ckpt_dir,
        model=model,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        global_step=1000,
        iteration=20,
        config=config,
        norm_stats=norm_stats,
        eval_history=eval_history,
        active_time_s=120.5,
    )

    data, _ = load_checkpoint_with_fallback(ckpt_path, current_config=config)
    assert data["lr_scheduler_state_dict"] is not None
    assert "last_epoch" in data["lr_scheduler_state_dict"]
    assert data["eval_history"] == eval_history
    assert torch.equal(data["norm_stats"]["mean"], torch.zeros(10))
    assert data["active_time_s"] == 120.5


def test_empty_directory_returns_none(tmp_ckpt_dir):
    """When directory has no checkpoints, load_checkpoint_with_fallback returns (None, None)."""
    data, path = load_checkpoint_with_fallback(tmp_ckpt_dir)
    assert data is None
    assert path is None


def test_orphan_tmp_files_ignored_and_cleaned_after_simulated_crash(tmp_ckpt_dir, sample_components):
    """
    Simulates a hard kill / crash mid-write:
    1. Saves checkpoint 1 at step 1000.
    2. Simulates an interrupted write of step 2000 leaving orphan *.tmp files.
    3. Confirms get_checkpoint_candidates ignores orphan *.tmp files.
    4. Confirms startup / load_checkpoint_with_fallback automatically purges orphan *.tmp files.
    """
    model, optimizer, config = sample_components
    ckpt1_path = save_checkpoint(tmp_ckpt_dir, model, optimizer, global_step=1000, config=config)

    # Simulate crash mid-write of step 2000
    orphan_tmp1 = os.path.join(tmp_ckpt_dir, ".ckpt_step_000002000.pt.tmp.9999.abcd1234")
    orphan_tmp2 = os.path.join(tmp_ckpt_dir, "ckpt_step_000002000.pt.tmp")
    orphan_tmp3 = os.path.join(tmp_ckpt_dir, ".ckpt_step_000002000.pt.tmp.9999.abcd1234.sha256")

    with open(orphan_tmp1, "wb") as f:
        f.write(b"PARTIAL_CRASHED_CHECKPOINT_BYTES" * 10)
    with open(orphan_tmp2, "wb") as f:
        f.write(b"HALF_WRITTEN_FILE")
    with open(orphan_tmp3, "w") as f:
        f.write("partial_sha\n")

    # 1. get_checkpoint_candidates must ignore all *.tmp files
    candidates = get_checkpoint_candidates(tmp_ckpt_dir)
    for c in candidates:
        assert ".tmp" not in c
    assert len(candidates) >= 1
    assert os.path.realpath(candidates[0]) == os.path.realpath(ckpt1_path)

    # 2. Loading with fallback cleans up orphan files and loads ckpt1
    data, loaded_path = load_checkpoint_with_fallback(tmp_ckpt_dir, current_config=config)
    assert data is not None
    assert data["global_step"] == 1000
    assert os.path.realpath(loaded_path) == os.path.realpath(ckpt1_path)

    # Verify orphan files were deleted
    assert not os.path.exists(orphan_tmp1)
    assert not os.path.exists(orphan_tmp2)
    assert not os.path.exists(orphan_tmp3)


def test_missing_sha256_accepted_if_cpu_load_passes_rejected_if_corrupt(tmp_ckpt_dir, sample_components):
    """
    Defines behavior for a checkpoint whose .sha256 companion file is missing:
    - If CPU load test passes and required keys exist: ACCEPTED.
    - If corrupted / unreadable: REJECTED.
    """
    model, optimizer, config = sample_components
    ckpt_path = save_checkpoint(tmp_ckpt_dir, model, optimizer, global_step=1000, config=config)

    sha_path = f"{ckpt_path}.sha256"
    assert os.path.exists(sha_path)

    # Delete .sha256 file
    os.remove(sha_path)
    assert not os.path.exists(sha_path)

    # Case A: Intact file with missing .sha256 -> ACCEPTED via CPU load test
    is_valid, data = validate_checkpoint(ckpt_path)
    assert is_valid, f"Intact checkpoint without .sha256 was rejected: {data}"
    assert data["global_step"] == 1000

    loaded_data, loaded_path = load_checkpoint_with_fallback(tmp_ckpt_dir, current_config=config)
    assert loaded_data is not None
    assert loaded_data["global_step"] == 1000
    assert os.path.realpath(loaded_path) == os.path.realpath(ckpt_path)

    # Case B: Corrupted file with missing .sha256 -> REJECTED
    with open(ckpt_path, "wb") as f:
        f.write(b"CORRUPT_BYTES_WITHOUT_SHA")

    is_valid_corrupt, err = validate_checkpoint(ckpt_path)
    assert not is_valid_corrupt
    assert isinstance(err, (CorruptCheckpointError, Exception))


def test_config_hash_sensitivity_and_exclusions():
    """
    Proves that:
    1. Changing total_steps, seed, num_envs, modality, or split CHANGES the config hash.
    2. Changing log paths, output dirs, or timestamps DOES NOT CHANGE the config hash.
    """
    base_config = {
        "seed": 0,
        "num_envs": 160,
        "modality": "rgbd",
        "total_steps": 20000000,
        "viewpoint_split": [0, 1, 2, 4, 5, 7, 8, 11, 12, 14],
        "learning_rate": 0.0002,
        "gamma": 0.8,
        "gae_lambda": 0.9,
        # Non-functional / path / timestamp keys:
        "output_dir": "/home/maniskill/runs/run_pilot_rgbd",
        "log_dir": "/home/maniskill/runs/run_pilot_rgbd/logs",
        "log_path": "/home/maniskill/runs/run_pilot_rgbd/logs/train.csv",
        "timestamp": 1771660000.0,
        "created_at": "2026-10-21 14:15:00",
        "saved_at": 1771660100.0,
    }

    base_hash = compute_config_hash(base_config)

    # --- Group 1: Non-functional fields MUST NOT change the hash ---
    for non_func_key, new_val in [
        ("output_dir", "/completely/different/output/path"),
        ("log_dir", "/var/log/rl_runs"),
        ("log_path", "/var/log/rl_runs/stdout.log"),
        ("timestamp", 1888888888.0),
        ("created_at", "2027-01-01 00:00:00"),
        ("saved_at", 1888888999.0),
    ]:
        cfg = dict(base_config)
        cfg[non_func_key] = new_val
        h = compute_config_hash(cfg)
        assert h == base_hash, f"Modifying non-functional key '{non_func_key}' altered the config hash!"

    # Modifying multiple non-functional keys at once
    cfg_multi_non_func = dict(base_config)
    cfg_multi_non_func["output_dir"] = "/new/dir"
    cfg_multi_non_func["log_path"] = "/new/path.log"
    cfg_multi_non_func["timestamp"] = 999999999.9
    assert compute_config_hash(cfg_multi_non_func) == base_hash

    # --- Group 2: Core experimental protocol fields MUST change the hash ---
    sensitive_changes = [
        ("total_steps", 30000000),
        ("seed", 1),
        ("num_envs", 128),
        ("modality", "rgb"),
        ("viewpoint_split", [0, 1, 2, 3, 5, 7, 8, 11, 12, 14]),
        ("reward_mode", "sparse"),
        ("max_episode_steps", 100),
        ("ignore_terminations", True),
        ("record_metrics", False),
        ("control_mode", "pd_joint_delta_pos"),
        ("sim_backend", "physx_cpu"),
        ("camera", {"width": 256, "height": 256, "fov": 1.57, "near": 0.01, "far": 100.0, "look_at": [-0.1, 0.0, 0.1]}),
    ]

    for sens_key, new_val in sensitive_changes:
        cfg = dict(base_config)
        cfg[sens_key] = new_val
        h = compute_config_hash(cfg)
        assert h != base_hash, f"Modifying critical parameter '{sens_key}' FAILED to alter the config hash!"

    # Nested camera subfields must each change the hash
    base_with_cam = dict(base_config)
    base_with_cam["camera"] = {
        "width": 128,
        "height": 128,
        "fov": 1.5707963267948966,
        "near": 0.01,
        "far": 100.0,
        "look_at": [-0.1, 0.0, 0.1],
    }
    cam_base_hash = compute_config_hash(base_with_cam)

    for cam_subfield, altered_val in [
        ("width", 64),
        ("height", 64),
        ("fov", 1.2),
        ("near", 0.05),
        ("far", 50.0),
        ("look_at", [0.0, 0.0, 0.0]),
    ]:
        cfg_cam = dict(base_with_cam)
        cfg_cam["camera"] = dict(base_with_cam["camera"])
        cfg_cam["camera"][cam_subfield] = altered_val
        assert compute_config_hash(cfg_cam) != cam_base_hash, f"Altering camera.{cam_subfield} did not alter hash!"

    # Nested model.state_activation must change the hash
    base_with_model = dict(base_config)
    base_with_model["model"] = {"state_activation": "none"}
    model_base_hash = compute_config_hash(base_with_model)

    cfg_model_relu = dict(base_with_model)
    cfg_model_relu["model"] = {"state_activation": "relu"}
    assert compute_config_hash(cfg_model_relu) != model_base_hash, "Altering model.state_activation did not alter hash!"

