"""
Integration unit tests for src/train_ppo.py pause, resume, and logging continuity.

Validates per SPEC Section 2:
1. Graceful pause on SIGTERM mid-run finishes iteration, saves checkpoint, and exits code 0.
2. Auto-resume from newest valid checkpoint continues training without manual flags.
3. Both RGB and RGB-D modalities tested.
4. Logging continuity: progress.csv has strictly monotonic iterations, continuous steps,
   and zero duplicate or missing rows across pause/resume cycles.
5. Pause file trigger (state/PAUSE or output_dir/PAUSE).

Python 3.9 compatible only.
"""

import csv
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest
import yaml

from src.checkpoint import validate_checkpoint
from src.fingerprint import ManiSkillFingerprintMismatchError
from src.train_ppo import write_or_verify_manifest, ManiSkillIntegrityError, get_maniskill_task_hashes

EXP_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = str(EXP_DIR / "configs" / "tiny_test.yaml")


def read_csv_log(csv_path: str) -> List[Dict[str, Any]]:
    """Reads progress.csv into a list of row dicts."""
    if not os.path.isfile(csv_path):
        return []
    rows = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def wait_for_iterations(csv_path: str, min_iters: int = 2, timeout: float = 30.0) -> bool:
    """Waits until progress.csv contains at least min_iters rows."""
    start_t = time.time()
    while time.time() - start_t < timeout:
        if os.path.isfile(csv_path):
            rows = read_csv_log(csv_path)
            if len(rows) >= min_iters:
                return True
        time.sleep(0.2)
    return False


def run_pause_resume_scenario(
    modality: str,
    output_dir: str,
    total_timesteps: int = 2000,
    pause_after_iters: int = 3,
) -> None:
    """
    Executes a pause/resume verification lifecycle for the given modality:
    1. Start run.
    2. Wait until pause_after_iters are logged.
    3. Send SIGTERM to process.
    4. Assert process finishes iteration and exits 0 cleanly.
    5. Assert checkpoint validates.
    6. Restart run with exact same command.
    7. Let it run to completion.
    8. Assert progress.csv has exact, continuous, monotonic sequence without duplicates.
    """
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, "progress.csv")
    ckpt_dir = os.path.join(output_dir, "checkpoints")

    cmd = [
        sys.executable,
        "-m",
        "src.train_ppo",
        "--config",
        CONFIG_PATH,
        "--modality",
        modality,
        "--output_dir",
        output_dir,
        "--total_timesteps",
        str(total_timesteps),
        "--seed",
        "42",
    ]

    sub_env = dict(os.environ, EXP_STATE_DIR=os.path.join(output_dir, "isolated_state"))

    # --- Phase 1: Launch and pause mid-run ---
    proc1 = subprocess.Popen(
        cmd,
        cwd=str(EXP_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=sub_env,
    )

    # Wait for process to record at least pause_after_iters iterations
    ready = wait_for_iterations(csv_path, min_iters=pause_after_iters, timeout=45.0)
    assert ready, f"Process did not complete {pause_after_iters} iterations before timeout"

    # Send SIGTERM cleanly
    proc1.send_signal(signal.SIGTERM)
    stdout1, _ = proc1.communicate(timeout=30.0)
    assert proc1.returncode == 0, f"Process did not exit with code 0 on SIGTERM: exit code {proc1.returncode}\nOutput:\n{stdout1}"

    # Verify checkpoint exists and validates
    latest_path = os.path.join(ckpt_dir, "latest.pt")
    assert os.path.exists(latest_path), f"latest.pt was not saved on pause: {os.listdir(ckpt_dir)}"
    valid, data = validate_checkpoint(latest_path)
    assert valid, f"Checkpoint validation failed on pause: {data}"

    paused_iter = int(data["iteration"])
    paused_step = int(data["global_step"])
    assert paused_iter >= pause_after_iters
    assert paused_step == paused_iter * 200

    rows_phase1 = read_csv_log(csv_path)
    assert len(rows_phase1) == paused_iter, f"Expected {paused_iter} rows in progress.csv, got {len(rows_phase1)}"

    # --- Phase 2: Resume with the exact same command ---
    proc2 = subprocess.Popen(
        cmd,
        cwd=str(EXP_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=sub_env,
    )
    stdout2, _ = proc2.communicate(timeout=60.0)
    assert proc2.returncode == 0, f"Resumed process did not exit with code 0:\nOutput:\n{stdout2}"
    assert "[Resume] Resumed successfully" in stdout2, f"Resume message not found in stdout:\n{stdout2}"

    # --- Phase 3: Verify final checkpoint and log continuity ---
    final_ckpt = os.path.join(ckpt_dir, f"ckpt_final_step_{total_timesteps:09d}.pt")
    assert os.path.exists(final_ckpt), f"Final checkpoint {final_ckpt} not found"
    valid_final, data_final = validate_checkpoint(final_ckpt)
    assert valid_final, f"Final checkpoint invalid: {data_final}"
    assert int(data_final["global_step"]) == total_timesteps

    # Verify progress.csv continuity
    final_rows = read_csv_log(csv_path)
    expected_iterations = total_timesteps // 200  # 10 iterations

    assert len(final_rows) == expected_iterations, (
        f"Expected exactly {expected_iterations} rows in progress.csv, got {len(final_rows)}: "
        f"{[r['iteration'] for r in final_rows]}"
    )

    iterations = [int(r["iteration"]) for r in final_rows]
    global_steps = [int(r["global_step"]) for r in final_rows]

    # Strictly monotonic iterations [1, 2, ..., expected_iterations]
    expected_iters = list(range(1, expected_iterations + 1))
    assert iterations == expected_iters, f"Iterations sequence mismatch: {iterations} != {expected_iters}"

    # Strictly monotonic steps [200, 400, ..., total_timesteps]
    expected_steps = [i * 200 for i in expected_iters]
    assert global_steps == expected_steps, f"Step sequence mismatch: {global_steps} != {expected_steps}"

    # Verify no duplicates
    assert len(iterations) == len(set(iterations)), "Duplicate iterations found in progress.csv!"
    assert len(global_steps) == len(set(global_steps)), "Duplicate global steps found in progress.csv!"


def test_train_ppo_pause_resume_rgb(tmp_path):
    """Validates SIGTERM pause, checkpointing, and resume continuity for RGB modality."""
    run_pause_resume_scenario(
        modality="rgb",
        output_dir=str(tmp_path / "test_pause_resume_rgb"),
        total_timesteps=2000,
        pause_after_iters=3,
    )


def test_train_ppo_pause_resume_rgbd(tmp_path):
    """Validates SIGTERM pause, checkpointing, and resume continuity for RGB-D modality."""
    run_pause_resume_scenario(
        modality="rgbd",
        output_dir=str(tmp_path / "test_pause_resume_rgbd"),
        total_timesteps=2000,
        pause_after_iters=3,
    )


def test_train_ppo_pause_file_trigger(tmp_path):
    """Validates that creating output_dir/PAUSE triggers a clean pause and exit code 0."""
    output_dir = str(tmp_path / "test_pause_file")
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, "progress.csv")
    pause_file = os.path.join(output_dir, "PAUSE")

    cmd = [
        sys.executable,
        "-m",
        "src.train_ppo",
        "--config",
        CONFIG_PATH,
        "--modality",
        "rgb",
        "--output_dir",
        output_dir,
        "--total_timesteps",
        "3000",
        "--seed",
        "42",
        "--pause_file",
        pause_file,
    ]

    sub_env = dict(os.environ, EXP_STATE_DIR=os.path.join(output_dir, "isolated_state"))

    proc = subprocess.Popen(
        cmd,
        cwd=str(EXP_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=sub_env,
    )

    # Wait for at least 2 iterations
    ready = wait_for_iterations(csv_path, min_iters=2, timeout=45.0)
    assert ready

    # Create PAUSE file
    with open(pause_file, "w") as f:
        f.write("pause requested\n")

    stdout, _ = proc.communicate(timeout=30.0)
    assert proc.returncode == 0, f"Process did not exit 0 on PAUSE file trigger:\n{stdout}"
    assert "[Pause]" in stdout, f"Pause log not in stdout:\n{stdout}"

    # Verify checkpoint exists
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    latest_path = os.path.join(ckpt_dir, "latest.pt")
    assert os.path.exists(latest_path)
    valid, data = validate_checkpoint(latest_path)
    assert valid


def test_refuses_to_start_with_null_total_timesteps(tmp_path):
    """
    CRITICAL PROTOCOL TEST:
    1. Running train_ppo with ppo_frozen.yaml without specifying total_timesteps must fail fast with ValueError.
    2. Attempting to start an unbudgeted main-grid run (--run_name main_rgb_seed0) where total_timesteps is null
       must fail fast with ValueError, preventing unbudgeted runs from starting.
    """
    # 1. Direct ppo_frozen.yaml without total_timesteps
    cmd1 = [
        sys.executable,
        "-m",
        "src.train_ppo",
        "--config",
        "configs/ppo_frozen.yaml",
        "--modality",
        "rgb",
        "--output_dir",
        str(tmp_path / "test_null_budget_refusal"),
    ]
    res1 = subprocess.run(cmd1, cwd=str(EXP_DIR), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert res1.returncode != 0
    assert "total_timesteps is None" in (res1.stdout + res1.stderr)

    # 2. Main-grid run from runs.yaml with total_timesteps: null
    dummy_runs_path = tmp_path / "dummy_null_runs.yaml"
    dummy_runs_path.write_text(yaml.dump({
        "runs": {
            "unbudgeted_run": {
                "modality": "rgb",
                "seed": 0,
                "total_timesteps": None,
                "output_dir": "runs/unbudgeted_run",
                "final_eval_groups": "all",
            }
        }
    }))
    cmd2 = [
        sys.executable,
        "-m",
        "src.train_ppo",
        "--config",
        "configs/ppo_frozen.yaml",
        "--runs_config",
        str(dummy_runs_path),
        "--run_name",
        "unbudgeted_run",
    ]
    res2 = subprocess.run(cmd2, cwd=str(EXP_DIR), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert res2.returncode != 0
    assert "The budget has not been frozen yet" in (res2.stdout + res2.stderr)


def test_maniskill_task_file_hash_mismatch_refusal(tmp_path):
    """
    CRITICAL PROTOCOL TEST:
    Manifest records SHA-256 hashes of pick_cube.py and pick_cube_cfgs.py.
    If the manifest hashes disagree with the current files on disk, resume must be strictly
    refused with ManiSkillIntegrityError.
    """
    from src.train_ppo import write_or_verify_manifest, ManiSkillIntegrityError, get_maniskill_task_hashes

    output_dir = str(tmp_path / "test_manifest_tamper_refusal")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Direct unit test of write_or_verify_manifest logic
    cfg = {"seed": 42, "total_timesteps": 1000}
    eff = {"reward_mode": "normalized_dense"}
    write_or_verify_manifest(output_dir, cfg, eff, is_resuming=False)

    # Resuming with identical hashes must succeed
    manifest_unmodified = write_or_verify_manifest(output_dir, cfg, eff, is_resuming=True)
    assert manifest_unmodified is not None

    # Tamper with recorded hash of pick_cube.py
    manifest_path = os.path.join(output_dir, "manifest.json")
    import json
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    manifest["maniskill_task_hashes"]["pick_cube.py"]["sha256"] = "tampered_bad_sha256_hash_value"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # Resuming must raise ManiSkillIntegrityError
    with pytest.raises(ManiSkillIntegrityError) as exc_info:
        write_or_verify_manifest(output_dir, cfg, eff, is_resuming=True)
    assert "ManiSkill source integrity violation" in str(exc_info.value)
    assert "pick_cube.py changed" in str(exc_info.value)

    # 2. Integration test via train_ppo subprocess
    pause_file = os.path.join(output_dir, "PAUSE")
    with open(pause_file, "w") as f:
        f.write("pause\n")

    # Run 1 iteration and pause
    cmd_pause = [
        sys.executable,
        "-m",
        "src.train_ppo",
        "--config",
        CONFIG_PATH,
        "--modality",
        "rgb",
        "--output_dir",
        output_dir,
        "--total_timesteps",
        "400",
        "--seed",
        "42",
        "--pause_file",
        pause_file,
    ]
    # Remove old manifest to let subprocess write fresh one
    os.remove(manifest_path)
    res_init = subprocess.run(cmd_pause, cwd=str(EXP_DIR), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert res_init.returncode == 0, f"Init run failed:\n{res_init.stdout}\n{res_init.stderr}"
    assert os.path.isfile(manifest_path)

    # Remove pause file so resume can proceed
    if os.path.exists(pause_file):
        os.remove(pause_file)

    # Tamper with manifest again
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    manifest["maniskill_task_hashes"]["pick_cube_cfgs.py"]["sha256"] = "tampered_bad_cfgs_sha256"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # Subprocess resume must fail fast with ManiSkillIntegrityError
    res_resume = subprocess.run(cmd_pause, cwd=str(EXP_DIR), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert res_resume.returncode != 0
    all_output = res_resume.stdout + res_resume.stderr
    assert "ManiSkillIntegrityError" in all_output or "ManiSkill source integrity violation" in all_output
    assert "pick_cube_cfgs.py changed" in all_output


def test_runs_yaml_spec_conformance():
    """
    Verifies configs/runs.yaml:
    - All 6 main grid runs (2 modalities x 3 seeds {0, 1, 2}) have total_timesteps: 40_000_000.
    """
    import yaml
    with open("configs/runs.yaml", "r") as f:
        data = yaml.safe_load(f)

    runs = data.get("runs", {})
    if "pilot_rgb" in runs:
        assert runs["pilot_rgb"]["seed"] == 100
    if "pilot_rgbd" in runs:
        assert runs["pilot_rgbd"]["seed"] == 100

    main_runs = [
        "main_rgb_seed0", "main_rgbd_seed0",
        "main_rgb_seed1", "main_rgbd_seed1",
        "main_rgb_seed2", "main_rgbd_seed2",
    ]
    for r in main_runs:
        assert r in runs, f"Missing {r} in configs/runs.yaml"
        assert runs[r]["total_timesteps"] == 40_000_000, f"{r} total_timesteps must be 40M!"


def test_maniskill_source_fingerprint_mismatch_refusal(tmp_path):
    """
    Protocol Requirement:
    Manifest stores maniskill_source_fingerprint (git HEAD + diff SHA256).
    If the source fingerprint differs upon resuming, write_or_verify_manifest
    must raise ManiSkillFingerprintMismatchError and refuse to resume.
    """
    from src.fingerprint import ManiSkillFingerprintMismatchError

    out_dir = str(tmp_path / "test_fp_refusal")
    ref_fp_file = str(tmp_path / "ref_fp.json")
    os.makedirs(out_dir, exist_ok=True)

    cfg = {"seed": 42, "total_timesteps": 1000}
    eff = {"reward_mode": "normalized_dense"}

    # Initial manifest creation succeeds and records fingerprint
    m1 = write_or_verify_manifest(out_dir, cfg, eff, is_resuming=False, ref_fingerprint_path=ref_fp_file)
    assert "maniskill_source_fingerprint" in m1
    assert os.path.isfile(ref_fp_file)

    # Resuming with untouched files succeeds
    m2 = write_or_verify_manifest(out_dir, cfg, eff, is_resuming=True, ref_fingerprint_path=ref_fp_file)
    assert m2 is not None

    # Tamper with recorded fingerprint in manifest.json
    manifest_path = os.path.join(out_dir, "manifest.json")
    with open(manifest_path, "r") as f:
        data = json.load(f)
    data["maniskill_source_fingerprint"]["fingerprint"] = "tampered_git_head:tampered_diff_sha256"
    with open(manifest_path, "w") as f:
        json.dump(data, f, indent=2)

    with pytest.raises(ManiSkillFingerprintMismatchError) as exc_info:
        write_or_verify_manifest(out_dir, cfg, eff, is_resuming=True, ref_fingerprint_path=ref_fp_file)

    assert "ManiSkill source fingerprint mismatch" in str(exc_info.value)
    assert "Refusing to resume run with altered ManiSkill source repository" in str(exc_info.value)

