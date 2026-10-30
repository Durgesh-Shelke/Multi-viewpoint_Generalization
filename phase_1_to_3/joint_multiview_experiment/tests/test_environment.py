"""
Unit and Integration Tests for Environment Fingerprint & Integrity Guard.

Tests per Protocol Requirements:
1. Environment metadata collection:
   - python_version, torch, CUDA build, mani_skill, gymnasium, numpy versions
   - SHA-256 of configs/requirements_frozen.txt
   - hashed output of pip freeze
2. Reference environment creation:
   - Created on the first real run, NEVER by tests
   - Tests isolate state via EXP_STATE_DIR
3. Refusal to start or resume if versions differ:
   - torch
   - CUDA build
   - mani_skill
   - gymnasium
   - numpy
   - Python
   with a clear EnvironmentMismatchError.
4. Manifest recording:
   - Every run's manifest.json contains all required environment fields.
5. Runner refusal:
   - Runner check_environment_integrity refuses mismatched environments.

Python 3.9 compatible only.
"""

import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import pytest

from src.environment import (
    CRITICAL_ENVIRONMENT_FIELDS,
    EnvironmentMismatchError,
    collect_environment_metadata,
    compare_environment_metadata,
    compute_file_sha256,
    get_reference_environment_path,
    save_reference_environment,
    verify_environment,
)
from src.runner import Runner
from src.train_ppo import write_or_verify_manifest

EXP_DIR = Path(__file__).resolve().parent.parent


def test_collect_environment_metadata():
    """Confirms collect_environment_metadata gathers all required keys and valid hashes."""
    meta = collect_environment_metadata(use_cache=False)

    required_keys = [
        "python_version",
        "python_full_version",
        "torch_version",
        "cuda_build",
        "mani_skill_version",
        "sapien_version",
        "gymnasium_version",
        "numpy_version",
        "requirements_frozen_sha256",
        "pip_freeze_sha256",
        "pip_packages",
    ]
    for key in required_keys:
        assert key in meta, f"Missing key in collected environment metadata: {key}"
        assert meta[key] is not None, f"Key {key} is None"
        assert len(str(meta[key])) > 0, f"Key {key} is empty"

    # Verify requirements_frozen_sha256 matches actual file
    req_file = str(EXP_DIR / "configs" / "requirements_frozen.txt")
    expected_req_sha = compute_file_sha256(req_file)
    assert meta["requirements_frozen_sha256"] == expected_req_sha
    assert len(meta["requirements_frozen_sha256"]) == 64

    # Verify pip_freeze_sha256 is 64 hex characters
    assert len(meta["pip_freeze_sha256"]) == 64
    int(meta["pip_freeze_sha256"], 16)  # Valid hex


def test_first_run_creates_reference_environment(tmp_path):
    """
    Confirms reference_environment.json is created on first run when missing,
    and tests using EXP_STATE_DIR never touch real state/.
    """
    isolated_state = tmp_path / "test_state"
    isolated_state.mkdir()
    ref_path = str(isolated_state / "reference_environment.json")

    assert not os.path.exists(ref_path)

    # First verify call creates reference environment
    is_match, err = verify_environment(ref_path=ref_path)
    assert is_match is True
    assert err is None
    assert os.path.isfile(ref_path)

    with open(ref_path, "r") as f:
        saved_meta = json.load(f)

    for field in CRITICAL_ENVIRONMENT_FIELDS:
        assert field in saved_meta

    # Real repository state/reference_environment.json MUST NOT exist
    real_ref = str(EXP_DIR / "state" / "reference_environment.json")
    assert not os.path.exists(real_ref), "Tests must NEVER create real state/reference_environment.json!"


def test_matching_environment_passes(tmp_path):
    """Confirms that when environment matches reference, verification succeeds."""
    ref_path = str(tmp_path / "reference_environment.json")
    verify_environment(ref_path=ref_path)

    # Second check against the same reference
    is_match, err = verify_environment(ref_path=ref_path)
    assert is_match is True
    assert err is None


@pytest.mark.parametrize("critical_field,altered_value", [
    ("torch_version", "1.12.1+cu113"),
    ("cuda_build", "11.7"),
    ("mani_skill_version", "2.1.0"),
    ("sapien_version", "2.2.0"),
    ("gymnasium_version", "0.28.1"),
    ("numpy_version", "1.21.6"),
    ("python_version", "3.10.12"),
])
def test_refuse_when_critical_version_differs(tmp_path, critical_field, altered_value):
    """Confirms refusal with clear error when any critical environment component differs."""
    ref_path = str(tmp_path / f"ref_{critical_field}.json")

    # Save a reference with the altered version
    meta = collect_environment_metadata(use_cache=False)
    tampered_meta = dict(meta)
    tampered_meta[critical_field] = altered_value
    save_reference_environment(env_meta=tampered_meta, ref_path=ref_path)

    # Verification of current environment against tampered reference must fail
    is_match, err = verify_environment(ref_path=ref_path, current_meta=meta)
    assert is_match is False
    assert err is not None
    assert "Environment version mismatch detected against reference!" in err
    assert critical_field in err
    assert altered_value in err
    assert str(meta[critical_field]) in err


def test_train_ppo_refuses_start_on_mismatched_reference_env(tmp_path):
    """Confirms write_or_verify_manifest refuses to start run when torch differs in reference."""
    ref_path = str(tmp_path / "reference_environment.json")
    meta = collect_environment_metadata(use_cache=False)
    tampered_meta = dict(meta, torch_version="0.0.1")
    save_reference_environment(env_meta=tampered_meta, ref_path=ref_path)

    run_dir = str(tmp_path / "run_refuse_start")
    os.makedirs(run_dir, exist_ok=True)

    dummy_cfg = {"exp_name": "test", "modality": "rgb", "seed": 42}

    with pytest.raises(EnvironmentMismatchError) as exc_info:
        write_or_verify_manifest(
            output_dir=run_dir,
            config=dummy_cfg,
            effective_env_settings={},
            is_resuming=False,
            ref_environment_path=ref_path,
        )
    assert "torch_version" in str(exc_info.value)
    assert "0.0.1" in str(exc_info.value)


def test_train_ppo_refuses_resume_on_mismatched_reference_env(tmp_path):
    """Confirms write_or_verify_manifest refuses to resume run when numpy differs in reference."""
    ref_path = str(tmp_path / "reference_environment.json")
    run_dir = str(tmp_path / "run_refuse_resume_ref")
    os.makedirs(run_dir, exist_ok=True)

    dummy_cfg = {"exp_name": "test", "modality": "rgb", "seed": 42}

    # First start run with valid reference
    write_or_verify_manifest(
        output_dir=run_dir,
        config=dummy_cfg,
        effective_env_settings={},
        is_resuming=False,
        ref_environment_path=ref_path,
    )
    assert os.path.exists(os.path.join(run_dir, "manifest.json"))

    # Alter reference environment numpy version
    with open(ref_path, "r") as f:
        saved_meta = json.load(f)
    saved_meta["numpy_version"] = "9.9.9"
    with open(ref_path, "w") as f:
        json.dump(saved_meta, f, indent=2)

    # Resume must refuse
    with pytest.raises(EnvironmentMismatchError) as exc_info:
        write_or_verify_manifest(
            output_dir=run_dir,
            config=dummy_cfg,
            effective_env_settings={},
            is_resuming=True,
            ref_environment_path=ref_path,
        )
    assert "numpy_version" in str(exc_info.value)
    assert "9.9.9" in str(exc_info.value)


def test_train_ppo_refuses_resume_on_tampered_manifest_env(tmp_path):
    """Confirms write_or_verify_manifest refuses to resume when manifest.json environment was altered."""
    ref_path = str(tmp_path / "reference_environment.json")
    run_dir = str(tmp_path / "run_tampered_manifest")
    os.makedirs(run_dir, exist_ok=True)

    dummy_cfg = {"exp_name": "test", "modality": "rgb", "seed": 42}

    # Start run cleanly
    manifest = write_or_verify_manifest(
        output_dir=run_dir,
        config=dummy_cfg,
        effective_env_settings={},
        is_resuming=False,
        ref_environment_path=ref_path,
    )

    manifest_path = os.path.join(run_dir, "manifest.json")
    with open(manifest_path, "r") as f:
        manifest_data = json.load(f)

    # Tamper with saved manifest environment torch_version
    manifest_data["environment"]["torch_version"] = "1.0.0"
    manifest_data["torch_version"] = "1.0.0"
    with open(manifest_path, "w") as f:
        json.dump(manifest_data, f, indent=2)

    # Resuming must refuse with EnvironmentMismatchError
    with pytest.raises(EnvironmentMismatchError) as exc_info:
        write_or_verify_manifest(
            output_dir=run_dir,
            config=dummy_cfg,
            effective_env_settings={},
            is_resuming=True,
            ref_environment_path=ref_path,
        )
    assert "Environment version mismatch against saved run manifest!" in str(exc_info.value)
    assert "torch_version" in str(exc_info.value)
    assert "1.0.0" in str(exc_info.value)


def test_manifest_records_all_environment_fields(tmp_path):
    """Verifies that manifest.json contains all required environment fields."""
    ref_path = str(tmp_path / "reference_environment.json")
    run_dir = str(tmp_path / "run_manifest_check")
    os.makedirs(run_dir, exist_ok=True)

    dummy_cfg = {"exp_name": "test_manifest", "modality": "rgb", "seed": 100}
    write_or_verify_manifest(
        output_dir=run_dir,
        config=dummy_cfg,
        effective_env_settings={},
        is_resuming=False,
        ref_environment_path=ref_path,
    )

    manifest_path = os.path.join(run_dir, "manifest.json")
    assert os.path.isfile(manifest_path)

    with open(manifest_path, "r") as f:
        manifest_data = json.load(f)

    # Verify top-level environment fields
    assert "python_version" in manifest_data
    assert "torch_version" in manifest_data
    assert "cuda_build" in manifest_data
    assert "mani_skill_version" in manifest_data
    assert "sapien_version" in manifest_data
    assert "gymnasium_version" in manifest_data
    assert "numpy_version" in manifest_data
    assert "requirements_frozen_sha256" in manifest_data
    assert "pip_freeze_sha256" in manifest_data

    # Verify structured environment sub-dictionary
    assert "environment" in manifest_data
    env_sub = manifest_data["environment"]
    for field in [
        "python_version",
        "torch_version",
        "cuda_build",
        "mani_skill_version",
        "sapien_version",
        "gymnasium_version",
        "numpy_version",
        "requirements_frozen_sha256",
        "pip_freeze_sha256",
    ]:
        assert field in env_sub
        assert len(str(env_sub[field])) > 0


def test_refuse_when_sapien_version_differs_in_train_ppo(tmp_path):
    """Confirms write_or_verify_manifest refuses when sapien version differs."""
    ref_path = str(tmp_path / "reference_environment.json")
    meta = collect_environment_metadata(use_cache=False)
    tampered_meta = dict(meta, sapien_version="2.0.0")
    save_reference_environment(env_meta=tampered_meta, ref_path=ref_path)

    run_dir = str(tmp_path / "run_refuse_sapien")
    os.makedirs(run_dir, exist_ok=True)

    dummy_cfg = {"exp_name": "test_sapien", "modality": "rgb", "seed": 42}

    with pytest.raises(EnvironmentMismatchError) as exc_info:
        write_or_verify_manifest(
            output_dir=run_dir,
            config=dummy_cfg,
            effective_env_settings={},
            is_resuming=False,
            ref_environment_path=ref_path,
        )
    assert "sapien_version" in str(exc_info.value)
    assert "2.0.0" in str(exc_info.value)


def test_unrelated_package_diff_warns_and_does_not_refuse(tmp_path, caplog):
    """
    Confirms that when pip_freeze_sha256 differs due to an unrelated package:
    - A WARNING is logged with package-level diff.
    - verify_environment does NOT refuse (returns True, None).
    - current_meta['pip_freeze_diff'] contains the package diff.
    """
    import logging
    ref_path = str(tmp_path / "reference_environment.json")
    meta = collect_environment_metadata(use_cache=False)

    # Reference has an older or extra package that is unrelated to critical list
    tampered_ref = dict(meta)
    tampered_ref["pip_freeze_sha256"] = "altered_sha256_unrelated_pkg_1234567890abcdef"
    ref_pkgs = dict(meta.get("pip_packages", {}))
    ref_pkgs["requests"] = "2.25.0"  # altered version
    ref_pkgs["old_tool"] = "1.0.0"   # extra package in reference
    tampered_ref["pip_packages"] = ref_pkgs

    save_reference_environment(env_meta=tampered_ref, ref_path=ref_path)

    with caplog.at_level(logging.WARNING):
        is_match, err = verify_environment(ref_path=ref_path, current_meta=meta)

    # Must NOT refuse!
    assert is_match is True
    assert err is None

    # Must have populated pip_freeze_diff
    assert "pip_freeze_diff" in meta
    diff = meta["pip_freeze_diff"]
    assert "diff_lines" in diff
    assert any("requests" in line for line in diff["diff_lines"])
    assert any("old_tool" in line for line in diff["diff_lines"])

    # Must have logged WARNING
    assert any("pip_freeze_sha256 differs from reference" in rec.message for rec in caplog.records)


def test_manifest_records_pip_freeze_diff_when_changed(tmp_path):
    """Confirms manifest.json records pip_freeze_diff when pip freeze hash differs."""
    ref_path = str(tmp_path / "reference_environment.json")
    meta = collect_environment_metadata(use_cache=False)

    tampered_ref = dict(meta)
    tampered_ref["pip_freeze_sha256"] = "different_hash_for_manifest_test"
    ref_pkgs = dict(meta.get("pip_packages", {}))
    ref_pkgs["some_unrelated_library"] = "0.9.1"
    tampered_ref["pip_packages"] = ref_pkgs
    save_reference_environment(env_meta=tampered_ref, ref_path=ref_path)

    run_dir = str(tmp_path / "run_pip_diff_manifest")
    os.makedirs(run_dir, exist_ok=True)

    dummy_cfg = {"exp_name": "test_diff_manifest", "modality": "rgb", "seed": 42}
    write_or_verify_manifest(
        output_dir=run_dir,
        config=dummy_cfg,
        effective_env_settings={},
        is_resuming=False,
        ref_environment_path=ref_path,
    )

    manifest_path = os.path.join(run_dir, "manifest.json")
    with open(manifest_path, "r") as f:
        manifest_data = json.load(f)

    assert "pip_freeze_diff" in manifest_data
    diff = manifest_data["pip_freeze_diff"]
    assert "diff_lines" in diff
    assert any("some_unrelated_library" in line for line in diff["diff_lines"])


def test_runner_refuses_when_environment_mismatched(tmp_path):
    """Confirms Runner.check_environment_integrity raises EnvironmentMismatchError if reference differs."""
    state_dir = tmp_path / "runner_state"
    state_dir.mkdir()
    ref_env_path = state_dir / "reference_environment.json"

    meta = collect_environment_metadata(use_cache=False)
    tampered_meta = dict(meta, gymnasium_version="0.10.0")
    save_reference_environment(env_meta=tampered_meta, ref_path=str(ref_env_path))

    runner = Runner(
        runs_config_path="configs/runs.yaml",
        base_config_path="configs/ppo_frozen.yaml",
        state_dir=str(state_dir),
    )

    with pytest.raises(EnvironmentMismatchError) as exc_info:
        runner.check_environment_integrity()

    assert "gymnasium_version" in str(exc_info.value)
    assert "0.10.0" in str(exc_info.value)


def test_real_state_dir_not_polluted():
    """Confirms real state/reference_environment.json does not exist (created by first real run, never tests)."""
    real_ref = EXP_DIR / "state" / "reference_environment.json"
    assert not real_ref.exists(), "real state/reference_environment.json must not be created by tests!"
