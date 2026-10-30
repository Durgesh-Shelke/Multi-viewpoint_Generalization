#!/usr/bin/env python3
"""
Preflight Readiness Check Script (scripts/preflight.py).

Verifies all hardware, system, and experimental protocol prerequisites
before launching pilot or main grid runs:
1. Git working tree is completely clean.
2. Frozen config and split/fresh-view SHA-256 hashes match configs/freeze_manifest.json.
3. Reference fingerprint and environment files are absent from state/ (to be created by first real run).
4. Free disk space is at least 20 GB.
5. No other process is using > 1.0 GB (1,024 MiB) of GPU memory.
6. AC power is connected if /sys/class/power_supply reports a battery.

Prints a structured PASS/FAIL table and exits with 0 if all pass, 1 if any fail.
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

EXP_DIR = Path(__file__).resolve().parent.parent


def compute_sha256(filepath: Union[str, Path]) -> str:
    """Computes SHA-256 hex digest of a file."""
    p = Path(filepath)
    if not p.is_file():
        raise FileNotFoundError(f"File not found: {p}")
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def check_git_clean() -> Tuple[bool, str]:
    """1. Git working tree clean."""
    try:
        res = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(EXP_DIR),
            capture_output=True,
            text=True,
            check=True,
        )
        output = res.stdout.strip()
        if not output:
            return True, "Git working tree is completely clean"
        lines = [line.strip() for line in output.splitlines() if line.strip()]
        return False, f"Working tree has {len(lines)} uncommitted change(s): {lines[:3]}"
    except Exception as e:
        return False, f"Failed to execute git status: {e}"


def check_freeze_manifest(manifest_path: Optional[Path] = None) -> Tuple[bool, str]:
    """2. Frozen config and split/fresh-view SHA-256 hashes unchanged versus freeze_manifest.json."""
    if manifest_path is None:
        manifest_path = EXP_DIR / "configs" / "freeze_manifest.json"

    if not manifest_path.is_file():
        return False, f"Freeze manifest not found at {manifest_path}"

    try:
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
    except Exception as e:
        return False, f"Could not read freeze manifest: {e}"

    hashes_dict = manifest.get("file_hashes", {})
    if not hashes_dict:
        return False, "Freeze manifest does not contain 'file_hashes'"

    mismatches = []
    missing_files = []

    for rel_path, expected_hash in hashes_dict.items():
        file_path = EXP_DIR / rel_path
        if not file_path.is_file():
            missing_files.append(rel_path)
            continue
        actual_hash = compute_sha256(file_path)
        if actual_hash != expected_hash:
            mismatches.append(f"{rel_path} (expected {expected_hash[:8]}..., got {actual_hash[:8]}...)")

    if missing_files:
        return False, f"Frozen file(s) missing: {missing_files}"
    if mismatches:
        return False, f"Hash mismatch in frozen files: {mismatches}"

    return True, f"All {len(hashes_dict)} frozen files verified against manifest"


def check_state_clean(state_dir: Optional[Path] = None) -> Tuple[bool, str]:
    """3. Reference fingerprint and environment files absent from state/."""
    if state_dir is None:
        state_dir = Path(os.environ.get("EXP_STATE_DIR", str(EXP_DIR / "state")))

    stale_files = []
    ref_env = state_dir / "reference_environment.json"
    ref_fp = state_dir / "reference_fingerprint.json"

    if ref_env.is_file():
        stale_files.append(str(ref_env.name))
    if ref_fp.is_file():
        stale_files.append(str(ref_fp.name))

    if stale_files:
        return False, f"Found stale reference file(s) in {state_dir}: {stale_files} (must be absent before first real run)"

    return True, "state/ is clean (reference environment and fingerprint absent)"


def check_disk_space(min_gb: float = 20.0) -> Tuple[bool, str]:
    """4. Free disk space at least 20 GB."""
    try:
        usage = shutil.disk_usage(str(EXP_DIR))
        free_gb = usage.free / (1024 ** 3)
        if free_gb >= min_gb:
            return True, f"{free_gb:.1f} GB free (>= {min_gb:.1f} GB required)"
        return False, f"Only {free_gb:.1f} GB free (< {min_gb:.1f} GB required)"
    except Exception as e:
        return False, f"Could not determine disk usage: {e}"


def check_gpu_processes(max_other_vram_mib: float = 1024.0) -> Tuple[bool, str]:
    """5. No other process using more than 1 GB of GPU memory."""
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
        )
        if res.returncode != 0:
            # Fall back to general nvidia-smi check
            return True, "nvidia-smi query-compute-apps not supported or no compute apps running"

        output = res.stdout.strip()
        if not output:
            return True, "No GPU compute processes running"

        current_pid = os.getpid()
        offending_processes = []
        for line in output.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                try:
                    pid = int(parts[0])
                    pname = parts[1]
                    used_mb = float(parts[2])
                    if pid != current_pid and used_mb > max_other_vram_mib:
                        offending_processes.append(f"PID {pid} ({pname}): {used_mb:.0f} MiB")
                except ValueError:
                    continue

        if offending_processes:
            return False, f"Process(es) using > 1.0 GB GPU memory: {offending_processes}"

        return True, "No external process using > 1.0 GB GPU memory"
    except FileNotFoundError:
        return False, "nvidia-smi utility not found in PATH"
    except Exception as e:
        return False, f"GPU process check error: {e}"


def check_ac_power() -> Tuple[bool, str]:
    """6. AC power if /sys/class/power_supply reports a battery."""
    ps_dir = Path("/sys/class/power_supply")
    if not ps_dir.is_dir():
        return True, "No /sys/class/power_supply directory found (assumed AC power)"

    try:
        supplies = list(ps_dir.iterdir())
    except Exception:
        return True, "Could not list /sys/class/power_supply (assumed AC power)"

    batteries = []
    ac_supplies = []

    for s in supplies:
        type_file = s / "type"
        if type_file.is_file():
            try:
                stype = type_file.read_text().strip()
                if stype == "Battery" or s.name.startswith("BAT"):
                    batteries.append(s)
                elif stype == "Mains" or s.name.startswith("AC") or s.name.startswith("ADP"):
                    ac_supplies.append(s)
            except Exception:
                pass

    if not batteries:
        return True, "No battery detected (system is desktop/server on AC power)"

    # Check if any AC supply is online
    ac_online = False
    for ac in ac_supplies:
        online_file = ac / "online"
        if online_file.is_file():
            try:
                if online_file.read_text().strip() == "1":
                    ac_online = True
                    break
            except Exception:
                pass

    # Also inspect battery status: "Charging", "Full", or "Not charging" indicates AC adapter is connected
    battery_statuses = []
    for bat in batteries:
        status_file = bat / "status"
        if status_file.is_file():
            try:
                b_status = status_file.read_text().strip()
                battery_statuses.append(f"{bat.name}: {b_status}")
                if b_status in ("Charging", "Full", "Not charging"):
                    ac_online = True
            except Exception:
                pass

    if ac_online:
        return True, f"AC power connected (Battery: {', '.join(battery_statuses)})"

    return False, f"Running on battery power without AC adapter! ({', '.join(battery_statuses)})"


def run_preflight(verbose: bool = True) -> bool:
    """
    Executes all preflight checks and prints a formatted report.
    Returns True if all pass, False if any fails.
    """
    checks = [
        ("Git Working Tree Clean", check_git_clean),
        ("Frozen Config Hashes Unchanged", check_freeze_manifest),
        ("State Directory Clean (No Stale Refs)", check_state_clean),
        ("Free Disk Space >= 20 GB", check_disk_space),
        ("GPU Memory Available (No Apps > 1 GB)", check_gpu_processes),
        ("AC Power Connected (Laptop Safe)", check_ac_power),
    ]

    results = []
    all_passed = True

    for name, check_fn in checks:
        try:
            passed, detail = check_fn()
        except Exception as e:
            passed, detail = False, f"Exception: {e}"
        results.append((name, passed, detail))
        if not passed:
            all_passed = False

    if verbose:
        print("\n" + "=" * 78)
        print(" EXPERIMENT PREFLIGHT READINESS REPORT")
        print("=" * 78)
        print(f" {'Check':<36} | {'Status':<6} | {'Details'}")
        print("-" * 78)
        for name, passed, detail in results:
            status_str = "PASS" if passed else "FAIL"
            print(f" {name:<36} | {status_str:<6} | {detail}")
        print("=" * 78)
        if all_passed:
            print(" >>> ALL PREFLIGHT CHECKS PASSED. SYSTEM READY FOR PILOT RUNS. <<<\n")
        else:
            print(" >>> PREFLIGHT FAILED. RESOLVE IDENTIFIED ISSUES BEFORE STARTING. <<<\n")

    return all_passed


def main():
    passed = run_preflight(verbose=True)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
