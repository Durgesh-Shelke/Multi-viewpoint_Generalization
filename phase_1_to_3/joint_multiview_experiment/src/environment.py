"""
Environment Fingerprint & Integrity Utility.

Per Protocol Requirements:
Records the environment in every run's manifest.json:
- Python version
- torch version
- CUDA build
- mani_skill version
- gymnasium version
- numpy version
- SHA-256 of configs/requirements_frozen.txt
- SHA-256 of `pip freeze` output

On start and on resume, compares these versions with state/reference_environment.json
(created by the first real run, never by tests; respects EXP_STATE_DIR in tests)
and refuses to start or resume if torch, CUDA build, mani_skill, gymnasium, numpy
or Python versions differ, raising EnvironmentMismatchError with a clear error.

Python 3.9 compatible only.
"""

import hashlib
import json
import logging
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

EXP_DIR = Path(__file__).resolve().parent.parent

# Critical fields that MUST match reference_environment.json and saved run manifest.
CRITICAL_ENVIRONMENT_FIELDS = (
    "python_version",
    "torch_version",
    "cuda_build",
    "mani_skill_version",
    "sapien_version",
    "gymnasium_version",
    "numpy_version",
)

FIELD_ALIASES = {
    "python_version": ["python_version", "python", "python_short_version"],
    "torch_version": ["torch_version", "torch"],
    "cuda_build": ["cuda_build", "cuda"],
    "mani_skill_version": ["mani_skill_version", "mani_skill"],
    "sapien_version": ["sapien_version", "sapien"],
    "gymnasium_version": ["gymnasium_version", "gymnasium"],
    "numpy_version": ["numpy_version", "numpy"],
}

_CACHED_ENV_META: Optional[Dict[str, Any]] = None


class EnvironmentMismatchError(RuntimeError):
    """Raised when the runtime environment differs from reference_environment.json or run manifest."""
    pass


def get_reference_environment_path() -> str:
    """Returns the reference environment path, respecting EXP_STATE_DIR if set."""
    custom_state = os.environ.get("EXP_STATE_DIR")
    if custom_state:
        return str(Path(custom_state) / "reference_environment.json")
    return str(EXP_DIR / "state" / "reference_environment.json")


REFERENCE_ENVIRONMENT_PATH = get_reference_environment_path()


def compute_file_sha256(path: str) -> str:
    """Computes SHA-256 hash of a file."""
    if not os.path.isfile(path):
        return f"file_not_found:{path}"
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def get_pip_freeze_raw() -> Tuple[str, str]:
    """
    Executes pip freeze, returns (raw_output_text, sha256_hash).
    """
    try:
        res = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        raw = res.stdout
        sha = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        return raw, sha
    except Exception as e:
        logger.error("Failed to compute pip freeze hash: %s", e)
        return "", f"unknown_{e}"


def compute_pip_freeze_sha256() -> str:
    """Computes SHA-256 hash of `pip freeze` output."""
    _, sha = get_pip_freeze_raw()
    return sha


def parse_pip_freeze(freeze_output: str) -> Dict[str, str]:
    """
    Parses pip freeze output into {package_name: version_or_spec}.
    """
    pkgs = {}
    for line in freeze_output.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "==" in line:
            parts = line.split("==", 1)
            pkgs[parts[0].lower()] = parts[1]
        elif " @ " in line:
            parts = line.split(" @ ", 1)
            pkgs[parts[0].lower()] = parts[1]
        else:
            pkgs[line.lower()] = "installed"
    return pkgs


def compute_package_diff(
    current_pkgs: Dict[str, str],
    reference_pkgs: Dict[str, str],
) -> Dict[str, Any]:
    """
    Computes package-level diff between current and reference pip packages.
    """
    added = {}
    removed = {}
    changed = {}

    all_names = sorted(set(current_pkgs.keys()).union(set(reference_pkgs.keys())))
    for name in all_names:
        curr_ver = current_pkgs.get(name)
        ref_ver = reference_pkgs.get(name)
        if ref_ver is None and curr_ver is not None:
            added[name] = curr_ver
        elif curr_ver is None and ref_ver is not None:
            removed[name] = ref_ver
        elif curr_ver != ref_ver:
            changed[name] = {"reference": ref_ver, "current": curr_ver}

    diff_lines = []
    for k in sorted(changed.keys()):
        v = changed[k]
        diff_lines.append(f"~ {k}: reference='{v['reference']}', current='{v['current']}'")
    for k in sorted(added.keys()):
        diff_lines.append(f"+ {k}=={added[k]}")
    for k in sorted(removed.keys()):
        diff_lines.append(f"- {k}=={removed[k]}")

    return {
        "added": added,
        "removed": removed,
        "changed": changed,
        "diff_lines": diff_lines,
        "total_differences": len(diff_lines),
    }


def get_package_versions() -> Dict[str, str]:
    """Inspects runtime versions for core dependencies."""
    py_ver = platform.python_version()

    # torch & CUDA build
    try:
        import torch
        torch_ver = getattr(torch, "__version__", "unknown")
        cuda_build = getattr(torch.version, "cuda", None)
        if cuda_build is None:
            cuda_build = "none"
    except Exception as e:
        torch_ver = f"unknown_{e}"
        cuda_build = f"unknown_{e}"

    # mani_skill
    try:
        import mani_skill
        maniskill_ver = getattr(mani_skill, "__version__", "unknown")
    except Exception as e:
        maniskill_ver = f"unknown_{e}"

    # sapien (physics and rendering engine)
    try:
        import sapien
        sapien_ver = getattr(sapien, "__version__", "unknown")
    except Exception as e:
        sapien_ver = f"unknown_{e}"

    # gymnasium
    try:
        import gymnasium
        gym_ver = getattr(gymnasium, "__version__", "unknown")
    except Exception as e:
        gym_ver = f"unknown_{e}"

    # numpy
    try:
        import numpy
        numpy_ver = getattr(numpy, "__version__", "unknown")
    except Exception as e:
        numpy_ver = f"unknown_{e}"

    return {
        "python_version": py_ver,
        "python_full_version": sys.version,
        "torch_version": torch_ver,
        "cuda_build": str(cuda_build),
        "mani_skill_version": maniskill_ver,
        "sapien_version": sapien_ver,
        "gymnasium_version": gym_ver,
        "numpy_version": numpy_ver,
    }


def collect_environment_metadata(
    requirements_path: Optional[str] = None,
    use_cache: bool = True,
) -> Dict[str, Any]:
    """
    Collects full environment metadata:
    - python_version, python_full_version
    - torch_version, cuda_build
    - mani_skill_version, sapien_version, gymnasium_version, numpy_version
    - requirements_frozen_sha256
    - pip_freeze_sha256
    - pip_packages
    """
    global _CACHED_ENV_META
    if use_cache and _CACHED_ENV_META is not None:
        return dict(_CACHED_ENV_META)

    req_path = requirements_path or str(EXP_DIR / "configs" / "requirements_frozen.txt")
    req_sha = compute_file_sha256(req_path)
    raw_pip, pip_sha = get_pip_freeze_raw()
    pip_pkgs = parse_pip_freeze(raw_pip)

    meta = get_package_versions()
    meta["requirements_frozen_sha256"] = req_sha
    meta["pip_freeze_sha256"] = pip_sha
    meta["pip_packages"] = pip_pkgs

    if use_cache:
        _CACHED_ENV_META = dict(meta)

    return meta


def _extract_field_value(data: Dict[str, Any], field: str) -> Optional[str]:
    """Extracts a version string from data using aliases."""
    aliases = FIELD_ALIASES.get(field, [field])
    for alias in aliases:
        if alias in data and data[alias] is not None:
            return str(data[alias]).strip()
    return None


def _normalize_version_for_cmp(field: str, val: str) -> str:
    """Normalizes version strings for robust comparison."""
    if field == "python_version":
        # Handle cases where sys.version full string was recorded:
        # e.g., '3.9.19 (main, May  6 2024, 19:43:03)...' -> '3.9.19'
        return val.split()[0]
    return val


def compare_environment_metadata(
    current_meta: Dict[str, Any],
    reference_meta: Dict[str, Any],
) -> List[str]:
    """
    Compares critical environment components between current and reference.
    Returns a list of human-readable mismatch strings.
    """
    mismatches = []
    for field in CRITICAL_ENVIRONMENT_FIELDS:
        curr_val = _extract_field_value(current_meta, field)
        ref_val = _extract_field_value(reference_meta, field)

        if ref_val is None:
            continue

        if curr_val is None:
            mismatches.append(f"{field}: missing in current environment (reference='{ref_val}')")
            continue

        curr_norm = _normalize_version_for_cmp(field, curr_val)
        ref_norm = _normalize_version_for_cmp(field, ref_val)

        if curr_norm != ref_norm:
            mismatches.append(f"{field}: reference='{ref_val}', current='{curr_val}'")

    return mismatches


def save_reference_environment(
    env_meta: Optional[Dict[str, Any]] = None,
    ref_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Saves environment metadata to state/reference_environment.json atomically with tmp + fsync.
    """
    target_path = ref_path or get_reference_environment_path()
    if env_meta is None:
        env_meta = collect_environment_metadata()

    os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)
    tmp_path = target_path + f".tmp.{os.getpid()}"
    with open(tmp_path, "w") as f:
        json.dump(env_meta, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, target_path)
    logger.info("Saved reference environment to: %s", target_path)
    return env_meta


def get_or_create_reference_environment(
    ref_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Loads reference environment if it exists; otherwise collects and saves it."""
    target_path = ref_path or get_reference_environment_path()
    if os.path.isfile(target_path):
        with open(target_path, "r") as f:
            return json.load(f)
    return save_reference_environment(ref_path=target_path)


def verify_environment(
    ref_path: Optional[str] = None,
    current_meta: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, Optional[str]]:
    """
    Verifies that the current environment matches reference_environment.json.
    - If reference does not exist, saves current_meta as reference (first real run or in EXP_STATE_DIR).
    - If reference exists, compares torch, CUDA build, mani_skill, gymnasium, numpy, and Python.
    - If any differ, returns (False, error_message).
    - Also warns if requirements_frozen.txt or pip freeze hashes differ.
    
    Returns:
        (is_match, error_message)
    """
    target_path = ref_path or get_reference_environment_path()
    if current_meta is None:
        current_meta = collect_environment_metadata()

    if not os.path.isfile(target_path):
        save_reference_environment(current_meta, ref_path=target_path)
        return True, None

    with open(target_path, "r") as f:
        ref_meta = json.load(f)

    mismatches = compare_environment_metadata(current_meta, ref_meta)
    if mismatches:
        err_msg = (
            "Environment version mismatch detected against reference!\n"
            f"Reference file: {target_path}\n"
            "The following environment components differ:\n"
            + "\n".join(f"  - {m}" for m in mismatches)
            + "\nRefusing to start or resume run with altered environment versions."
        )
        return False, err_msg

    # Informative warnings for requirements / pip freeze if different
    if "requirements_frozen_sha256" in ref_meta and current_meta.get("requirements_frozen_sha256") != ref_meta["requirements_frozen_sha256"]:
        logger.warning(
            "requirements_frozen.txt SHA-256 differs from reference (reference=%s, current=%s)",
            ref_meta["requirements_frozen_sha256"],
            current_meta.get("requirements_frozen_sha256"),
        )
    if "pip_freeze_sha256" in ref_meta and current_meta.get("pip_freeze_sha256") != ref_meta["pip_freeze_sha256"]:
        curr_pkgs = current_meta.get("pip_packages") or {}
        ref_pkgs = ref_meta.get("pip_packages") or {}
        diff_info = compute_package_diff(curr_pkgs, ref_pkgs)
        current_meta["pip_freeze_diff"] = diff_info

        logger.warning(
            "[Environment Warning] pip_freeze_sha256 differs from reference (ref=%s, cur=%s)!\n"
            "Package-level diff:\n%s",
            ref_meta["pip_freeze_sha256"],
            current_meta.get("pip_freeze_sha256"),
            "\n".join(f"  {line}" for line in diff_info["diff_lines"]) if diff_info["diff_lines"] else "  (no individual packages identified)",
        )

    return True, None
