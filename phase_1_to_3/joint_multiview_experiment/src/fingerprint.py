"""
ManiSkill Source Fingerprint Utility.

Per Protocol Requirements:
Computes a tamper-proof fingerprint of the parent ManiSkill source repository:
- git HEAD of /home/maniskill
- sha256 of `git -C /home/maniskill diff -- mani_skill`
(Uses strictly read-only git commands; never modifies /home/maniskill).

Stores fingerprint in each run's manifest.json, saves first run's fingerprint
to state/reference_fingerprint.json, and refuses to start/resume any run if altered.

Python 3.9 compatible only.
"""

import hashlib
import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

EXP_DIR = Path(__file__).resolve().parent.parent
PARENT_REPO_DIR = "/home/maniskill"


def get_reference_fingerprint_path() -> str:
    """Returns the reference fingerprint path, respecting EXP_STATE_DIR if set."""
    custom_state = os.environ.get("EXP_STATE_DIR")
    if custom_state:
        return str(Path(custom_state) / "reference_fingerprint.json")
    return str(EXP_DIR / "state" / "reference_fingerprint.json")


REFERENCE_FINGERPRINT_PATH = get_reference_fingerprint_path()


class ManiSkillFingerprintMismatchError(RuntimeError):
    """Raised when the ManiSkill source repository fingerprint differs from reference."""
    pass


def compute_maniskill_source_fingerprint(repo_dir: str = PARENT_REPO_DIR) -> Dict[str, str]:
    """
    Computes fingerprint of the parent ManiSkill repository:
    1. git HEAD of /home/maniskill
    2. sha256 of `git -C /home/maniskill diff -- mani_skill`
    
    Returns:
        Dict with keys:
        - "git_head": Git commit hash of /home/maniskill
        - "mani_skill_diff_sha256": SHA-256 of diff in mani_skill package
        - "fingerprint": Canonical combined fingerprint string f"{git_head}:{diff_sha256}"
    """
    try:
        res_head = subprocess.run(
            ["git", "-C", repo_dir, "rev-parse", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        git_head = res_head.stdout.strip()
    except Exception as e:
        logger.error("Failed to read git HEAD of %s: %s", repo_dir, e)
        git_head = f"unknown_{e}"

    try:
        res_diff = subprocess.run(
            ["git", "-C", repo_dir, "diff", "--", "mani_skill"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        diff_text = res_diff.stdout
        diff_sha256 = hashlib.sha256(diff_text.encode("utf-8")).hexdigest()
    except Exception as e:
        logger.error("Failed to run git diff in %s: %s", repo_dir, e)
        diff_sha256 = f"unknown_{e}"

    composite = f"{git_head}:{diff_sha256}"
    return {
        "git_head": git_head,
        "mani_skill_diff_sha256": diff_sha256,
        "fingerprint": composite,
    }


def save_reference_fingerprint(
    fingerprint: Optional[Dict[str, str]] = None,
    ref_path: Optional[str] = None,
) -> Dict[str, str]:
    """
    Saves the reference fingerprint to state/reference_fingerprint.json if not present.
    """
    target_path = ref_path or get_reference_fingerprint_path()
    if fingerprint is None:
        fingerprint = compute_maniskill_source_fingerprint()

    os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)
    tmp_path = target_path + f".tmp.{os.getpid()}"
    with open(tmp_path, "w") as f:
        json.dump(fingerprint, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, target_path)
    logger.info("Saved reference ManiSkill fingerprint to: %s (%s)", target_path, fingerprint["fingerprint"])
    return fingerprint


def get_or_create_reference_fingerprint(
    ref_path: Optional[str] = None,
) -> Dict[str, str]:
    """
    Loads state/reference_fingerprint.json if it exists; otherwise computes and saves it.
    """
    target_path = ref_path or get_reference_fingerprint_path()
    if os.path.isfile(target_path):
        with open(target_path, "r") as f:
            return json.load(f)
    return save_reference_fingerprint(ref_path=target_path)


def verify_maniskill_source_fingerprint(
    ref_path: Optional[str] = None,
    current_fp: Optional[Dict[str, str]] = None,
) -> Tuple[bool, Optional[str]]:
    """
    Verifies that current ManiSkill source matches state/reference_fingerprint.json.
    
    Returns:
        (is_match, error_message)
    """
    target_path = ref_path or get_reference_fingerprint_path()
    if current_fp is None:
        current_fp = compute_maniskill_source_fingerprint()

    if not os.path.isfile(target_path):
        save_reference_fingerprint(current_fp, ref_path=target_path)
        return True, None

    with open(target_path, "r") as f:
        ref_fp = json.load(f)

    if current_fp["fingerprint"] != ref_fp["fingerprint"]:
        msg = (
            f"ManiSkill source fingerprint mismatch!\n"
            f"  Reference fingerprint: {ref_fp['fingerprint']}\n"
            f"  Current fingerprint:   {current_fp['fingerprint']}\n"
            f"  (HEAD: ref={ref_fp.get('git_head')}, cur={current_fp.get('git_head')})\n"
            f"  (diff SHA: ref={ref_fp.get('mani_skill_diff_sha256')}, cur={current_fp.get('mani_skill_diff_sha256')})\n"
            f"Refusing to start or resume any run with altered ManiSkill source repository."
        )
        return False, msg

    return True, None
