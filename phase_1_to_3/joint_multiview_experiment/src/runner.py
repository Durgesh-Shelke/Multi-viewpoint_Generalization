"""
Multi-Run Orchestrator for Joint Multi-View PickCube Training.

Per SPEC Section 2.4, 2.5:
- Manages queue of runs from configs/runs.yaml.
- Tracks state in state/queue.json: pending | running | paused | done | failed.
- Enforces single active instance via state/lock with PID; detects stale locks.
- Refuses to start any run whose total_timesteps is null.
- Enforces ManiSkill source repository fingerprint integrity against state/reference_fingerprint.json.
- Resumes interrupted/paused runs and proceeds sequentially without re-running completed runs.

Python 3.9 compatible only.
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.environment import (
    EnvironmentMismatchError,
    collect_environment_metadata,
    get_reference_environment_path,
    verify_environment,
)
from src.fingerprint import (
    ManiSkillFingerprintMismatchError,
    compute_maniskill_source_fingerprint,
    get_or_create_reference_fingerprint,
    verify_maniskill_source_fingerprint,
    REFERENCE_FINGERPRINT_PATH,
)

logger = logging.getLogger("runner")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

EXP_DIR = Path(__file__).resolve().parent.parent


def get_state_dir() -> Path:
    custom_state = os.environ.get("EXP_STATE_DIR")
    if custom_state:
        return Path(custom_state)
    return EXP_DIR / "state"


def get_runs_dir() -> Path:
    """Returns the runs directory, respecting EXP_RUNS_DIR if configured."""
    custom_runs = os.environ.get("EXP_RUNS_DIR")
    if custom_runs:
        return Path(custom_runs)
    return EXP_DIR / "runs"


def resolve_output_dir(raw_output_dir: Optional[str]) -> Optional[str]:
    """Resolves output_dir, re-rooting under EXP_RUNS_DIR if set."""
    if not raw_output_dir:
        return raw_output_dir
    custom_runs = os.environ.get("EXP_RUNS_DIR")
    if custom_runs:
        rel_name = raw_output_dir
        if rel_name.startswith("runs/"):
            rel_name = rel_name[len("runs/"):]
        return str(Path(custom_runs) / rel_name)
    return raw_output_dir


STATE_DIR = get_state_dir()
LOCK_PATH = str(STATE_DIR / "lock")
QUEUE_PATH = str(STATE_DIR / "queue.json")
DEFAULT_RUNS_CONFIG = str(EXP_DIR / "configs" / "runs.yaml")
DEFAULT_BASE_CONFIG = str(EXP_DIR / "configs" / "ppo_frozen.yaml")


class RunnerLockedError(RuntimeError):
    """Raised when another runner instance is already active."""
    pass


class LockManager:
    """
    Manages process lock in state/lock with PID recording and stale lock detection.
    """
    def __init__(self, lock_path: Optional[str] = None):
        self.lock_path = lock_path if lock_path else str(get_state_dir() / "lock")
        self.acquired = False

    def is_pid_alive(self, pid: int) -> bool:
        """Checks if a process ID is currently running on the system (not a zombie)."""
        if pid <= 0:
            return False
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True

        # Check if process is a zombie (terminated but not reaped by parent)
        proc_status = Path(f"/proc/{pid}/status")
        if proc_status.is_file():
            try:
                with open(proc_status, "r") as f:
                    for line in f:
                        if line.startswith("State:"):
                            if "Z" in line.split()[1]:
                                return False
                            break
            except (OSError, IndexError):
                pass
        return True

    def acquire(self) -> None:
        """Acquires the lockfile. Cleans stale locks if PID no longer exists."""
        os.makedirs(os.path.dirname(os.path.abspath(self.lock_path)), exist_ok=True)
        if os.path.exists(self.lock_path):
            try:
                with open(self.lock_path, "r") as f:
                    data = json.load(f)
                active_pid = data.get("pid")
                if active_pid is not None and self.is_pid_alive(active_pid):
                    raise RunnerLockedError(
                        f"Another runner instance is already active (PID {active_pid}). "
                        f"Lockfile: {self.lock_path}"
                    )
                else:
                    logger.warning("Detected stale lockfile from dead PID %s. Removing lock.", active_pid)
                    os.remove(self.lock_path)
            except (json.JSONDecodeError, ValueError):
                os.remove(self.lock_path)

        lock_data = {
            "pid": os.getpid(),
            "acquired_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        }
        tmp_path = self.lock_path + f".tmp.{os.getpid()}"
        with open(tmp_path, "w") as f:
            json.dump(lock_data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, self.lock_path)
        self.acquired = True

    def release(self) -> None:
        """Releases the lockfile if owned by this process."""
        if self.acquired and os.path.exists(self.lock_path):
            try:
                with open(self.lock_path, "r") as f:
                    data = json.load(f)
                if data.get("pid") == os.getpid():
                    os.remove(self.lock_path)
            except Exception:
                pass
            self.acquired = False

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


class Runner:
    """
    Orchestrates sequential execution of training runs in the experimental queue.
    """
    def __init__(
        self,
        runs_config_path: str = DEFAULT_RUNS_CONFIG,
        base_config_path: str = DEFAULT_BASE_CONFIG,
        state_dir: Optional[str] = None,
    ):
        self.runs_config_path = runs_config_path
        self.base_config_path = base_config_path
        self.state_dir = Path(state_dir) if state_dir else get_state_dir()
        self.queue_path = self.state_dir / "queue.json"
        self.lock_path = self.state_dir / "lock"
        self.ref_fp_path = self.state_dir / "reference_fingerprint.json"
        self.ref_env_path = self.state_dir / "reference_environment.json"

    def check_fingerprint_integrity(self) -> Dict[str, str]:
        """
        Ensures reference fingerprint exists (saving it on first run),
        and asserts current ManiSkill source matches reference.
        """
        current_fp = compute_maniskill_source_fingerprint()
        is_match, err = verify_maniskill_source_fingerprint(
            ref_path=str(self.ref_fp_path),
            current_fp=current_fp,
        )
        if not is_match:
            logger.error(err)
            raise ManiSkillFingerprintMismatchError(err)
        return current_fp

    def check_environment_integrity(self) -> Dict[str, Any]:
        """
        Ensures reference environment exists (saving it on first real run),
        and asserts current environment matches reference.
        """
        current_env = collect_environment_metadata()
        is_match, err = verify_environment(
            ref_path=str(self.ref_env_path),
            current_meta=current_env,
        )
        if not is_match:
            logger.error(err)
            raise EnvironmentMismatchError(err)
        return current_env

    def load_or_init_queue(self) -> Dict[str, Any]:
        """Loads or creates state/queue.json from configs/runs.yaml."""
        import yaml
        with open(self.runs_config_path, "r") as f:
            runs_yaml = yaml.safe_load(f)
        runs_dict = runs_yaml.get("runs", {})

        queue = {}
        if os.path.isfile(str(self.queue_path)):
            try:
                with open(str(self.queue_path), "r") as f:
                    queue = json.load(f)
            except Exception:
                queue = {}

        updated = False
        for run_id, run_info in runs_dict.items():
            if run_id not in queue:
                queue[run_id] = {
                    "status": "pending",
                    "modality": run_info.get("modality"),
                    "seed": run_info.get("seed"),
                    "total_timesteps": run_info.get("total_timesteps"),
                    "output_dir": run_info.get("output_dir"),
                    "config": run_info.get("config"),
                    "final_eval_groups": run_info.get("final_eval_groups"),
                    "is_pilot": run_info.get("is_pilot"),
                    "description": run_info.get("description", ""),
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
                }
                updated = True
            else:
                if queue[run_id].get("total_timesteps") != run_info.get("total_timesteps"):
                    queue[run_id]["total_timesteps"] = run_info.get("total_timesteps")
                    updated = True
                if "final_eval_groups" in run_info and queue[run_id].get("final_eval_groups") != run_info.get("final_eval_groups"):
                    queue[run_id]["final_eval_groups"] = run_info.get("final_eval_groups")
                    updated = True
                if "is_pilot" in run_info and queue[run_id].get("is_pilot") != run_info.get("is_pilot"):
                    queue[run_id]["is_pilot"] = run_info.get("is_pilot")
                    updated = True
                    queue[run_id]["total_timesteps"] = run_info.get("total_timesteps")
                    updated = True
                if run_info.get("config") and queue[run_id].get("config") != run_info.get("config"):
                    queue[run_id]["config"] = run_info.get("config")
                    updated = True

        if updated or not os.path.isfile(str(self.queue_path)):
            self.save_queue(queue)

        return queue

    def save_queue(self, queue: Dict[str, Any]) -> None:
        """Atomically saves state/queue.json."""
        os.makedirs(os.path.dirname(os.path.abspath(str(self.queue_path))), exist_ok=True)
        tmp = str(self.queue_path) + f".tmp.{os.getpid()}"
        with open(tmp, "w") as f:
            json.dump(queue, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, str(self.queue_path))

    def run_single(self, run_id: str) -> None:
        """Runs a single run from the queue."""
        queue = self.load_or_init_queue()
        if run_id not in queue:
            raise KeyError(f"Run '{run_id}' not found in queue. Available: {list(queue.keys())}")

        run_info = queue[run_id]

        # 1. Budget check
        total_steps = run_info.get("total_timesteps")
        if total_steps is None or int(total_steps) <= 0:
            raise ValueError(
                f"Run '{run_id}' has total_timesteps: {total_steps}. "
                f"The budget B has not been frozen yet; refusing to start unbudgeted main-grid run."
            )

        # 2. Check if already done
        if run_info.get("status") == "done":
            logger.info("Run '%s' is already marked 'done'. Skipping (completed runs are never repeated).", run_id)
            return

        out_dir = resolve_output_dir(run_info.get("output_dir"))
        final_summary_file = Path(out_dir) / "final_eval" / "final_eval_summary.json"
        if final_summary_file.is_file():
            try:
                with open(final_summary_file, "r") as f:
                    summ = json.load(f)
                if summ.get("status") == "completed":
                    logger.info("Run '%s' already has completed final_eval_summary.json. Marking done.", run_id)
                    queue[run_id]["status"] = "done"
                    queue[run_id]["completed_at"] = summ.get("completed_at", time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()))
                    self.save_queue(queue)
                    return
            except Exception:
                pass

        # 3. Fingerprint & environment integrity check
        self.check_fingerprint_integrity()
        self.check_environment_integrity()

        # Check for active alert
        alert_file = self.state_dir / "ALERT.txt"
        if alert_file.is_file():
            logger.error(
                "Runner halted: state/ALERT.txt exists. Early sanity alert triggered. "
                "Review alert and resume explicitly via 'python src/ctl.py resume'."
            )
            queue[run_id]["status"] = "paused"
            self.save_queue(queue)
            return

        # 4. Mark running
        queue[run_id]["status"] = "running"
        queue[run_id]["started_at"] = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
        self.save_queue(queue)

        # 5. Launch run via train_ppo
        from src.train_ppo import train_ppo, load_yaml_config
        from src.evaluate import run_final_evaluation
        base_cfg_path = run_info.get("config") or self.base_config_path
        base_cfg = load_yaml_config(base_cfg_path)

        # Enforce pilot restrictions on final_eval_groups (SPEC Section 4.3)
        is_pilot = run_id.startswith("pilot_") or run_info.get("is_pilot", False)
        eval_groups = run_info.get("final_eval_groups", ["train", "val"] if is_pilot else "all")
        if is_pilot:
            base_cfg["is_pilot"] = True
            check_groups = [eval_groups] if isinstance(eval_groups, str) else list(eval_groups)
            for disallowed in ["test_interp", "test_extrap", "fresh"]:
                if disallowed in check_groups or "all" in check_groups:
                    raise ValueError(
                        f"Protocol violation: pilot run '{run_id}' cannot evaluate '{disallowed}' or 'all'. "
                        "SPEC Section 4.3 strictly requires pilot evaluations on train + val views only."
                    )

        pause_paths = [
            str(self.state_dir / "PAUSE"),
            str(EXP_DIR / "PAUSE"),
            os.path.join(out_dir, "PAUSE"),
        ]

        try:
            train_result = train_ppo(
                config=base_cfg,
                output_dir=out_dir,
                modality=run_info.get("modality"),
                seed=run_info.get("seed"),
                total_timesteps=int(total_steps),
                ref_fingerprint_path=str(self.ref_fp_path),
                ref_environment_path=str(self.ref_env_path),
                run_name=run_id,
            )

            if train_result.get("status") == "completed":
                # Final evaluation after training completion (SPEC Section 5)
                final_ckpt = train_result.get("checkpoint_path")
                if not final_ckpt or not os.path.isfile(final_ckpt):
                    ckpt_dir = Path(out_dir) / "checkpoints"
                    final_candidates = sorted(ckpt_dir.glob("ckpt_final_*.pt"))
                    if final_candidates:
                        final_ckpt = str(final_candidates[-1])
                    else:
                        from src.checkpoint import get_newest_checkpoint
                        newest = get_newest_checkpoint(str(ckpt_dir))
                        final_ckpt = newest[0] if newest else None

                if not final_ckpt:
                    raise FileNotFoundError(f"Training completed for '{run_id}', but no checkpoint found in {out_dir}/checkpoints")

                logger.info("Training completed for '%s'. Starting final evaluation on %s...", run_id, final_ckpt)
                final_eval_res = run_final_evaluation(
                    checkpoint_path=final_ckpt,
                    output_dir=out_dir,
                    modality=run_info.get("modality"),
                    config=base_cfg,
                    pause_paths=pause_paths,
                    eval_groups=eval_groups,
                )

                if final_eval_res.get("status") == "completed":
                    queue[run_id]["status"] = "done"
                    queue[run_id]["completed_at"] = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
                    logger.info("Run '%s' fully completed (training + final evaluation).", run_id)
                else:
                    queue[run_id]["status"] = "paused"
                    logger.info("Run '%s' paused during final evaluation.", run_id)
            else:
                queue[run_id]["status"] = "paused"
                if alert_file.is_file():
                    logger.error("Run '%s' paused due to early sanity alert.", run_id)
                else:
                    logger.info("Run '%s' paused during training.", run_id)
        except SystemExit as e:
            if e.code == 0:
                queue[run_id]["status"] = "paused"
            else:
                queue[run_id]["status"] = "failed"
            self.save_queue(queue)
            raise
        except Exception as e:
            queue[run_id]["status"] = "failed"
            queue[run_id]["error"] = str(e)
            self.save_queue(queue)
            raise

        self.save_queue(queue)

    def run_all(self) -> None:
        """Executes all pending or paused runs in order."""
        lock = LockManager(lock_path=str(self.lock_path))
        with lock:
            self.check_fingerprint_integrity()
            queue = self.load_or_init_queue()

            for run_id, run_info in queue.items():
                status = run_info.get("status")
                if status == "done":
                    continue
                if run_info.get("total_timesteps") is None:
                    logger.info("Skipping unbudgeted run '%s' (total_timesteps is null).", run_id)
                    continue

                logger.info("Runner executing run '%s' (status: %s, steps: %s)...", run_id, status, run_info.get("total_timesteps"))
                self.run_single(run_id)


def parse_args() -> argparse.Namespace:
    """Parses CLI arguments for the multi-run runner."""
    parser = argparse.ArgumentParser(description="Multi-Run Orchestrator for Joint Multi-View PickCube")
    parser.add_argument(
        "--runs-file", "--runs_file", "--runs-config", "--runs_config",
        dest="runs_config",
        type=str,
        default=DEFAULT_RUNS_CONFIG,
        help="Path to runs configuration YAML (e.g. configs/runs.yaml or configs/tiny_runs.yaml)",
    )
    parser.add_argument(
        "--base-config", "--base_config",
        dest="base_config",
        type=str,
        default=DEFAULT_BASE_CONFIG,
        help="Path to base frozen config",
    )
    parser.add_argument("--run-id", type=str, default=None, help="Execute a specific run ID")
    parser.add_argument("--status", action="store_true", help="Print queue status and exit")
    return parser.parse_args()


def print_status(runner: Runner) -> None:
    """Prints formatted queue status table."""
    queue = runner.load_or_init_queue()
    print("\n" + "=" * 80)
    print(f"{'Run ID':<18} | {'Modality':<8} | {'Seed':<5} | {'Budget':<10} | {'Status':<10} | {'Output Dir':<22}")
    print("-" * 80)
    for rid, info in queue.items():
        budget_str = f"{info.get('total_timesteps'):,}" if info.get('total_timesteps') else "null"
        status_str = info.get("status", "unknown")
        print(f"{rid:<18} | {str(info.get('modality')):<8} | {str(info.get('seed')):<5} | {budget_str:<10} | {status_str:<10} | {str(info.get('output_dir')):<22}")
    print("=" * 80 + "\n")


def main() -> None:
    args = parse_args()
    runner = Runner(
        runs_config_path=args.runs_config,
        base_config_path=args.base_config,
    )

    if args.status:
        print_status(runner)
        return

    if args.run_id:
        lock = LockManager(lock_path=str(runner.lock_path))
        with lock:
            runner.run_single(args.run_id)
    else:
        runner.run_all()


if __name__ == "__main__":
    main()
