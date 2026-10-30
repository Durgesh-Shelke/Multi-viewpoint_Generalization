"""
CLI Control Utility for Joint Multi-View PickCube Training.

Per SPEC Section 2.1:
- pause: Creates state/PAUSE, waits for the training process to exit, prints confirmation.
- resume: Deletes stale PAUSE files in state/, experiment root, and run output dirs,
          launches runner.py detached with nohup/detached session and nice -n 10,
          writes PID to state/runner.pid.
- status: Prints current run, step/budget, % done, SPS, ETA from active time,
          last checkpoint age, and GPU memory via nvidia-smi.

Python 3.9 compatible only.
"""

import argparse
import csv
import datetime
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("ctl")

EXP_DIR = Path(__file__).resolve().parent.parent
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))


def get_state_dir() -> Path:
    """Returns the active state directory, respecting EXP_STATE_DIR if configured."""
    custom_state = os.environ.get("EXP_STATE_DIR")
    if custom_state:
        return Path(custom_state)
    return EXP_DIR / "state"


def get_runs_dir() -> Path:
    """Returns the active runs directory, respecting EXP_RUNS_DIR if configured."""
    custom_runs = os.environ.get("EXP_RUNS_DIR")
    if custom_runs:
        return Path(custom_runs)
    return EXP_DIR / "runs"


def resolve_output_dir(raw_output_dir: Optional[str]) -> Optional[Path]:
    """Resolves output_dir, respecting EXP_RUNS_DIR if configured."""
    if not raw_output_dir:
        return None
    custom_runs = os.environ.get("EXP_RUNS_DIR")
    if custom_runs:
        rel_name = raw_output_dir
        if rel_name.startswith("runs/"):
            rel_name = rel_name[len("runs/"):]
        return Path(custom_runs) / rel_name
    return EXP_DIR / raw_output_dir if not os.path.isabs(raw_output_dir) else Path(raw_output_dir)


def is_pid_alive(pid: int) -> bool:
    """Checks if a process with the given PID is currently active and running (not a zombie)."""
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
                        state_val = line.split()[1]
                        if "Z" in state_val:
                            return False
                        break
        except (OSError, IndexError):
            pass

    return True


def get_active_runner_pid(state_dir: Optional[Path] = None) -> Optional[int]:
    """Inspects state/lock and state/runner.pid to find active runner PID."""
    s_dir = state_dir or get_state_dir()

    # 1. Check state/lock
    lock_file = s_dir / "lock"
    if lock_file.is_file():
        try:
            with open(lock_file, "r") as f:
                data = json.load(f)
            pid = data.get("pid")
            if pid and is_pid_alive(int(pid)):
                return int(pid)
        except Exception:
            pass

    # 2. Check state/runner.pid
    pid_file = s_dir / "runner.pid"
    if pid_file.is_file():
        try:
            with open(pid_file, "r") as f:
                pid_str = f.read().strip()
            if pid_str:
                pid = int(pid_str)
                if is_pid_alive(pid):
                    return pid
        except Exception:
            pass

    return None


def format_duration(seconds: float) -> str:
    """Formats seconds into human-readable H:MM:SS or M:SS."""
    sec = max(0, int(seconds))
    return str(datetime.timedelta(seconds=sec))


def format_age(seconds: float) -> str:
    """Formats age in seconds into 'Xs ago', 'Xm Ys ago', or 'Xh Ym ago'."""
    sec = max(0, int(seconds))
    if sec < 60:
        return f"{sec}s ago"
    elif sec < 3600:
        mins = sec // 60
        s = sec % 60
        return f"{mins}m {s}s ago"
    else:
        hrs = sec // 3600
        mins = (sec % 3600) // 60
        return f"{hrs}h {mins}m ago"


def get_gpu_memory_info() -> Dict[str, Any]:
    """Queries GPU memory and utilization via nvidia-smi."""
    info = {
        "used_mib": None,
        "total_mib": None,
        "util_pct": None,
        "raw": "N/A",
    }
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
            timeout=5.0,
        )
        line = res.stdout.strip().split("\n")[0]
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            used = int(parts[0])
            total = int(parts[1])
            util = int(parts[2])
            pct = (used / total * 100.0) if total > 0 else 0.0
            info["used_mib"] = used
            info["total_mib"] = total
            info["util_pct"] = util
            info["raw"] = f"{used:,} MiB / {total:,} MiB ({pct:.1f}% VRAM used, {util}% GPU util)"
    except Exception as e:
        info["raw"] = f"Unavailable ({e})"
    return info


def get_frozen_code_info(exp_dir: Optional[Path] = None) -> Tuple[Optional[Path], Optional[str]]:
    """
    Returns (frozen_dir, frozen_commit) if a frozen code directory exists.
    Checks EXP_FROZEN_CODE_DIR or FROZEN_CODE_DIR environment variables,
    then defaults to <exp_dir>/frozen_code/v2-freeze (falling back to pre-pilot-freeze).
    """
    e_dir = exp_dir or EXP_DIR
    custom_frozen = os.environ.get("EXP_FROZEN_CODE_DIR") or os.environ.get("FROZEN_CODE_DIR")
    if custom_frozen:
        f_dir = Path(custom_frozen)
        if f_dir.is_dir():
            commit_file = f_dir / "COMMIT_HASH"
            f_commit = commit_file.read_text().strip() if commit_file.is_file() else os.environ.get("FROZEN_COMMIT")
            return f_dir, f_commit

    for tag in ["v2-freeze", "pre-pilot-freeze"]:
        candidate = e_dir / "frozen_code" / tag
        if candidate.is_dir():
            commit_file = candidate / "COMMIT_HASH"
            f_commit = commit_file.read_text().strip() if commit_file.is_file() else os.environ.get("FROZEN_COMMIT")
            return candidate, f_commit

    return None, None


def pause(state_dir: Optional[Path] = None, timeout: float = 60.0, manual: bool = True) -> bool:
    """
    Creates state/PAUSE, waits for any active training process to exit,
    and prints confirmation. If manual is True (the default for user CLI invocations),
    also creates state/MANUAL_PAUSE which prevents automated supervisor resumption.
    """
    s_dir = state_dir or get_state_dir()
    os.makedirs(s_dir, exist_ok=True)

    pause_path = s_dir / "PAUSE"
    tmp_pause = str(pause_path) + f".tmp.{os.getpid()}"
    with open(tmp_pause, "w") as f:
        f.write(f"PAUSE requested by ctl.py at {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_pause, str(pause_path))

    if manual:
        manual_path = s_dir / "MANUAL_PAUSE"
        tmp_manual = str(manual_path) + f".tmp.{os.getpid()}"
        with open(tmp_manual, "w") as f:
            f.write(f"MANUAL_PAUSE created by user via ctl.py at {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_manual, str(manual_path))
        print(f"[ctl pause] Created manual pause marker: {manual_path}")

    # Also place at experiment root as secondary trigger if not in isolated state dir
    if not os.environ.get("EXP_STATE_DIR"):
        root_pause = EXP_DIR / "PAUSE"
        try:
            with open(root_pause, "w") as f:
                f.write("PAUSE\n")
        except Exception:
            pass

    active_pid = get_active_runner_pid(s_dir)
    if active_pid is not None:
        print(f"[ctl pause] Created {pause_path}. Waiting for process {active_pid} to cleanly finish iteration and exit...")
        start_t = time.time()
        while is_pid_alive(active_pid):
            if time.time() - start_t > timeout:
                print(f"[ctl pause] Warning: Process {active_pid} did not exit within {timeout}s.")
                return False
            time.sleep(0.5)

        print(f"[ctl pause] Confirmed: Training process (PID {active_pid}) has cleanly exited and paused.")
        print("[ctl pause] GPU memory has been fully released.")
        return True
    else:
        print(f"[ctl pause] Created {pause_path}. No active training process currently running.")
        return True


def resume(
    state_dir: Optional[Path] = None,
    runs_dir: Optional[Path] = None,
    runs_file: Optional[str] = None,
) -> bool:
    """
    Deletes stale PAUSE files, launches runner.py detached with nohup and nice -n 10,
    and writes the PID to state/runner.pid.
    """
    s_dir = state_dir or get_state_dir()
    r_dir = runs_dir or get_runs_dir()
    os.makedirs(s_dir, exist_ok=True)

    # 1. Clean stale PAUSE files
    stale_files: List[Path] = []
    state_pause = s_dir / "PAUSE"
    if state_pause.exists():
        stale_files.append(state_pause)

    # Also clear manual pause marker if present
    manual_pause = s_dir / "MANUAL_PAUSE"
    if manual_pause.is_file():
        try:
            manual_pause.unlink()
            print(f"[ctl resume] Cleared manual pause marker: {manual_pause}")
        except Exception as e:
            logger.warning("Could not remove %s: %s", manual_pause, e)

    # Also archive any active alert on resume
    alert_file = s_dir / "ALERT.txt"
    if alert_file.is_file():
        logger.info("Resuming run with active alert. Archiving %s -> ALERT.txt.acknowledged", alert_file)
        try:
            alert_file.rename(s_dir / "ALERT.txt.acknowledged")
        except Exception:
            try:
                alert_file.unlink()
            except Exception:
                pass

    if not os.environ.get("EXP_STATE_DIR"):
        root_pause = EXP_DIR / "PAUSE"
        if root_pause.exists():
            stale_files.append(root_pause)

    if r_dir.is_dir():
        for run_sub in r_dir.iterdir():
            if run_sub.is_dir():
                sub_pause = run_sub / "PAUSE"
                if sub_pause.exists():
                    stale_files.append(sub_pause)

    for sf in stale_files:
        try:
            sf.unlink()
            print(f"[ctl resume] Removed stale pause file: {sf}")
        except Exception as e:
            logger.warning("Could not remove %s: %s", sf, e)

    # 2. Check if runner is already active
    active_pid = get_active_runner_pid(s_dir)
    if active_pid is not None:
        print(f"[ctl resume] Runner is already active (PID {active_pid}). Resume aborted.")
        return False

    # 3. Clean any stale lock if present
    lock_file = s_dir / "lock"
    if lock_file.exists():
        try:
            lock_file.unlink()
        except Exception:
            pass

    # 4. Launch runner.py detached with nohup & nice -n 10
    runner_log_path = s_dir / "runner.log"
    log_fd = open(runner_log_path, "a")

    cmd = ["nice", "-n", "10", sys.executable, "-m", "src.runner"]
    if runs_file:
        cmd.extend(["--runs-file", str(runs_file)])

    launch_env = os.environ.copy()
    launch_env["EXP_STATE_DIR"] = str(s_dir)
    launch_env["EXP_RUNS_DIR"] = str(r_dir)
    f_dir, f_commit = get_frozen_code_info(EXP_DIR)
    if f_dir is not None and f_dir.is_dir():
        launch_env["PYTHONPATH"] = f"{f_dir}:{launch_env.get('PYTHONPATH', '')}"
        launch_env["FROZEN_CODE_DIR"] = str(f_dir)
        if f_commit:
            launch_env["FROZEN_COMMIT"] = f_commit
        launch_env["EXP_ROOT_DIR"] = str(EXP_DIR)
        print(f"[ctl resume] Using frozen code export: {f_dir} (commit {f_commit[:8] if f_commit else 'unknown'})")

    proc = subprocess.Popen(
        cmd,
        cwd=str(EXP_DIR),
        stdout=log_fd,
        stderr=subprocess.STDOUT,
        start_new_session=True,  # detached from terminal session (nohup)
        env=launch_env,
    )

    pid_path = s_dir / "runner.pid"
    with open(pid_path, "w") as f:
        f.write(str(proc.pid) + "\n")
        f.flush()
        os.fsync(f.fileno())

    print(f"[ctl resume] Launched runner detached (PID {proc.pid}) with nice -n 10.")
    if runs_file:
        print(f"[ctl resume] Runs configuration: {runs_file}")
    print(f"[ctl resume] Runner log: {runner_log_path}")
    print(f"[ctl resume] PID saved to: {pid_path}")
    return True


def status(
    state_dir: Optional[Path] = None,
    runs_dir: Optional[Path] = None,
    runs_file: Optional[str] = None,
) -> None:
    """
    Displays formatted status:
    Current run, step/budget, % done, SPS, ETA from active time, last checkpoint age, GPU memory.
    """
    s_dir = state_dir or get_state_dir()
    r_dir = runs_dir or get_runs_dir()
    active_pid = get_active_runner_pid(s_dir)
    gpu_info = get_gpu_memory_info()

    # Read queue state if present
    queue_file = s_dir / "queue.json"
    queue: Dict[str, Any] = {}
    if queue_file.is_file():
        try:
            with open(queue_file, "r") as f:
                queue = json.load(f)
        except Exception:
            pass

    # Fallback to reading runs_file if queue not yet created
    if not queue and runs_file and os.path.isfile(runs_file):
        try:
            import yaml
            with open(runs_file, "r") as f:
                r_yaml = yaml.safe_load(f)
            for rid, rinfo in r_yaml.get("runs", {}).items():
                queue[rid] = {
                    "status": "pending",
                    "modality": rinfo.get("modality"),
                    "seed": rinfo.get("seed"),
                    "total_timesteps": rinfo.get("total_timesteps"),
                    "output_dir": rinfo.get("output_dir"),
                }
        except Exception:
            pass

    # Identify active or most relevant run
    current_run_id = None
    current_run_info: Dict[str, Any] = {}

    # Look for running first
    for rid, rinfo in queue.items():
        if rinfo.get("status") == "running":
            current_run_id = rid
            current_run_info = rinfo
            break

    # If none running, look for paused
    if current_run_id is None:
        for rid, rinfo in queue.items():
            if rinfo.get("status") == "paused":
                current_run_id = rid
                current_run_info = rinfo
                break

    # If none, take the first run from queue
    if current_run_id is None and queue:
        current_run_id = list(queue.keys())[0]
        current_run_info = queue[current_run_id]

    run_status_str = current_run_info.get("status", "idle")
    if active_pid is not None and run_status_str != "running":
        run_status_str = f"running (PID {active_pid})"
    elif active_pid is None and run_status_str == "running":
        run_status_str = "paused / stopped"

    # Step, budget, SPS, and active time from progress.csv
    out_dir_path = current_run_info.get("output_dir")
    resolved_out_dir = resolve_output_dir(out_dir_path)

    global_step = 0
    active_time_s = 0.0
    sps = 0.0
    mean_return = 0.0
    last_ckpt_age_str = "None"
    csv_rows = []

    if resolved_out_dir and resolved_out_dir.is_dir():
        csv_file = resolved_out_dir / "progress.csv"
        if csv_file.is_file():
            try:
                with open(csv_file, "r", newline="") as f:
                    reader = csv.DictReader(f)
                    for r in reader:
                        csv_rows.append(r)
                if csv_rows:
                    last_row = csv_rows[-1]
                    global_step = int(last_row.get("global_step", 0))
                    active_time_s = float(last_row.get("active_time_s", 0.0))
                    sps = float(last_row.get("sps", 0.0))
                    mean_return = float(last_row.get("mean_return", 0.0))
            except Exception:
                pass

        # Inspect checkpoint age
        ckpt_dir = resolved_out_dir / "checkpoints"
        if ckpt_dir.is_dir():
            latest_pt = ckpt_dir / "latest.pt"
            if latest_pt.exists():
                try:
                    mtime = os.path.getmtime(latest_pt)
                    last_ckpt_age_str = format_age(time.time() - mtime)
                except Exception:
                    pass
            else:
                pts = list(ckpt_dir.glob("ckpt_step_*.pt"))
                if pts:
                    newest = max(pts, key=lambda p: p.stat().st_mtime)
                    mtime = newest.stat().st_mtime
                    last_ckpt_age_str = format_age(time.time() - mtime)

    budget = current_run_info.get("total_timesteps")
    if budget is not None and isinstance(budget, (int, float)) and budget > 0:
        budget_val = int(budget)
        budget_str = f"{budget_val:,}"
        pct_done = (global_step / budget_val * 100.0)
        if sps > 0 and global_step < budget_val:
            remaining_steps = budget_val - global_step
            eta_seconds = remaining_steps / sps
            eta_str = format_duration(eta_seconds)
        elif global_step >= budget_val:
            eta_str = "00:00:00 (Completed)"
        else:
            eta_str = "N/A"
    else:
        budget_str = "budget not set"
        pct_done = 0.0
        eta_str = "N/A"

    alert_file = s_dir / "ALERT.txt"
    alert_banner = ""
    if alert_file.is_file():
        alert_content = alert_file.read_text().strip()
        alert_banner = (
            "\n" + "!" * 70 + "\n"
            " ACTIVE EARLY SANITY ALERT DETECTED:\n"
            f"{alert_content}\n"
            + "!" * 70 + "\n"
        )

    watchdog_lock = s_dir / "watchdog.lock"
    watchdog_status_str = "INACTIVE"
    if watchdog_lock.is_file():
        try:
            with open(watchdog_lock, "r") as f:
                w_data = json.load(f)
            w_pid = w_data.get("pid")
            if w_pid and is_pid_alive(int(w_pid)):
                watchdog_status_str = f"ACTIVE (PID {w_pid})"
            else:
                watchdog_status_str = "STALE LOCK"
        except Exception:
            watchdog_status_str = "UNKNOWN"

    manual_pause_file = s_dir / "MANUAL_PAUSE"
    manual_pause_str = "ACTIVE (supervisor auto-resume disabled)" if manual_pause_file.is_file() else "None (auto-resume enabled)"

    print("\n" + "=" * 70)
    print(" EXPERIMENT CONTROL STATUS (ctl.py)")
    print("=" * 70)
    if alert_banner:
        print(alert_banner)
    print(f"  Runner Process:     {'ACTIVE (PID ' + str(active_pid) + ')' if active_pid else 'INACTIVE / PAUSED'}")
    print(f"  Supervisor (Watchdog): {watchdog_status_str}")
    print(f"  Manual Pause Marker: {manual_pause_str}")
    print(f"  Current Run:        {current_run_id or 'None'}")
    print(f"  Run Status:         {run_status_str.upper()}")
    print(f"  Modality:           {current_run_info.get('modality', 'N/A')}")
    print(f"  Step / Budget:      {global_step:,} / {budget_str}")
    print(f"  Progress:           {pct_done:.1f}%")
    print(f"  Throughput (SPS):   {sps:.0f} steps/sec")
    print(f"  Active Time:        {format_duration(active_time_s)}")
    print(f"  Estimated ETA:      {eta_str}")
    print(f"  Latest Checkpoint:  {last_ckpt_age_str}")
    print(f"  GPU Memory / Util:  {gpu_info['raw']}")
    print("=" * 70 + "\n")


def generate_overnight_report(
    state_dir: Optional[Path] = None,
    runs_dir: Optional[Path] = None,
    runs_file: Optional[str] = None,
    output_path: Optional[Path] = None,
) -> str:
    """
    Generates a comprehensive Markdown overnight report per SPEC Section 2 & prompt:
    - Runs and steps progressed
    - SPS
    - Restarts and guard pauses count
    - Alerts
    - Latest evaluation rows
    - Checkpoint status and ages
    """
    s_dir = state_dir or get_state_dir()
    r_dir = runs_dir or get_runs_dir()
    now_str = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    local_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S %Z")

    active_pid = get_active_runner_pid(s_dir)
    gpu_info = get_gpu_memory_info()

    # Read queue
    queue_data = {}
    queue_file = s_dir / "queue.json"
    if queue_file.is_file():
        try:
            with open(queue_file, "r") as f:
                queue_data = json.load(f)
        except Exception:
            pass

    # Read alert if present
    alert_text = None
    alert_file = s_dir / "ALERT.txt"
    if alert_file.is_file():
        try:
            alert_text = alert_file.read_text().strip()
        except Exception:
            pass

    # Read supervisor log to parse restarts and guard events
    restarts_count = 0
    guard_pauses_count = 0
    watchdog_log_file = s_dir / "watchdog.log"
    if watchdog_log_file.is_file():
        try:
            with open(watchdog_log_file, "r") as f:
                for line in f:
                    if "Restarted runner after unexpected death" in line:
                        restarts_count += 1
                    elif "Guard tripped:" in line or "guard pause" in line.lower():
                        guard_pauses_count += 1
        except Exception:
            pass

    watchdog_lock = s_dir / "watchdog.lock"
    w_status = "ACTIVE" if (watchdog_lock.is_file()) else "INACTIVE / STANDBY"

    lines = [
        "# Overnight Supervision Report",
        "",
        f"**Report Generated:** {local_str} ({now_str})  ",
        f"**Runner Process:** {'ACTIVE (PID ' + str(active_pid) + ')' if active_pid else 'INACTIVE / PAUSED'}  ",
        f"**Supervisor Status:** {w_status}  ",
        f"**GPU Memory:** {gpu_info.get('raw', 'N/A')}  ",
        "",
        "---",
        "",
        "## 1. Supervisor & Guard Event Counters",
        "",
        f"- **Unexpected Runner Restarts:** {restarts_count}",
        f"- **Guard Pauses Tonight:** {guard_pauses_count} (Limit: 3/night)",
        f"- **Manual Pause Marker:** {'ACTIVE (state/MANUAL_PAUSE - auto-resume paused)' if (s_dir / 'MANUAL_PAUSE').is_file() else 'None (auto-resume enabled)'}",
        f"- **Alert Status:** {'ACTIVE ALERT DETECTED' if alert_text else 'Clean (no alerts)'}",
        "",
    ]

    if alert_text:
        lines.extend([
            "### Active Alert Details",
            "```",
            alert_text,
            "```",
            "",
        ])

    lines.extend([
        "---",
        "",
        "## 2. Experimental Queue & Run Progress",
        "",
        "| Run ID | Status | Modality | Step Progress | % Done | SPS | Active Time | Return | Success Rate |",
        "|---|---|---|---|---|---|---|---|---|",
    ])

    run_dirs_to_check = []
    if queue_data:
        for run_id, info in queue_data.items():
            status_val = info.get("status", "unknown")
            modality = info.get("modality", "N/A")
            total_timesteps = info.get("total_timesteps")
            out_dir = resolve_output_dir(info.get("output_dir"))
            if out_dir:
                run_dirs_to_check.append((run_id, status_val, modality, total_timesteps, out_dir))
    else:
        if r_dir.is_dir():
            for sub in sorted(r_dir.iterdir()):
                if sub.is_dir():
                    run_dirs_to_check.append((sub.name, "unknown", "N/A", None, sub))

    latest_eval_sections = []
    checkpoint_sections = []

    for run_id, r_status, modality, total_timesteps, run_path in run_dirs_to_check:
        global_step = 0
        sps = 0.0
        active_time_s = 0.0
        mean_return = 0.0
        success_rate = 0.0

        prog_file = run_path / "progress.csv"
        if prog_file.is_file():
            try:
                with open(prog_file, "r") as f:
                    r = csv.DictReader(f)
                    for row in r:
                        global_step = int(row.get("global_step", global_step))
                        sps = float(row.get("sps", sps))
                        active_time_s = float(row.get("active_time_s", active_time_s))
                        mean_return = float(row.get("mean_return", mean_return))
                        success_rate = float(row.get("iter_success_rate", row.get("success_rate", success_rate)))
            except Exception:
                pass

        if total_timesteps is not None and isinstance(total_timesteps, (int, float)) and total_timesteps > 0:
            b_val = int(total_timesteps)
            pct = (global_step / b_val * 100.0)
            b_str = f"{b_val:,}"
        else:
            pct = 0.0
            b_str = "budget not set"
        dur_str = format_duration(active_time_s)

        lines.append(
            f"| `{run_id}` | `{r_status.upper()}` | {modality} | {global_step:,} / {b_str} | {pct:.1f}% | {sps:.0f} | {dur_str} | {mean_return:.2f} | {success_rate * 100.0:.1f}% |"
        )

        ckpt_dir = run_path / "checkpoints"
        latest_ckpt_str = "None"
        if ckpt_dir.is_dir():
            ckpts = sorted(ckpt_dir.glob("ckpt_step_*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
            if ckpts:
                latest = ckpts[0]
                age_s = time.time() - latest.stat().st_mtime
                size_mb = latest.stat().st_size / (1024 * 1024)
                latest_ckpt_str = f"`{latest.name}` ({size_mb:.1f} MB, {format_age(age_s)})"
        checkpoint_sections.append(f"- **`{run_id}` Checkpoint:** {latest_ckpt_str}")

        eval_sum = run_path / "logs" / "eval_summary.csv"
        if eval_sum.is_file():
            try:
                with open(eval_sum, "r") as f:
                    r = csv.DictReader(f)
                    eval_rows = list(r)
                if eval_rows:
                    last_eval = eval_rows[-1]
                    latest_eval_sections.append(
                        f"- **`{run_id}` (Step {int(last_eval.get('step', 0)):,}):** Train Mean = {float(last_eval.get('train_mean', 0.0))*100.0:.1f}%, Val Mean = {float(last_eval.get('val_mean', 0.0))*100.0:.1f}%"
                    )
            except Exception:
                pass

    lines.extend([
        "",
        "---",
        "",
        "## 3. Checkpoint Status & Ages",
        "",
    ])
    lines.extend(checkpoint_sections or ["No active checkpoints found."])

    lines.extend([
        "",
        "---",
        "",
        "## 4. Latest Periodic Evaluation Results",
        "",
    ])
    lines.extend(latest_eval_sections or ["No evaluation logs recorded yet."])
    lines.append("")

    report_content = "\n".join(lines)

    target_path = output_path or (s_dir / "OVERNIGHT_REPORT.md")
    tmp_out = str(target_path) + f".tmp.{os.getpid()}"
    with open(tmp_out, "w") as f:
        f.write(report_content)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_out, str(target_path))

    return report_content


def report(
    state_dir: Optional[Path] = None,
    runs_dir: Optional[Path] = None,
    runs_file: Optional[str] = None,
    output_path: Optional[Path] = None,
) -> None:
    """Generates and displays the overnight supervision report."""
    content = generate_overnight_report(
        state_dir=state_dir,
        runs_dir=runs_dir,
        runs_file=runs_file,
        output_path=output_path,
    )
    print(content)


def parse_args() -> argparse.Namespace:
    """Parses ctl.py CLI command."""
    parser = argparse.ArgumentParser(description="CLI Control for Joint Multi-View PickCube Training")
    parser.add_argument("--runs-file", "--runs_file", type=str, default=None, help="Path to runs configuration YAML (e.g. configs/tiny_runs.yaml)")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    pause_p = subparsers.add_parser("pause", help="Create state/PAUSE and wait for training process to exit")
    pause_p.add_argument("--runs-file", "--runs_file", type=str, default=argparse.SUPPRESS, help="Path to runs configuration YAML")
    pause_p.add_argument("--no-manual", action="store_true", default=False, help="Do not create state/MANUAL_PAUSE (used by automated supervisors)")

    start_p = subparsers.add_parser("start", help="Start or resume runner detached")
    start_p.add_argument("--runs-file", "--runs_file", type=str, default=argparse.SUPPRESS, help="Path to runs configuration YAML")

    resume_p = subparsers.add_parser("resume", help="Clean stale PAUSE files and launch runner detached")
    resume_p.add_argument("--runs-file", "--runs_file", type=str, default=argparse.SUPPRESS, help="Path to runs configuration YAML")

    status_p = subparsers.add_parser("status", help="Print experiment and GPU status")
    status_p.add_argument("--runs-file", "--runs_file", type=str, default=argparse.SUPPRESS, help="Path to runs configuration YAML")

    report_p = subparsers.add_parser("report", help="Generate and display overnight supervision report")
    report_p.add_argument("--runs-file", "--runs_file", type=str, default=argparse.SUPPRESS, help="Path to runs configuration YAML")
    report_p.add_argument("--output", type=str, default=None, help="Custom output path for OVERNIGHT_REPORT.md")

    preflight_p = subparsers.add_parser("preflight", help="Run preflight readiness checks")
    preflight_p.add_argument("--runs-file", "--runs_file", type=str, default=argparse.SUPPRESS, help="Path to runs configuration YAML")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs_file = getattr(args, "runs_file", None)

    if args.command == "pause":
        manual = not getattr(args, "no_manual", False)
        pause(manual=manual)
    elif args.command in ("start", "resume"):
        resume(runs_file=runs_file)
    elif args.command == "status":
        status(runs_file=runs_file)
    elif args.command == "report":
        out_path = Path(args.output) if getattr(args, "output", None) else None
        report(runs_file=runs_file, output_path=out_path)
    elif args.command == "preflight":
        from scripts.preflight import run_preflight
        passed = run_preflight(verbose=True)
        sys.exit(0 if passed else 1)
    else:
        status(runs_file=runs_file)


if __name__ == "__main__":
    main()
