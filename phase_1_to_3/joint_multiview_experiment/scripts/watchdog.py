#!/usr/bin/env python3
"""
Overnight Supervisor Watchdog (scripts/watchdog.py).

Implements autonomous supervision of joint training:
1. Supervises runner.py through ctl.py on a configurable check loop (default: 30 s).
2. Enforces run window from configs/schedule.yaml (default: all day, every day):
   - Outside window: pauses runner (non-manual) and does not restart.
   - Inside window: resumes runner unless a manual pause marker (state/MANUAL_PAUSE) exists.
   - Emits state/OVERNIGHT_REPORT.md at the end of each window.
3. Hardware Safety Guards:
   - GPU temperature > 85 C for 2 continuous minutes.
   - Running on battery or battery below 30% (/sys/class/power_supply, skip if none).
   - Free disk space below 10 GB.
   Auto-resumes when conditions clear; limits to at most 3 guard pauses per night before writing ALERT.txt.
4. Auto-restarts runner on unexpected death/crash with exponential backoff:
   - Never restarts if state/PAUSE, state/ALERT.txt, state/MANUAL_PAUSE exist or run failed.
   - After 3 restarts without checkpoint progress, writes state/ALERT.txt and halts.
5. Single-instance lockfile (state/watchdog.lock).
6. Handles SIGTERM/SIGINT by exiting cleanly without terminating training.

Python 3.9 standard library only.
"""

import argparse
import datetime
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

EXP_DIR = Path(__file__).resolve().parent.parent
frozen_dir = EXP_DIR / "frozen_code" / "v2-freeze"
if frozen_dir.is_dir() and str(frozen_dir) not in sys.path:
    sys.path.insert(0, str(frozen_dir))
    os.environ["FROZEN_CODE_DIR"] = str(frozen_dir)
    commit_file = frozen_dir / "COMMIT_HASH"
    if commit_file.is_file():
        os.environ["FROZEN_COMMIT"] = commit_file.read_text().strip()
elif str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))


def get_state_dir() -> Path:
    custom_state = os.environ.get("EXP_STATE_DIR")
    if custom_state:
        return Path(custom_state)
    return EXP_DIR / "state"


def get_runs_dir() -> Path:
    custom_runs = os.environ.get("EXP_RUNS_DIR")
    if custom_runs:
        return Path(custom_runs)
    return EXP_DIR / "runs"


def is_pid_alive(pid: int) -> bool:
    """Checks if a process with PID is currently active and running (not a zombie)."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True

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


def get_active_runner_pid(state_dir: Path) -> Optional[int]:
    """Inspects state/lock and state/runner.pid to find active runner PID."""
    lock_file = state_dir / "lock"
    if lock_file.is_file():
        try:
            with open(lock_file, "r") as f:
                data = json.load(f)
            pid = data.get("pid")
            if pid and is_pid_alive(int(pid)):
                return int(pid)
        except Exception:
            pass

    pid_file = state_dir / "runner.pid"
    if pid_file.is_file():
        try:
            pid = int(pid_file.read_text().strip())
            if is_pid_alive(pid):
                return pid
        except Exception:
            pass

    return None


class WatchdogLock:
    """Single-instance lock management for watchdog process."""
    def __init__(self, lock_path: Path):
        self.lock_path = lock_path
        self.acquired = False

    def acquire(self) -> bool:
        if self.lock_path.is_file():
            try:
                with open(self.lock_path, "r") as f:
                    data = json.load(f)
                old_pid = data.get("pid")
                if old_pid and is_pid_alive(int(old_pid)):
                    return False
                else:
                    try:
                        self.lock_path.unlink()
                    except Exception:
                        pass
            except Exception:
                try:
                    self.lock_path.unlink()
                except Exception:
                    pass

        lock_data = {
            "pid": os.getpid(),
            "acquired_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        }
        tmp_path = str(self.lock_path) + f".tmp.{os.getpid()}"
        with open(tmp_path, "w") as f:
            json.dump(lock_data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, str(self.lock_path))
        self.acquired = True
        return True

    def release(self) -> None:
        if self.acquired and self.lock_path.is_file():
            try:
                with open(self.lock_path, "r") as f:
                    data = json.load(f)
                if data.get("pid") == os.getpid():
                    self.lock_path.unlink()
            except Exception:
                pass
            self.acquired = False


def load_schedule(schedule_path: Path) -> Dict[str, Any]:
    """
    Parses configs/schedule.yaml using standard library.
    Returns dict with start, end, timezone, days.
    """
    default_sched = {
        "start": "00:00",
        "end": "00:00",
        "timezone": "local",
        "days": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
    }
    if not schedule_path.is_file():
        return default_sched

    res = dict(default_sched)
    try:
        with open(schedule_path, "r") as f:
            lines = f.readlines()

        in_days = False
        parsed_days = []
        for line in lines:
            line_str = line.strip()
            if not line_str or line_str.startswith("#"):
                continue
            if line_str.startswith("start:"):
                res["start"] = line_str.split(":", 1)[1].strip().strip("\"'")
            elif line_str.startswith("end:"):
                res["end"] = line_str.split(":", 1)[1].strip().strip("\"'")
            elif line_str.startswith("timezone:"):
                res["timezone"] = line_str.split(":", 1)[1].strip().strip("\"'")
            elif line_str.startswith("days:"):
                in_days = True
                parsed_days = []
            elif in_days and line_str.startswith("-"):
                day_val = line_str.lstrip("-").strip().strip("\"'")
                parsed_days.append(day_val)
            elif in_days and not line.startswith(" ") and not line.startswith("\t"):
                in_days = False

        if parsed_days:
            res["days"] = parsed_days
    except Exception:
        return default_sched

    return res


def is_time_in_window(dt: datetime.datetime, schedule: Dict[str, Any]) -> bool:
    """
    Evaluates whether datetime dt falls within the scheduled run window.
    Equal start and end times mean a full day; also handles windows spanning midnight.
    """
    try:
        sh, sm = map(int, schedule["start"].split(":"))
        eh, em = map(int, schedule["end"].split(":"))
    except Exception:
        sh, sm = 0, 0
        eh, em = 0, 0

    start_time = datetime.time(sh, sm)
    end_time = datetime.time(eh, em)

    curr_time = dt.time()
    curr_weekday = dt.strftime("%a")
    prev_weekday = (dt - datetime.timedelta(days=1)).strftime("%a")

    allowed_days = schedule.get("days", ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])

    if start_time == end_time:
        return curr_weekday in allowed_days

    if start_time < end_time:
        # Same-day window (e.g. 09:00 to 17:00)
        return (curr_weekday in allowed_days) and (start_time <= curr_time < end_time)
    else:
        # Overnight window crossing midnight (e.g. 23:30 to 07:30)
        if curr_time >= start_time and curr_weekday in allowed_days:
            return True
        if curr_time < end_time and prev_weekday in allowed_days:
            return True
        return False


class HardwareGuards:
    """
    Monitors hardware conditions per SPEC:
    - GPU temperature > 85 C for 2 continuous minutes.
    - Battery: running on battery or battery below 30%.
    - Free disk space < 10 GB.
    """
    def __init__(self, exp_dir: Path):
        self.exp_dir = exp_dir
        self.high_temp_start_time: Optional[float] = None

    def read_gpu_temperature(self) -> Optional[int]:
        """Queries GPU temperature in Celsius via nvidia-smi."""
        try:
            res = subprocess.run(
                ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=5.0,
            )
            if res.returncode == 0:
                line = res.stdout.strip().split("\n")[0]
                return int(line.strip())
        except Exception:
            pass
        return None

    def read_battery_status(self) -> Tuple[bool, bool, Optional[int]]:
        """
        Reads /sys/class/power_supply.
        Returns: (has_battery, is_on_battery, battery_capacity_pct)
        """
        ps_dir = Path("/sys/class/power_supply")
        if not ps_dir.is_dir():
            return False, False, None

        batteries = []
        ac_supplies = []

        try:
            for s in ps_dir.iterdir():
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
        except Exception:
            return False, False, None

        if not batteries:
            return False, False, None

        is_on_battery = False
        for ac in ac_supplies:
            online_file = ac / "online"
            if online_file.is_file():
                try:
                    if online_file.read_text().strip() == "0":
                        is_on_battery = True
                except Exception:
                    pass

        min_cap: Optional[int] = None
        for b in batteries:
            status_file = b / "status"
            if status_file.is_file():
                try:
                    if status_file.read_text().strip().lower() == "discharging":
                        is_on_battery = True
                except Exception:
                    pass
            cap_file = b / "capacity"
            if cap_file.is_file():
                try:
                    cap = int(cap_file.read_text().strip())
                    if min_cap is None or cap < min_cap:
                        min_cap = cap
                except Exception:
                    pass

        return True, is_on_battery, min_cap

    def read_free_disk_gb(self) -> float:
        """Determines free disk space on experiment volume in GB."""
        try:
            usage = shutil.disk_usage(str(self.exp_dir))
            return usage.free / (1024 ** 3)
        except Exception:
            return 999.0

    def check(self, now: Optional[float] = None) -> Tuple[bool, Optional[str]]:
        """
        Evaluates all guards.
        Returns: (is_tripped, reason_message)
        """
        t = now or time.time()

        # 1. GPU Temperature (> 85 C for 2 continuous minutes)
        gpu_temp = self.read_gpu_temperature()
        if gpu_temp is not None:
            if gpu_temp > 85:
                if self.high_temp_start_time is None:
                    self.high_temp_start_time = t
                elif (t - self.high_temp_start_time) >= 120.0:
                    dur_s = int(t - self.high_temp_start_time)
                    return True, f"GPU temperature stayed above 85 C for {dur_s}s (current: {gpu_temp} C)"
            else:
                self.high_temp_start_time = None
        else:
            self.high_temp_start_time = None

        # 2. Battery Guard (on battery or capacity < 30%)
        has_bat, on_bat, bat_cap = self.read_battery_status()
        if has_bat:
            if on_bat:
                cap_str = f" ({bat_cap}% charge)" if bat_cap is not None else ""
                return True, f"System running on battery power{cap_str}"
            if bat_cap is not None and bat_cap < 30:
                return True, f"Battery capacity below 30% ({bat_cap}%)"

        # 3. Disk Space Guard (< 10 GB)
        free_gb = self.read_free_disk_gb()
        if free_gb < 10.0:
            return True, f"Free disk space below 10 GB ({free_gb:.2f} GB available)"

        return False, None


def get_latest_checkpoint_stats(runs_dir: Path) -> Tuple[Optional[str], Optional[float], Optional[int]]:
    """
    Inspects all run directories under runs_dir to find newest checkpoint.
    Returns: (checkpoint_path_str, mtime, global_step)
    """
    if not runs_dir.is_dir():
        return None, None, None

    newest_ckpt = None
    newest_mtime = 0.0
    newest_step = None

    for sub in runs_dir.iterdir():
        if sub.is_dir():
            ckpt_dir = sub / "checkpoints"
            if ckpt_dir.is_dir():
                for pt in ckpt_dir.glob("ckpt_step_*.pt"):
                    try:
                        mtime = pt.stat().st_mtime
                        if mtime > newest_mtime:
                            newest_mtime = mtime
                            newest_ckpt = str(pt)
                            # Parse step from filename e.g. ckpt_step_000100000.pt
                            m = re.search(r"ckpt_step_(\d+)\.pt", pt.name)
                            if m:
                                newest_step = int(m.group(1))
                    except Exception:
                        pass

    return newest_ckpt, (newest_mtime if newest_ckpt else None), newest_step


def is_run_marked_failed(state_dir: Path) -> Tuple[bool, Optional[str]]:
    """Checks if current or active run in state/queue.json is marked failed."""
    queue_file = state_dir / "queue.json"
    if not queue_file.is_file():
        return False, None

    try:
        with open(queue_file, "r") as f:
            queue = json.load(f)
        for run_id, info in queue.items():
            if info.get("status") == "failed":
                return True, f"Run '{run_id}' failed: {info.get('error', 'unknown error')}"
    except Exception:
        pass

    return False, None


def is_queue_all_done(state_dir: Path) -> bool:
    """Returns True if state/queue.json exists and all registered runs are marked done."""
    queue_file = state_dir / "queue.json"
    if not queue_file.is_file():
        return False
    try:
        with open(queue_file, "r") as f:
            queue = json.load(f)
        if not queue:
            return False
        return all(info.get("status") == "done" for info in queue.values())
    except Exception:
        return False


class Supervisor:
    """
    Orchestrates watchdog supervisor loop.
    """
    def __init__(
        self,
        exp_dir: Optional[Path] = None,
        state_dir: Optional[Path] = None,
        runs_dir: Optional[Path] = None,
        schedule_path: Optional[Path] = None,
        runs_file: Optional[str] = None,
        check_interval: float = 30.0,
    ):
        self.exp_dir = exp_dir or EXP_DIR
        self.state_dir = state_dir or get_state_dir()
        self.runs_dir = runs_dir or get_runs_dir()
        self.schedule_path = schedule_path or (self.exp_dir / "configs" / "schedule.yaml")
        self.runs_file = runs_file
        self.check_interval = check_interval

        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.lock = WatchdogLock(self.state_dir / "watchdog.lock")
        self.guards = HardwareGuards(self.exp_dir)

        # Supervisor state tracking
        self.guard_pauses_count = 0
        self.restarts_without_progress = 0
        self.total_restarts = 0
        self.backoff_delay = 5.0
        self.last_known_step: Optional[int] = None
        self.last_known_mtime: Optional[float] = None
        self.was_in_window: Optional[bool] = None
        self.shutdown_requested = False

        # Configure logging to state/watchdog.log and stderr
        self.log_file = self.state_dir / "watchdog.log"
        self._init_logger()

    def _init_logger(self) -> None:
        self.logger = logging.getLogger(f"watchdog_{id(self)}")
        self.logger.setLevel(logging.INFO)
        self.logger.handlers = []

        # File handler
        fh = logging.FileHandler(str(self.log_file))
        fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        self.logger.addHandler(fh)

        # Stream handler
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(logging.Formatter("%(asctime)s [watchdog] %(message)s"))
        self.logger.addHandler(sh)

    def write_alert(self, title: str, details: str) -> None:
        """Atomically writes structured ALERT.txt and triggers PAUSE."""
        alert_file = self.state_dir / "ALERT.txt"
        content = (
            "=" * 70 + "\n"
            f" SUPERVISOR ALERT: {title}\n"
            "=" * 70 + "\n"
            f" Timestamp:   {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}\n"
            f" Details:     {details}\n"
            " Action:      Supervision halted. Training paused.\n"
            "              Review issue and resume manually via 'python src/ctl.py resume'.\n"
            "=" * 70 + "\n"
        )
        tmp_alert = str(alert_file) + f".tmp.{os.getpid()}"
        with open(tmp_alert, "w") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_alert, str(alert_file))

        # Set pause file
        pause_file = self.state_dir / "PAUSE"
        with open(str(pause_file), "w") as f:
            f.write(f"PAUSE requested by Watchdog: {title}\n")

        self.logger.critical("SUPERVISOR ALERT CREATED:\n%s", content)

    def ctl_pause(self, manual: bool = False) -> bool:
        """Calls ctl.py pause via module or subprocess."""
        try:
            from src.ctl import pause as run_pause
            return run_pause(state_dir=self.state_dir, manual=manual)
        except Exception as e:
            self.logger.error("Failed to execute ctl_pause: %s", e)
            return False

    def ctl_resume(self) -> bool:
        """Calls ctl.py resume via module or subprocess."""
        try:
            from src.ctl import resume as run_resume
            return run_resume(
                state_dir=self.state_dir,
                runs_dir=self.runs_dir,
                runs_file=self.runs_file,
            )
        except Exception as e:
            self.logger.error("Failed to execute ctl_resume: %s", e)
            return False

    def generate_report(self) -> None:
        """Calls ctl.py report generation writing state/OVERNIGHT_REPORT.md."""
        try:
            from src.ctl import generate_overnight_report
            generate_overnight_report(
                state_dir=self.state_dir,
                runs_dir=self.runs_dir,
                runs_file=self.runs_file,
                output_path=self.state_dir / "OVERNIGHT_REPORT.md",
            )
            self.logger.info("Generated overnight report: %s", self.state_dir / "OVERNIGHT_REPORT.md")
        except Exception as e:
            self.logger.error("Failed to generate report: %s", e)

    def handle_signal(self, signum: int, frame: Any) -> None:
        """Handles SIGTERM / SIGINT: exits cleanly without terminating training."""
        sig_name = "SIGTERM" if signum == signal.SIGTERM else "SIGINT"
        self.logger.info("Received %s signal. Stopping watchdog supervisor (training left active).", sig_name)
        self.shutdown_requested = True

    def run_iteration(self, now_dt: Optional[datetime.datetime] = None) -> bool:
        """
        Executes one evaluation cycle of supervisor logic.
        Returns False if supervisor should stop supervising, True to continue.
        """
        dt = now_dt or datetime.datetime.now()
        schedule = load_schedule(self.schedule_path)
        in_window = is_time_in_window(dt, schedule)

        # 1. Window transition detection
        if self.was_in_window is not None:
            if self.was_in_window and not in_window:
                # Window just closed!
                self.logger.info("Schedule window closed (%s -> %s). Pausing training and generating report.", schedule["start"], schedule["end"])
                active_pid = get_active_runner_pid(self.state_dir)
                if active_pid is not None:
                    self.ctl_pause(manual=False)
                self.generate_report()
                # Reset nightly counters for next window
                self.guard_pauses_count = 0
            elif not self.was_in_window and in_window:
                self.logger.info("Schedule window opened (%s -> %s).", schedule["start"], schedule["end"])

        self.was_in_window = in_window

        # 2. Check for terminal alert
        if (self.state_dir / "ALERT.txt").is_file():
            self.logger.warning("Active alert in state/ALERT.txt. Supervisor holding (not restarting).")
            return True

        # 3. Check for failed run or all runs completed
        is_failed, fail_msg = is_run_marked_failed(self.state_dir)
        if is_failed:
            self.logger.warning("Run marked failed: %s. Supervisor holding (not restarting).", fail_msg)
            return True

        if is_queue_all_done(self.state_dir):
            # All runs completed!
            return True

        # 4. Check hardware guards
        guard_tripped, guard_reason = self.guards.check()
        active_pid = get_active_runner_pid(self.state_dir)

        if guard_tripped:
            self.logger.warning("Safety guard tripped: %s", guard_reason)
            if active_pid is not None:
                self.ctl_pause(manual=False)
                self.guard_pauses_count += 1
                self.logger.info("Guard pause executed (pause %d/3 tonight).", self.guard_pauses_count)

            if self.guard_pauses_count > 3:
                self.write_alert(
                    "MAXIMUM GUARD PAUSES EXCEEDED",
                    f"4 guard pauses occurred tonight (limit: 3). Last condition: {guard_reason}"
                )
                return False
            return True

        # 5. Outside window logic
        if not in_window:
            if active_pid is not None:
                self.logger.info("Currently outside schedule window (%s to %s). Pausing training.", schedule["start"], schedule["end"])
                self.ctl_pause(manual=False)
            return True

        # 6. Inside window logic
        manual_pause = (self.state_dir / "MANUAL_PAUSE").is_file()
        if manual_pause:
            # User manually paused; respect it
            return True

        # Update checkpoint progress tracker if runner is active
        latest_ckpt, latest_mtime, latest_step = get_latest_checkpoint_stats(self.runs_dir)
        if latest_step is not None:
            if self.last_known_step is None or latest_step > self.last_known_step or (latest_mtime and self.last_known_mtime and latest_mtime > self.last_known_mtime):
                self.last_known_step = latest_step
                self.last_known_mtime = latest_mtime
                if self.restarts_without_progress > 0:
                    self.logger.info("Checkpoint progress confirmed (step %d). Resetting crash loop counter.", latest_step)
                self.restarts_without_progress = 0
                self.backoff_delay = 5.0

        if active_pid is not None:
            # Runner is healthy and progressing
            return True

        # Runner is NOT active inside window with guards clear and no manual pause:
        pause_file = self.state_dir / "PAUSE"
        if pause_file.is_file():
            # Paused earlier by window or guard that is now cleared: resume!
            self.logger.info("Window open and guards cleared; auto-resuming training.")
            self.ctl_resume()
            return True

        # Runner died unexpectedly without a pause file! (Crash / SIGKILL)
        self.logger.warning("Runner process died unexpectedly without PAUSE file.")
        if self.restarts_without_progress >= 3:
            self.write_alert(
                "CRASH LOOP DETECTED",
                f"Runner died 3 consecutive times without checkpoint progress. Last checkpoint step: {self.last_known_step}"
            )
            return False

        self.restarts_without_progress += 1
        self.total_restarts += 1
        self.logger.warning(
            "Restarting runner after crash (restart %d/3 without progress; backoff %ds)...",
            self.restarts_without_progress,
            int(self.backoff_delay),
        )
        time.sleep(self.backoff_delay)
        self.backoff_delay = min(self.backoff_delay * 2.0, 120.0)
        self.ctl_resume()
        return True

    def loop(self) -> None:
        """Main supervisor loop running every check_interval seconds."""
        if not self.lock.acquire():
            print(f"Error: Another watchdog supervisor is already active for {self.state_dir}. Lock exists.")
            sys.exit(1)

        signal.signal(signal.SIGTERM, self.handle_signal)
        signal.signal(signal.SIGINT, self.handle_signal)

        self.logger.info("=" * 70)
        self.logger.info("Overnight supervisor started (PID %d)", os.getpid())
        self.logger.info("State Dir:       %s", self.state_dir)
        self.logger.info("Runs Dir:        %s", self.runs_dir)
        self.logger.info("Schedule Config: %s", self.schedule_path)
        self.logger.info("Check Interval:  %ds", int(self.check_interval))
        self.logger.info("=" * 70)

        try:
            while not self.shutdown_requested:
                should_continue = self.run_iteration()
                if not should_continue:
                    self.logger.critical("Supervisor stopping due to guard or crash limit.")
                    break

                # Sleep check interval in short increments to remain responsive to SIGTERM
                sleep_start = time.time()
                while time.time() - sleep_start < self.check_interval:
                    if self.shutdown_requested:
                        break
                    time.sleep(0.5)
        finally:
            self.lock.release()
            self.logger.info("Watchdog supervisor lock released. Exited cleanly.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Overnight Supervisor Watchdog")
    parser.add_argument("--interval", type=float, default=30.0, help="Check loop interval in seconds (default: 30)")
    parser.add_argument("--schedule", type=str, default=None, help="Path to schedule.yaml")
    parser.add_argument("--runs-file", "--runs_file", type=str, default=None, help="Path to runs YAML")
    return parser.parse_args()


def main():
    args = parse_args()
    sched_path = Path(args.schedule) if args.schedule else (EXP_DIR / "configs" / "schedule.yaml")
    sup = Supervisor(
        schedule_path=sched_path,
        runs_file=args.runs_file,
        check_interval=args.interval,
    )
    sup.loop()


if __name__ == "__main__":
    main()
