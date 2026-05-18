"""
monitor.py — ProactiveMonitor for FRIDAY.

Runs as a background daemon thread, polling system health every 30 seconds.
Fires spoken alerts via skills._announce_fn when thresholds are breached.

Alert conditions:
  - CPU  > 90% sustained for 2 consecutive polls (~60s)
  - RAM  > 90%
  - Battery < 15% and not plugged in
  - Disk C: < 5 GB free
  - Ollama unreachable for 60s (2 consecutive polls)

Each alert type has a 10-minute cooldown so Friday doesn't repeat herself.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Optional
from urllib import error, request as urllib_request

try:
    import psutil
    _PSUTIL_OK = True
except ImportError:
    psutil = None          # type: ignore
    _PSUTIL_OK = False


# ── Thresholds ────────────────────────────────────────────────────────────────

CPU_THRESHOLD       = 90.0     # % — trigger after 2 consecutive polls
RAM_THRESHOLD       = 90.0     # %
BATTERY_THRESHOLD   = 15.0     # % — only when unplugged
DISK_FREE_GB        = 5.0      # GB free on C:
OLLAMA_TIMEOUT_S    = 5.0      # seconds per reachability check
POLL_INTERVAL_S     = 30       # seconds between polls
COOLDOWN_S          = 600      # 10 minutes between repeated alerts of the same type
OLLAMA_URL          = "http://127.0.0.1:11434/api/tags"


@dataclass
class ProactiveMonitor:
    """
    Background health monitor.  Call start() once after VoiceIO is ready.
    announce_fn must be set to a callable (e.g. voice.speak) before starting.
    """
    announce_fn:  Optional[Callable[[str], None]] = None
    ollama_url:   str  = OLLAMA_URL
    debug:        bool = False

    def __post_init__(self) -> None:
        # Private mutable state — kept out of @dataclass fields entirely
        # so the machinery doesn't touch them and repr stays clean.
        self._thread:           Optional[threading.Thread] = None
        self._stop_event:       threading.Event            = threading.Event()
        self._last_alert:       dict                       = {}
        self._cpu_high_runs:    int                        = 0
        self._ollama_fail_runs: int                        = 0

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the monitor daemon thread."""
        if not _PSUTIL_OK:
            print("[monitor] psutil not available — ProactiveMonitor disabled.")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="friday-monitor"
        )
        self._thread.start()
        if self.debug:
            print("[monitor] ProactiveMonitor started.")

    def stop(self) -> None:
        """Signal the monitor to stop (called on shutdown)."""
        self._stop_event.set()

    # ── Internal loop ─────────────────────────────────────────────────────────

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._poll()
            except Exception as exc:
                if self.debug:
                    print(f"[monitor] poll error: {exc}")
            self._stop_event.wait(timeout=POLL_INTERVAL_S)

    def _poll(self) -> None:
        self._check_cpu()
        self._check_ram()
        self._check_battery()
        self._check_disk()
        self._check_ollama()

    # ── Checks ────────────────────────────────────────────────────────────────

    def _check_cpu(self) -> None:
        cpu = psutil.cpu_percent(interval=1)
        if self.debug:
            print(f"[monitor] CPU={cpu:.1f}%")
        if cpu > CPU_THRESHOLD:
            self._cpu_high_runs += 1
            if self._cpu_high_runs >= 2:          # sustained ~60s
                self._alert(
                    "cpu",
                    f"Sir, CPU usage has been above {CPU_THRESHOLD:.0f} percent "
                    f"for over a minute. Currently at {cpu:.0f} percent."
                )
        else:
            self._cpu_high_runs = 0

    def _check_ram(self) -> None:
        vm = psutil.virtual_memory()
        pct = vm.percent
        if self.debug:
            print(f"[monitor] RAM={pct:.1f}%")
        if pct > RAM_THRESHOLD:
            self._alert(
                "ram",
                f"Sir, RAM usage is at {pct:.0f} percent. "
                f"You may want to close some applications."
            )

    def _check_battery(self) -> None:
        bat = psutil.sensors_battery()
        if bat is None:
            return   # desktop / no battery sensor
        if self.debug:
            print(f"[monitor] Battery={bat.percent:.1f}% plugged={bat.power_plugged}")
        if not bat.power_plugged and bat.percent < BATTERY_THRESHOLD:
            self._alert(
                "battery",
                f"Sir, battery is at {bat.percent:.0f} percent and not plugged in. "
                f"Please connect your charger."
            )

    def _check_disk(self) -> None:
        try:
            usage = psutil.disk_usage("C:\\")
        except (PermissionError, FileNotFoundError):
            return
        free_gb = usage.free / (1024 ** 3)
        if self.debug:
            print(f"[monitor] Disk C: free={free_gb:.1f}GB")
        if free_gb < DISK_FREE_GB:
            self._alert(
                "disk",
                f"Sir, drive C has only {free_gb:.1f} gigabytes free. "
                f"Consider cleaning up some space."
            )

    def _check_ollama(self) -> None:
        try:
            req = urllib_request.Request(self.ollama_url, method="GET")
            with urllib_request.urlopen(req, timeout=OLLAMA_TIMEOUT_S):
                pass
            self._ollama_fail_runs = 0    # reachable — reset counter
        except (error.URLError, OSError):
            self._ollama_fail_runs += 1
            if self.debug:
                print(f"[monitor] Ollama unreachable (run {self._ollama_fail_runs})")
            if self._ollama_fail_runs >= 2:    # ~60s unreachable
                self._alert(
                    "ollama",
                    "Sir, Ollama appears to be unreachable. "
                    "My language model may not respond until it restarts."
                )

    # ── Alert dispatch ────────────────────────────────────────────────────────

    def _alert(self, key: str, message: str) -> None:
        """Fire an alert if the cooldown for this key has expired."""
        now = time.monotonic()
        last = self._last_alert.get(key, 0.0)
        if now - last < COOLDOWN_S:
            if self.debug:
                print(f"[monitor] alert '{key}' suppressed (cooldown)")
            return

        self._last_alert[key] = now
        timestamp = datetime.now().strftime("%I:%M %p")
        if self.debug:
            print(f"[monitor] ALERT [{key}] @ {timestamp}: {message}")

        if self.announce_fn is not None:
            try:
                self.announce_fn(message)
            except Exception as exc:
                if self.debug:
                    print(f"[monitor] announce_fn failed: {exc}")
        else:
            # Fallback: print if no voice available
            print(f"\n[FRIDAY MONITOR] {message}")
