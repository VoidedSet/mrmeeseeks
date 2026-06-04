"""
kernel_listener.py — Mr Meeseeks Kernel Listener
Background asyncio tasks that poll OS state and keep KernelState fresh.

Polling rates (tunable):
    active_window : 800ms  — fast, user notices latency on window queries
    open_windows  : 3s     — moderate, list changes infrequently
    battery       : 30s    — slow, power level rarely jumps

On value change: calls brain.handle_proactive_event() so the brain can
optionally alert the user (e.g. low battery warning).

Usage (in main.py):
    from kernel.kernel_listener import start as start_kernel
    kernel_task = asyncio.create_task(start_kernel(brain))
    # on shutdown:
    kernel_task.cancel()
"""

import asyncio
import glob
import logging
import subprocess

from kernel.kernel_state import state

log = logging.getLogger("kernel")

# ── Poll intervals (seconds) ──────────────────────────────────────────────────
POLL_ACTIVE_WINDOW = 0.8
POLL_OPEN_WINDOWS  = 3.0
POLL_BATTERY       = 30.0


# ── Helpers ───────────────────────────────────────────────────────────────────

def _run(cmd: str) -> str:
    """Run a quick shell command, return stdout stripped. Never raises."""
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=3)
        return r.stdout.strip()
    except Exception:
        return ""


def _parse_windows(wmctrl_output: str) -> list[str]:
    """
    Parse `wmctrl -l` output into a list of window titles.
    Format: <hex_id> <desktop> <hostname> <title...>
    Desktop -1 = sticky/system windows (plank, desktop). We include them
    but mark clearly, or filter if desired.
    """
    titles = []
    for line in wmctrl_output.splitlines():
        parts = line.split(None, 3)
        if len(parts) >= 4:
            title = parts[3].strip()
            if title:
                titles.append(title)
    return titles


def _read_battery() -> tuple[str, str]:
    """Read battery level and status from sysfs."""
    try:
        bats = glob.glob("/sys/class/power_supply/BAT*")
        if not bats:
            return "no battery", "unknown"
        bat = bats[0]
        level  = open(f"{bat}/capacity").read().strip()
        status = open(f"{bat}/status").read().strip()
        return f"{level}%", status
    except Exception:
        return "unknown", "unknown"


# ── Poll loops ────────────────────────────────────────────────────────────────

async def _poll_active_window(brain_ref):
    """Poll active window title every POLL_ACTIVE_WINDOW seconds."""
    while True:
        try:
            title = _run("xdotool getactivewindow getwindowname 2>/dev/null") or "unknown"
            changed = state.set_active_window(title)
            if changed:
                log.debug(f"Active window → {title!r}")
                if brain_ref is not None:
                    await brain_ref.handle_proactive_event({
                        "type":   "window_focus",
                        "window": title,
                    })
        except Exception as e:
            log.warning(f"Active window poll error: {e}")
        await asyncio.sleep(POLL_ACTIVE_WINDOW)


async def _poll_open_windows(brain_ref):
    """Poll all open windows every POLL_OPEN_WINDOWS seconds."""
    while True:
        try:
            raw     = _run("wmctrl -l 2>/dev/null")
            titles  = _parse_windows(raw)
            changed = state.set_open_windows(titles)
            if changed:
                log.debug(f"Open windows changed: {len(titles)} windows")
                if brain_ref is not None:
                    await brain_ref.handle_proactive_event({
                        "type":    "windows_changed",
                        "windows": titles,
                    })
        except Exception as e:
            log.warning(f"Open windows poll error: {e}")
        await asyncio.sleep(POLL_OPEN_WINDOWS)


async def _poll_battery(brain_ref):
    """Poll battery every POLL_BATTERY seconds. Fire alert on low battery."""
    LOW_BATTERY_THRESHOLD = 15  # percent
    while True:
        try:
            level_str, status = _read_battery()
            changed = state.set_battery(level_str, status)

            if changed:
                log.debug(f"Battery → {level_str} ({status})")
                # Low battery alert
                try:
                    level_int = int(level_str.replace("%", ""))
                    if level_int <= LOW_BATTERY_THRESHOLD and status == "Discharging":
                        if brain_ref is not None:
                            await brain_ref.handle_proactive_event({
                                "type":   "low_battery",
                                "level":  level_str,
                                "status": status,
                            })
                except ValueError:
                    pass
        except Exception as e:
            log.warning(f"Battery poll error: {e}")
        await asyncio.sleep(POLL_BATTERY)


# ── Entry point ───────────────────────────────────────────────────────────────

async def start(brain_ref=None):
    """
    Start all polling loops as concurrent tasks under one parent task.
    brain_ref: the Brain singleton (or None for standalone testing).

    Cancel this task to stop all polling.
    """
    log.info("Kernel listener starting...")

    # Do one immediate pass to warm up state before brain starts processing
    try:
        title  = _run("xdotool getactivewindow getwindowname 2>/dev/null") or "unknown"
        state.set_active_window(title)

        raw    = _run("wmctrl -l 2>/dev/null")
        titles = _parse_windows(raw)
        state.set_open_windows(titles)

        level, status = _read_battery()
        state.set_battery(level, status)

        log.info(f"Kernel state warm: window={title!r}, {len(titles)} windows, battery={level}")
    except Exception as e:
        log.warning(f"Kernel warm-up error: {e}")

    # Launch all polling loops
    await asyncio.gather(
        _poll_active_window(brain_ref),
        _poll_open_windows(brain_ref),
        _poll_battery(brain_ref),
    )
