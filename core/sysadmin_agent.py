"""
sysadmin_agent.py — Mr Meeseeks System Admin Agent
Silent read-only commands run here via subprocess.
Visible terminal commands open real gnome-terminal.
Registers all handlers with ipc_bus.
"""

import asyncio
import subprocess
import shlex
import logging
from ipc_bus import bus
from schema_registry import READ_ONLY_CMDS, is_destructive

log = logging.getLogger("sysadmin")


async def handle_run_bg_cmd(args: dict) -> dict:
    cmd = args.get("cmd", "").strip()
    first_word = cmd.split()[0] if cmd else ""

    if first_word not in READ_ONLY_CMDS:
        return {"error": f"'{first_word}' not in read-only whitelist. Use open_visible_terminal."}

    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=10
        )
        output = result.stdout.strip() or result.stderr.strip() or "(no output)"
        return {"output": output, "exit_code": result.returncode}
    except subprocess.TimeoutExpired:
        return {"error": "Command timed out after 10s"}
    except Exception as e:
        return {"error": str(e)}


async def handle_open_visible_terminal(args: dict) -> dict:
    cmd = args.get("cmd", "").strip()

    if is_destructive(cmd):
        # pause — let safety gate handle (for now just log + return)
        log.warning(f"DESTRUCTIVE CMD BLOCKED: {cmd}")
        return {"error": f"Destructive command detected: '{cmd}'. Needs explicit user confirmation."}

    try:
        # open gnome-terminal, run cmd, keep window open after
        terminal_cmd = f'gnome-terminal -- bash -c "{cmd}; exec bash"'
        subprocess.Popen(terminal_cmd, shell=True)
        return {"output": f"Opened terminal running: {cmd}"}
    except Exception as e:
        return {"error": str(e)}


async def handle_check_battery(args: dict) -> dict:
    try:
        result = subprocess.run(
            "cat /sys/class/power_supply/BAT0/capacity && cat /sys/class/power_supply/BAT0/status",
            shell=True, capture_output=True, text=True, timeout=5
        )
        lines = result.stdout.strip().split("\n")
        level  = lines[0] if len(lines) > 0 else "unknown"
        status = lines[1] if len(lines) > 1 else "unknown"
        return {"level": f"{level}%", "status": status}
    except Exception as e:
        return {"error": str(e)}


async def handle_get_active_window(args: dict) -> dict:
    try:
        result = subprocess.run(
            "xdotool getactivewindow getwindowname",
            shell=True, capture_output=True, text=True, timeout=5
        )
        return {"window": result.stdout.strip() or "unknown"}
    except Exception as e:
        return {"error": str(e)}


async def handle_read_notifications(args: dict) -> dict:
    # placeholder — real impl needs dbus
    return {"notifications": "notification reading not yet implemented"}


def register():
    bus.register("run_bg_cmd",            handle_run_bg_cmd)
    bus.register("open_visible_terminal", handle_open_visible_terminal)
    bus.register("check_battery",         handle_check_battery)
    bus.register("get_active_window",     handle_get_active_window)
    bus.register("read_notifications",    handle_read_notifications)
    log.info("SysAdmin agent registered ✓")
