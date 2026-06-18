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
import os
import glob
from subsystems.sentinel.kernel_state import state
from subsystems.tool_calling.ipc_bus import bus
from subsystems.tool_calling.schema_registry import READ_ONLY_CMDS, is_destructive
import safety.gate as gate

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
        if len(output) > 2000:
            output = output[:2000] + "\n...[Output truncated due to length]"
        return {"output": output, "exit_code": result.returncode}
    except subprocess.TimeoutExpired:
        return {"error": "Command timed out after 10s"}
    except Exception as e:
        return {"error": str(e)}


async def handle_open_visible_terminal(args: dict) -> dict:
    cmd = args.get("cmd", "").strip()

    if is_destructive(cmd):
        approved = await gate.confirm_destructive(cmd)
        if not approved:
            return {"error": f"User denied destructive command: '{cmd}'"}

    try:
        import tempfile
        temp_log = tempfile.mktemp(prefix="meeseeks_")
        
        # open gnome-terminal, run cmd, capture output, wait 5s, close
        bash_script = f"{cmd} 2>&1 | tee {temp_log}; echo ''; echo '[Command finished. Closing in 5 seconds...]'; sleep 5"
        terminal_cmd = f"gnome-terminal --wait -- bash -c {shlex.quote(bash_script)}"
        
        proc = await asyncio.create_subprocess_shell(terminal_cmd)
        await proc.communicate()
        
        output = "(no output)"
        if os.path.exists(temp_log):
            with open(temp_log, "r") as f:
                output = f.read().strip() or "(no output)"
            os.remove(temp_log)
            
        return {"output": output}
    except Exception as e:
        return {"error": str(e)}


async def handle_check_battery(args: dict) -> dict:
    try:
        bats = glob.glob("/sys/class/power_supply/BAT*")
        if not bats:
            return {"error": "No battery found (/sys/class/power_supply/BAT*)."}
        bat = bats[0]
        result = subprocess.run(
            f"cat {bat}/capacity && cat {bat}/status",
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


SYSTEM_WINDOWS = {"plank", "Desktop Icons", "Desktop Icons 1", "gjs", "gnome-shell"}

async def handle_list_open_windows(args: dict) -> dict:
    try:
        # Use state.open_windows since it's already updated and parsed
        windows = [w for w in state.open_windows if w not in SYSTEM_WINDOWS]
        if not windows:
            return {"windows": "No open windows found."}
        return {"windows": "\n".join(windows)}
    except Exception as e:
        return {"error": str(e)}


def register():
    bus.register("run_bg_cmd",            handle_run_bg_cmd)
    bus.register("open_visible_terminal", handle_open_visible_terminal)
    bus.register("check_battery",         handle_check_battery)
    bus.register("get_active_window",     handle_get_active_window)
    bus.register("list_open_windows",     handle_list_open_windows)
    bus.register("read_notifications",    handle_read_notifications)
    log.info("SysAdmin agent registered ✓")
