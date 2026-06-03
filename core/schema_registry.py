"""
schema_registry.py — Mr Meeseeks Tool Schema Registry
Brain reads this at startup. All tool calls validated here before dispatch.
"""

from typing import Any

# ── Tool Schemas (injected into system prompt) ────────────────────────────────
TOOL_SCHEMAS = {
    # ── Eyes ──
    "get_ui_elements": {
        "description": "Read all visible UI elements on screen. Returns list with name, role, x, y, id.",
        "args": {}
    },
    "read_element_text": {
        "description": "Read text from a specific UI element by id.",
        "args": {"id": "string — element id from get_ui_elements"}
    },
    "find_element_by_label": {
        "description": "Find a UI element by its visible label text.",
        "args": {"label": "string — text to search for"}
    },

    # ── Hands ──
    "move_mouse": {
        "description": "Move AI cursor to x,y. Use get_ui_elements first to find coords.",
        "args": {"x": "int", "y": "int"}
    },
    "click": {
        "description": "Click at current cursor position. btn: left|right|middle.",
        "args": {"btn": "string — left|right|middle", "action": "string — click|double|hold|release"}
    },
    "type_text": {
        "description": "Type text via physical keystrokes.",
        "args": {"text": "string"}
    },
    "key_press": {
        "description": "Press a keyboard shortcut. e.g. ctrl+c, ctrl+z, Return.",
        "args": {"keys": "string — key combo e.g. ctrl+s"}
    },
    "scroll": {
        "description": "Scroll at current cursor position.",
        "args": {"direction": "string — up|down", "amount": "int — number of scroll steps"}
    },

    # ── SysAdmin — silent read only ──
    "run_bg_cmd": {
        "description": "Run a READ-ONLY command silently in background. Only: cat, ls, grep, pwd, echo, ps, df, free, uname, which, find, head, tail, wc, stat, env, printenv. NO write/execute commands.",
        "args": {"cmd": "string — read-only shell command"}
    },
    "check_battery": {
        "description": "Get current battery level and charging status.",
        "args": {}
    },
    "get_active_window": {
        "description": "Get title and class of currently focused window.",
        "args": {}
    },
    "read_notifications": {
        "description": "Read current desktop notifications.",
        "args": {}
    },

    # ── SysAdmin — visible terminal ──
    "open_visible_terminal": {
        "description": "Open a real gnome-terminal and type a command. User watches and can kill. Use for ANY write/install/execute command.",
        "args": {"cmd": "string — command to execute visibly"}
    },

    # ── Memory ──
    "update_memory": {
        "description": "Save a fact or preference to persistent memory.",
        "args": {"key": "string — topic key", "data": "any — data to store"}
    },
    "fetch_memory": {
        "description": "Retrieve facts from memory by keys.",
        "args": {"keys": "list of strings — keys to retrieve"}
    },

    # ── Web ──
    "simple_scrape": {
        "description": "Fast background web search. Returns summary of results.",
        "args": {"query": "string"}
    },
    "gui_research": {
        "description": "Open browser, physically navigate, read results interactively. Slow but powerful.",
        "args": {"task": "string — what to research"}
    },

    # ── Voice ──
    "speak": {
        "description": "Speak text aloud via Kokoro TTS.",
        "args": {"text": "string"}
    },

    # ── Done ──
    "done": {
        "description": "Signal end of ReAct loop. Always emit this last with your spoken response.",
        "args": {"speech": "string — what to say to the user"}
    },
}

# ── Required args per tool ─────────────────────────────────────────────────────
REQUIRED_ARGS: dict[str, list[str]] = {
    "get_ui_elements":      [],
    "read_element_text":    ["id"],
    "find_element_by_label":["label"],
    "move_mouse":           ["x", "y"],
    "click":                ["btn", "action"],
    "type_text":            ["text"],
    "key_press":            ["keys"],
    "scroll":               ["direction", "amount"],
    "run_bg_cmd":           ["cmd"],
    "check_battery":        [],
    "get_active_window":    [],
    "read_notifications":   [],
    "open_visible_terminal":["cmd"],
    "update_memory":        ["key", "data"],
    "fetch_memory":         ["keys"],
    "simple_scrape":        ["query"],
    "gui_research":         ["task"],
    "speak":                ["text"],
    "done":                 ["speech"],
}

# ── Read-only commands whitelist (for run_bg_cmd safety gate) ─────────────────
READ_ONLY_CMDS = {
    # ── Core shell utils ──
    "cat", "ls", "grep", "pwd", "echo", "ps", "df", "free",
    "uname", "which", "find", "head", "tail", "wc", "stat",
    "env", "printenv", "whoami", "id", "uptime", "date", "file",
    "sort", "uniq", "cut", "awk", "sed", "tr", "diff", "less",
    "realpath", "dirname", "basename",

    # ── Process / system info ──
    "top", "htop", "lsof", "pgrep", "pstree",
    "lscpu", "lsmem", "lsblk", "lspci", "lsusb", "lshw",
    "dmidecode", "sensors",

    # ── Network (read only) ──
    "netstat", "ss", "hostname", "ip", "ifconfig",
    "nslookup", "dig", "host", "ping",
    "traceroute", "route", "arp",

    # ── X11 / window management (read) ──
    "xdotool",   # xdotool search/getwindowpid/getwindowname etc.
    "wmctrl",    # wmctrl -l  (list all windows)
    "xprop",     # xprop -root  (read window properties)
    "xrandr",    # display config info
    "xwininfo",  # window geometry info
    "xlsfonts",

    # ── Audio (read) ──
    "pactl",     # pactl list sinks / sources
    "amixer",    # amixer get
    "aplay",     # aplay -l  (list devices)

    # ── Disk / filesystem (read) ──
    "du", "mount", "findmnt", "blkid",
    "smartctl",  # disk health

    # ── Package / system info (read) ──
    "dpkg", "apt-cache", "snap",
    "systemctl", "journalctl", "dmesg",
    "timedatectl", "localectl", "hostnamectl",

    # ── Git (read) ──
    "git",       # git status / log / diff  (brain should not git push/commit)

    # ── Python / dev info ──
    "python", "python3", "pip", "node", "npm",
}

# ── Destructive command keywords (always → visible terminal + confirm) ────────
DESTRUCTIVE_CMDS = {
    "rm", "rmdir", "dd", "mkfs", "fdisk", "parted",
    "shred", "wipe", "format", "kill", "killall",
    "chmod", "chown", "mv", "truncate", "> /",
}


def validate_tool_call(tool_call: dict) -> tuple[bool, str]:
    """
    Validate a tool call dict against schema.
    Returns (is_valid, error_message).
    """
    if not isinstance(tool_call, dict):
        return False, "Tool call must be a JSON object"

    tool_name = tool_call.get("tool")
    if not tool_name:
        return False, "Missing 'tool' field"

    if tool_name not in TOOL_SCHEMAS:
        known = ", ".join(TOOL_SCHEMAS.keys())
        return False, f"Unknown tool '{tool_name}'. Known tools: {known}"

    args = tool_call.get("args", {})
    if not isinstance(args, dict):
        return False, f"'args' must be a JSON object, got {type(args).__name__}"

    required = REQUIRED_ARGS.get(tool_name, [])
    for req in required:
        if req not in args:
            return False, f"Tool '{tool_name}' missing required arg '{req}'"

    # extra safety: run_bg_cmd must use read-only command
    if tool_name == "run_bg_cmd":
        cmd = args.get("cmd", "").strip().split()[0]
        if cmd not in READ_ONLY_CMDS:
            return False, (
                f"run_bg_cmd only allows read-only commands. "
                f"'{cmd}' is not allowed. Use open_visible_terminal instead."
            )

    return True, ""


def is_destructive(cmd: str) -> bool:
    """Check if a shell command contains destructive operations."""
    first_word = cmd.strip().split()[0] if cmd.strip() else ""
    return first_word in DESTRUCTIVE_CMDS or any(d in cmd for d in {"> /", "sudo rm", "sudo dd"})
