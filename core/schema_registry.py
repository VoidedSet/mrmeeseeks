"""
schema_registry.py — Mr Meeseeks Tool Schema Registry
Brain reads this at startup. All tool calls validated here before dispatch.
"""

from typing import Any

# ── Tool Schemas (injected into system prompt) ────────────────────────────────
TOOL_SCHEMAS = {
    # ── Eyes ──
    "get_ui_elements": {
        "description": (
            "Get interactive UI elements in an app or screen region. "
            "IMPORTANT: app= takes the PROCESS name (e.g. 'code' for VS Code, 'firefox' for Firefox), NOT the window title. "
            "If unsure of the process name, call list_at_spi_apps first. "
            "region={x1,y1,x2,y2} filters by screen area (e.g. top bar: {x1:0,y1:0,x2:1920,y2:50}). "
            "Returns elements grouped by category: buttons, menu, tabs, inputs, links, content."
        ),
        "args": {
            "app":    "string (optional) — PROCESS name e.g. 'code', 'firefox', 'gnome-terminal-server'",
            "region": "object (optional) — {x1: int, y1: int, x2: int, y2: int} screen region bounding box",
        }
    },
    "list_at_spi_apps": {
        "description": "List all apps on the AT-SPI accessibility bus with their process names. Call this first when you don't know the exact app name for get_ui_elements.",
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
    "click_at": {
        "description": (
            "Move cursor to (x, y) and click. ALWAYS use this instead of separate move+click. "
            "Get coords from get_ui_elements first. btn defaults to 'left'."
        ),
        "args": {
            "x":   "int — screen x coordinate (from get_ui_elements)",
            "y":   "int — screen y coordinate (from get_ui_elements)",
            "btn": "string (optional, default 'left') — left|right|middle",
        }
    },
    "double_click_at": {
        "description": "Move cursor to (x, y) and double-click.",
        "args": {"x": "int", "y": "int"}
    },
    "type_text": {
        "description": "Type text at current focus. Supports Unicode.",
        "args": {"text": "string"}
    },
    "key_press": {
        "description": "Press a keyboard shortcut. e.g. ctrl+c, ctrl+z, Return, ctrl+alt+t.",
        "args": {"keys": "string — key combo e.g. ctrl+s"}
    },
    "scroll": {
        "description": "Scroll at current cursor position.",
        "args": {"direction": "string — up|down", "amount": "int — scroll steps (default 3)"}
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
    "list_open_windows": {
        "description": "Get a list of all currently open window titles.",
        "args": {}
    },
    "read_notifications": {
        "description": "Read current desktop notifications.",
        "args": {}
    },

    # ── SysAdmin — visible terminal ──
    "open_visible_terminal": {
        "description": "Open a visible terminal for the user. USE ONLY when the user asks to launch a GUI app, open a URL explicitly, or run a dangerous command. Do NOT use this for background web searches or info fetching.",
        "args": {"cmd": "string — command to execute visibly"}
    },

    # ── Memory ──
    "list_memory_keys": {
        "description": "List all keys stored in memory. Call this first when you don't know the exact key name.",
        "args": {}
    },
    "update_memory": {
        "description": "Save a fact or preference to persistent memory.",
        "args": {"key": "string — topic key", "data": "any — data to store"}
    },
    "fetch_memory": {
        "description": "Retrieve facts from memory by keys. Uses fuzzy matching — approximate key names work.",
        "args": {"keys": "list of strings — keys to retrieve"}
    },

    # ── Web ──
    "simple_scrape": {
        "description": "Fast silent background web search. USE THIS ALWAYS when asked to find latest news, info, or search the web.",
        "args": {"query": "string — what to search for"}
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
    "list_at_spi_apps":     [],
    "read_element_text":    ["id"],
    "find_element_by_label":["label"],
    "click_at":             ["x", "y"],
    "double_click_at":      ["x", "y"],
    "type_text":            ["text"],
    "key_press":            ["keys"],
    "scroll":               [],
    "run_bg_cmd":           ["cmd"],
    "check_battery":        [],
    "get_active_window":    [],
    "list_open_windows":    [],
    "read_notifications":   [],
    "open_visible_terminal":["cmd"],
    "list_memory_keys":     [],
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

    # ── Config (Safe mutation) ──
    "gsettings", "dbus-send",


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

def get_openai_tools() -> list[dict]:
    """Convert TOOL_SCHEMAS to OpenAI/Ollama native tools array format."""
    tools = []
    for name, schema in TOOL_SCHEMAS.items():
        required = REQUIRED_ARGS.get(name, [])
        properties = {}
        for arg_name, arg_desc in schema.get("args", {}).items():
            arg_type = "string"
            if "int" in arg_desc: arg_type = "integer"
            elif "object" in arg_desc: arg_type = "object"
            elif "list" in arg_desc: arg_type = "array"
            elif "any" in arg_desc: arg_type = "string"  # fallback
            
            properties[arg_name] = {"type": arg_type, "description": arg_desc}
        
        tools.append({
            "type": "function",
            "function": {
                "name": name,
                "description": schema["description"],
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            }
        })
    return tools
