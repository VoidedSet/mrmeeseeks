"""
schema_registry.py — Mr Meeseeks Tool Schema Registry
Brain reads this at startup. All tool calls validated here before dispatch.
"""

from typing import Any

# ── Tool Schemas (injected into system prompt) ────────────────────────────────
TOOL_SCHEMAS = {
    # ── Web ──
    "simple_scrape": {
        "description": "Search the web for real-time news, latest sports scores, football/Barca updates, tech news, or current facts.",
        "args": {"query": "string — web search query"}
    },
    "web_fetch": {
        "description": "Fetch text content from a specific URL.",
        "args": {"url": "string — URL to fetch"}
    },

    # ── Email ──
    "fetch_inbox": {
        "description": "Fetch and refresh recent unread emails from Gmail inbox.",
        "args": {"max": "int (optional, default 20) — max emails to fetch"}
    },
    "get_email_summary": {
        "description": "Get brief summaries of recent unread non-promotional emails.",
        "args": {"count": "int (optional, default 5) — number of emails to summarize"}
    },
    "read_email": {
        "description": "Read full body content of a specific email by UID.",
        "args": {"uid": "string — email UID"}
    },
    "search_emails": {
        "description": "Search inbox emails by sender name, company name, or subject keywords.",
        "args": {"query": "string — sender or keyword search query"}
    },

    # ── Memory ──
    "list_memory_keys": {
        "description": "List stored user preference memory keys.",
        "args": {}
    },
    "update_memory": {
        "description": "Save a personal user fact or preference to memory.",
        "args": {"key": "string — topic key", "data": "any — data to store"}
    },
    "fetch_memory": {
        "description": "Retrieve stored personal user facts or preferences by key (e.g. user_name, city). DO NOT use for emails or web news.",
        "args": {"keys": "list of strings — keys to retrieve"}
    },

    # ── Voice ──
    "speak": {
        "description": "Speak text aloud via Kokoro TTS.",
        "args": {"text": "string"}
    },

    # ── Done ──
    "done": {
        "description": "Emit this last with your spoken response at the end of ReACT Loop.",
        "args": {"speech": "string — what to say to the user"}
    },
}

# ── Required args per tool ─────────────────────────────────────────────────────
REQUIRED_ARGS: dict[str, list[str]] = {
    "simple_scrape":        ["query"],
    "web_fetch":            ["url"],
    "fetch_inbox":          [],
    "get_email_summary":    [],
    "read_email":           ["uid"],
    "search_emails":        ["query"],
    "list_memory_keys":     [],
    "update_memory":        ["key", "data"],
    "fetch_memory":         ["keys"],
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


EXCLUDED_TOOLS = {
    "list_memory_keys"
}

VISIBLE_TOOL_SCHEMAS = {k: v for k, v in TOOL_SCHEMAS.items() if k not in EXCLUDED_TOOLS}


def get_openai_tools() -> list[dict]:
    """Convert VISIBLE_TOOL_SCHEMAS to OpenAI/Ollama native tools array format."""
    tools = []
    for name, schema in VISIBLE_TOOL_SCHEMAS.items():
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
