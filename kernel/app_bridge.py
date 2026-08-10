"""
kernel/app_bridge.py — Window Title ↔ AT-SPI Process Name Bridge

Uses wmctrl -lp (PID per window) + ps (PID → process name) + pyatspi
to build a live mapping so the model can say "Firefox" or "Visual Studio Code"
and we resolve it to the correct AT-SPI process name ("firefox", "code").

Also detects apps that have no AT-SPI registration (accessibility disabled)
and returns a helpful fix message.
"""

import subprocess
import logging

log = logging.getLogger("app_bridge")

# Known app name aliases — user-friendly name → possible process names (ordered by priority)
_KNOWN_ALIASES: dict[str, list[str]] = {
    "visual studio code": ["code"],
    "vscode":             ["code"],
    "vs code":            ["code"],
    "code":               ["code"],
    "firefox":            ["firefox"],
    "mozilla firefox":    ["firefox"],
    "chrome":             ["google-chrome", "chrome"],
    "google chrome":      ["google-chrome", "chrome"],
    "chromium":           ["chromium", "chromium-browser"],
    "terminal":           ["gnome-terminal-server", "gnome-terminal"],
    "gnome terminal":     ["gnome-terminal-server"],
    "nautilus":           ["org.gnome.Nautilus", "nautilus"],
    "files":              ["org.gnome.Nautilus"],
    "libreoffice":        ["soffice"],
    "writer":             ["soffice"],
    "calc":               ["soffice"],
    "gedit":              ["gedit", "org.gnome.gedit"],
    "vlc":                ["vlc"],
    "spotify":            ["spotify"],
    "slack":              ["slack"],
    "discord":            ["discord"],
}


class AppBridge:
    """
    Singleton bridge table. Call refresh() to rebuild from live OS data.

    Table structure:
      proc_name (str) → {
        "atspi": str | None,   # AT-SPI app name if accessible, else None
        "windows": [str],      # wmctrl window titles for this process
        "accessible": bool,    # True if app is on the AT-SPI bus
      }
    """

    def __init__(self):
        self._table: dict[str, dict] = {}

    def refresh(self):
        """Rebuild bridge from wmctrl -lp + ps + AT-SPI registry."""
        try:
            self._table = _build_table()
        except Exception as e:
            log.warning(f"AppBridge refresh failed: {e}")

    def resolve(self, user_query: str) -> tuple[str | None, str | None]:
        """
        Resolve a user-given app name to an AT-SPI process name.

        Returns:
            (atspi_name, error_msg)
            - (atspi_name, None)  → found, use this name with pyatspi
            - (None, error_msg)   → not found or accessibility disabled
        """
        q = user_query.lower().strip()

        # 1. Check known alias table first
        candidates = _KNOWN_ALIASES.get(q, [])
        if not candidates:
            # Partial match in alias keys
            for alias_key, procs in _KNOWN_ALIASES.items():
                if q in alias_key or alias_key in q:
                    candidates = procs
                    break

        # 2. If we have candidates, look them up in the live table
        for proc in candidates:
            entry = self._table.get(proc.lower())
            if entry:
                if entry["accessible"]:
                    return entry["atspi"], None
                else:
                    return None, (
                        f"'{proc}' is running (windows: {entry['windows']}) "
                        f"but is NOT on the AT-SPI accessibility bus. "
                        f"To fix: gsettings set org.gnome.desktop.interface "
                        f"toolkit-accessibility true — then restart the app."
                    )

        # 3. Fuzzy match directly against live process names in table
        for proc_name, entry in self._table.items():
            if q in proc_name or proc_name in q:
                if entry["accessible"]:
                    return entry["atspi"], None
                else:
                    return None, (
                        f"'{proc_name}' is running but has no accessibility. "
                        f"Run: gsettings set org.gnome.desktop.interface "
                        f"toolkit-accessibility true — then restart the app."
                    )

        # 4. Fuzzy match against window titles
        for proc_name, entry in self._table.items():
            for title in entry["windows"]:
                if q in title.lower():
                    if entry["accessible"]:
                        return entry["atspi"], None
                    else:
                        return None, (
                            f"Found window '{title}' (proc: {proc_name}) "
                            f"but accessibility is disabled. "
                            f"Run: gsettings set org.gnome.desktop.interface "
                            f"toolkit-accessibility true — then restart the app."
                        )

        return None, f"No app matching '{user_query}' found in open windows or AT-SPI bus."

    def resolve_active_window(self, active_window_title: str) -> str | None:
        """
        Given the active window title (from xdotool), return its AT-SPI name.
        Returns None if not found or not accessible.
        """
        title_lower = active_window_title.lower()
        for proc_name, entry in self._table.items():
            for title in entry["windows"]:
                if title.lower() in title_lower or title_lower in title.lower():
                    if entry["accessible"]:
                        return entry["atspi"]
        return None

    def get_table(self) -> dict:
        return dict(self._table)


def _run(cmd: str) -> str:
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
        return r.stdout.strip()
    except Exception:
        return ""


def _build_table() -> dict[str, dict]:
    """Build mapping: proc_name → {atspi, windows, accessible}"""

    # Step 1: Get wmctrl -lp (window list with PIDs)
    wmctrl_out = _run("wmctrl -lp")
    pid_titles: dict[str, list[str]] = {}  # pid → [titles]
    for line in wmctrl_out.splitlines():
        parts = line.split(None, 4)
        if len(parts) >= 5:
            pid   = parts[2]
            title = parts[4].strip()
            if pid and title:
                pid_titles.setdefault(pid, []).append(title)

    # Step 2: PID → process name via ps
    pid_proc: dict[str, str] = {}
    all_pids = list(pid_titles.keys())
    if all_pids:
        pids_str = ",".join(all_pids)
        ps_out = _run(f"ps -p {pids_str} -o pid=,comm= 2>/dev/null")
        for line in ps_out.splitlines():
            parts = line.split(None, 1)
            if len(parts) == 2:
                pid_proc[parts[0].strip()] = parts[1].strip()

    # Step 3: Get AT-SPI app names + their pids
    atspi_names: set[str] = set()
    atspi_pid_name: dict[str, str] = {}  # pid → atspi name
    try:
        import pyatspi
        desktop = pyatspi.Registry.getDesktop(0)
        for app in desktop:
            if app and app.name:
                atspi_names.add(app.name)
                try:
                    # AT-SPI apps expose their PID
                    pid = str(app.get_process_id())
                    atspi_pid_name[pid] = app.name
                except Exception:
                    pass
    except Exception:
        pass

    # Step 4: Build table keyed by process name (lowercased)
    table: dict[str, dict] = {}
    for pid, titles in pid_titles.items():
        proc = pid_proc.get(pid, "").strip().lower()
        if not proc:
            continue

        # Try PID-based AT-SPI match first (most accurate)
        atspi_name = atspi_pid_name.get(pid)

        # Fall back to name-based fuzzy match
        if not atspi_name:
            for aname in atspi_names:
                if proc in aname.lower() or aname.lower() in proc:
                    atspi_name = aname
                    break

        entry = table.setdefault(proc, {
            "atspi":      None,
            "windows":    [],
            "accessible": False,
        })
        entry["windows"].extend(t for t in titles if t not in entry["windows"])
        if atspi_name:
            entry["atspi"]      = atspi_name
            entry["accessible"] = True

    return table


# ── Module-level singleton ────────────────────────────────────────────────────
bridge = AppBridge()
