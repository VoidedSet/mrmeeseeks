"""
kernel_state.py — Mr Meeseeks Kernel State
Singleton that holds real-time OS state updated by the KernelListener.

brain.build_context() reads from here instead of dispatching tool calls,
saving ~50ms + LLM tokens on every single turn.

Fields:
    active_window  : title of the currently focused window
    open_windows   : list of all open window titles (from wmctrl)
    battery        : {"level": "85%", "status": "Discharging"}
    last_updated   : per-field timestamps (epoch float)
"""

import time
import logging

log = logging.getLogger("kernel_state")


class KernelState:
    """
    Single shared object. All fields are plain Python values — asyncio-safe
    because Python GIL protects reads/writes of individual attributes.

    Use get_snapshot() to get a consistent dict for context injection.
    """

    def __init__(self):
        self.active_window: str        = "unknown"
        self.open_windows:  list[str]  = []
        self.battery:       dict       = {"level": "unknown", "status": "unknown"}
        self.app_bridge:    dict       = {}  # proc_name → {atspi, windows, accessible}
        self.last_updated:  dict[str, float] = {
            "active_window": 0.0,
            "open_windows":  0.0,
            "battery":       0.0,
            "app_bridge":    0.0,
        }

    # ── Setters ───────────────────────────────────────────────────────────────

    def set_active_window(self, title: str) -> bool:
        """Returns True if value changed."""
        if title != self.active_window:
            self.active_window = title
            self.last_updated["active_window"] = time.time()
            return True
        return False

    def set_open_windows(self, titles: list[str]) -> bool:
        """Returns True if list changed."""
        if titles != self.open_windows:
            self.open_windows = titles
            self.last_updated["open_windows"] = time.time()
            return True
        return False

    def set_app_bridge(self, table: dict) -> bool:
        """Returns True if table changed."""
        if table != self.app_bridge:
            self.app_bridge = table
            self.last_updated["app_bridge"] = time.time()
            return True
        return False

    def set_battery(self, level: str, status: str) -> bool:
        """Returns True if value changed."""
        new = {"level": level, "status": status}
        if new != self.battery:
            self.battery = new
            self.last_updated["battery"] = time.time()
            return True
        return False

    # ── Snapshot ──────────────────────────────────────────────────────────────

    def get_snapshot(self) -> dict:
        """
        Returns a copy safe to inject into build_context().
        Always returns valid data — falls back to 'unknown' if not yet populated.
        """
        return {
            "active_window": self.active_window,
            "open_windows":  list(self.open_windows),
            "battery":       dict(self.battery),
            "last_updated":  dict(self.last_updated),
        }

    def is_fresh(self, field: str, max_age_seconds: float = 5.0) -> bool:
        """Check if a field has been updated recently."""
        ts = self.last_updated.get(field, 0.0)
        return (time.time() - ts) < max_age_seconds

    def __repr__(self) -> str:
        return (
            f"KernelState("
            f"window={self.active_window!r}, "
            f"windows={len(self.open_windows)}, "
            f"battery={self.battery})"
        )


# ── Global singleton ──────────────────────────────────────────────────────────
state = KernelState()
