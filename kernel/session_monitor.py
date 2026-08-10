"""
session_monitor.py — Laptop-open / session-resume detection.

Tracks last_seen timestamp in ~/.meeseeks/last_seen.json.
On startup: if gap > BRIEF_TRIGGER_MINUTES → morning brief should run.
Updates last_seen periodically while running.
"""
import asyncio
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

log = logging.getLogger("session_monitor")

MEESEEKS_DIR = Path.home() / ".meeseeks"
LAST_SEEN_FILE = MEESEEKS_DIR / "last_seen.json"
UPDATE_INTERVAL = 300  # 5 minutes


class SessionMonitor:
    """
    Tracks session timestamps to determine if a morning brief should play.
    """

    def __init__(self):
        self.trigger_minutes = int(os.environ.get("BRIEF_TRIGGER_MINUTES", "30"))
        MEESEEKS_DIR.mkdir(parents=True, exist_ok=True)

    def _load_last_seen(self) -> Optional[datetime]:
        """Load last_seen timestamp from disk."""
        if not LAST_SEEN_FILE.exists():
            return None
        try:
            with open(LAST_SEEN_FILE, "r") as f:
                data = json.load(f)
            ts = data.get("timestamp")
            if ts:
                return datetime.fromisoformat(ts)
        except Exception:
            pass
        return None

    def _save_last_seen(self):
        """Write current timestamp to disk."""
        with open(LAST_SEEN_FILE, "w") as f:
            json.dump({"timestamp": datetime.now().isoformat()}, f)

    async def should_run_brief(self) -> bool:
        """
        Returns True if enough time has passed since last session
        to warrant a morning brief.
        Updates last_seen timestamp.
        """
        last_seen = await asyncio.to_thread(self._load_last_seen)
        now = datetime.now()

        # Update last_seen immediately
        await asyncio.to_thread(self._save_last_seen)

        if last_seen is None:
            log.info("[SessionMonitor] First run → trigger morning brief.")
            return True

        gap_minutes = (now - last_seen).total_seconds() / 60
        log.info(f"[SessionMonitor] Session gap: {gap_minutes:.1f} min (threshold: {self.trigger_minutes} min)")

        return gap_minutes >= self.trigger_minutes

    async def start_heartbeat(self):
        """Background task: update last_seen every 5 minutes."""
        while True:
            await asyncio.sleep(UPDATE_INTERVAL)
            try:
                await asyncio.to_thread(self._save_last_seen)
            except Exception as e:
                log.debug(f"[SessionMonitor] Heartbeat error: {e}")
