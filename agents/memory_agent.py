"""
memory_agent.py — Mr Meeseeks Memory Agent
Lightweight RAG using flat JSON files. No vector DB.

Storage layout:
  memory/store/{key}.json → {data: ..., updated_at: ...}

Brain injects this agent via brain.inject_memory_agent(agent).
Bus handlers registered via register().
"""

import json
import os
import logging
from datetime import datetime
from pathlib import Path

from core.ipc_bus import bus

log = logging.getLogger("memory_agent")

STORE_DIR = Path("memory/store")


def _key_to_path(key: str) -> Path:
    """Sanitize key → safe filename."""
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in key)
    return STORE_DIR / f"{safe}.json"


class MemoryAgent:
    def __init__(self):
        STORE_DIR.mkdir(parents=True, exist_ok=True)
        log.info(f"Memory store at: {STORE_DIR.resolve()}")

    # ── Core Operations ──────────────────────────────────────────────────────

    async def update_memory(self, key: str, data) -> dict:
        """
        Write or merge data into memory/store/{key}.json.
        If existing value is a dict and new data is a dict → shallow merge.
        Otherwise → replace.
        """
        path = _key_to_path(key)

        existing = {}
        if path.exists():
            try:
                with open(path) as f:
                    existing = json.load(f)
            except (json.JSONDecodeError, OSError):
                existing = {}

        if isinstance(existing.get("data"), dict) and isinstance(data, dict):
            merged = {**existing["data"], **data}
        else:
            merged = data

        payload = {
            "data":       merged,
            "updated_at": datetime.now().isoformat(),
        }

        try:
            with open(path, "w") as f:
                json.dump(payload, f, indent=2)
            log.info(f"Memory updated: {key}")
            return {"ok": True, "key": key}
        except OSError as e:
            log.error(f"Memory write failed for '{key}': {e}")
            return {"error": str(e)}

    async def fetch_memory(self, keys: list[str]) -> dict:
        """
        Read memory for a list of keys.
        Returns {key: data} for each key that exists.
        Missing keys are silently skipped.
        """
        result = {}
        for key in keys:
            path = _key_to_path(key)
            if path.exists():
                try:
                    with open(path) as f:
                        payload = json.load(f)
                    result[key] = payload.get("data")
                except (json.JSONDecodeError, OSError) as e:
                    log.warning(f"Memory read failed for '{key}': {e}")
        return result

    def list_keys(self) -> list[str]:
        """Return all stored memory keys."""
        if not STORE_DIR.exists():
            return []
        return [p.stem for p in STORE_DIR.glob("*.json")]


# ── Bus Handlers ──────────────────────────────────────────────────────────────

_agent: MemoryAgent | None = None


async def _handle_update_memory(args: dict) -> dict:
    key  = args.get("key", "")
    data = args.get("data")
    if not key:
        return {"error": "Missing 'key' argument"}
    return await _agent.update_memory(key, data)


async def _handle_fetch_memory(args: dict) -> dict:
    keys = args.get("keys", [])
    if not isinstance(keys, list):
        return {"error": "'keys' must be a list of strings"}
    return await _agent.fetch_memory(keys)


def register() -> MemoryAgent:
    """
    Create the MemoryAgent singleton, register bus handlers, and return it.
    Call from main.py:
        memory = memory_agent.register()
        brain.inject_memory_agent(memory)
    """
    global _agent
    _agent = MemoryAgent()
    bus.register("update_memory", _handle_update_memory)
    bus.register("fetch_memory",  _handle_fetch_memory)
    log.info("Memory agent registered ✓")
    return _agent
