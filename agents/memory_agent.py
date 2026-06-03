"""
memory_agent.py — Mr Meeseeks Memory Agent
Lightweight persistent store using flat JSON files.

Storage layout:
  memory/store/{key}.json → {data: ..., updated_at: ...}

Fetch uses fuzzy key matching:
  If model asks for "username" but "name" is stored → returns "name"'s data.
  Matching order: exact → substring → word overlap → closest stored key.

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


def _fuzzy_match(requested_key: str, stored_keys: list[str]) -> str | None:
    """
    Find the best matching stored key for a requested key.

    Priority:
      1. Exact match
      2. Stored key contains requested key (e.g. stored="user_name", req="name")
      3. Requested key contains stored key (e.g. stored="name", req="username")
      4. Word overlap (e.g. stored="user_name", req="user name")
      5. No match → None

    All comparisons are case-insensitive.
    """
    req = requested_key.lower().replace("_", " ").replace("-", " ")

    # 1. Exact
    for k in stored_keys:
        if k.lower() == requested_key.lower():
            return k

    # 2. Stored key is a substring of requested key
    for k in stored_keys:
        k_norm = k.lower().replace("_", " ").replace("-", " ")
        if k_norm in req:
            return k

    # 3. Requested key is a substring of stored key
    for k in stored_keys:
        k_norm = k.lower().replace("_", " ").replace("-", " ")
        if req in k_norm:
            return k

    # 4. Word overlap — at least one common word
    req_words = set(req.split())
    best_k    = None
    best_overlap = 0
    for k in stored_keys:
        k_words  = set(k.lower().replace("_", " ").replace("-", " ").split())
        overlap  = len(req_words & k_words)
        if overlap > best_overlap:
            best_overlap = overlap
            best_k = k

    return best_k if best_overlap > 0 else None


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
        Retrieve memory for a list of keys using fuzzy matching.

        For each requested key:
          1. Try exact match first.
          2. If no exact match, fuzzy-match against all stored keys.
          3. Return {requested_key: data} so the model gets what it asked for.

        Missing keys with no fuzzy match are silently skipped.
        """
        stored_keys = self.list_keys()
        result = {}

        for requested_key in keys:
            # Try exact first
            exact_path = _key_to_path(requested_key)
            if exact_path.exists():
                try:
                    with open(exact_path) as f:
                        payload = json.load(f)
                    result[requested_key] = payload.get("data")
                    continue
                except (json.JSONDecodeError, OSError) as e:
                    log.warning(f"Memory read failed for '{requested_key}': {e}")
                    continue

            # Fuzzy match
            matched_key = _fuzzy_match(requested_key, stored_keys)
            if matched_key:
                matched_path = _key_to_path(matched_key)
                try:
                    with open(matched_path) as f:
                        payload = json.load(f)
                    log.info(f"Fuzzy match: '{requested_key}' → '{matched_key}'")
                    # Return under the requested key so model finds it naturally
                    result[requested_key] = payload.get("data")
                except (json.JSONDecodeError, OSError) as e:
                    log.warning(f"Memory read failed for fuzzy match '{matched_key}': {e}")
            else:
                log.info(f"No match found for key '{requested_key}' (stored: {stored_keys})")

        return result

    def list_keys(self) -> list[str]:
        """Return all stored memory keys (filename stems)."""
        if not STORE_DIR.exists():
            return []
        return [p.stem for p in STORE_DIR.glob("*.json")]


# ── Bus Handlers ──────────────────────────────────────────────────────────────

_agent: MemoryAgent | None = None


async def _handle_list_memory_keys(args: dict) -> dict:
    keys = _agent.list_keys()
    return {"keys": keys}


async def _handle_update_memory(args: dict) -> dict:
    key  = args.get("key", "")
    data = args.get("data")
    if not key:
        return {"error": "Missing 'key' argument"}
    return await _agent.update_memory(key, data)


async def _handle_fetch_memory(args: dict) -> dict:
    keys = args.get("keys", [])
    if not isinstance(keys, list):
        # Tolerate model passing a single string
        if isinstance(keys, str):
            keys = [keys]
        else:
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
    bus.register("list_memory_keys", _handle_list_memory_keys)
    bus.register("update_memory", _handle_update_memory)
    bus.register("fetch_memory",  _handle_fetch_memory)
    log.info("Memory agent registered ✓")
    return _agent
