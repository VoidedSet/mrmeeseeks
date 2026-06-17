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
import uuid
import asyncio
from datetime import datetime
from pathlib import Path

from core.ipc_bus import bus
from core.supermemory_client import SupermemoryClient

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
        self.session_id = f"session_{uuid.uuid4().hex[:8]}"
        self.sm_client = SupermemoryClient()

    # ── Core Operations ──────────────────────────────────────────────────────

    async def update_memory(self, key: str, data) -> dict:
        """
        Write or merge data into memory/store/{key}.json.
        Also writes/ingests into the Supermemory Graph DB.
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

        # 1. Update local cache/file
        try:
            with open(path, "w") as f:
                json.dump(payload, f, indent=2)
            log.info(f"Memory updated locally: {key}")
        except OSError as e:
            log.error(f"Memory local write failed for '{key}': {e}")
            return {"error": str(e)}

        # 2. Update Supermemory Graph DB
        try:
            if key == "last_interaction":
                # Ingest conversation turn
                if isinstance(data, dict):
                    user_text = data.get("user", "")
                    response_text = data.get("response", "")
                    if user_text and response_text:
                        messages = [
                            {"role": "user", "content": user_text},
                            {"role": "assistant", "content": response_text}
                        ]
                        def run_ingest():
                            return self.sm_client.ingest_conversation(
                                conversation_id=self.session_id,
                                messages=messages,
                                container_tags=["personal_notes"]
                            )
                        # Run sync requests inside a thread pool
                        asyncio.create_task(asyncio.to_thread(run_ingest))
            else:
                # Format fact and add memory
                fact_str = f"{key}: {json.dumps(merged)}" if isinstance(merged, (dict, list)) else f"{key}: {merged}"
                def run_add_mem():
                    return self.sm_client.add_memories(
                        memories=[fact_str],
                        container_tag="personal_notes"
                    )
                asyncio.create_task(asyncio.to_thread(run_add_mem))
        except Exception as e:
            log.warning(f"Failed to propagate memory '{key}' to Supermemory Graph DB: {e}")

        return {"ok": True, "key": key}

    async def fetch_memory(self, keys: list[str]) -> dict:
        """
        Retrieve memory for a list of keys using fuzzy matching.
        Combines exact/fuzzy local JSON retrieval with semantic Supermemory Graph DB lookup.
        """
        stored_keys = self.list_keys()
        result = {}

        for requested_key in keys:
            # 1. Local exact/fuzzy lookup
            exact_path = _key_to_path(requested_key)
            local_found = False
            if exact_path.exists():
                try:
                    with open(exact_path) as f:
                        payload = json.load(f)
                    result[requested_key] = payload.get("data")
                    local_found = True
                except (json.JSONDecodeError, OSError) as e:
                    log.warning(f"Local memory read failed for '{requested_key}': {e}")

            if not local_found:
                matched_key = _fuzzy_match(requested_key, stored_keys)
                if matched_key:
                    matched_path = _key_to_path(matched_key)
                    try:
                        with open(matched_path) as f:
                            payload = json.load(f)
                        result[requested_key] = payload.get("data")
                        local_found = True
                        log.info(f"Local fuzzy match: '{requested_key}' → '{matched_key}'")
                    except (json.JSONDecodeError, OSError) as e:
                        log.warning(f"Local memory read failed for fuzzy match '{matched_key}': {e}")

            # 2. Supermemory Graph DB lookup
            try:
                def query_sm():
                    return self.sm_client.search_memories(query=requested_key, container_tag="personal_notes", limit=3)
                
                sm_results = await asyncio.to_thread(query_sm)
                facts = [m.get("content") for m in sm_results if m.get("content")]
                if facts:
                    # Merge Graph DB results with local KV results
                    if requested_key in result:
                        val = result[requested_key]
                        if isinstance(val, dict):
                            val["semantic_facts"] = facts
                        elif isinstance(val, list):
                            result[requested_key] = val + facts
                        else:
                            result[requested_key] = [val] + facts
                    else:
                        result[requested_key] = facts
            except Exception as e:
                log.warning(f"Supermemory search failed for '{requested_key}': {e}")

            if requested_key not in result:
                log.info(f"No match found for key '{requested_key}' locally or in Graph DB.")

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
