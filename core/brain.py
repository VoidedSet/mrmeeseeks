"""
brain.py — Mr Meeseeks Core Brain
ReAct loop coordinator. Provider-agnostic (Groq / Ollama via llm_provider.py).

Routing:
  Unified first call → model outputs plain text → return directly (conversational)
                     → model outputs JSON tool call → enter ReAct loop (agentic)

No regex classifier. The model decides. One lightweight LLM call for simple queries,
one call (reused as react step 1) + N tool calls for agentic tasks.
"""

import asyncio
import json
import os
import re
import time
import logging
from datetime import datetime
from typing import Optional

import httpx

from core.schema_registry import TOOL_SCHEMAS, validate_tool_call
from core.ipc_bus import bus
from core.state_machine import StateMachine, State
import core.llm_provider as llm_mod

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [BRAIN] %(message)s")
log = logging.getLogger("brain")

# ── Config ───────────────────────────────────────────────────────────────────
MAX_REACT_STEPS = 10
TOKEN_LIMIT     = 4000
COMPRESS_EVERY  = 15 * 60  # seconds

# ── JSON parse retry hint ─────────────────────────────────────────────────────
_RETRY_HINT = (
    "Your last response was not valid JSON. "
    "Output ONLY a single JSON object — no other text, no markdown:\n"
    '{"thought": "your reasoning", "tool": "tool_name", "args": {...}}\n'
    "Exact arg names matter. Examples:\n"
    '  run_bg_cmd    -> {"tool": "run_bg_cmd", "args": {"cmd": "head -n 10 /path/file"}}\n'
    '  fetch_memory  -> {"tool": "fetch_memory", "args": {"keys": ["name"]}}\n'
    '  update_memory -> {"tool": "update_memory", "args": {"key": "name", "data": "kshayik"}}\n'
    '  done          -> {"tool": "done", "args": {"speech": "your spoken reply"}}'
)


# ── Unified System Prompt (first call — lightweight) ─────────────────────────
def build_unified_prompt(context: dict) -> str:
    """
    Lightweight first-call prompt. Includes tool NAMES only (no full schemas).
    Full schemas injected only when entering ReAct loop.
    The model decides: plain text reply = conversational, JSON tool call = agentic.
    """
    available    = bus.registered_tools()
    memory_str   = json.dumps(context.get("memory", {}), indent=2)
    open_windows = context.get("open_windows", [])
    windows_str  = "\n".join(f"  - {w}" for w in open_windows) if open_windows else "  (none detected)"

    return (
        "You are Mr Meeseeks — a local AI OS companion running on Ubuntu.\n"
        "You are helpful, direct, and have personality.\n"
        "\n"
        "=== HOW TO RESPOND ===\n"
        "You have two modes — pick based on what the user needs:\n"
        "\n"
        "1. PLAIN TEXT — for greetings, small talk, factual questions you already know.\n"
        "   Just write your reply naturally. No JSON.\n"
        "   Examples: hi, how are you, thanks, you da goat, tell me about yourself\n"
        "\n"
        "2. JSON TOOL CALL — for tasks needing system access, commands, or memory.\n"
        "   Output ONLY this JSON — NO text before or after it:\n"
        '   {"thought": "...", "tool": "tool_name", "args": {...}}\n'
        "   Examples: open netflix, check battery, take a screenshot, click a button\n"
        "\n"
        "CRITICAL for mode 2: output ONLY the JSON. Zero intro text. Zero explanation.\n"
        "\n"
        "=== WHAT'S ALREADY KNOWN (NO TOOL CALL NEEDED) ===\n"
        "The following is live OS state — use it directly without calling any tool:\n"
        f"  active_window : {context.get('active_window', 'unknown')}\n"
        f"  battery       : {context.get('battery', {})}\n"
        f"  time          : {context.get('time', '')}\n"
        f"  open_windows  :\n{windows_str}\n"
        "\n"
        "If the user asks 'what apps are open?', 'list open windows', 'what is my battery?', etc.\n"
        "→ Answer directly from the above. Do NOT call list_open_windows or check_battery.\n"
        "\n"
        "=== TOOL RULES ===\n"
        "run_bg_cmd           — ALL read operations: head, cat, grep, ls, find, ps, df, wmctrl\n"
        '                       Read file: {"tool": "run_bg_cmd", "args": {"cmd": "head -n 20 /path/to/file"}}\n'
        "open_visible_terminal — ONLY for commands that MODIFY or LAUNCH: install, xdg-open, scripts\n"
        "get_ui_elements      — see screen elements. PROCESS names: VS Code='code', Firefox='firefox'.\n"
        "                       CRITICAL: app= takes the PROCESS name, NOT the window title.\n"
        "                       VS Code  → app='code'  |  Firefox → app='firefox'  |  Terminal → app='gnome-terminal-server'\n"
        "                       If unsure: call list_at_spi_apps first to see all registered process names.\n"
        '                       VS Code: {"tool": "get_ui_elements", "args": {"app": "code"}}\n'
        '                       Firefox top bar: {"tool": "get_ui_elements", "args": {"app": "firefox", "region": {"x1":0,"y1":0,"x2":1920,"y2":80}}}\n'
        "\n"
        "list_at_spi_apps     — USE THIS when unsure of process name for get_ui_elements.\n"
        '                       {"tool": "list_at_spi_apps", "args": {}}\n'
        "\n"
        f"=== AVAILABLE TOOLS ===\n{available}\n"
        "\n"
        "Memory workflow:\n"
        "  - Don't know the key? Call list_memory_keys first to see what's stored.\n"
        '  - Recall: {"tool": "fetch_memory", "args": {"keys": ["name"]}}\n'
        '  - Save:   {"tool": "update_memory", "args": {"key": "name", "data": "kshayik"}}\n'
        "\n"
        "=== CURRENT CONTEXT ===\n"
        f"working_dir   : {context.get('cwd', 'unknown')}\n"
        f"logs_dir      : {context.get('logs_dir', 'unknown')}\n"
        "\n"
        "=== INJECTED MEMORY ===\n"
        + (memory_str if memory_str != "{}" else "(empty — nothing stored yet)")
    )


# ── Agentic System Prompt (ReAct loop — full schemas) ────────────────────────
def build_system_prompt(context: dict) -> str:
    available   = bus.registered_tools()
    schemas_str = json.dumps(TOOL_SCHEMAS, indent=2)
    memory_str  = json.dumps(context.get("memory", {}), indent=2)
    events_str  = json.dumps(context.get("recent_events", [])[-5:], indent=2)

    return (
        "You are Mr Meeseeks — a local AI OS companion running on Ubuntu.\n"
        "You assist with coding, system tasks, and research.\n"
        "\n"
        "=== STRICT OUTPUT RULES ===\n"
        "1. Output ONLY valid JSON. Zero free text. Zero markdown.\n"
        "2. One JSON object per response.\n"
        "3. When done, emit the 'done' tool with your spoken response.\n"
        "4. NEVER guess coordinates — call get_ui_elements first.\n"
        "5. Never emit destructive commands in run_bg_cmd.\n"
        "6. If a tool returns an error, try a DIFFERENT approach.\n"
        "7. If you cannot complete a task, emit done and explain honestly.\n"
        "8. When a tool returns data, DESCRIBE IT in your done speech — do NOT ignore it.\n"
        "\n"
        "=== TOOL USAGE RULES ===\n"
        "run_bg_cmd:\n"
        "  USE for ALL read-only ops: head, cat, grep, ls, find, ps, df, free, wmctrl, etc.\n"
        '  Read file: {"tool": "run_bg_cmd", "args": {"cmd": "head -n 20 /path/to/file"}}\n'
        "  DO NOT use for write/install/execute operations.\n"
        "\n"
        "open_visible_terminal:\n"
        "  USE ONLY for commands that modify the system or launch GUI apps.\n"
        '  Open URL: {"tool": "open_visible_terminal", "args": {"cmd": "xdg-open https://..."}}\n'
        "  DO NOT use for read-only commands.\n"
        "  DO NOT invent commands that don't exist (e.g. xdg-query does not exist).\n"
        "\n"
        "list_open_windows:\n"
        "  USE THIS (not get_active_window) when user asks for ALL windows.\n"
        '  {"tool": "list_open_windows", "args": {}}\n'
        "\n"
        "get_ui_elements:\n"
        "  Returns AT-SPI accessibility tree. Large output — summarize key items in done speech.\n"
        "  Only return element names/roles that are relevant to user's question.\n"
        "\n"
        "=== WHEN NOT TO USE TOOLS ===\n"
        "Answer from context/history WITHOUT tools for:\n"
        "- Questions about yourself or your capabilities -> emit done immediately\n"
        "- Questions about past conversation -> conversation history is in your messages\n"
        "- Greetings, factual knowledge -> emit done immediately\n"
        "\n"
        "=== MEMORY WORKFLOW ===\n"
        "If unsure of exact key name:\n"
        '  Step 1: {"tool": "list_memory_keys", "args": {}}   <- see all stored keys\n'
        '  Step 2: {"tool": "fetch_memory", "args": {"keys": ["the_right_key"]}}\n'
        "Never guess a key — always list first if uncertain.\n"
        "\n"
        "=== EXACT ARG NAMES ===\n"
        "run_bg_cmd           -> args.cmd    (string)\n"
        "open_visible_terminal -> args.cmd   (string)\n"
        "update_memory        -> args.key, args.data\n"
        "fetch_memory         -> args.keys   (LIST of strings, NOT a single string)\n"
        "done                 -> args.speech (string)\n"
        "\n"
        f"=== REGISTERED TOOLS ===\n{available}\n"
        "\n"
        f"=== FULL TOOL SCHEMAS ===\n{schemas_str}\n"
        "\n"
        "=== CURRENT CONTEXT ===\n"
        f"active_window : {context.get('active_window', 'unknown')}\n"
        f"battery       : {context.get('battery', 'unknown')}\n"
        f"time          : {context.get('time', datetime.now().strftime('%H:%M'))}\n"
        f"working_dir   : {context.get('cwd', 'unknown')}\n"
        f"logs_dir      : {context.get('logs_dir', 'unknown')}\n"
        f"recent_events : {events_str}\n"
        "\n"
        "=== INJECTED MEMORY ===\n"
        + (memory_str if memory_str != "{}" else "(empty)")
        + "\n\n"
        "=== EXAMPLES ===\n"
        'User: "list all open windows"\n'
        '-> {"thought": "Use list_open_windows for all windows.", "tool": "list_open_windows", "args": {}}\n'
        '   (gets result) -> {"thought": "Got window list.", "tool": "done", "args": {"speech": "Open windows: Firefox, VS Code, Terminal."}}\n'
        "\n"
        'User: "read first 10 lines of /home/user/file.txt"\n'
        '-> {"thought": "head is read-only, use run_bg_cmd.", "tool": "run_bg_cmd", "args": {"cmd": "head -n 10 /home/user/file.txt"}}\n'
        '   (gets output) -> {"thought": "Got file contents.", "tool": "done", "args": {"speech": "The file starts with: ..."}}\n'
        "\n"
        'User: "open youtube"\n'
        '-> {"thought": "xdg-open launches browser.", "tool": "open_visible_terminal", "args": {"cmd": "xdg-open https://youtube.com"}}\n'
        "\n"
        'User: "what is my city?"\n'
        '-> {"thought": "Not sure of key, list memory keys first.", "tool": "list_memory_keys", "args": {}}\n'
        '   (sees {"keys": ["city_name", "name"]})\n'
        '-> {"thought": "Key is city_name.", "tool": "fetch_memory", "args": {"keys": ["city_name"]}}\n'
        '   (gets result) -> {"thought": "Got city.", "tool": "done", "args": {"speech": "Your city is Mumbai."}}\n'
    )


# ── Conversation History ──────────────────────────────────────────────────────
class ConversationHistory:
    def __init__(self):
        self.messages: list[dict] = []
        self._last_compress_time = time.time()

    def add(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})

    def token_estimate(self) -> int:
        total = sum(len(m["content"]) for m in self.messages)
        return total // 4

    def needs_compression(self) -> bool:
        time_elapsed = (time.time() - self._last_compress_time) > COMPRESS_EVERY
        token_heavy  = self.token_estimate() > TOKEN_LIMIT
        return time_elapsed or token_heavy

    async def compress(self):
        """Summarize conversation → save raw log → wipe to summary only."""
        if not self.messages:
            return

        provider = llm_mod.provider
        if provider is None:
            log.warning("Compression skipped — provider not initialized.")
            self.messages = self.messages[-5:]
            return

        log.info("Compressing context...")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        raw_path  = f"logs/raw/chat_{timestamp}.json"
        try:
            os.makedirs("logs/raw", exist_ok=True)
            with open(raw_path, "w") as f:
                json.dump(self.messages, f, indent=2)
            log.info(f"Raw log saved → {raw_path}")
        except Exception as e:
            log.warning(f"Failed to save raw log: {e}")

        summary_prompt = (
            "Summarize this conversation in bullet points. "
            "Capture: what user asked, what was done, key facts learned, errors hit. "
            "Be dense. No fluff. Output plain text summary only.\n\n"
            + "\n".join(f"{m['role']}: {m['content']}" for m in self.messages)
        )

        try:
            summary = await provider.complete(
                system_prompt="You are a summarizer. Output plain text only. No JSON.",
                messages=[{"role": "user", "content": summary_prompt}],
                temperature=0.1,
                max_tokens=512,
            )
        except Exception as e:
            log.warning(f"Compression LLM call failed: {e}. Keeping last 10 messages.")
            self.messages = self.messages[-10:]
            return

        self.messages = [{
            "role": "system",
            "content": f"[CONVERSATION SUMMARY — {datetime.now().strftime('%H:%M')}]\n{summary}"
        }]
        self._last_compress_time = time.time()
        log.info("Context compressed ✓")


# ── JSON Parser (hardened) ────────────────────────────────────────────────────
def parse_tool_call(raw: str) -> Optional[dict]:
    """
    Extract the first valid JSON tool call from model output.

    Handles:
    - Markdown code fences (```json ... ```)
    - Leading <think>...</think> blocks (CoT models)
    - Reasoning text before the JSON object
    - Multiple JSON blobs (takes first valid one with a "tool" key)
    """
    # 1. Strip <think>...</think> blocks
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

    # 2. Strip markdown fences
    if "```" in raw:
        raw = re.sub(r"```(?:json)?", "", raw).replace("```", "").strip()

    # 3. Find all {...} blobs and try each for a valid tool call
    candidates = []
    depth = 0
    start = None
    for i, ch in enumerate(raw):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                candidates.append(raw[start:i+1])
                start = None

    for blob in candidates:
        try:
            parsed = json.loads(blob)
            if isinstance(parsed, dict) and "tool" in parsed:
                return parsed
        except json.JSONDecodeError:
            continue

    return None


# ── Unified First Call ────────────────────────────────────────────────────────
async def unified_first_call(
    user_input: str,
    history: ConversationHistory,
    context: dict,
) -> tuple[Optional[str], Optional[dict]]:
    """
    Single lightweight LLM call. The model decides its own routing.

    Returns:
      (plain_text, None)   — conversational response, return directly
      (None, tool_call)    — model wants to use a tool, enter ReAct loop
      (error_text, None)   — LLM failure
    """
    provider = llm_mod.provider
    if provider is None:
        return "LLM provider not initialized.", None

    system_prompt = build_unified_prompt(context)

    messages = history.messages.copy()
    messages.append({"role": "user", "content": user_input})

    log.info(f"Unified call for: {user_input[:60]}")

    try:
        raw = await provider.complete(
            system_prompt=system_prompt,
            messages=messages,
            temperature=0.5,
            max_tokens=300,
            force_json=False,  # model picks plain text or JSON naturally
        )
    except Exception as e:
        log.error(f"Unified LLM call failed: {e}")
        return f"Sorry, something went wrong: {e}", None

    raw = raw.strip()
    log.info(f"Unified raw: {raw[:200]}")

    # Try to parse as a tool call
    tool_call = parse_tool_call(raw)

    if tool_call is not None:
        valid, error = validate_tool_call(tool_call)
        if valid:
            log.info(f"Unified → AGENTIC (tool: {tool_call.get('tool')})")
            return None, tool_call
        else:
            # Invalid tool call — enter react to self-correct with schema feedback
            log.warning(f"Unified tool call invalid: {error} — entering react to self-correct")
            return None, tool_call

    # Plain text — conversational
    log.info("Unified → CONVERSATIONAL (plain text)")

    # Handle edge case: Groq json_object mode forces JSON even with force_json=False
    if raw.startswith("{"):
        try:
            parsed = json.loads(raw)
            text = (
                parsed.get("response")
                or parsed.get("speech")
                or parsed.get("args", {}).get("speech")
                or raw
            )
            return str(text), None
        except json.JSONDecodeError:
            pass

    return raw, None


# ── ReAct Loop ────────────────────────────────────────────────────────────────
async def react_loop(
    user_input: str,
    history: ConversationHistory,
    context: dict,
    first_tool_call: Optional[dict] = None,
) -> str:
    """
    Full ReAct loop for agentic inputs.
    Think → emit tool call → observe result → repeat → done.
    Returns the final speech text.

    If first_tool_call is provided (from unified_first_call), it is used as
    step 1's output — no extra LLM call wasted.

    Loop protections:
    - MAX_REACT_STEPS hard limit
    - MAX_PARSE_FAIL abort on repeated parse failures
    - Repeated action detection: counter-based
        count==2: warn model, don't dispatch
        count>=3: set force_done flag, exit cleanly next iteration
    """
    provider = llm_mod.provider
    if provider is None:
        return "LLM provider not initialized. Call init_provider() first."

    if history.needs_compression():
        await history.compress()

    history.add("user", user_input)

    system_prompt  = build_system_prompt(context)
    steps          = 0
    observations   = []
    parse_failures = 0
    MAX_PARSE_FAIL = 3

    # action_key -> number of times dispatched
    seen_actions: dict[str, int] = {}
    force_done = False  # hard abort: exits loop cleanly without extra LLM call

    # ── Optionally inject the pre-parsed first tool call ─────────────────────
    if first_tool_call is not None:
        steps += 1
        log.info(f"ReAct step {steps}/{MAX_REACT_STEPS} (pre-parsed from unified call)")

        tool_name = first_tool_call.get("tool")
        tool_args = first_tool_call.get("args", {})
        thought   = first_tool_call.get("thought", "")

        if thought:
            log.info(f"Thought: {thought}")

        # Validate (unified call may have passed invalid tool call)
        valid, error = validate_tool_call(first_tool_call)
        if not valid:
            log.warning(f"Pre-parsed tool invalid: {error}")
            observations.append(
                f"ERROR: {error}. "
                f"Registered tools: {bus.registered_tools()}\n"
                f"{_RETRY_HINT}"
            )
        elif tool_name == "done":
            speech = tool_args.get("speech", "Done.")
            history.add("assistant", json.dumps(first_tool_call))
            log.info(f"ReAct done (immediate). Speech: {speech}")
            return speech
        else:
            action_key = f"{tool_name}:{json.dumps(tool_args, sort_keys=True)}"
            seen_actions[action_key] = 1

            log.info(f"Dispatching → {tool_name}({tool_args})")
            result = await bus.dispatch(tool_name, tool_args)
            log.info(f"Result: {str(result)[:300]}")

            result_str = json.dumps(result)
            observations.append(
                f"[Result from {tool_name}]: {result_str}\n"
                "Use this result to answer the user. Do NOT ignore it."
            )
            history.add("assistant", json.dumps(first_tool_call))
            history.add("user", f"[Tool result from {tool_name}]: {result_str}")

    # ── Main ReAct loop ───────────────────────────────────────────────────────
    while steps < MAX_REACT_STEPS:
        steps += 1
        log.info(f"ReAct step {steps}/{MAX_REACT_STEPS}")

        # Hard abort: repeated action fired 3x — skip LLM call, return immediately
        if force_done:
            log.warning("Force-done: aborting after repeated action")
            return "I got stuck repeating the same action and couldn't finish. Please try rephrasing."

        messages = history.messages.copy()
        if observations:
            obs_text = "\n".join(
                f"[Observation {i+1}]: {o}" for i, o in enumerate(observations)
            )
            messages.append({"role": "user", "content": obs_text})

        try:
            raw_output = await provider.complete(
                system_prompt,
                messages,
                force_json=True,  # agentic — must emit strict JSON
            )
        except httpx.HTTPStatusError as e:
            log.error(f"LLM HTTP error: {e.response.status_code}")
            if e.response.status_code == 429:
                return "I'm being rate-limited. Try again in a moment."
            return f"LLM call failed: {e.response.status_code}"
        except Exception as e:
            log.error(f"LLM call failed: {e}")
            return f"Sorry, I couldn't reach the LLM backend: {e}"

        log.info(f"Raw output: {raw_output[:300]}")

        # ── Parse ────────────────────────────────────────────────────────────
        tool_call = parse_tool_call(raw_output)

        if tool_call is None:
            parse_failures += 1
            log.warning(f"Parse failure {parse_failures}/{MAX_PARSE_FAIL}: {raw_output[:200]}")
            if parse_failures >= MAX_PARSE_FAIL:
                log.error("Too many parse failures. Aborting.")
                return "I kept producing malformed responses. Please try rephrasing."
            observations.append(_RETRY_HINT)
            continue
        else:
            parse_failures = 0

        # ── Schema validation ─────────────────────────────────────────────────
        valid, error = validate_tool_call(tool_call)
        if not valid:
            log.warning(f"Schema validation failed: {error}")
            observations.append(
                f"ERROR: {error}. "
                f"Registered tools: {bus.registered_tools()}"
            )
            continue

        tool_name = tool_call.get("tool")
        tool_args = tool_call.get("args", {})
        thought   = tool_call.get("thought", "")

        if thought:
            log.info(f"Thought: {thought}")

        # ── DONE ─────────────────────────────────────────────────────────────
        if tool_name == "done":
            speech = tool_args.get("speech", "Done.")
            history.add("assistant", raw_output)
            log.info(f"ReAct done. Speech: {speech}")
            return speech

        # ── Repeated action detection ─────────────────────────────────────────
        action_key = f"{tool_name}:{json.dumps(tool_args, sort_keys=True)}"
        count = seen_actions.get(action_key, 0) + 1
        seen_actions[action_key] = count

        if count == 2:
            # First repeat — warn and don't dispatch
            log.warning(f"Repeated action (2nd time): {action_key}")
            observations.append(
                f"LOOP WARNING: You already called '{tool_name}' with these exact args "
                "and got a result. Repeating it will give the SAME output. "
                "Try a DIFFERENT tool/approach, or emit 'done' and tell the user what you found."
            )
            continue
        elif count >= 3:
            # Second repeat — hard abort, will exit on next while iteration
            log.error(f"Repeated action {count}x — hard abort: {action_key}")
            force_done = True
            continue

        # ── Dispatch ──────────────────────────────────────────────────────────
        log.info(f"Dispatching → {tool_name}({tool_args})")
        result = await bus.dispatch(tool_name, tool_args)
        log.info(f"Result: {str(result)[:300]}")

        result_str = json.dumps(result)
        observations.append(
            f"[Result from {tool_name}]: {result_str}\n"
            "Describe/use this result in your done speech. Do NOT confabulate or ignore it."
        )
        history.add("assistant", raw_output)
        history.add("user", f"[Tool result from {tool_name}]: {result_str}")

    log.warning("Hit MAX_REACT_STEPS — forcing done")
    return "I ran out of steps. The task may require tools I don't have yet."


# ── Context Builder ───────────────────────────────────────────────────────────
async def build_context(memory_agent, kernel_events: list) -> dict:
    """
    Pull OS context + memory to inject into system prompt.

    Active window, all windows, and battery are read from KernelState (zero latency,
    updated by the background KernelListener). Falls back to tool dispatch if the
    listener hasn't warmed up yet (first few ms after startup).
    """
    from kernel.kernel_state import state as kstate

    snap = kstate.get_snapshot()

    # Active window — from KernelState (no subprocess needed)
    active_window = snap["active_window"]

    # Battery — from KernelState
    battery = snap["battery"]

    # Fall back to dispatching if KernelState is empty (listener not started yet)
    if active_window == "unknown" and not kstate.is_fresh("active_window", max_age_seconds=2.0):
        try:
            result = await bus.dispatch("get_active_window", {})
            active_window = result.get("window", "unknown")
        except Exception:
            pass

    if battery.get("level") == "unknown" and not kstate.is_fresh("battery", max_age_seconds=5.0):
        try:
            battery = await bus.dispatch("check_battery", {})
        except Exception:
            pass

    keywords = extract_keywords_from_events(kernel_events)

    memory = {}
    if memory_agent and keywords:
        memory = await memory_agent.fetch_memory(keywords)

    cwd = os.getcwd()

    return {
        "active_window": active_window,
        "open_windows":  snap["open_windows"],
        "battery":       battery,
        "time":          datetime.now().strftime("%H:%M"),
        "recent_events": kernel_events[-10:],
        "memory":        memory,
        "cwd":           cwd,
        "logs_dir":      os.path.join(cwd, "logs", "outputs"),
    }


def extract_keywords_from_events(events: list) -> list[str]:
    keywords = []
    for ev in events:
        if isinstance(ev, dict):
            keywords.extend([
                str(v) for v in ev.values()
                if isinstance(v, str) and len(v) > 3
            ])
    seen = set()
    out  = []
    for k in keywords:
        if k.lower() not in seen:
            seen.add(k.lower())
            out.append(k)
        if len(out) >= 10:
            break
    return out


# ── Brain Class ───────────────────────────────────────────────────────────────
class Brain:
    def __init__(self):
        self.history       = ConversationHistory()
        self.state_machine = StateMachine()
        self.kernel_events: list = []
        self.memory_agent  = None

    def inject_memory_agent(self, agent):
        self.memory_agent = agent

    def push_kernel_event(self, event: dict):
        self.kernel_events.append({**event, "ts": time.time()})
        if len(self.kernel_events) > 50:
            self.kernel_events = self.kernel_events[-50:]

    async def process(self, user_input: str) -> str:
        """
        Main entry point. Unified routing:
          1. Single lightweight LLM call (unified_first_call)
          2a. If plain text -> return directly (conversational path, 1 call total)
          2b. If tool call -> enter react_loop reusing the tool call (agentic path)
        """
        await self.state_machine.transition(State.THINKING)

        context = await build_context(self.memory_agent, self.kernel_events)

        plain_text, tool_call = await unified_first_call(user_input, self.history, context)

        if plain_text is not None:
            # Conversational: model replied in plain text — done
            self.history.add("user", user_input)
            self.history.add("assistant", plain_text)
            speech = plain_text
        else:
            # Agentic: model emitted a tool call — enter ReAct loop
            speech = await react_loop(
                user_input=user_input,
                history=self.history,
                context=context,
                first_tool_call=tool_call,
            )

        if self.memory_agent:
            await self.memory_agent.update_memory(
                "last_interaction",
                {
                    "user":     user_input,
                    "response": speech,
                    "ts":       datetime.now().isoformat(),
                }
            )

        await self.state_machine.transition(State.IDLE)
        return speech

    async def handle_proactive_event(self, event: dict):
        """Called by kernel listeners for proactive alerts."""
        self.push_kernel_event(event)

        if self.state_machine.current != State.IDLE:
            log.info(f"Proactive event queued (busy): {event}")
            return

        event_prompt = (
            f"OS event detected: {json.dumps(event)}. "
            f"Decide: should I alert the user? If yes, what do I say and do? "
            f"If not urgent, emit done with empty speech."
        )

        await self.state_machine.transition(State.THINKING)
        context = await build_context(self.memory_agent, self.kernel_events)

        speech = await react_loop(
            user_input=event_prompt,
            history=ConversationHistory(),  # fresh context — don't pollute main history
            context=context,
        )

        if speech.strip():
            await bus.dispatch("speak", {"text": speech})

        await self.state_machine.transition(State.IDLE)


# ── Singleton ─────────────────────────────────────────────────────────────────
brain = Brain()
