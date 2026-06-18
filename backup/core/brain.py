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
from typing import Optional, Callable

from core.schema_registry import TOOL_SCHEMAS, VISIBLE_TOOL_SCHEMAS, validate_tool_call, get_openai_tools
from core.ipc_bus import bus
from core.state_machine import StateMachine, State
import core.llm_provider as llm_mod
from core import profiler_emitter

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [BRAIN] %(message)s")
log = logging.getLogger("brain")

# ── Config ───────────────────────────────────────────────────────────────────
MAX_REACT_STEPS = 10
TOKEN_LIMIT     = 3000
COMPRESS_EVERY_TURNS = 16  # 8 user turns

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


def format_memory_context(memory_dict: dict) -> str:
    if not memory_dict:
        return "(empty)"
    
    parts = []
    profile = memory_dict.get("user_profile", "")
    if isinstance(profile, dict):
        items = []
        for k, v in profile.items():
            if isinstance(v, list):
                items.extend([str(item).strip() for item in v if item])
            elif v:
                items.append(str(v).strip())
        profile = "\n  ".join(items)
    
    profile = str(profile).strip()
    if profile:
        parts.append(f"User Profile:\n  {profile}")
        
    memories_raw = memory_dict.get("relevant_memories", [])
    if isinstance(memories_raw, list):
        memories = [str(m).strip() for m in memories_raw if str(m).strip()]
    else:
        memories = [str(memories_raw).strip()] if str(memories_raw).strip() else []
        
    if memories:
        parts.append("Relevant Facts/Memories:")
        for m in memories:
            parts.append(f"  - {m}")
            
    docs_raw = memory_dict.get("relevant_documents", [])
    if isinstance(docs_raw, list):
        docs = [str(d).strip() for d in docs_raw if str(d).strip()]
    else:
        docs = [str(docs_raw).strip()] if str(docs_raw).strip() else []
        
    if docs:
        parts.append("Relevant Document Snippets:")
        for d in docs:
            flat = d.replace("\n", " ")
            parts.append(f"  - {flat}")
            
    return "\n".join(parts) if parts else "(empty)"


def process_time_words(text: str) -> str:
    """Replaces temporal words like 'yesterday' or 'today' with absolute date strings before search."""
    if not text:
        return text
    from datetime import date, timedelta
    today = date.today()
    
    # Yesterday
    if "yesterday" in text.lower():
        val = today - timedelta(days=1)
        text = re.sub(r"\byesterday\b", f"on {val.strftime('%A, %Y-%m-%d')}", text, flags=re.IGNORECASE)
    
    # Today
    if "today" in text.lower():
        text = re.sub(r"\btoday\b", f"on {today.strftime('%A, %Y-%m-%d')}", text, flags=re.IGNORECASE)
        
    return text


def should_query_supermemory(prompt: str) -> bool:
    """Deterministic check to skip Supermemory queries for greetings and simple OS/system actions."""
    if not prompt:
        return False
    lowered = prompt.lower().strip().strip("?!.")
    
    # Greetings/simple chat
    greetings = {
        "hi", "hello", "hey", "yo", "good morning", "good afternoon", "good evening",
        "who are you", "what is your name", "tell me about yourself", "bye", "goodbye", "exit", "quit"
    }
    if lowered in greetings:
        return False
        
    # Simple OS actions/status checks that don't need semantic memory
    os_keywords = {
        "battery", "window", "screenshot", "screen shot", "click", "type", 
        "press", "scroll", "open", "launch", "restart", "shutdown", "mute", "volume"
    }
    
    # If it contains any OS keywords and is relatively short, skip Supermemory
    words = set(lowered.split())
    if words.intersection(os_keywords) and len(words) <= 6:
        return False
        
    return True


def format_user_message_with_context(user_input: str, context: dict) -> str:
    """Formats the dynamic session context to be sent inside the user message (caching-friendly)."""
    cwd = context.get("cwd", "unknown")
    logs_dir = context.get("logs_dir", "unknown")
    memory_str = format_memory_context(context.get("memory", {}))
    
    parts = []
    parts.append("=== DYNAMIC CONTEXT ===")
    parts.append(f"working_dir : {cwd}")
    parts.append(f"logs_dir    : {logs_dir}")
    if memory_str and memory_str != "(empty)":
        parts.append(f"=== INJECTED MEMORY ===\n{memory_str}")
    
    parts.append(f"\nUser Query: {user_input}")
    return "\n".join(parts)


# ── Unified System Prompt (first call — lightweight) ─────────────────────────
def build_unified_prompt(context: dict = None) -> str:
    """
    Lightweight first-call prompt. Includes tool NAMES only (no full schemas).
    Full schemas injected only when entering ReAct loop.
    The model decides: plain text reply = conversational, JSON tool call = agentic.
    """
    from core.schema_registry import EXCLUDED_TOOLS
    available = [t for t in bus.registered_tools() if t not in EXCLUDED_TOOLS]
    current_date = datetime.now().strftime("%A, %Y-%m-%d")

    return (
        "You are Mr Meeseeks — a local AI OS companion running on Ubuntu.\n"
        f"Current Date: {current_date}\n"
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
        "   Examples: open netflix, check battery, take a screenshot\n"
        "\n"
        "CRITICAL for mode 2: output ONLY the JSON. Zero intro text. Zero explanation.\n"
        "\n"
        "=== TOOL RULES ===\n"
        "run_bg_cmd           — ALL read operations: head, cat, grep, ls, find, ps, df, wmctrl\n"
        '                       Read file: {"tool": "run_bg_cmd", "args": {"cmd": "head -n 20 /path/to/file"}}\n'
        "open_visible_terminal — ONLY for commands that MODIFY or LAUNCH: install, xdg-open, scripts\n"
        "\n"
        f"=== AVAILABLE TOOLS ===\n{available}\n"
        "\n"
        "Memory workflow:\n"
        "  - Don't know the key? Call list_memory_keys first to see what's stored.\n"
        '  - Recall: {"tool": "fetch_memory", "args": {"keys": ["name"]}}\n'
        '  - Save:   {"tool": "update_memory", "args": {"key": "name", "data": "kshayik"}}\n'
    )


# ── Agentic System Prompt (ReAct loop — full schemas) ────────────────────────
def build_system_prompt(context: dict = None) -> str:
    from core.schema_registry import EXCLUDED_TOOLS, VISIBLE_TOOL_SCHEMAS
    available = [t for t in bus.registered_tools() if t not in EXCLUDED_TOOLS]
    schemas_str = json.dumps(VISIBLE_TOOL_SCHEMAS, indent=2)
    current_date = datetime.now().strftime("%A, %Y-%m-%d")

    return (
        "You are Mr Meeseeks — a local AI OS companion running on Ubuntu.\n"
        f"Current Date: {current_date}\n"
        "You assist with coding, system tasks, and research.\n"
        "\n"
        "=== STRICT OUTPUT RULES ===\n"
        "1. Output ONLY valid JSON. Zero free text. Zero markdown.\n"
        "2. One JSON object per response.\n"
        "3. When done, emit the 'done' tool with your spoken response.\n"
        "4. Never guess coordinates.\n"
        "5. Never emit destructive commands in run_bg_cmd.\n"
        "6. If a tool returns an error, try a DIFFERENT approach.\n"
        "7. If you cannot complete a task, emit done and explain honestly.\n"
        "8. When a tool returns data, DESCRIBE IT in your done speech — do NOT ignore it.\n"
        "\n"
        "=== TOOL USAGE RULES ===\n"
        "run_bg_cmd:\n"
        "  USE THIS to run ANY terminal command to get its output (e.g., cat, grep, ls, python scripts).\n"
        '  {"tool": "run_bg_cmd", "args": {"cmd": "uname -a"}}\n'
        "\n"
        "open_visible_terminal:\n"
        "  USE ONLY when the user explicitly asks to open a terminal or launch a GUI app.\n"
        '  Open app: {"tool": "open_visible_terminal", "args": {"cmd": "xdg-open https://..."}}\n'
        "  DO NOT use for read-only commands or web searches.\n"
        "\n"
        "simple_scrape:\n"
        "  USE THIS silently in the background whenever you need to search the web for latest knowledge or news.\n"
        '  {"tool": "simple_scrape", "args": {"query": "latest nvidia news"}}\n'
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
        '-> {"thought": "User wants to open a GUI app, use open_visible_terminal.", "tool": "open_visible_terminal", "args": {"cmd": "xdg-open https://youtube.com"}}\n'
        "\n"
        'User: "what is the latest news about NVIDIA?"\n'
        '-> {"thought": "I need to search the web for recent info.", "tool": "simple_scrape", "args": {"query": "latest NVIDIA news"}}\n'
        '   (gets result) -> {"thought": "Got the news.", "tool": "done", "args": {"speech": "Here is the latest news..."}}\n'
        "\n"
        'User: "what is my city?"\n'
        '-> {"thought": "Not sure of key, list memory keys first.", "tool": "list_memory_keys", "args": {}}\n'
        '   (sees {"keys": ["city_name", "name"]})\n'
        '-> {"thought": "Key is city_name.", "tool": "fetch_memory", "args": {"keys": ["city_name"]}}\n'
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
        # Check based on message count/tokens; do not gate by time
        turns_heavy  = len(self.messages) > COMPRESS_EVERY_TURNS
        token_heavy  = self.token_estimate() > TOKEN_LIMIT
        return turns_heavy or token_heavy

    async def compress(self):
        """Summarize conversation → save raw log → wipe to summary only."""
        if not self.messages:
            return

        profiler_emitter.emit("function_call", name="ConversationHistory.compress", status="start")
        try:
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
        finally:
            profiler_emitter.emit("function_call", name="ConversationHistory.compress", status="end")


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
            if isinstance(parsed, dict):
                # Standardize OpenAI/Ollama function calling schema keys to Mr Meeseeks format
                if "name" in parsed and "tool" not in parsed:
                    parsed["tool"] = parsed["name"]
                if "arguments" in parsed and "args" not in parsed:
                    parsed["args"] = parsed["arguments"]
                
                if "tool" in parsed:
                    return parsed
        except json.JSONDecodeError:
            continue

    return None


class SentenceStreamer:
    def __init__(self, callback):
        self.callback = callback
        self.buffer = ""
        self.delimiters = {".", "?", "!", "\n", ";"}

    def add_chunk(self, chunk: str):
        self.buffer += chunk
        while True:
            earliest_idx = -1
            for d in self.delimiters:
                idx = self.buffer.find(d)
                if idx != -1:
                    if earliest_idx == -1 or idx < earliest_idx:
                        earliest_idx = idx
            
            if earliest_idx == -1:
                break
                
            sentence = self.buffer[:earliest_idx + 1].strip()
            self.buffer = self.buffer[earliest_idx + 1:]
            if sentence:
                self.callback(sentence)
                
    def flush(self):
        sentence = self.buffer.strip()
        if sentence:
            self.callback(sentence)
        self.buffer = ""


async def unified_stream_call(
    user_input: str,
    history: ConversationHistory,
    context: dict,
    on_text_chunk: Callable[[str], None],
    on_sentence: Callable[[str], None],
) -> tuple[Optional[str], Optional[dict]]:
    """
    Streams the unified call.
    Automatically detects if response is agentic (tool call) or conversational (plain text).
    - If agentic: gathers full response, parses tool call, and returns (None, tool_call).
    - If conversational: streams content to stdout/voice, and returns (full_text, None).
    """
    provider = llm_mod.provider
    if provider is None:
        return "LLM provider not initialized.", None

    profiler_emitter.emit("llm_start", prompt=user_input, provider=provider.__class__.__name__)

    system_prompt = build_unified_prompt()
    messages = history.messages.copy()
    user_content = format_user_message_with_context(user_input, context)
    messages.append({"role": "user", "content": user_content})

    log.info(f"Unified stream call for: {user_input[:60]}")

    full_content = ""
    tool_calls_list = []
    is_agentic = None  # None = undecided, True = tool calling, False = plain text
    
    sentence_streamer = SentenceStreamer(on_sentence)

    try:
        async for chunk in provider.stream_complete(
            system_prompt=system_prompt,
            messages=messages,
            temperature=0.5,
            max_tokens=300,
            tools=get_openai_tools()
        ):
            # Check if this chunk indicates a tool call
            if chunk.get("tool_calls"):
                if is_agentic is None:
                    is_agentic = True
                    log.info("Unified stream → AGENTIC mode detected (native tool calls)")
                tool_calls_list.extend(chunk["tool_calls"])
            
            content_chunk = chunk.get("content", "")
            if content_chunk:
                profiler_emitter.emit("llm_chunk", chunk=content_chunk)
                full_content += content_chunk
                
                # If we haven't decided if it's agentic or not, check for typical JSON/tool-call starts in the text
                if is_agentic is None:
                    stripped = full_content.strip()
                    # If it starts like a JSON object or markdown block, it's likely textual tool call
                    if stripped.startswith("{") or stripped.startswith("```"):
                        pass
                    elif len(stripped) > 10:
                        # Definitely plain text conversational
                        is_agentic = False
                        log.info("Unified stream → CONVERSATIONAL mode detected (plain text)")
                        if on_text_chunk:
                            on_text_chunk(full_content)
                        sentence_streamer.add_chunk(full_content)
                elif is_agentic == False:
                    if on_text_chunk:
                        on_text_chunk(content_chunk)
                    sentence_streamer.add_chunk(content_chunk)

        # Flush any remaining text in sentence buffer
        if is_agentic == False:
            sentence_streamer.flush()

    except Exception as e:
        log.exception(f"Unified stream LLM call failed: {e}")
        profiler_emitter.emit("llm_end", error=str(e), is_agentic=False)
        return f"Sorry, something went wrong: {e}", None

    full_content = full_content.strip()

    # Now let's finalise
    # 1. Native tool call check
    if tool_calls_list:
        tc = tool_calls_list[0]
        tool_call = {
            "thought": full_content,
            "tool": tc.get("name"),
            "args": tc.get("args", {})
        }
        profiler_emitter.emit("llm_end", is_agentic=True, tool_call=tool_call, full_content=full_content)
        valid, error = validate_tool_call(tool_call)
        if valid:
            log.info(f"Unified stream → AGENTIC tool: {tool_call.get('tool')}")
            return None, tool_call
        else:
            log.warning(f"Unified stream tool call invalid: {error}")
            return None, tool_call

    # 2. Textual tool call check (fallback)
    tool_call = parse_tool_call(full_content)
    if tool_call is not None:
        profiler_emitter.emit("llm_end", is_agentic=True, tool_call=tool_call, full_content=full_content)
        valid, error = validate_tool_call(tool_call)
        if valid:
            log.info(f"Unified stream (text fallback) → AGENTIC tool: {tool_call.get('tool')}")
            return None, tool_call
        else:
            log.warning(f"Unified stream (text fallback) tool call invalid: {error}")
            return None, tool_call

    # 3. Plain text response
    log.info("Unified stream → CONVERSATIONAL complete")
    if full_content.startswith("{"):
        try:
            parsed = json.loads(full_content)
            text = (
                parsed.get("response")
                or parsed.get("speech")
                or parsed.get("args", {}).get("speech")
                or full_content
            )
            profiler_emitter.emit("llm_end", is_agentic=False, full_content=str(text))
            return str(text), None
        except json.JSONDecodeError:
            pass

    profiler_emitter.emit("llm_end", is_agentic=False, full_content=full_content)
    return full_content, None


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

    profiler_emitter.emit("llm_start", prompt=user_input, provider=provider.__class__.__name__)

    system_prompt = build_unified_prompt()

    messages = history.messages.copy()
    user_content = format_user_message_with_context(user_input, context)
    messages.append({"role": "user", "content": user_content})

    log.info(f"Unified call for: {user_input[:60]}")

    try:
        response = await provider.complete(
            system_prompt=system_prompt,
            messages=messages,
            temperature=0.5,
            max_tokens=300,
            force_json=False,  # model picks plain text or JSON naturally
            tools=get_openai_tools()
        )
    except Exception as e:
        log.error(f"Unified LLM call failed: {e}")
        profiler_emitter.emit("llm_end", error=str(e), is_agentic=False)
        return f"Sorry, something went wrong: {e}", None

    raw_content = response.get("content", "").strip()
    log.info(f"Unified raw content: {raw_content[:200]}")

    # Check for native tool calls
    tool_call = None
    if response.get("tool_calls"):
        tc = response["tool_calls"][0]
        tool_call = {
            "thought": raw_content,
            "tool": tc.get("name"),
            "args": tc.get("args", {})
        }
    else:
        # Fallback to textual parsing just in case model ignores native tools
        tool_call = parse_tool_call(raw_content)

    if tool_call is not None:
        profiler_emitter.emit("llm_end", is_agentic=True, tool_call=tool_call, full_content=raw_content)
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
    if raw_content.startswith("{"):
        try:
            parsed = json.loads(raw_content)
            text = (
                parsed.get("response")
                or parsed.get("speech")
                or parsed.get("args", {}).get("speech")
                or raw_content
            )
            profiler_emitter.emit("llm_end", is_agentic=False, full_content=str(text))
            return str(text), None
        except json.JSONDecodeError:
            pass

    profiler_emitter.emit("llm_end", is_agentic=False, full_content=raw_content)

    return raw_content, None


# ── ReAct Loop ────────────────────────────────────────────────────────────────
async def react_loop(
    user_input: str,
    history: ConversationHistory,
    context: dict,
    first_tool_call: Optional[dict] = None,
) -> tuple[str, list, bool]:
    profiler_emitter.emit("function_call", name="brain.react_loop", status="start")
    try:
        speech, all_tool_calls, success = await _react_loop_impl(user_input, history, context, first_tool_call)
        return speech, all_tool_calls, success
    finally:
        profiler_emitter.emit("function_call", name="brain.react_loop", status="end")

async def _react_loop_impl(
    user_input: str,
    history: ConversationHistory,
    context: dict,
    first_tool_call: Optional[dict] = None,
) -> tuple[str, list, bool]:
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
        return "LLM provider not initialized. Call init_provider() first.", [], False

    user_content = format_user_message_with_context(user_input, context)
    history.add("user", user_content)

    system_prompt  = build_system_prompt()
    steps          = 0
    observations   = []
    parse_failures = 0
    MAX_PARSE_FAIL = 3

    # action_key -> number of times dispatched
    seen_actions: dict[str, int] = {}
    force_done = False  # hard abort: exits loop cleanly without extra LLM call
    all_tool_calls = []

    # ── Optionally inject the pre-parsed first tool call ─────────────────────
    if first_tool_call is not None:
        all_tool_calls.append(first_tool_call)
        steps += 1
        log.info(f"ReAct step {steps}/{MAX_REACT_STEPS} (pre-parsed from unified call)")

        tool_name = first_tool_call.get("tool")
        tool_args = first_tool_call.get("args", {})
        thought   = first_tool_call.get("thought", "")

        profiler_emitter.emit("react_step", step=steps, thought=thought, tool=tool_name, args=tool_args)

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
            profiler_emitter.emit("agent_response", speech=speech)
            return speech, all_tool_calls, True
        else:
            action_key = f"{tool_name}:{json.dumps(tool_args, sort_keys=True)}"
            seen_actions[action_key] = 1

            log.info(f"Dispatching → {tool_name}({tool_args})")
            profiler_emitter.emit("tool_dispatch", tool=tool_name, args=tool_args)
            await brain.state_machine.transition(State.ACTING)
            result = await bus.dispatch(tool_name, tool_args)
            await brain.state_machine.transition(State.THINKING)
            profiler_emitter.emit("tool_result", tool=tool_name, result=str(result))
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
            return "I got stuck repeating the same action and couldn't finish. Please try rephrasing.", all_tool_calls, False

        messages = history.messages.copy()
        if observations:
            obs_text = "\n".join(
                f"[Observation {i+1}]: {o}" for i, o in enumerate(observations)
            )
            messages.append({"role": "user", "content": obs_text})

        try:
            response = await provider.complete(
                system_prompt=system_prompt,
                messages=messages,
                temperature=0.2,
                force_json=True,
                tools=get_openai_tools()
            )
        except httpx.HTTPStatusError as e:
            log.error(f"LLM HTTP error: {e.response.status_code}")
            if e.response.status_code == 429:
                return "I'm being rate-limited. Try again in a moment.", all_tool_calls, False
            return f"LLM call failed: {e.response.status_code}", all_tool_calls, False
        except Exception as e:
            log.error(f"LLM call failed: {e}")
            return f"Sorry, I couldn't reach the LLM backend: {e}", all_tool_calls, False

        raw_output = response.get("content", "")
        log.info(f"Raw output: {raw_output[:300]}")

        # ── Parse ────────────────────────────────────────────────────────────
        tool_call = None
        if response.get("tool_calls"):
            tc = response["tool_calls"][0]
            tool_call = {
                "thought": raw_output,
                "tool": tc.get("name"),
                "args": tc.get("args", {})
            }
        else:
            tool_call = parse_tool_call(raw_output)

        if not tool_call:
            parse_failures += 1
            log.warning(f"Parse failure {parse_failures}/{MAX_PARSE_FAIL}: {raw_output[:200]}")
            if parse_failures >= MAX_PARSE_FAIL:
                log.error("Too many parse failures. Aborting.")
                return "I kept producing malformed responses. Please try rephrasing.", all_tool_calls, False
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

        profiler_emitter.emit("react_step", step=steps, thought=thought, tool=tool_name, args=tool_args)

        if thought:
            log.info(f"Thought: {thought}")

        if tool_name == "done":
            all_tool_calls.append(tool_call)
            speech = tool_args.get("speech", "Done.")
            history.add("assistant", raw_output)
            log.info(f"ReAct done. Speech: {speech}")
            profiler_emitter.emit("agent_response", speech=speech)
            return speech, all_tool_calls, True

        if tool_name not in ("done",):
            all_tool_calls.append(tool_call)

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
        profiler_emitter.emit("tool_dispatch", tool=tool_name, args=tool_args)
        await brain.state_machine.transition(State.ACTING)
        result = await bus.dispatch(tool_name, tool_args)
        await brain.state_machine.transition(State.THINKING)
        profiler_emitter.emit("tool_result", tool=tool_name, result=str(result))
        log.info(f"Result: {str(result)[:300]}")

        result_str = json.dumps(result)
        observations.append(
            f"[Result from {tool_name}]: {result_str}\n"
            "Describe/use this result in your done speech. Do NOT confabulate or ignore it."
        )
        history.add("assistant", raw_output)
        history.add("user", f"[Tool result from {tool_name}]: {result_str}")

    log.warning("Hit MAX_REACT_STEPS — forcing done")
    return "I ran out of steps. The task may require tools I don't have yet.", all_tool_calls, False


# ── Context Builder ───────────────────────────────────────────────────────────
async def build_context(memory_agent, kernel_events: list, user_prompt: str = "") -> dict:
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

    memory = {}
    if user_prompt and should_query_supermemory(user_prompt):
        try:
            processed_prompt = process_time_words(user_prompt)
            from core.supermemory_client import SupermemoryClient
            sm = SupermemoryClient()

            async def fetch_docs():
                try:
                    return await asyncio.to_thread(
                        sm.search_all,
                        query=processed_prompt,
                        container_tags=["personal_notes", "projects"],
                        limit=2
                    )
                except Exception as ex:
                    log.warning(f"Error fetching docs from SM: {ex}")
                    return []

            async def fetch_mems():
                try:
                    return await asyncio.to_thread(
                        sm.search_memories,
                        query=processed_prompt,
                        container_tag="chat_memory",
                        limit=2
                    )
                except Exception as ex:
                    log.warning(f"Error fetching memories from SM: {ex}")
                    return []

            async def fetch_prof():
                try:
                    return await asyncio.to_thread(
                        sm.get_profile,
                        container_tag="chat_memory",
                        query=processed_prompt
                    )
                except Exception as ex:
                    log.warning(f"Error fetching profile from SM: {ex}")
                    return {}

            doc_chunks, memory_results, profile_data = await asyncio.gather(
                fetch_docs(),
                fetch_mems(),
                fetch_prof()
            )

            documents = []
            for item in doc_chunks:
                content = item.get("content", "")
                if content:
                    content_clean = content.strip()
                    if len(content_clean) > 300:
                        content_clean = content_clean[:300] + "... [truncated]"
                    documents.append(content_clean)

            memories = [m.get("content", "").strip() for m in memory_results if m.get("content")]
            profile_text = profile_data.get("profile", "") or profile_data.get("summary", "") or ""

            if documents or memories or profile_text:
                memory = {
                    "relevant_documents": documents,
                    "relevant_memories": memories,
                    "user_profile": profile_text
                }
        except Exception as e:
            log.warning(f"Failed to fetch semantic context from Supermemory: {e}")

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
        self.last_interaction: Optional[dict] = None

    def inject_memory_agent(self, agent):
        self.memory_agent = agent

    def push_kernel_event(self, event: dict):
        self.kernel_events.append({**event, "ts": time.time()})
        if len(self.kernel_events) > 50:
            self.kernel_events = self.kernel_events[-50:]

    async def process(
        self,
        user_input: str,
        on_chunk: Callable[[str], None] = None,
        on_sentence: Callable[[str], None] = None,
    ) -> str:
        """
        Main entry point. Unified streaming routing:
          1. Single lightweight streaming LLM call (unified_stream_call)
          2a. If plain text -> stream directly to console and TTS
          2b. If tool call -> enter react_loop (agentic path)
        """
        profiler_emitter.emit("user_input", prompt=user_input)
        profiler_emitter.emit("function_call", name="brain.process", status="start")
        await self.state_machine.transition(State.THINKING)

        profiler_emitter.emit("function_call", name="build_context", status="start")
        context = await build_context(self.memory_agent, self.kernel_events, user_input)
        profiler_emitter.emit("function_call", name="build_context", status="end")

        plain_text, tool_call = await unified_stream_call(
            user_input=user_input,
            history=self.history,
            context=context,
            on_text_chunk=on_chunk,
            on_sentence=on_sentence,
        )

        if plain_text is not None:
            # Conversational: model replied in plain text — done
            self.history.add("user", user_input)
            self.history.add("assistant", plain_text)
            speech = plain_text
            self.last_interaction = None
            profiler_emitter.emit("agent_response", speech=plain_text)
        else:
            # Agentic: model emitted a tool call — enter ReAct loop
            speech, tool_calls, success = await react_loop(
                user_input=user_input,
                history=self.history,
                context=context,
                first_tool_call=tool_call,
            )
            # Since react loop was not streamed, print/speak final response now
            if speech:
                if on_chunk:
                    on_chunk(speech + "\n")
                if on_sentence:
                    on_sentence(speech)

            self.last_interaction = {
                "input": user_input,
                "context": {
                    "active_window": context.get("active_window"),
                    "battery": context.get("battery", {}).get("level"),
                    "time": context.get("time"),
                },
                "tool_calls": tool_calls,
                "success": success,
                "timestamp": datetime.now().isoformat()
            }

        if self.memory_agent:
            await self.memory_agent.update_memory(
                "last_interaction",
                {
                    "user":     user_input,
                    "response": speech,
                    "ts":       datetime.now().isoformat(),
                }
            )

        if self.history.needs_compression():
            await self.history.compress()

        await self.state_machine.transition(State.IDLE)
        profiler_emitter.emit("function_call", name="brain.process", status="end")
        return speech



    async def handle_proactive_event(self, event: dict):
        """Called by kernel listeners for proactive alerts."""
        self.push_kernel_event(event)

        # Only trigger proactive LLM loops for urgent alerts (e.g. low battery)
        # to prevent background loops on frequent events (window focus, window list updates).
        if event.get("type") not in {"low_battery"}:
            return

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

        speech, _, _ = await react_loop(
            user_input=event_prompt,
            history=ConversationHistory(),  # fresh context — don't pollute main history
            context=context,
        )

        if speech.strip():
            await bus.dispatch("speak", {"text": speech})

        await self.state_machine.transition(State.IDLE)


# ── Singleton ─────────────────────────────────────────────────────────────────
brain = Brain()
