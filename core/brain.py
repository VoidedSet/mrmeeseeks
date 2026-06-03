"""
brain.py — Mr Meeseeks Core Brain
ReAct loop coordinator. Provider-agnostic (Groq / Ollama via llm_provider.py).
Outputs strict JSON tool calls only. No free text dispatch.

Routing:
  CONVERSATIONAL → single LLM call, no tools, returns text directly
  AGENTIC        → full ReAct loop with tool dispatch

Both paths share the same ConversationHistory, so context is always preserved.
"""

import asyncio
import json
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
    "Or if finished:\n"
    '{"thought": "done", "tool": "done", "args": {"speech": "your spoken reply"}}'
)


# ── Intent Classification ─────────────────────────────────────────────────────
# Patterns that clearly do NOT need tool calls.
# Matched against lowercase-stripped input. Ordering matters — more specific first.
_CONVERSATIONAL_PATTERNS = [
    # Greetings / social
    r"^(hi|hello|hey|howdy|sup|yo|hiya|greetings)\b",
    r"^(good (morning|evening|afternoon|night))\b",
    r"^(how are you|how's it going|what's up|how do you do)\b",
    r"^(thanks|thank you|cheers|appreciated|thx|ty)\b",
    r"^(ok|okay|cool|got it|understood|sure|alright|sounds good|great|nice)\b",
    r"^(bye|goodbye|cya|see you|exit|quit)\b",

    # Self-knowledge — answer from system prompt, no tools needed
    r"(what tools|which tools|what can you|what are your capabilities)",
    r"(what agents|which agents|what do you have access to)",
    r"(who are you|what are you|describe yourself|introduce yourself)",
    r"(how do you work|what is your purpose|what can you help with)",

    # History questions — answer from conversation context, no tools needed
    r"(what (did|have) (i|we)|questions? i asked|our conversation|what (was|were) (my|the) (question|request))",
    r"(summarize (our|this|the) conversation|recap|list (everything|what) (we|i))",
    r"(what did (you|mr meeseeks) (say|tell|respond|answer))",
]

_CONVERSATIONAL_RE = [re.compile(p, re.IGNORECASE) for p in _CONVERSATIONAL_PATTERNS]


def classify_intent(user_input: str) -> str:
    """
    Fast regex classifier. Returns 'conversational' or 'agentic'.
    No LLM call — zero latency overhead.
    Ambiguous inputs default to 'agentic' (safe to over-route to ReAct).
    """
    stripped = user_input.strip()
    # very short inputs with no action verb are usually conversational
    if len(stripped.split()) <= 3:
        for pattern in _CONVERSATIONAL_RE:
            if pattern.search(stripped):
                return "conversational"
    else:
        for pattern in _CONVERSATIONAL_RE:
            if pattern.search(stripped):
                return "conversational"
    return "agentic"


# ── Conversational System Prompt ──────────────────────────────────────────────
def build_conversational_prompt(context: dict) -> str:
    return f"""You are Mr Meeseeks — a local AI OS companion running on Ubuntu.
You are helpful, direct, and have a personality. You're talkative but not verbose.

This is a CONVERSATIONAL response — no tools needed. Respond naturally in plain text.

Rules:
- Answer from your own knowledge and the conversation history below.
- If asked what tools or agents you have, tell them from the list: {bus.registered_tools()}
- If asked about conversation history, look through the messages provided and summarize accurately.
- Do NOT ask "Is there anything else I can help with?" — it's annoying. Just answer.
- Be concise. 1-3 sentences max unless a longer answer is clearly needed.

Current context:
  time: {context.get("time", datetime.now().strftime("%H:%M"))}
  battery: {context.get("battery", "unknown")}
  active window: {context.get("active_window", "unknown")}
"""


# ── Agentic System Prompt ─────────────────────────────────────────────────────
def build_system_prompt(context: dict) -> str:
    available   = bus.registered_tools()
    schemas_str = json.dumps(TOOL_SCHEMAS, indent=2)
    memory_str  = json.dumps(context.get("memory", {}), indent=2)
    events_str  = json.dumps(context.get("recent_events", [])[-5:], indent=2)

    return f"""You are Mr Meeseeks — a local AI OS companion running on Ubuntu.
You are helpful, direct, and talkative. You assist with coding, system tasks, and research.

═══ STRICT OUTPUT RULES ═══
1. Output ONLY valid JSON. Zero free text. Zero markdown. Zero explanation outside JSON.
2. One JSON object per response.
3. When all tasks done, emit the "done" tool with your spoken response.
4. NEVER guess coordinates — call get_ui_elements first if you need x,y.
5. Safety gate handles destructive commands — never emit them in run_bg_cmd.
6. If a tool returns an error, reason about it and try a DIFFERENT approach.
7. If you cannot complete a task with available tools, emit done and explain honestly.

═══ WHEN NOT TO USE TOOLS ═══
Answer from context/history WITHOUT tool calls for:
- Questions about yourself, your tools, your capabilities → emit done immediately with answer
- Questions about past conversation → the full conversation history is in your messages context
- Greetings, acknowledgements, factual knowledge questions → emit done immediately
- Anything you already know the answer to without running a command

═══ TOOL CALL FORMAT ═══
{{"thought": "I need to check disk usage", "tool": "run_bg_cmd", "args": {{"cmd": "df -h"}}}}
{{"thought": "I have the answer, no more tools needed", "tool": "done", "args": {{"speech": "Your disk is 60% full."}}}}

═══ REGISTERED TOOLS (only call these) ═══
{available}

═══ FULL TOOL SCHEMAS ═══
{schemas_str}

═══ CURRENT CONTEXT ═══
active_window : {context.get("active_window", "unknown")}
battery       : {context.get("battery", "unknown")}
time          : {context.get("time", datetime.now().strftime("%H:%M"))}
recent_events : {events_str}

═══ INJECTED MEMORY ═══
{memory_str}

═══ EXAMPLES ═══
User: "what tools do you have?"
→ {{"thought": "This is a self-knowledge question. I know my tools from the registered list.", "tool": "done", "args": {{"speech": "I have access to: run_bg_cmd, check_battery, get_active_window, open_visible_terminal, update_memory, fetch_memory."}}}}

User: "what did I ask you before?"
→ {{"thought": "The user is asking about conversation history. I can see all prior messages in my context.", "tool": "done", "args": {{"speech": "You've asked about X, Y, and Z so far."}}}}

User: "list all open windows"
→ {{"thought": "I can use wmctrl or xdotool to list open windows.", "tool": "run_bg_cmd", "args": {{"cmd": "wmctrl -l"}}}}
"""


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
            import os
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


# ── Direct Conversational Response ───────────────────────────────────────────
async def direct_response(
    user_input: str,
    history: ConversationHistory,
    context: dict,
) -> str:
    """
    Single LLM call for conversational inputs.
    No tool dispatch. No ReAct loop.
    Uses full conversation history so the model can answer "what did I ask?" accurately.
    """
    provider = llm_mod.provider
    if provider is None:
        return "LLM provider not initialized."

    system_prompt = build_conversational_prompt(context)

    # Include full history so model can answer history questions accurately
    messages = history.messages.copy()
    messages.append({"role": "user", "content": user_input})

    log.info("Routing → CONVERSATIONAL (no tools)")

    try:
        # For Groq (json_object mode), we ask for JSON with a "response" field.
        # For Ollama, the model returns plain text naturally.
        # We try plain text first and only parse JSON if it looks like it.
        response = await provider.complete(
            system_prompt=system_prompt,
            messages=messages,
            temperature=0.5,
            max_tokens=300,
            force_json=False,   # conversational — plain text is fine
        )
    except Exception as e:
        log.error(f"Conversational LLM call failed: {e}")
        return f"Sorry, something went wrong: {e}"

    # If provider forced JSON (Groq json_object mode), extract the response field
    response = response.strip()
    if response.startswith("{"):
        try:
            parsed = json.loads(response)
            # Handle both {"response": "..."} and {"speech": "..."} and {"tool": "done", ...}
            text = (
                parsed.get("response")
                or parsed.get("speech")
                or parsed.get("args", {}).get("speech")
                or response
            )
            return str(text)
        except json.JSONDecodeError:
            pass

    return response


# ── ReAct Loop ────────────────────────────────────────────────────────────────
async def react_loop(
    user_input: str,
    history: ConversationHistory,
    context: dict,
) -> str:
    """
    Full ReAct loop for agentic inputs.
    Think → emit tool call → observe result → repeat → done.
    Returns the final speech text.

    Protections:
    - MAX_REACT_STEPS hard limit
    - MAX_PARSE_FAIL abort on repeated parse failures
    - Repeated action detection: same (tool, args) twice → force wrap-up
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

    # Repeated action detection: track (tool_name, serialized_args) pairs
    seen_actions: set[str] = set()

    while steps < MAX_REACT_STEPS:
        steps += 1
        log.info(f"ReAct step {steps}/{MAX_REACT_STEPS}")

        messages = history.messages.copy()
        if observations:
            obs_text = "\n".join(
                f"[Tool result {i+1}]: {o}" for i, o in enumerate(observations)
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
        if action_key in seen_actions:
            log.warning(f"Repeated action detected: {action_key}")
            # Inject strong corrective signal and force wrap-up in 2 more steps
            observations.append(
                f"LOOP DETECTED: You already tried '{tool_name}' with these exact args "
                f"and got the same result. Do NOT repeat it again. "
                f"Either try a completely different approach, or emit 'done' and "
                f"honestly tell the user what you cannot do with the tools available."
            )
            steps = MAX_REACT_STEPS - 2  # leave 2 steps to wrap up
            continue
        seen_actions.add(action_key)

        # ── Dispatch ──────────────────────────────────────────────────────────
        log.info(f"Dispatching → {tool_name}({tool_args})")
        result = await bus.dispatch(tool_name, tool_args)
        log.info(f"Result: {str(result)[:300]}")

        observations.append(f"{tool_name} → {json.dumps(result)}")
        history.add("assistant", raw_output)
        history.add("user", f"[Tool result]: {json.dumps(result)}")

    log.warning("Hit MAX_REACT_STEPS — forcing done")
    return "I ran out of steps. The task may require tools I don't have yet."


# ── Context Builder ───────────────────────────────────────────────────────────
async def build_context(memory_agent, kernel_events: list) -> dict:
    """Pull OS context + memory to inject into system prompt."""
    try:
        active_window = await bus.dispatch("get_active_window", {})
        battery       = await bus.dispatch("check_battery", {})
    except Exception:
        active_window = "unknown"
        battery       = "unknown"

    keywords = extract_keywords_from_events(kernel_events)

    memory = {}
    if memory_agent and keywords:
        memory = await memory_agent.fetch_memory(keywords)

    return {
        "active_window": active_window,
        "battery":       battery,
        "time":          datetime.now().strftime("%H:%M"),
        "recent_events": kernel_events[-10:],
        "memory":        memory,
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
        Main entry point. Routes to conversational or agentic path.
        Conversational: 1 LLM call, no tools, ~0.5-1s
        Agentic: full ReAct loop, N LLM calls
        """
        await self.state_machine.transition(State.THINKING)

        context = await build_context(self.memory_agent, self.kernel_events)

        intent = classify_intent(user_input)
        log.info(f"Intent: {intent} for: {user_input[:60]}")

        if intent == "conversational":
            speech = await direct_response(user_input, self.history, context)
            # Still add to history so model has full context for follow-ups
            self.history.add("user", user_input)
            self.history.add("assistant", speech)
        else:
            speech = await react_loop(
                user_input=user_input,
                history=self.history,
                context=context,
            )

        if self.memory_agent:
            await self.memory_agent.update_memory(
                "last_interaction",
                {
                    "user":     user_input,
                    "response": speech,
                    "intent":   intent,
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
