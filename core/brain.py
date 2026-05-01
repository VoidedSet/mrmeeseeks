"""
brain.py — Mr Meeseeks Core Brain
ReAct loop coordinator. Qwen2.5:3b via Ollama.
Outputs strict JSON tool calls only. No free text dispatch.
"""

import asyncio
import json
import time
import logging
from datetime import datetime
from typing import Optional

import httpx

from schema_registry import TOOL_SCHEMAS, validate_tool_call
from ipc_bus import bus
from state_machine import StateMachine, State

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [BRAIN] %(message)s")
log = logging.getLogger("brain")

# ── Config ───────────────────────────────────────────────────────────────────
OLLAMA_URL      = "http://localhost:11434/api/chat"
MODEL           = "qwen2.5:3b"
MAX_REACT_STEPS = 10       # prevent infinite loops
TOKEN_LIMIT     = 3000     # trigger context compression before this
COMPRESS_EVERY  = 15 * 60  # seconds — also compress on timer


# ── System Prompt ─────────────────────────────────────────────────────────────
def build_system_prompt(context: dict) -> str:
    # in build_system_prompt()
    available = bus.registered_tools()
    schemas_str = json.dumps(TOOL_SCHEMAS, indent=2)
    memory_str  = json.dumps(context.get("memory", {}), indent=2)
    events_str  = json.dumps(context.get("recent_events", [])[-5:], indent=2)  # last 5

    return f"""You are Mr Meeseeks — a local AI OS companion running on Ubuntu.
You are helpful, direct, and talkative. You assist with coding, system tasks, and research.

STRICT OUTPUT RULES:
1. Output ONLY valid JSON. Zero free text. Zero markdown. Zero explanation outside JSON.
2. One JSON object per response.
3. When all tasks done, emit the done tool with your spoken response.
4. NEVER guess coordinates — call get_ui_elements first if you need x,y.
5. NEVER emit destructive commands directly — safety gate handles confirmation.
6. If a tool returns an error, reason about it and try a different approach.

CURRENTLY ONLINE TOOLS (only use these): {available}
AVAILABLE TOOLS:
{schemas_str}


CURRENT CONTEXT:
active_window : {context.get("active_window", "unknown")}
battery       : {context.get("battery", "unknown")}
time          : {context.get("time", datetime.now().strftime("%H:%M"))}
recent_events : {events_str}

INJECTED MEMORY:
{memory_str}

REACT LOOP FORMAT — your internal reasoning goes inside the JSON as "thought":
{{"thought": "I need to open a terminal first", "tool": "run_bg_cmd", "args": {{"cmd": "pwd"}}}}
{{"thought": "Got the path. Now I can answer.", "tool": "done", "args": {{"speech": "Your current directory is /home/user."}}}}
"""


# ── Conversation History ──────────────────────────────────────────────────────
class ConversationHistory:
    def __init__(self):
        self.messages: list[dict] = []
        self._last_compress_time = time.time()

    def add(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})

    def token_estimate(self) -> int:
        # rough: 1 token ≈ 4 chars
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

        log.info("Compressing context...")

        # save raw log first
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        raw_path = f"logs/raw/chat_{timestamp}.json"
        try:
            import os
            os.makedirs("logs/raw", exist_ok=True)
            with open(raw_path, "w") as f:
                json.dump(self.messages, f, indent=2)
            log.info(f"Raw log saved → {raw_path}")
        except Exception as e:
            log.warning(f"Failed to save raw log: {e}")

        # ask brain to summarize
        summary_prompt = (
            "Summarize this conversation in bullet points. "
            "Capture: what user asked, what was done, key facts learned, errors hit. "
            "Be dense. No fluff. Output plain text summary only.\n\n"
            + "\n".join(f"{m['role']}: {m['content']}" for m in self.messages)
        )

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(OLLAMA_URL, json={
                    "model": MODEL,
                    "messages": [{"role": "user", "content": summary_prompt}],
                    "stream": False,
                })
                summary = resp.json()["message"]["content"]
        except Exception as e:
            log.warning(f"Compression LLM call failed: {e}. Keeping last 5 messages.")
            self.messages = self.messages[-5:]
            return

        # wipe → inject summary as single system context message
        self.messages = [{
            "role": "system",
            "content": f"[CONVERSATION SUMMARY — {datetime.now().strftime('%H:%M')}]\n{summary}"
        }]
        self._last_compress_time = time.time()
        log.info("Context compressed ✓")


# ── Ollama Call ───────────────────────────────────────────────────────────────
async def call_ollama(system_prompt: str, messages: list[dict]) -> str:
    """Single Ollama inference call. Returns raw string."""
    payload = {
        "model": MODEL,
        "messages": [{"role": "system", "content": system_prompt}] + messages,
        "stream": False,
        "options": {
            "temperature": 0.2,      # low temp = more deterministic JSON
            "num_predict": 512,
        },
    }

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(OLLAMA_URL, json=payload)
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()


# ── JSON Parser ───────────────────────────────────────────────────────────────
def parse_tool_call(raw: str) -> Optional[dict]:
    """
    Extract JSON from model output.
    Model sometimes wraps in ```json ... ``` — strip it.
    """
    raw = raw.strip()

    # strip markdown fences if present
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(
            line for line in lines
            if not line.strip().startswith("```")
        ).strip()

    # find first { ... }
    start = raw.find("{")
    end   = raw.rfind("}") + 1
    if start == -1 or end == 0:
        return None

    try:
        return json.loads(raw[start:end])
    except json.JSONDecodeError:
        return None


# ── ReAct Loop ────────────────────────────────────────────────────────────────
async def react_loop(
    user_input: str,
    history: ConversationHistory,
    context: dict,
) -> str:
    """
    Core ReAct loop.
    Think → emit tool call → observe result → repeat → done.
    Returns the final speech text.
    """

    # compress if needed
    if history.needs_compression():
        await history.compress()

    # add user message to history
    history.add("user", user_input)

    system_prompt = build_system_prompt(context)
    steps = 0
    observations = []

    while steps < MAX_REACT_STEPS:
        steps += 1
        log.info(f"ReAct step {steps}/{MAX_REACT_STEPS}")

        # build messages: history + any observations from this turn
        messages = history.messages.copy()
        if observations:
            obs_text = "\n".join(
                f"[Tool result {i+1}]: {o}" for i, o in enumerate(observations)
            )
            messages.append({"role": "user", "content": obs_text})

        # call LLM
        try:
            raw_output = await call_ollama(system_prompt, messages)
        except httpx.HTTPError as e:
            log.error(f"Ollama call failed: {e}")
            return "Sorry, I couldn't reach my brain. Is Ollama running?"

        log.info(f"Raw output: {raw_output[:200]}")

        # parse tool call
        tool_call = parse_tool_call(raw_output)

        if tool_call is None:
            log.warning(f"Could not parse JSON from: {raw_output}")
            observations.append(f"ERROR: Your last response was not valid JSON. Output only JSON.")
            continue

        # validate against schema
        valid, error = validate_tool_call(tool_call)
        if not valid:
            log.warning(f"Schema validation failed: {error}")
            observations.append(f"ERROR: Invalid tool call — {error}. Check available tools.")
            continue

        tool_name = tool_call.get("tool")
        tool_args = tool_call.get("args", {})
        thought   = tool_call.get("thought", "")

        if thought:
            log.info(f"Thought: {thought}")

        # ── DONE ──────────────────────────────────────────────────────────────
        if tool_name == "done":
            speech = tool_args.get("speech", "Done.")
            history.add("assistant", raw_output)
            log.info(f"ReAct done. Speech: {speech}")
            return speech

        # ── DISPATCH TOOL ─────────────────────────────────────────────────────
        log.info(f"Dispatching → {tool_name}({tool_args})")
        result = await bus.dispatch(tool_name, tool_args)
        log.info(f"Result: {str(result)[:200]}")

        observations.append(f"{tool_name} → {json.dumps(result)}")

        # push tool call + result into history for next step
        history.add("assistant", raw_output)
        history.add("user", f"[Tool result]: {json.dumps(result)}")

    # hit step limit
    log.warning("Hit MAX_REACT_STEPS — forcing done")
    return "I ran out of steps trying to complete that. Can you simplify the request?"


# ── Context Builder ───────────────────────────────────────────────────────────
async def build_context(memory_agent, kernel_events: list) -> dict:
    """Pull OS context + memory to inject into system prompt."""

    # get active window + battery from sysadmin agent
    try:
        active_window = await bus.dispatch("get_active_window", {})
        battery       = await bus.dispatch("check_battery", {})
    except Exception:
        active_window = "unknown"
        battery       = "unknown"

    # extract keywords from recent kernel events for memory lookup
    keywords = extract_keywords_from_events(kernel_events)

    # fetch relevant memory
    memory = {}
    if memory_agent and keywords:
        memory = await memory_agent.fetch_memory(keywords)

    return {
        "active_window":  active_window,
        "battery":        battery,
        "time":           datetime.now().strftime("%H:%M"),
        "recent_events":  kernel_events[-10:],
        "memory":         memory,
    }


def extract_keywords_from_events(events: list) -> list[str]:
    """Simple keyword extraction from event list for memory lookup."""
    keywords = []
    for ev in events:
        if isinstance(ev, dict):
            keywords.extend([
                str(v) for v in ev.values()
                if isinstance(v, str) and len(v) > 3
            ])
    # deduplicate + limit
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
        self.kernel_events: list = []          # filled by kernel listeners
        self.memory_agent  = None              # injected by main.py

    def inject_memory_agent(self, agent):
        self.memory_agent = agent

    def push_kernel_event(self, event: dict):
        """Called by kernel listener daemon when event fires."""
        self.kernel_events.append({**event, "ts": time.time()})
        # keep last 50 events in memory
        if len(self.kernel_events) > 50:
            self.kernel_events = self.kernel_events[-50:]

    async def process(self, user_input: str) -> str:
        """
        Main entry point.
        Called by buddy_agent after Whisper transcription.
        Returns speech text → buddy_agent passes to Kokoro.
        """
        await self.state_machine.transition(State.THINKING)

        context = await build_context(self.memory_agent, self.kernel_events)

        speech = await react_loop(
            user_input=user_input,
            history=self.history,
            context=context,
        )

        # memory agent updates happen inside dispatcher after tool calls
        # but also update after full turn with the final speech
        if self.memory_agent:
            await self.memory_agent.update_memory(
                "last_interaction",
                {
                    "user": user_input,
                    "response": speech,
                    "ts": datetime.now().isoformat(),
                }
            )

        return speech

    async def handle_proactive_event(self, event: dict):
        """
        Called by kernel listeners for proactive alerts
        (battery low, suspicious process, download complete, etc.)
        Brain decides whether to alert user or stay silent.
        """
        self.push_kernel_event(event)

        # only interrupt if not already mid-task
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
            history=ConversationHistory(),  # fresh context for proactive — don't pollute main history
            context=context,
        )

        if speech.strip():
            await bus.dispatch("speak", {"text": speech})

        await self.state_machine.transition(State.IDLE)


# ── Singleton ─────────────────────────────────────────────────────────────────
brain = Brain()
