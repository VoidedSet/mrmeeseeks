import re
import time
from enum import Enum
from collections import deque
from typing import Optional, Tuple, Deque
from dataclasses import dataclass, field
from rapidfuzz import process, fuzz

class Intent(Enum):
    CASUAL = "CASUAL"
    TOOL_CALLING = "TOOL_CALLING"

def normalize_voice_input(text: str) -> str:
    if not text:
        return ""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s']", " ", text)  # strip ALL punctuation, not just edges
    text = re.sub(r"(.)\1{2,}", r"\1", text)  # hiiiiii -> hi
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ---------------------------------------------------------------------------
# STRATEGIC MATCHING PATTERNS
# ---------------------------------------------------------------------------
_GREET_SUFFIX = r"( there| guys| everyone| friend| meeseeks)?"
_FILLER_PREFIX = r"(okay |alright |ok |so |well )?"
CASUAL_EXACT = [
    rf"^h[ei]y?{_GREET_SUFFIX}$", rf"^hello{_GREET_SUFFIX}$", r"^yo$", r"^sup$",
    r"^good (morning|afternoon|evening|night)$",
    r"^who are you$", r"^what('s| is) your name$", r"^tell me about yourself$",
    rf"^{_FILLER_PREFIX}(bye|goodbye|see ya|see you|exit|quit)$",
    r"^thanks?( you)?( so much| a lot)?$",
    r"^you('re| are)? (da )?goat$",
    r"^how('s| is) it going$", r"^everything good$", r"^wyd$", r"^what('s| is) up$"
]

CASUAL_FUZZ_TARGETS = [
    "hi", "hello", "hey", "yo", "sup", "good morning", "good afternoon",
    "good evening", "who are you", "what is your name", "tell me about yourself",
    "bye", "goodbye", "exit", "quit", "thanks", "thank you", "you da goat",
    "hows it going", "what can you do", "thanks for the help", "everything good",
    "what are you doing", "what is up", "how are you",
    # code-mixed greetings — extend this as more show up in your logs
    "kem cho", "kya haal hai", "kaise ho", "tum kaise ho", "namaste", "kya kar raha hai",
]

CANCEL_PATTERNS = r"^(no|nope|nah|cancel|stop|never mind( (it|that|about (it|that)))?|forget it|forget that|forget what i (just )?said|scratch that)$"
CONFIRM_PATTERNS = r"^(yes|yeah|yep|yup|sure|okay|ok|go ahead|do it|please do|do that|correct|exactly|that('s| is) right|sounds good)$"
ANAPHORA_TOKENS = {"it", "that", "this", "again", "same", "also", "too", "one"}

BARE_VERBS = {"open", "launch", "close", "click", "press", "type", "scroll",
              "restart", "reboot", "shutdown", "mute", "unmute", "search", "find", "check"}

HARD_KEYWORDS = {
    # unambiguous outside a command context — safe to fire alone
    "battery", "screenshot", "volume", "mute", "unmute", "terminal",
    "browser", "chrome", "icon", "link", "key", "enter", "settings",
}

SOFT_KEYWORDS = {
    # common in plain conversation too — only count WITH an info-seeking verb
    "window", "system", "computer", "pc", "laptop", "screen",
    "code", "notes", "file", "resume", "deadline", "deadlines", "log", "logs",
    "meeting", "meetings", "project", "pitch", "nas", "yesterday",
}

INFO_VERB_PATTERN = re.compile(
    r"\b(what|check|find|search|show me|recall|recollect|did i|do i have|"
    r"where('s| is)|pull up|look up|remind me|look at)\b"
)

# explicit verb+object proximity — bare verb presence is NOT enough
# ("open up about feelings" / "window shopping" must not fire)
COMMAND_PATTERNS = [
    r"\b(open|close|switch|resize|minimize|maximize)\b.{0,12}\bwindow\b",
    r"\b(open|launch)\b.{0,15}\b(browser|chrome|terminal|app|application|file manager|settings)\b",
    r"\bclick\b.{0,15}\b(on|the|button|icon|link)\b",
    r"\b(press|hit)\b.{0,15}\b(key|enter|button|the)\b",
    r"\btype\b.{0,20}\b(this|that|in|into|the)\b",
    r"\bscroll\b.{0,10}\b(up|down|page)\b",
    r"\b(restart|reboot|shut ?down)\b.{0,12}\b(computer|system|pc|laptop|now|it)?\b",
]

# Added standalone "no" with a trailing space boundary to catch opening sentence negations
NEGATION_MARKERS = ["no ", "don't", "do not", "dont", "never", "without", "stop", "cancel", "no need to"]

def _negated(text: str, span_start: int, window: int = 40) -> bool:
    left = text[max(0, span_start - window):span_start]
    return any(neg in left for neg in NEGATION_MARKERS)

def _classify_base(cleaned: str) -> Tuple[Intent, float]:
    """Pure single-utterance baseline logic."""
    if not cleaned:
        return Intent.CASUAL, 1.0
        
    for pat in CASUAL_EXACT:
        if re.fullmatch(pat, cleaned):
            return Intent.CASUAL, 1.0
            
    tokens = cleaned.split()
    if len(tokens) <= 5:
        match = process.extractOne(cleaned, CASUAL_FUZZ_TARGETS, scorer=fuzz.token_sort_ratio)
        if match and match[1] >= 85:
            return Intent.CASUAL, match[1] / 100.0

    all_kw = HARD_KEYWORDS | SOFT_KEYWORDS
    for word in all_kw:
        m = re.search(rf"\b{word}\b", cleaned)
        if m and _negated(cleaned, m.start()):
            return Intent.CASUAL, 0.95
    for pat in COMMAND_PATTERNS:
        m = re.search(pat, cleaned)
        if m and _negated(cleaned, m.start()):
            return Intent.CASUAL, 0.95

    if any(re.search(pat, cleaned) for pat in COMMAND_PATTERNS):
        return Intent.TOOL_CALLING, 0.95

    if any(word in HARD_KEYWORDS for word in tokens):
        return Intent.TOOL_CALLING, 0.95

    if any(word in SOFT_KEYWORDS for word in tokens) and INFO_VERB_PATTERN.search(cleaned):
        return Intent.TOOL_CALLING, 0.9

    # No confident signal either way. This must default to TOOL_CALLING, not CASUAL.
    # A missed command = a broken task. A wrongly-routed casual line = wasted latency.
    # Recall on keyword lists is always incomplete (typos, code-mixed languages,
    # abbreviations, new vocabulary) — the DEFAULT DIRECTION matters more than
    # any single keyword you can add.
    return Intent.TOOL_CALLING, 0.3

# ---------------------------------------------------------------------------
# ROUTER STATE MACHINE
# ---------------------------------------------------------------------------
HISTORY_DECAY = [0.6, 0.4, 0.25]
CURRENT_WEIGHT = 1.0
CONTEXT_TTL_SECONDS = 30.0  # stale history can't keep flipping new turns to TOOL_CALLING

@dataclass
class RouterState:
    tool_call_history: Deque[Tuple[str, Intent, float]] = field(default_factory=lambda: deque(maxlen=3))
    pending_verb: Optional[str] = None
    pending_at: float = 0.0

def route_intent(user_input: str, state: Optional[RouterState] = None) -> Intent:
    state = state if state is not None else RouterState()
    cleaned = normalize_voice_input(user_input)
    if not cleaned:
        return Intent.CASUAL
    tokens = cleaned.split()

    # ---- Layer 1: Short-Circuits ----
    if re.fullmatch(CANCEL_PATTERNS, cleaned):
        state.pending_verb = None
        return Intent.CASUAL
        
    for pat in CASUAL_EXACT:
        if re.fullmatch(pat, cleaned):
            return Intent.CASUAL

    # ---- Layer 2: One-Shot Split-Command Pending Slots ----
    if state.pending_verb and (time.time() - state.pending_at) <= 8.0:
        verb = state.pending_verb
        state.pending_verb = None  # Consume slot
        intent_slot, conf_slot = _classify_base(f"{verb} {cleaned}")
        if intent_slot == Intent.TOOL_CALLING and conf_slot >= 0.80:
            state.tool_call_history.append((f"{verb} {cleaned}", Intent.TOOL_CALLING, time.time()))
            return Intent.TOOL_CALLING

    # ---- Layer 3: Contextual Carry-Forward ----
    now = time.time()
    fresh_history = [h for h in state.tool_call_history if (now - h[2]) <= CONTEXT_TTL_SECONDS]
    last_label = fresh_history[-1][1] if fresh_history else None
    if last_label is not None:
        if re.fullmatch(CONFIRM_PATTERNS, cleaned):
            # confirmations don't introduce new evidence — return label, don't re-anchor history
            return last_label
        if len(tokens) <= 4 and ANAPHORA_TOKENS.intersection(tokens):
            # Short ambiguous statements lean on the historical context to decide tracking path
            scores = {Intent.CASUAL: 0.0, Intent.TOOL_CALLING: 0.0}
            for i, (_, past_label, _ts) in enumerate(reversed(fresh_history)):
                if i < len(HISTORY_DECAY):
                    scores[past_label] += HISTORY_DECAY[i]
            if scores[Intent.TOOL_CALLING] > scores[Intent.CASUAL]:
                # inherited guess, not new evidence — don't re-append (avoids self-reinforcing loop)
                return Intent.TOOL_CALLING

    # ---- Layer 4: Bare Verb Interception ----
    if cleaned in BARE_VERBS:
        state.pending_verb = cleaned
        state.pending_at = time.time()
        return Intent.CASUAL

    # ---- Layer 5: Classify Current Utterance ----
    intent_c, conf_c = _classify_base(cleaned)
    if conf_c >= 0.7:
        if intent_c == Intent.TOOL_CALLING:
            state.tool_call_history.append((user_input, intent_c, time.time()))
        return intent_c

    # No confident match in either direction. Default to TOOL_CALLING (safe-slow),
    # NOT CASUAL. Don't anchor history on a low-confidence guess.
    return Intent.TOOL_CALLING

# ---------------------------------------------------------------------------
# INTERACTIVE REPL
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    session_state = RouterState()
    print("=" * 60)
    # Testing script to verify accuracy before wiring into brain.py
    print("  MEESEEKS GATE v4 (CASUAL vs TOOL_CALLING) — TEST REPL")
    print("=" * 60)
    test_prompts = [
    # Casual
    "Hi", "Hello there", "Yo, how's it going?", "Good morning!", "What can you do?",
    "Tell me about yourself.", "Who are you?", "Thanks for the help.", "You da goat.",
    "Sup.", "Okay, goodbye!", "Hey Meeseeks.", "Everything good?", "Never mind.",
    "Forget what I just said.",
    
    # System Command
    "Battery level?", "Take a screenshot.", "Volume up.", "Mute system.",
    "Open window.", "Close that window.", "Resize the window.", "Open browser.",
    "Launch terminal.", "Restart the computer.", "Shut down now.", "Click on the icon.",
    "Press enter.", "Scroll down.", "Open settings.",
    
    # Split Command / Pending Slot Testing
    "Open", "Chrome", "Type", "this is a test", "Close", "window",
    "Launch", "file manager", "Click", "on the button",
    
    # Anaphora & Context Carry-Forward
    "Do that again.", "Yes.", "Please do.", "Do it again.", "Also this one.",
    
    # Deep Search / Memory Triggers
    "What did I code yesterday regarding the netlink loop?",
    "Recollect my notes on advanced network design.",
    "Search for the last file I edited.", "Find my resume.",
    "What are my project deadlines?", "Look up the logs for the system daemon.",
    "Do I have any meetings today?", "Check my notes on the ESP32 project.",
    "Show me the project pitch I wrote.", "What was that thing I said about the NAS?",
    
    # Negation Traps
    "No, don't take a screenshot.", "Do not shutdown the computer.",
    "Don't open the window.", "No volume changes.", "Never mind about the battery."
]
    
    for p in test_prompts:
        intent = route_intent(p, session_state)
        print(f"[{intent.value}] {p}")

    # IMPORTANT: fresh state for the live session — canned test history must not
    # bleed into interactive testing (this was the actual cause of the "u in?" bug)
    session_state = RouterState()
    while True:
        try:
            user_raw = input("\nYou (Prompt): ")
            if user_raw.strip().lower() in {"exit", "quit", "bye"}:
                break
                
            start = time.perf_counter()
            classification = route_intent(user_raw, session_state)
            elapsed = (time.perf_counter() - start) * 1000.0
            
            color = "\033[92m" if classification == Intent.CASUAL else "\033[91m"
            print(f"Classification : {color}{classification.value}\033[0m")
            print(f"Latency        : \033[93m{elapsed:.2f} ms\033[0m")
            
            if session_state.pending_verb:
                print(f"[State Slot] Armed Pending Verb: '{session_state.pending_verb}'")
                
        except (KeyboardInterrupt, EOFError):
            break