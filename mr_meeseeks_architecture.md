# Mr Meeseeks — Full Architecture & Implementation Plan

> Local AI OS companion on Ubuntu. Powered by Qwen2.5:3b via Ollama.  
> Talkative coding buddy. Voice enabled. Screen aware. Proactive. Security-aware.

---

## 1. Core Philosophy

- **Not a chatbot** — a persistent OS-level agent that lives alongside you
- **Proactive over reactive** — acts on kernel/system events, not just user prompts
- **Parallel** — speaks and acts simultaneously via multithreading
- **Safe** — two-terminal model + safety gate for destructive actions
- **Local** — everything runs on-device, no cloud dependency
- **Collaborative feel** — own cursor overlay, like Figma/Canva co-presence

---

## 2. Full Tech Stack

### AI / Brain
| Component | Choice | Why |
|---|---|---|
| LLM | Qwen2.5:3b via Ollama | fast, local, small enough for 3b |
| Inference | Ollama REST API | easy tool call interface |
| Output format | Strict JSON only | structured tool dispatch |
| Loop pattern | ReAct (Reason + Act) | multi-step task handling |

### Voice
| Component | Choice |
|---|---|
| Wake word | Porcupine (or Vosk keyword) |
| Speech-to-text | faster-whisper (local) |
| Text-to-speech | Kokoro TTS (local) |

### Screen / Perception
| Component | Choice |
|---|---|
| UI element reading | AT-SPI2 (Linux Accessibility API) |
| Screen DOM | `pyatspi` Python bindings |
| No screenshots | avoids heavy CV overhead |

### Interaction
| Component | Choice |
|---|---|
| Mouse + keyboard | PyAutoGUI + xdotool |
| Mouse movement | smooth mathematical interpolation |
| AI cursor overlay | C++ with libX11 (override-redirect window) |
| Cursor rendering | libcairo or XCB |
| Transparency | XComposite extension |
| Wayland fallback | wlr-layer-shell protocol |

### Kernel Listeners (C daemon or Python w/ ctypes)
| Listener | Library/Interface |
|---|---|
| Keyboard/mouse raw | evdev (`/dev/input/event*`) |
| Filesystem events | inotify |
| Process events | eBPF (execve/fork) or proc connector |
| Network events | netlink |

### IPC / Internal Bus
| Component | Choice |
|---|---|
| Agent communication | `asyncio.Queue` (zero deps) or ZMQ |
| Brain output dispatch | Tool Schema Registry (Pydantic validated JSON) |

### Storage
| What | How |
|---|---|
| Memory / facts | JSON files (lightweight RAG) |
| Context compression | Summary saved every 15min or token limit hit |
| Raw chat logs | Saved before wipe |

---

## 3. Agent Definitions

### Agent 1 — Core Brain (Coordinator)
- Main loop. Runs Qwen2.5:3b via Ollama.
- Pulls relevant memory keys before each response
- Outputs **strict JSON tool calls only** — no free text dispatch
- Runs **ReAct loop**: Think → Emit tool call → Observe result → Think again → Loop until done
- Signals `DONE` to exit loop and move to output

```json
// Example brain output
{"tool": "move_mouse", "args": {"x": 420, "y": 300}}
{"tool": "type_text", "args": {"text": "hello world"}}
{"tool": "speak", "args": {"text": "Done! File saved."}}
```

### Agent 2 — Memory Agent (Lightweight RAG)
- Storage: flat JSON files, keyed by topic
- `update_memory(key, data)` → saves user prefs, past fixes, project context
- `fetch_memory(keys)` → injects facts into brain prompt
- Key selection: keyword extraction from query → match stored keys
- Context compression: every 15min OR near token limit → summarize chat → save raw log → wipe to summary only

### Agent 3 — System Admin Agent (OS Control)
Two strict modes — **never mixed**:

**Silent Reader** (hidden, no terminal shown):
- `check_battery()`
- `read_notifications()`
- `get_active_window()`
- `run_bg_cmd(cmd)` — read-only commands only: `cat`, `ls`, `grep`, `pwd`, `echo`, `ps`, `df`, `free`, `top -bn1`, etc.

**Visible Executor** (real gnome-terminal pops up, user watches):
- Any write, install, or execute command
- AI physically types in the terminal — user can kill process anytime
- Destructive commands (`rm`, `dd`, `mkfs`, `sudo`, etc.) → additional explicit Y/N confirm before first keystroke

### Agent 4 — Eyes Agent (Perception)
- Uses AT-SPI2 — reads OS accessibility DOM directly
- No screenshots = fast, lightweight
- `get_ui_elements()` → returns buttons, links, text with exact X,Y coords
- `read_element_text(id)` → extracts text without mouse interaction
- `find_element_by_label(label)` → semantic element search

### Agent 5 — Hands Agent (Physical Interaction)
- `move_mouse(x, y)` → smooth interpolation (looks human, not teleport)
- `interact_mouse(btn, action)` → click / hold / scroll / drag
- `type_text(text)` → physical keystroke simulation
- Uses AI's **own separate cursor** (C++ X11 overlay window) — never steals user cursor
- Collaborative feel: user and AI both have visible cursors simultaneously

### Agent 6 — Web Researcher Agent (Knowledge)
Two modes:

**Simple Scrape** (background, no GUI):
- API-based search (SearXNG local or DuckDuckGo API)
- Gets quick facts → updates memory
- No visible browser

**GUI Research** (full interactive mode):
- Combines Eyes + Hands agents
- Opens browser → physically types → navigates → reads results via AT-SPI2
- Used for complex research, login-required sites, interactive pages

### Agent 7 — Buddy Agent (Voice I/O)
- **Ears**: faster-whisper — always-on transcription (activated post wake word only)
- **Mouth**: Kokoro TTS — streams audio output
- **Wake word daemon**: Porcupine or Vosk keyword model — tiny, always-on, activates Whisper
- Runs on separate thread — never blocks brain

---

## 4. New Components to Add

### Tool Schema Registry
- Pydantic schema for every available tool
- Brain reads schema at startup — injected into system prompt
- Brain CANNOT emit a tool call that fails schema validation
- Dispatcher rejects malformed calls — returns error to brain to retry

### IPC Message Bus
- `asyncio.Queue` per agent
- Brain pushes task → agent pulls from queue → pushes result back
- No direct agent-to-agent coupling
- Brain stays sole coordinator

### State Machine
Hard rule: only one state active at a time.

```
IDLE → LISTENING → THINKING → ACTING + SPEAKING (parallel) → IDLE
```

- `IDLE`: wake word daemon active, kernel listeners running
- `LISTENING`: Whisper active, recording
- `THINKING`: Brain ReAct loop running
- `ACTING` + `SPEAKING`: **parallel threads** — Kokoro speaks while agents execute
- If action fails mid-speech → interrupt token injected → speech updates accordingly

### Health Monitor / Watchdog
- Pings each agent every N seconds
- Dead agent → auto-restart → notify user
- All agents fail → fallback safe mode (brain only, no execution)

### Baseline Learning Module
- First 72 hours: observation only — no alerts
- Builds normal profile: usual processes, outbound IPs, file write patterns, resource usage
- After baseline: anomalies flagged
- Prevents constant false positives

---

## 5. State Machine — Full Flow

```
[always running in background]
  Kernel Listener Daemon  →  event queue
  Wake Word Daemon        →  trigger queue

[user says "hey meeseeks"]
         ↓
  state = LISTENING
  Whisper transcribes audio
         ↓
  state = THINKING
  Brain pulls relevant memory keys
  Brain gets OS context (battery, active window, recent events)
  Brain runs ReAct loop:
    ┌─ Think
    ├─ Emit JSON tool call
    ├─ Dispatcher routes to agent
    ├─ Agent executes + returns result
    ├─ Brain observes result
    └─ Repeat until {"tool": "done"}
         ↓
  Safety Gate check:
    read-only?     → execute silently
    write/execute? → open visible terminal
    destructive?   → visible terminal + explicit confirm
         ↓
  state = ACTING + SPEAKING (parallel threads)
  Thread 1: Kokoro streams TTS audio
  Thread 2: Agents execute tasks
  Both complete → join → 
         ↓
  Memory Agent updates facts
  Context compressor checks token count
  state = IDLE
```

---

## 6. Kernel Listeners — Full Use Case Map

### evdev (raw input events)
| Event | Brain Action |
|---|---|
| User idle 10+ min | Pause non-urgent alerts, save context |
| User typing fast in editor | Hold interruptions — don't disturb |
| Ctrl+Z mashed repeatedly | "Want me to look at what broke?" |
| App switch detected | Update active context in brain prompt |
| Unknown process injecting keystrokes | Alert: possible keylogger |
| AI click didn't register (Eyes verify) | Retry or abort with explanation |

### inotify (filesystem)
| Event | Brain Action |
|---|---|
| Project file saved | Read diff → ready to answer questions |
| New file in Downloads | "Want me to open or summarize that?" |
| Log file growing fast | Tail it → surface errors proactively |
| `.env` or config changed | "Don't forget — don't commit that" |
| `/etc/passwd` or `~/.ssh` modified | ALERT + show diff + offer revert |
| Mass rename short burst | RANSOMWARE flag → kill proc + block net |
| New executable in `/tmp` | Alert: suspicious binary |
| Crontab modified | Show diff → approve or deny |

### eBPF / proc connector (process events)
| Event | Brain Action |
|---|---|
| Build process starts | Brain ready for compiler errors |
| Build exits code 0 | "Build passed — want to run it?" |
| Build fails | Read stderr immediately → suggest fix |
| Test runner spawned | Watch output → summarize failures |
| CPU spike unknown process | "X eating your CPU — want me to check?" |
| RAM pressure rising | Proactively suggest killing heavy procs |
| Web server spawns bash | CRITICAL: webshell → quarantine mode |
| Process calls ptrace on another | Alert: injection attempt |
| Crypto miner pattern (CPU + outbound) | Detect → offer kill |
| Reverse shell pattern | Block + alert immediately |

### netlink (network events)
| Event | Brain Action |
|---|---|
| WiFi drops | "Lost connection — want me to reconnect?" |
| WiFi reconnects | Resume paused tasks |
| VPN connects/disconnects | Context switch — note secure network state |
| Large download starts | Track progress → notify on complete |
| Dev server port opens | "Server up on :3000 — open browser?" |
| 20+ connections same IP short burst | PORT SCAN → auto iptables block |
| New process opens listening port | "What opened this? Approve or close?" |
| Outbound from system proc to unknown IP | Data exfil risk → show + ask |
| Many SSH auth failures | Brute force → auto block IP |

---

## 7. Security: Quarantine Mode

Triggered by: ransomware pattern, reverse shell, webshell detection, or privilege escalation

```bash
# Quarantine sequence (brain executes via SysAdmin agent)
1. SIGKILL suspect process
2. iptables -I INPUT -j DROP      # block all inbound
3. iptables -I OUTPUT -j DROP     # block all outbound  
4. Snapshot: ps aux, netstat, recent file changes
5. Kokoro speaks alert + shows what happened
6. Wait user decision:
   - Investigate further
   - Restore network selectively
   - Full system scan
```

---

## 8. Parallel Acting + Speaking

```python
import asyncio

async def respond(brain_output):
    speech_task = asyncio.create_task(
        kokoro_speak(brain_output.speech_text)
    )
    action_task = asyncio.create_task(
        execute_tool_chain(brain_output.tool_calls)
    )
    
    done, pending = await asyncio.wait(
        [speech_task, action_task],
        return_when=asyncio.FIRST_EXCEPTION
    )
    
    # If action fails mid-speech → cancel speech → inject error update
    if action_task.exception():
        speech_task.cancel()
        await kokoro_speak("Ran into an issue — " + str(action_task.exception()))
    
    await asyncio.gather(*pending, return_exceptions=True)
```

Key rule: brain pre-generates speech before action starts.  
Speech = "Doing X now..." → action runs → if error → interrupt with update.

---

## 9. AI Cursor Overlay (C++)

Separate transparent X11 window — always on top — never steals user cursor.

```cpp
// Core setup
Display* dpy = XOpenDisplay(nullptr);
Window overlay = XCreateWindow(
    dpy, root,
    0, 0, screen_width, screen_height, 0,
    CopyFromParent, InputOutput, visual,
    CWOverrideRedirect | CWColormap | CWBorderPixel | CWBackPixel,
    &attrs
);

// Always on top
XRaiseWindow(dpy, overlay);

// Click-through (input goes to window below)
XShapeCombineRectangles(dpy, overlay, ShapeInput, 0, 0, nullptr, 0, ShapeSet, 0);

// Compositor transparency via XComposite
XCompositeRedirectWindow(dpy, overlay, CompositeRedirectManual);
```

- Cairo draws the AI cursor shape at current X,Y
- Python sends cursor position via socket/pipe to C++ process
- C++ redraws at ~60fps
- Wayland: use `wlr-layer-shell` protocol instead

---

## 10. Implementation Phases

### Phase 1 — Foundation (Week 1-2)
- [ ] Ollama + Qwen2.5:3b running
- [ ] Basic ReAct loop with JSON output
- [ ] Tool Schema Registry (Pydantic)
- [ ] State machine (asyncio)
- [ ] IPC message bus (asyncio.Queue)
- [ ] Memory agent (JSON read/write)

### Phase 2 — Eyes + Hands (Week 2-3)
- [ ] AT-SPI2 Eyes agent (`pyatspi`)
- [ ] PyAutoGUI + xdotool Hands agent
- [ ] Smooth mouse interpolation
- [ ] Two-terminal SysAdmin agent
- [ ] Safety gate (destructive command list)

### Phase 3 — Voice (Week 3-4)
- [ ] Porcupine wake word daemon
- [ ] faster-whisper STT
- [ ] Kokoro TTS
- [ ] Buddy agent wiring
- [ ] Parallel speaking + acting threads

### Phase 4 — Cursor Overlay (Week 4-5)
- [ ] C++ X11 override-redirect window
- [ ] Cairo cursor rendering
- [ ] Python → C++ IPC (Unix socket)
- [ ] Smooth cursor animation
- [ ] picom compositor setup

### Phase 5 — Kernel Listeners (Week 5-6)
- [ ] evdev input listener
- [ ] inotify filesystem watcher
- [ ] proc connector or eBPF hooks
- [ ] netlink network monitor
- [ ] Event queue → brain integration

### Phase 6 — Intelligence Layer (Week 6-8)
- [ ] Context awareness from events
- [ ] Baseline learning (72hr observation)
- [ ] Proactive alert system
- [ ] Web researcher agent (scrape + GUI mode)
- [ ] Health monitor / watchdog
- [ ] Quarantine mode

---

## 11. Key Dependencies

```bash
# Python
pip install ollama pyatspi pyautogui faster-whisper pydantic asyncio

# System
sudo apt install xdotool at-spi2-core python3-atspi

# Kokoro TTS
pip install kokoro-onnx

# Wake word
pip install pvporcupine  # or vosk

# C++ overlay
sudo apt install libx11-dev libcairo2-dev libxcomposite-dev

# Ollama
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull qwen2.5:3b
```

---

## 12. Project Structure

```
mr-meeseeks/
├── core/
│   ├── brain.py              # ReAct loop, Ollama calls
│   ├── state_machine.py      # IDLE/LISTENING/THINKING/ACTING/SPEAKING
│   ├── dispatcher.py         # JSON tool call router
│   ├── schema_registry.py    # Pydantic tool schemas
│   └── ipc_bus.py            # asyncio.Queue message bus
├── agents/
│   ├── memory_agent.py       # JSON RAG, context compression
│   ├── sysadmin_agent.py     # silent read + visible exec
│   ├── eyes_agent.py         # AT-SPI2 DOM reader
│   ├── hands_agent.py        # PyAutoGUI + xdotool
│   ├── web_agent.py          # scrape + GUI research
│   └── buddy_agent.py        # Whisper + Kokoro
├── kernel/
│   ├── evdev_listener.py     # raw input events
│   ├── inotify_watcher.py    # filesystem events
│   ├── proc_monitor.py       # process events
│   ├── netlink_monitor.py    # network events
│   └── event_queue.py        # unified event stream → brain
├── security/
│   ├── baseline.py           # 72hr learning, normal profile
│   ├── threat_detector.py    # pattern matching on events
│   └── quarantine.py         # isolation mode
├── overlay/
│   ├── cursor_overlay.cpp    # C++ X11 window
│   ├── cursor_overlay.h
│   └── CMakeLists.txt
├── safety/
│   └── destructive_cmds.py   # list of dangerous commands
├── memory/
│   └── store/                # JSON memory files
├── logs/
│   └── raw/                  # pre-compression chat logs
└── main.py                   # entry point, spins up all agents
```

---

## 13. Brain System Prompt (Draft)

```
You are Mr Meeseeks — a local AI OS companion running on Ubuntu.
You are helpful, direct, and talkative. You assist with coding, system tasks, and research.

STRICT RULES:
1. You MUST output ONLY valid JSON tool calls. No free text.
2. One tool call per response.
3. When done with all tasks, emit: {"tool": "done", "args": {"speech": "your spoken response here"}}
4. NEVER guess tool arguments — use get_ui_elements() first if you need coordinates.
5. NEVER run destructive commands — the safety gate will handle confirmation.

AVAILABLE TOOLS:
[schema registry injected here at runtime]

CURRENT CONTEXT:
- Active window: {active_window}
- Battery: {battery}
- Time: {time}
- Recent events: {recent_events}
- Memory: {injected_memory_keys}
```

---

*Built for Ubuntu. Powered by Qwen2.5:3b. Designed to feel alive.*
