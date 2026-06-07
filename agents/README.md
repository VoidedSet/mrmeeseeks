# Mr Meeseeks - Agents Subsystem

The `agents/` directory houses the discrete components that execute tasks routed by the Core Brain. The agents do not communicate with each other directly; instead, they register handlers on the global `ipc_bus` and the Brain acts as the sole coordinator.

## Agent Registry & Descriptions

### 1. Coordinator (Core Brain)
*   **File**: [core/brain.py](../core/brain.py)
*   **Role**: Orchestrates the ReAct loop. Decides which tool call to emit next and observes the results until a `done` signal is reached.

### 2. Memory Agent (Lightweight RAG)
*   **File**: [agents/memory_agent.py](memory_agent.py)
*   **Role**: Manages local JSON persistence. Stores summaries and caches facts, pulling them before brain queries. Supports fuzzy search matching.

### 3. System Admin Agent (OS Control)
*   **File**: [agents/sysadmin_agent.py](sysadmin_agent.py)
*   **Role**: Controls command execution. Operates in two distinct modes:
    *   *Silent Reader*: Runs read-only background queries.
    *   *Visible Executor*: Opens a visible Gnome Terminal window and physically types out commands so the user can audit actions.

### 4. Eyes Agent (Perception)
*   **File**: [agents/eyes_agent.py](eyes_agent.py)
*   **Role**: Integrates with the Linux accessibility bus (`pyatspi`). Scans and crawls UI elements, active frame contents, coordinates, and buttons without generating screen overhead.

### 5. Hands Agent (Physical Interaction)
*   **File**: [agents/hands_agent.py](hands_agent.py)
*   **Role**: Uses `PyAutoGUI` and `xdotool` to simulate keyboard/mouse inputs and scroll events. Positions are interpolated smoothly to look natural.

### 6. Web Researcher Agent (Knowledge)
*   **File**: [agents/web_agent.py](web_agent.py)
*   **Role**: Scrapes websites or navigates active pages using SearXNG/DuckDuckGo or Chromium/accessibility nodes to fetch information.

### 7. Voice Agent (Buddy Agent)
*   **File**: [agents/voice_agent.py](voice_agent.py)
*   **Role**: Interfaces with the microphone and speakers. Handles faster-whisper STT, Kokoro TTS audio stream generation, and wake-word listeners.
