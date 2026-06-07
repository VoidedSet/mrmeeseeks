# Mr Meeseeks - Core Framework

The `core/` directory contains the foundational components of the Mr Meeseeks AI OS companion. It coordinates message routing, state machine transitions, interface schemas, and LLM backend interactions.

## Subdirectories and Files

-   **[brain.py](brain.py)**: The central ReAct loop processor. Parses the LLM's streaming outputs, maps tools to registry schemas, handles CoT (Chain of Thought), and detects execution loops.
-   **[ipc_bus.py](ipc_bus.py)**: The asynchronous IPC message bus. It registers agent tool routes, dispatches execution calls, and aggregates return buffers.
-   **[llm_provider.py](llm_provider.py)**: Provider wrappers for LLM backends (Ollama or Groq). Standardizes streaming requests, structured outputs, and token limits.
-   **[schema_registry.py](schema_registry.py)**: Contains schemas for all tools. Validates that tool inputs emitted by the LLM strictly comply with expected signatures.
-   **[state_machine.py](state_machine.py)**: Controls the agent lifecycle states. The framework permits only one state at a time:
    ```
    IDLE ──> LISTENING ──> THINKING ──> ACTING + SPEAKING (Parallel) ──> IDLE
    ```
-   **[voice_input.py](voice_input.py)**: Manages recording loops, VAD (Voice Activity Detection), and audio transcription using faster-whisper.
-   **`ui/`**: Directory managing overlays, cursors, panels, system tray controllers, and hotkey actions.
