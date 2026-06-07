# Mr Meeseeks - Tests Directory

This folder contains verification and diagnostic test scripts used during development. Because these tests import modules from the main packages (`core`, `agents`, `kernel`), they should be executed from the **repository root directory**.

## Execution Methods

You can run any test from the project root using one of the following methods:

### Method A: Run as a Python Module (Recommended)
```bash
# Example: Running the web research scraper test
python -m tests.test_web

# Example: Running the core brain loop test
python -m tests.core_tester
```

### Method B: Set PYTHONPATH explicitly
```bash
PYTHONPATH=. python tests/test_web.py
```

---

## Test Directory Manifest

-   **[core_tester.py](core_tester.py)**: Simulates a complete Brain task sequence (e.g. creating files and checking battery) in dry-run mode without spinning up the full GUI overlay.
-   **[test_list_ui.py](test_list_ui.py)**: Spawns the accessibility crawler to traverse active windows (like VS Code) and output DOM accessibility nodes to verify Eyes agent functionality.
-   **[test_dbus_qt.py](test_dbus_qt.py)**: Checks local D-Bus IPC links and QT application signal-slot bindings.
-   **[test_native_pill.py](test_native_pill.py)**: Spawns the desktop overlay status pill window to check UI styles and transparencies.
-   **[test_native_pill_loop.py](test_native_pill_loop.py)**: Tests animations, fades, and interactive state triggers for the overlay status pill.
-   **[test_voice_loop.py](test_voice_loop.py)**: Tests local faster-whisper recording, barge-in voice interruption thresholds, and transcription streams.
-   **[test_voice_streaming.py](test_voice_streaming.py)**: Audits streaming capabilities of the Kokoro TTS engine on local hardware.
-   **[test_web.py](test_web.py)**: Verifies DuckDuckGo API or SearXNG scraping processes by mock testing the Web Researcher agent query pipelines.
