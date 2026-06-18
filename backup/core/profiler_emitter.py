import json
import os
import time
import threading

PROFILER_FILE = "/tmp/meeseeks_profiler.jsonl"
_lock = threading.Lock()

def clear():
    """Clears the profiler event log file."""
    with _lock:
        try:
            if os.path.exists(PROFILER_FILE):
                os.remove(PROFILER_FILE)
        except Exception as e:
            print(f"[ProfilerEmitter] Warning: Could not clear profiler file: {e}")

def emit(event_type: str, **kwargs):
    """
    Emits an event to the profiler log.
    Includes timestamp, type, and any other extra metadata.
    """
    with _lock:
        try:
            event = {
                "ts": time.time(),
                "type": event_type,
                **kwargs
            }
            with open(PROFILER_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(event) + "\n")
        except Exception as e:
            # We don't want the profiler to crash the main application
            print(f"[ProfilerEmitter] Warning: Failed to emit event {event_type}: {e}")
