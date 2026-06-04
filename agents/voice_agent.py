"""
voice_agent.py — Mr Meeseeks Voice Output Agent
Handles the `speak` tool. Uses kokoro-onnx and sounddevice to play speech.
Uses a background worker thread and queue so playbacks are sequential and non-blocking.
"""

import os
import logging
import queue
import threading
from core.ipc_bus import bus

log = logging.getLogger("voice_agent")

_audio_queue = queue.Queue()
_worker_thread = None
_kokoro_engine = None

def _play_worker():
    global _kokoro_engine
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(project_root, "models", "kokoro-v1.0.int8.onnx")
    voices_path = os.path.join(project_root, "models", "voices-v1.0.bin")
    
    log.info(f"Initializing Kokoro TTS engine from {model_path}...")
    try:
        import sounddevice as sd
        from kokoro_onnx import Kokoro
        
        if not os.path.exists(model_path) or not os.path.exists(voices_path):
            log.error(f"Kokoro model or voices file not found at {model_path} / {voices_path}")
            return
            
        _kokoro_engine = Kokoro(model_path, voices_path)
        log.info("Kokoro TTS engine initialized successfully ✓")
    except Exception as e:
        log.exception(f"Failed to initialize Kokoro TTS engine: {e}")
        return

    while True:
        try:
            text = _audio_queue.get()
            if text is None:  # Shutdown signal
                break
                
            if not text.strip():
                _audio_queue.task_done()
                continue
                
            log.info(f"Synthesizing speech: '{text[:60]}...'")
            try:
                samples, sample_rate = _kokoro_engine.create(
                    text,
                    voice="af_sarah",
                    speed=1.0,
                    lang="en-us"
                )
                log.info("Playing synthesized audio...")
                sd.play(samples, sample_rate)
                sd.wait()
            except Exception as ex:
                log.error(f"Error during speech synthesis or playback: {ex}")
                
            _audio_queue.task_done()
        except Exception as e:
            log.error(f"Voice worker thread loop hit an error: {e}")

async def handle_speak(args: dict) -> dict:
    text = args.get("text", "").strip()
    if not text:
        return {"error": "Missing 'text' argument."}
        
    log.info(f"Queuing speak request: '{text[:60]}...'")
    _audio_queue.put(text)
    return {"status": "queued"}

def register():
    global _worker_thread
    bus.register("speak", handle_speak)
    
    # Start playback worker thread
    _worker_thread = threading.Thread(target=_play_worker, daemon=True)
    _worker_thread.start()
    log.info("Voice agent registered ✓")
