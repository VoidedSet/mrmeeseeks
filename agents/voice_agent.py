"""
voice_agent.py — Mr Meeseeks Voice Output Agent
Handles the `speak` tool. Uses kokoro-onnx and sounddevice to play speech.
Uses a background worker thread and queue so playbacks are sequential and non-blocking.
"""

import os
import logging
import queue
import threading
import asyncio
from core.ipc_bus import bus
from core.state_machine import State
from core.brain import brain

log = logging.getLogger("voice_agent")

_audio_queue = queue.Queue()
_worker_thread = None
_kokoro_engine = None
_main_loop = None

def _preload_cuda_libs():
    import sys
    import ctypes
    if not sys.platform.startswith("linux"):
        return
        
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    site_packages = os.path.join(project_root, "venv", "lib", f"python{sys.version_info.major}.{sys.version_info.minor}", "site-packages")
    
    # Preload the nvidia package library files if they exist in venv
    libs_to_load = [
        ("cublas", "libcublasLt.so.12"),
        ("cublas", "libcublas.so.12"),
        ("cudnn", "libcudnn.so.9"),
    ]
    
    for pkg, libname in libs_to_load:
        lib_path = os.path.join(site_packages, "nvidia", pkg, "lib", libname)
        if os.path.exists(lib_path):
            try:
                # Load with RTLD_GLOBAL so other loaded libraries (like ORT) can resolve these symbols
                ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
                log.info(f"Preloaded CUDA library: {libname} ✓")
            except Exception as ex:
                log.warning(f"Could not preload library {libname} from {lib_path}: {ex}")


def _play_worker():
    global _kokoro_engine, _main_loop
    
    # Try preloading any CUDA libraries installed in venv
    _preload_cuda_libs()
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(project_root, "models", "kokoro-v1.0.int8.onnx")
    voices_path = os.path.join(project_root, "models", "voices-v1.0.bin")
    
    log.info(f"Initializing Kokoro TTS engine from {model_path}...")
    try:
        import sounddevice as sd
        from kokoro_onnx import Kokoro
        
        import onnxruntime as ort
        from onnxruntime import InferenceSession
        
        if not os.path.exists(model_path) or not os.path.exists(voices_path):
            log.error(f"Kokoro model or voices file not found at {model_path} / {voices_path}")
            return
            
        # Detect and configure execution providers (GPU / CPU)
        available = ort.get_available_providers()
        selected_providers = []
        if "CUDAExecutionProvider" in available:
            selected_providers.append("CUDAExecutionProvider")
            log.info("Kokoro TTS: CUDA detected. Running on GPU ✓")
        else:
            log.info("Kokoro TTS: CUDA not detected. Falling back to CPU.")
        selected_providers.append("CPUExecutionProvider")
 
        session = InferenceSession(model_path, providers=selected_providers)
        _kokoro_engine = Kokoro.from_session(session, voices_path)
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
            
            # Transition to IDLE state once queue becomes empty
            if _audio_queue.empty() and _main_loop:
                _main_loop.call_soon_threadsafe(
                    lambda: asyncio.create_task(brain.state_machine.transition(State.IDLE))
                )
        except Exception as e:
            log.error(f"Voice worker thread loop hit an error: {e}")
 
async def handle_speak(args: dict) -> dict:
    text = args.get("text", "").strip()
    if not text:
        return {"error": "Missing 'text' argument."}
        
    log.info(f"Queuing speak request: '{text[:60]}...'")
    await brain.state_machine.transition(State.SPEAKING)
    _audio_queue.put(text)
    return {"status": "queued"}
 
def register():
    global _worker_thread, _main_loop
    bus.register("speak", handle_speak)
    
    try:
        _main_loop = asyncio.get_running_loop()
    except RuntimeError:
        _main_loop = None
        
    # Start playback worker thread
    _worker_thread = threading.Thread(target=_play_worker, daemon=True)
    _worker_thread.start()
    log.info("Voice agent registered ✓")

