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
_active_stream = None

libs_cuda13 = [
    ("cuda_runtime", "libcudart.so.13"),
    ("nvjitlink", "libnvjitlink.so.13"),
    ("cublas", "libcublasLt.so.13"),
    ("cublas", "libcublas.so.13"),
    ("cufft", "libcufft.so.12"),
    ("curand", "libcurand.so.10"),
    ("cusparse", "libcusparse.so.12"),
    ("cusolver", "libcusolver.so.12"),
    ("cudnn", "libcudnn.so.9"),
]

libs_cuda12 = [
    ("cuda_runtime", "libcudart.so.12"),
    ("nvjitlink", "libnvjitlink.so.12"),
    ("cublas", "libcublasLt.so.12"),
    ("cublas", "libcublas.so.12"),
    ("cufft", "libcufft.so.11"),
    ("curand", "libcurand.so.10"),
    ("cusparse", "libcusparse.so.12"),
    ("cusolver", "libcusolver.so.11"),
    ("cudnn", "libcudnn.so.9"),
]

def _preload_cuda_libs():
    import sys
    import ctypes
    import glob
    
    if not sys.platform.startswith("linux"):
        return
        
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    site_packages = os.path.join(project_root, "venv", "lib", f"python{sys.version_info.major}.{sys.version_info.minor}", "site-packages")
    nvidia_dir = os.path.join(site_packages, "nvidia")
    
    search_dirs = []
    if os.path.exists(nvidia_dir):
        search_dirs.append(nvidia_dir)
    search_dirs.append("/usr/local/lib/ollama/mlx_cuda_v13")
    search_dirs.append("/usr/local/lib/ollama/cuda_v13")
    search_dirs.append("/usr/local/lib/ollama/cuda_v12")
    
    use_cuda13 = False
    for search_dir in search_dirs:
        if glob.glob(os.path.join(search_dir, "**", "libcublasLt.so.13"), recursive=True) or \
           glob.glob(os.path.join(search_dir, "libcublasLt.so.13")):
            use_cuda13 = True
            break
            
    libs_to_load = libs_cuda13 if use_cuda13 else libs_cuda12
    
    for pkg, libname in libs_to_load:
        lib_path = None
        for search_dir in search_dirs:
            path1 = os.path.join(search_dir, libname)
            if os.path.exists(path1):
                lib_path = path1
                break
            path2 = os.path.join(search_dir, pkg, "lib", libname)
            if os.path.exists(path2):
                lib_path = path2
                break
            pattern = os.path.join(search_dir, "**", libname)
            found = glob.glob(pattern, recursive=True)
            if found:
                lib_path = found[0]
                break
                
        if lib_path and os.path.exists(lib_path):
            try:
                ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
                log.info(f"Preloaded CUDA library: {libname} ✓")
            except Exception:
                pass


def _play_worker():
    global _kokoro_engine, _main_loop, _active_stream
    
    # Preload dynamic CUDA 12/13 libraries
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
                
                # Reshape mono 1D array to 2D
                if samples.ndim == 1:
                    samples_2d = samples.reshape(-1, 1)
                else:
                    samples_2d = samples
                
                log.info("Playing synthesized audio...")
                _active_stream = sd.OutputStream(samplerate=sample_rate, channels=1, dtype='float32')
                with _active_stream:
                    _active_stream.write(samples_2d)
                _active_stream = None
            except Exception as ex:
                log.error(f"Error during speech synthesis or playback: {ex}")
                _active_stream = None
                
            _audio_queue.task_done()
            
            # Transition to IDLE state once queue becomes empty
            if _audio_queue.empty() and _main_loop:
                _main_loop.call_soon_threadsafe(
                    lambda: asyncio.create_task(brain.state_machine.transition(State.IDLE))
                )
        except Exception as e:
            log.error(f"Voice worker thread loop hit an error: {e}")
            _active_stream = None
 
async def handle_speak(args: dict) -> dict:
    text = args.get("text", "").strip()
    if not text:
        return {"error": "Missing 'text' argument."}
        
    log.info(f"Queuing speak request: '{text[:60]}...'")
    await brain.state_machine.transition(State.SPEAKING)
    _audio_queue.put(text)
    return {"status": "queued"}

async def handle_stop_speak(args: dict) -> dict:
    global _active_stream
    log.info("Stopping speak playback...")
    
    # 1. Abort stream if active
    if _active_stream is not None:
        try:
            _active_stream.abort()
        except Exception as e:
            log.warning(f"Error aborting stream: {e}")
            
    # 2. Clear the audio queue
    while not _audio_queue.empty():
        try:
            _audio_queue.get_nowait()
            _audio_queue.task_done()
        except Exception:
            pass
            
    return {"status": "stopped"}
 
def register():
    global _worker_thread, _main_loop
    bus.register("speak", handle_speak)
    bus.register("stop_speak", handle_stop_speak)
    
    try:
        _main_loop = asyncio.get_running_loop()
    except RuntimeError:
        _main_loop = None
        
    # Start playback worker thread
    _worker_thread = threading.Thread(target=_play_worker, daemon=True)
    _worker_thread.start()
    log.info("Voice agent registered ✓")
