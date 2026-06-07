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
import re
from core.ipc_bus import bus
from core.state_machine import State
from core.brain import brain

log = logging.getLogger("voice_agent")

_text_queue = queue.Queue()
_audio_queue = queue.Queue(maxsize=3)
_synthesis_thread = None
_playback_thread = None
_kokoro_engine = None
_main_loop = None
_active_stream = None
_active_generation_id = 0
_generation_lock = threading.Lock()

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


def clean_text_for_tts(text: str) -> str:
    """
    Cleans text formatting, modifiers, and fixes pronunciations of / symbol.
    - Removes formatting markers (** or \n)
    - Replaces word/word with "word or word"
    - Replaces number/number with "number on number" (e.g. 3/4 -> 3 on 4)
    """
    # 1. Remove markdown styling (**, *, __, _, `, etc.)
    text = re.sub(r"\*\*|__|\*|_|`", "", text)
    
    # 2. Replace / between two numbers (with or without spaces) with "on"
    text = re.sub(r"(\d+)\s*/\s*(\d+)", r"\1 on \2", text)
    
    # 3. Replace / between two words (with or without spaces) with "or"
    text = re.sub(r"([a-zA-Z]+)\s*/\s*([a-zA-Z]+)", r"\1 or \2", text)
    
    # 4. Replace any remaining forward slashes with spaces
    text = text.replace("/", " ")
    
    # 5. Collapse multiple spaces and newlines
    text = re.sub(r"\s+", " ", text).strip()
    
    return text


def _synthesis_worker():
    global _kokoro_engine
    
    # Preload dynamic CUDA 12/13 libraries
    _preload_cuda_libs()
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(project_root, "models", "kokoro-v1.0.fp16.onnx")
    voices_path = os.path.join(project_root, "models", "voices-v1.0.bin")
    
    log.info(f"Initializing Kokoro TTS engine from {model_path}...")
    try:
        import onnxruntime as ort
        from kokoro_onnx import Kokoro
        from onnxruntime import InferenceSession
        
        if not os.path.exists(model_path) or not os.path.exists(voices_path):
            log.error(f"Kokoro model or voices file not found at {model_path} / {voices_path}")
            return
            
        # Detect and configure execution providers (GPU / CPU)
        available = ort.get_available_providers()
        selected_providers = []
        if "CUDAExecutionProvider" in available:
            gpu_opts = {
                "device_id": 0,
                "gpu_mem_limit": 1024 * 1024 * 1024,  # Cap VRAM at 1 GB
                "cudnn_conv_algo_search": "HEURISTIC",
                "arena_extend_strategy": "kSameAsRequested"
            }
            selected_providers.append(("CUDAExecutionProvider", gpu_opts))
            log.info("Kokoro TTS: CUDA detected. Running on GPU (1 GB VRAM Limit) ✓")
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
            item = _text_queue.get()
            if item is None:  # Shutdown signal
                _audio_queue.put(None)
                break
                
            text, gen_id = item
            
            with _generation_lock:
                if gen_id != _active_generation_id:
                    # Discard old generation
                    _text_queue.task_done()
                    continue
            
            cleaned_chunk = clean_text_for_tts(text)
            if not cleaned_chunk.strip():
                _text_queue.task_done()
                continue
                
            log.info(f"Synthesizing speech: '{cleaned_chunk[:60]}...'")
            try:
                samples, sample_rate = _kokoro_engine.create(
                    cleaned_chunk,
                    voice="af_sarah",
                    speed=1.0,
                    lang="en-us"
                )
                
                with _generation_lock:
                    if gen_id == _active_generation_id:
                        _audio_queue.put((samples, sample_rate, gen_id))
            except Exception as ex:
                log.error(f"Error during speech synthesis: {ex}")
                
            _text_queue.task_done()
        except Exception as e:
            log.error(f"Voice synthesis worker thread loop hit an error: {e}")


def _play_worker():
    global _active_stream, _main_loop
    import sounddevice as sd
    
    while True:
        try:
            item = _audio_queue.get()
            if item is None:  # Shutdown signal
                break
                
            samples, sample_rate, gen_id = item
            
            with _generation_lock:
                if gen_id != _active_generation_id:
                    _audio_queue.task_done()
                    continue
            
            # Reshape mono 1D array to 2D
            if samples.ndim == 1:
                samples_2d = samples.reshape(-1, 1)
            else:
                samples_2d = samples
            
            log.info("Playing synthesized audio...")
            try:
                _active_stream = sd.OutputStream(samplerate=sample_rate, channels=1, dtype='float32')
                with _active_stream:
                    _active_stream.write(samples_2d)
                _active_stream = None
            except Exception as ex:
                log.error(f"Error during audio playback: {ex}")
                _active_stream = None
                
            _audio_queue.task_done()
            
            # Transition to IDLE state once queues become empty
            if _text_queue.empty() and _audio_queue.empty() and _main_loop:
                _main_loop.call_soon_threadsafe(
                    lambda: asyncio.create_task(brain.state_machine.transition(State.IDLE))
                )
        except Exception as e:
            log.error(f"Voice playback worker thread loop hit an error: {e}")
            _active_stream = None
 
async def handle_speak(args: dict) -> dict:
    text = args.get("text", "").strip()
    if not text:
        return {"error": "Missing 'text' argument."}
        
    log.info(f"Queuing speak request: '{text[:60]}...'")
    await brain.state_machine.transition(State.SPEAKING)
    with _generation_lock:
        _text_queue.put((text, _active_generation_id))
    return {"status": "queued"}

async def handle_stop_speak(args: dict) -> dict:
    global _active_stream, _active_generation_id
    log.info("Stopping speak playback...")
    
    with _generation_lock:
        # Increment generation to invalidate all active and future queue items from previous run
        _active_generation_id += 1
    
    # 1. Abort stream if active
    if _active_stream is not None:
        try:
            _active_stream.abort()
        except Exception as e:
            log.warning(f"Error aborting stream: {e}")
            
    # 2. Clear the text queue
    while not _text_queue.empty():
        try:
            _text_queue.get_nowait()
            _text_queue.task_done()
        except Exception:
            pass
            
    # 3. Clear the audio queue
    while not _audio_queue.empty():
        try:
            _audio_queue.get_nowait()
            _audio_queue.task_done()
        except Exception:
            pass
            
    return {"status": "stopped"}
 
def register():
    global _synthesis_thread, _playback_thread, _main_loop
    bus.register("speak", handle_speak)
    bus.register("stop_speak", handle_stop_speak)
    
    try:
        _main_loop = asyncio.get_running_loop()
    except RuntimeError:
        _main_loop = None
        
    # Start synthesis worker thread
    _synthesis_thread = threading.Thread(target=_synthesis_worker, daemon=True)
    _synthesis_thread.start()
    
    # Start playback worker thread
    _playback_thread = threading.Thread(target=_play_worker, daemon=True)
    _playback_thread.start()
    log.info("Voice agent registered ✓")
