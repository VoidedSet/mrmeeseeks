import os
import sys

# Add project root to sys.path so core and other packages can be imported
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import time
import queue
import threading
import argparse
import asyncio
import json
import httpx
import numpy as np
import re

# Load .env variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Timing variables
start_time = 0
first_token_time = None
first_audio_played_time = None
total_tokens_received = 0

# Sentence counters for dynamic compute capping
sentence_count = 0

def preload_cuda_libs():
    """
    Programmatically preload CUDA and cuDNN libraries from venv or system
    to make sure onnxruntime-gpu can load CUDAExecutionProvider correctly.
    Supports both CUDA 12 and CUDA 13 libraries.
    """
    import ctypes
    import glob
    
    if not sys.platform.startswith("linux"):
        return
        
    project_root = os.path.dirname(os.path.abspath(__file__))
    if os.path.basename(project_root) == "tests":
        project_root = os.path.dirname(project_root)
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
            except Exception:
                pass

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

# ── Text Cleaning ────────────────────────────────────────────────────────────
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

# ── Workers ───────────────────────────────────────────────────────────────────
def playback_worker(audio_queue, playback_stream_ref):
    global first_audio_played_time
    import sounddevice as sd
    
    stream = None
    try:
        while True:
            item = audio_queue.get()
            if item is None:
                break
                
            samples, sample_rate = item
            
            if stream is None:
                # Open output stream once for the session using correct sample rate
                stream = sd.OutputStream(samplerate=sample_rate, channels=1, dtype='float32')
                stream.start()
                playback_stream_ref[0] = stream
                
            if first_audio_played_time is None:
                first_audio_played_time = time.time()
                print(f"\n[Playback] Playing audio (latency: {first_audio_played_time - start_time:.3f}s from LLM generation start)")
                
            # Reshape mono 1D array to 2D (frames, 1) for sounddevice OutputStream
            if samples.ndim == 1:
                samples_2d = samples.reshape(-1, 1)
            else:
                samples_2d = samples
                
            stream.write(samples_2d)
            audio_queue.task_done()
    except Exception:
        pass
    finally:
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass
            playback_stream_ref[0] = None

def synthesis_worker(kokoro_gpu, kokoro_cpu, use_dynamic_cap, voice, speed, text_queue, audio_queue):
    global sentence_count
    
    while True:
        chunk = text_queue.get()
        if chunk is None:
            audio_queue.put(None)  # Signal playback thread to stop
            break
            
        cleaned_chunk = clean_text_for_tts(chunk)
        if not cleaned_chunk.strip():
            text_queue.task_done()
            continue
            
        words = cleaned_chunk.split()
        is_long_sentence = len(words) >= 5
        
        # Decide which compute engine to use
        # GPU for first 2 long sentences (instant startup), CPU thereafter (free up GPU for local LLM)
        if use_dynamic_cap and sentence_count >= 2:
            engine = kokoro_cpu
            engine_name = "CPU (Low Priority)"
        else:
            engine = kokoro_gpu
            engine_name = "GPU (High Priority)"
            
        print(f"\n[Synthesis] Generating sentence on {engine_name} (Queue size: {audio_queue.qsize()}/3): '{cleaned_chunk}'")
        
        try:
            samples, sample_rate = engine.create(
                cleaned_chunk,
                voice=voice,
                speed=speed,
                lang="en-us"
            )
            
            # Put synthesized audio into the queue (will block if queue is full at maxsize=3)
            audio_queue.put((samples, sample_rate))
            
            if is_long_sentence:
                sentence_count += 1
                
        except Exception as e:
            print(f"\n[Synthesis] Error: {e}")
            
        text_queue.task_done()

# ── LLM Streaming Clients ─────────────────────────────────────────────────────
async def stream_groq(user_text, api_key, model):
    global first_token_time
    
    base_url = "https://api.groq.com/openai/v1"
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are Mr Meeseeks, a helpful assistant. Keep your responses highly conversational, short, and punchy. Avoid long paragraphs. Act friendly and energetic, like a helpful buddy."
            },
            {"role": "user", "content": user_text}
        ],
        "temperature": 0.7,
        "stream": True,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    
    async with httpx.AsyncClient(timeout=60) as client:
        async with client.stream("POST", f"{base_url}/chat/completions", json=payload, headers=headers) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        content = data["choices"][0].get("delta", {}).get("content", "")
                        if content:
                            if first_token_time is None:
                                first_token_time = time.time()
                            yield content
                    except Exception:
                        pass

async def stream_ollama(user_text, url, model):
    global first_token_time
    
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are Mr Meeseeks, a helpful assistant. Keep your responses highly conversational, short, and punchy. Avoid long paragraphs. Act friendly and energetic, like a helpful buddy."
            },
            {"role": "user", "content": user_text}
        ],
        "options": {
            "temperature": 0.7
        },
        "stream": True,
    }
    
    async with httpx.AsyncClient(timeout=60) as client:
        async with client.stream("POST", f"{url}/api/chat", json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    content = data.get("message", {}).get("content", "")
                    if content:
                        if first_token_time is None:
                            first_token_time = time.time()
                        yield content
                except Exception:
                    pass

# ── Text Chunking Loop ────────────────────────────────────────────────────────
async def run_streaming_pipeline(user_text, backend, api_key, ollama_url, model, text_queue):
    buffer = ""
    sentence_ends = {".", "!", "?", "\n"}
    
    print("\n[Meeseeks] ", end="", flush=True)
    
    if backend == "groq":
        stream_generator = stream_groq(user_text, api_key, model)
    else:
        stream_generator = stream_ollama(user_text, ollama_url, model)
        
    async for text in stream_generator:
        print(text, end="", flush=True)
        buffer += text
        
        # Sentence-based splitting
        split_idx = -1
        for i, char in enumerate(buffer):
            if char in sentence_ends:
                split_idx = i
                break
                
        if split_idx != -1:
            chunk_to_send = buffer[:split_idx + 1].strip()
            if chunk_to_send:
                text_queue.put(chunk_to_send)
            buffer = buffer[split_idx + 1:]
        elif len(buffer) > 120:  # Fallback: chunk anyway if sentence is too long
            space_idx = buffer.rfind(" ")
            if space_idx != -1:
                chunk_to_send = buffer[:space_idx].strip()
                if chunk_to_send:
                    text_queue.put(chunk_to_send)
                buffer = buffer[space_idx + 1:]
                
    print("\n")
    
    # Process anything remaining in the buffer
    if buffer.strip():
        text_queue.put(buffer.strip())
        
    # Signal synthesis thread that text stream is done
    text_queue.put(None)

def voice_interrupt_worker(whisper_model, voice_interrupt_flag, stop_listening_event, shared_audio_data, audio_lock):
    """
    Background worker that listens to the microphone while Mr Meeseeks is speaking/thinking,
    appends audio to shared_audio_data, and sets voice_interrupt_flag if it hears the user speak 3+ words.
    """
    import sounddevice as sd
    
    def callback(indata, frames, time, status):
        with audio_lock:
            shared_audio_data.append(indata.copy())
        
    sample_rate = 16000
    # Keep last 1.5 seconds of audio for VAD checking (1.5 * 16000 = 24000 samples)
    max_samples = int(sample_rate * 1.5)
    
    try:
        stream = sd.InputStream(samplerate=sample_rate, channels=1, callback=callback)
        with stream:
            # First phase: check for user speaking while AI is responding
            while not stop_listening_event.is_set() and not voice_interrupt_flag.is_set():
                time.sleep(0.4)
                
                # Make a thread-safe copy of the shared list
                with audio_lock:
                    if not shared_audio_data:
                        continue
                    shared_audio_data_copy = list(shared_audio_data)
                    
                # Concatenate buffer contents
                current_audio = np.concatenate(shared_audio_data_copy, axis=0).flatten()
                
                # Slice to retain only the last 1.5s
                if len(current_audio) > max_samples:
                    current_audio = current_audio[-max_samples:]
                    
                # Run transcription if we have at least 0.5s of audio
                if len(current_audio) >= sample_rate * 0.5:
                    segments, _ = whisper_model.transcribe(current_audio, beam_size=1)
                    text = " ".join([seg.text for seg in segments]).strip()
                    
                    # Clean transcription and check for 3+ words
                    words = text.split()
                    if len(words) >= 3:
                        # Ignore common Whisper tiny hallucinations on silence
                        hallucinations = {"thank you", "thanks for", "you for watching", "subtitles by", "please subscribe", "watching"}
                        is_hallucination = any(h in text.lower() for h in hallucinations)
                        if not is_hallucination:
                            print(f"\n🗣️  [Voice Interrupt] Heard: '{text}'")
                            voice_interrupt_flag.set()
                            break
                            
            # Second phase: if we were interrupted by voice, we keep the stream open
            # and just append audio until stop_listening_event is set by the main thread.
            while not stop_listening_event.is_set():
                time.sleep(0.1)
    except Exception as e:
        print(f"\n[Error in Voice Interrupt Worker] {e}")

import select

def check_stdin_interrupt() -> bool:
    """
    Checks if the user has pressed ENTER in the terminal to request an interrupt.
    Returns True if an input is pending, otherwise False.
    """
    if not sys.platform.startswith("linux"):
        return False
    # Use select to check if stdin has data ready to read (non-blocking)
    rlist, _, _ = select.select([sys.stdin], [], [], 0.0)
    if rlist:
        # Consume the typed line so it doesn't affect subsequent prompt inputs
        sys.stdin.readline()
        return True
    return False

# ── Main ──────────────────────────────────────────────────────────────────────
async def main():
    global start_time, first_token_time, first_audio_played_time, total_tokens_received, sentence_count
    
    parser = argparse.ArgumentParser(description="Interactive Voice REPL with GPU Kokoro TTS & Local Whisper STT")
    parser.add_argument("--backend", choices=["groq", "ollama"], default="groq",
                        help="LLM backend choice: groq or ollama (default: groq)")
    parser.add_argument("--voice", default="af_sarah", help="Kokoro voice (default: af_sarah)")
    parser.add_argument("--speed", type=float, default=1.1, help="Speech speed multiplier (default: 1.1)")
    parser.add_argument("--model", default=None,
                        help="LLM model name (defaults to llama-3.1-8b-instant for groq, qwen2.5:3b for ollama)")
    parser.add_argument("--no-dynamic-cap", action="store_true",
                        help="Disable dynamic CPU fallback compute cap for TTS after 2 sentences")
    parser.add_argument("--no-voice-interrupt", action="store_true",
                        help="Disable voice-activated barge-in / interrupt")
    args = parser.parse_args()
    
    # Set backend-specific defaults
    if args.backend == "groq":
        llm_model = args.model if args.model else "llama-3.1-8b-instant"
    else:
        llm_model = args.model if args.model else "qwen2.5:3b"
        
    use_dynamic_cap = not args.no_dynamic_cap
    voice_interrupt_enabled = not args.no_voice_interrupt
    
    print("=========================================================")
    print("          MR MEESEEKS — INTERACTIVE VOICE LOOP           ")
    print(f"   LLM Backend       : {args.backend.upper()} ({llm_model})")
    print(f"   TTS GPU VRAM Limit: 1 GB Cap (Arena limit) ✓")
    print(f"   Dynamic Compute   : {'ENABLED (First 2 sentences GPU, then CPU)' if use_dynamic_cap else 'DISABLED (Pure GPU)'}")
    print(f"   Voice Interrupt   : {'ENABLED (Speak 3+ words or press ENTER)' if voice_interrupt_enabled else 'DISABLED (Press ENTER only)'}")
    print(f"   Audio Queue Cap   : Capped at 3 sentences max size ✓")
    print("=========================================================")
    
    # 1. Preload CUDA libraries
    preload_cuda_libs()
    
    # 2. Get API credentials
    api_key = os.environ.get("GROQ_API_KEY", "")
    if args.backend == "groq" and not api_key:
        print("ERROR: GROQ_API_KEY not found in environment or .env file.")
        sys.exit(1)
        
    ollama_url = os.environ.get("OLLAMA_URL", "http://localhost:11434")
    
    # 3. Initialize Kokoro ONNX on GPU (with 500MB VRAM limit) and CPU
    project_root = os.path.dirname(os.path.abspath(__file__))
    if os.path.basename(project_root) == "tests":
        project_root = os.path.dirname(project_root)
    model_path = os.path.join(project_root, "models", "kokoro-v1.0.fp16.onnx")
    voices_path = os.path.join(project_root, "models", "voices-v1.0.bin")
    
    if not os.path.exists(model_path) or not os.path.exists(voices_path):
        print(f"ERROR: Kokoro files not found at {model_path} or {voices_path}")
        sys.exit(1)
        
    print("[Init] Loading Kokoro TTS Engine on GPU (with 1 GB VRAM Arena Limit)...")
    try:
        import onnxruntime as ort
        from kokoro_onnx import Kokoro
        
        # Configure GPU session options and mem limit
        gpu_opts = {
            "device_id": 0,
            "gpu_mem_limit": 2 * 1024 * 1024 * 1024,  # Cap VRAM at 2 GB (2048 MB)
            "cudnn_conv_algo_search": "HEURISTIC",
            "arena_extend_strategy": "kSameAsRequested"
        }
        
        gpu_sess = ort.InferenceSession(model_path, providers=[("CUDAExecutionProvider", gpu_opts), "CPUExecutionProvider"])
        kokoro_gpu = Kokoro.from_session(gpu_sess, voices_path)
        print("[Init] Kokoro GPU Session ready ✓")
        
        kokoro_cpu = None
        if use_dynamic_cap:
            print("[Init] Loading Kokoro TTS Engine on CPU (for compute capping fallback)...")
            cpu_sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
            kokoro_cpu = Kokoro.from_session(cpu_sess, voices_path)
            print("[Init] Kokoro CPU Session ready ✓")
            
    except Exception as e:
        print(f"ERROR: Failed to initialize Kokoro engine: {e}")
        sys.exit(1)
        
    # 4. Initialize Whisper STT (CPU)
    print("[Init] Loading Local Whisper STT model (tiny.en)...")
    try:
        from core.voice_input import VoiceInputManager
        voice_input_mgr = VoiceInputManager()
        voice_input_mgr.load_model()
        print("[Init] Whisper STT ready ✓")
    except Exception as e:
        print(f"ERROR: Failed to initialize Whisper model: {e}")
        sys.exit(1)
        
    print("\nInitialization completed successfully! Entering interactive loop.\n")
    print("Instructions:")
    print("  1. Press ENTER to start recording your speech.")
    print("  2. Speak into your microphone.")
    print("  3. Press ENTER again to stop recording and send your query.")
    print("  4. Type 'exit' and press Enter to quit.\n")
    
    # Event loop state
    interrupted_by_voice = False
    shared_audio_data = []
    vi_thread = None
    stop_listening_event = None
    audio_lock = threading.Lock()
    
    while True:
        try:
            if not interrupted_by_voice:
                cmd = input("Press ENTER to speak (or type 'exit' to quit): ").strip().lower()
                if cmd == "exit":
                    break
                    
                # Record and transcribe
                user_text = voice_input_mgr.record_and_transcribe()
                if not user_text:
                    print("[System] No speech detected. Try again.")
                    continue
            else:
                # We were interrupted by voice, so we are already recording!
                # We just wait for the user to press ENTER to stop recording.
                print("\n🎙️  [Meeseeks] Listening... Press ENTER to stop recording.")
                input()  # Blocks until user presses Enter
                
                # Signal the listener thread to stop (this closes the stream)
                if stop_listening_event:
                    stop_listening_event.set()
                if vi_thread:
                    vi_thread.join(timeout=2.0)
                
                # Transcribe the accumulated audio data
                with audio_lock:
                    if shared_audio_data:
                        audio = np.concatenate(shared_audio_data, axis=0).flatten()
                        print("[System] Transcribing your full input...")
                        segments, _ = voice_input_mgr.model.transcribe(audio, beam_size=1)
                        user_text = " ".join([seg.text for seg in segments]).strip()
                    else:
                        user_text = ""
                
                # Reset event loop state
                interrupted_by_voice = False
                shared_audio_data = []
                vi_thread = None
                stop_listening_event = None
                
                if not user_text:
                    print("[System] No speech detected. Try again.")
                    continue
                print(f"\nYou (Voice): {user_text}")
                
            print("[Meeseeks] Thinking...", flush=True)
            
            # Reset timing & sentence variables
            first_token_time = None
            first_audio_played_time = None
            start_time = time.time()
            sentence_count = 0
            
            # Create session-specific queues
            session_text_queue = queue.Queue()
            session_audio_queue = queue.Queue(maxsize=3)
            
            playback_stream_ref = [None]
            
            # Start worker threads
            s_thread = threading.Thread(
                target=synthesis_worker,
                args=(kokoro_gpu, kokoro_cpu, use_dynamic_cap, args.voice, args.speed, session_text_queue, session_audio_queue),
                daemon=True
            )
            p_thread = threading.Thread(
                target=playback_worker,
                args=(session_audio_queue, playback_stream_ref),
                daemon=True
            )
            s_thread.start()
            p_thread.start()
            
            # Start background voice interrupt listener if enabled
            voice_interrupt_flag = threading.Event()
            stop_listening_event = threading.Event()
            shared_audio_data = []
            vi_thread = None
            if voice_interrupt_enabled:
                vi_thread = threading.Thread(
                    target=voice_interrupt_worker,
                    args=(voice_input_mgr.model, voice_interrupt_flag, stop_listening_event, shared_audio_data, audio_lock),
                    daemon=True
                )
                vi_thread.start()
                
            # Run the generation pipeline in the background
            pipeline_task = asyncio.create_task(
                run_streaming_pipeline(user_text, args.backend, api_key, ollama_url, llm_model, session_text_queue)
            )
            
            interrupted = False
            try:
                # Non-blocking wait loop that monitors for user ENTER interrupts
                print(f"--- {'Press ENTER or speak 3+ words to interrupt' if voice_interrupt_enabled else 'Press ENTER to interrupt'} ---")
                if voice_interrupt_enabled:
                    print("[System] Voice interrupt active. (Use headphones to prevent speaker feedback!)")
                    
                while True:
                    key_interrupt = check_stdin_interrupt()
                    voice_interrupt = voice_interrupt_flag.is_set()
                    
                    if key_interrupt or voice_interrupt:
                        interrupted = True
                        
                        # Stop audio playback immediately without affecting recording
                        if playback_stream_ref[0] is not None:
                            try:
                                playback_stream_ref[0].abort()
                            except Exception:
                                pass
                        
                        # Clear queues
                        while not session_text_queue.empty():
                            try:
                                session_text_queue.get_nowait()
                                session_text_queue.task_done()
                            except (queue.Empty, ValueError):
                                pass
                        while not session_audio_queue.empty():
                            try:
                                session_audio_queue.get_nowait()
                                session_audio_queue.task_done()
                            except (queue.Empty, ValueError):
                                pass
                                
                        # Force exit worker loops
                        try:
                            session_text_queue.put_nowait(None)
                        except queue.Full:
                            pass
                        try:
                            session_audio_queue.put_nowait(None)
                        except queue.Full:
                            pass
                        
                        # Cancel the streaming pipeline
                        pipeline_task.cancel()
                        
                        if voice_interrupt:
                            interrupted_by_voice = True
                            print("\n🛑 [Voice Interrupt] Stopping Meeseeks speech. Continue speaking...")
                        else:
                            interrupted_by_voice = False
                            print("\n🛑 [Manual Interrupt] Stopping Meeseeks speech...")
                        break
                    
                    # Exit loop if both worker threads have naturally finished
                    if not s_thread.is_alive() and not p_thread.is_alive():
                        break
                        
                    await asyncio.sleep(0.05)
            finally:
                # Always ensure the background listener is stopped, unless it's a voice interrupt
                if not interrupted_by_voice:
                    if stop_listening_event:
                        stop_listening_event.set()
                
            if not interrupted:
                # Wait for threads to naturally clean up
                s_thread.join(timeout=1.0)
                p_thread.join(timeout=1.0)
            
            print(f"[System] Completed in {time.time() - start_time:.2f}s.\n")
            
        except KeyboardInterrupt:
            if stop_listening_event:
                stop_listening_event.set()
            print("\nExiting...")
            break
        except Exception as e:
            if stop_listening_event:
                stop_listening_event.set()
            print(f"\n[System] Loop encountered an error: {e}\n")
            
    print("Goodbye!")

if __name__ == "__main__":
    asyncio.run(main())
