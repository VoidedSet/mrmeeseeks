"""
main.py — Mr Meeseeks Entry Point
Interactive CLI REPL. Loads env, wires all agents, starts the brain.

Usage:
    python main.py
    python main.py --debug
"""

import asyncio
import logging
import os
import sys
import argparse

# ── Load .env before anything else ───────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv optional — user can export vars manually

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Mr Meeseeks — AI OS Companion")
parser.add_argument("--debug", action="store_true", help="Verbose logging")
parser.add_argument("--cli", action="store_true", help="Force command-line REPL mode")
parser.add_argument("--voice", default="af_sarah", help="Kokoro voice (default: af_sarah)")
parser.add_argument("--speed", type=float, default=1.1, help="Speech speed multiplier (default: 1.1)")
parser.add_argument("--no-dynamic-cap", action="store_true",
                    help="Disable dynamic CPU fallback compute cap for TTS after 2 sentences")
parser.add_argument("--no-voice-interrupt", action="store_true",
                    help="Disable voice-activated barge-in / interrupt")
args = parser.parse_args()

# ── Logging ───────────────────────────────────────────────────────────────────
log_level = logging.DEBUG if args.debug else logging.INFO
logging.basicConfig(
    level=log_level,
    format="%(asctime)s [%(name)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger("main")

# ── Log output to file too ────────────────────────────────────────────────────
import os as _os
_os.makedirs("logs/outputs", exist_ok=True)
from datetime import datetime
_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
_fh = logging.FileHandler(f"logs/outputs/run_{_ts}.txt")
_fh.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(message)s"))
logging.getLogger().addHandler(_fh)
log.info(f"Logging to logs/outputs/run_{_ts}.txt")


import concurrent.futures
import queue
import threading
import time
import re
import select
import numpy as np

_input_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

def _blocking_input(prompt: str) -> str:
    return input(prompt)

async def async_input(prompt: str) -> str:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_input_executor, _blocking_input, prompt)

# Timing variables
start_time = 0
first_token_time = None
first_audio_played_time = None
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
        if use_dynamic_cap and sentence_count >= 2 and kokoro_cpu is not None:
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


async def main():
    # ── Init LLM provider ────────────────────────────────────────────────────
    from core.llm_provider import init_provider
    try:
        provider = init_provider()
        log.info(f"LLM backend: {provider.name}")
    except ValueError as e:
        log.error(f"Failed to initialize LLM provider: {e}")
        sys.exit(1)

    # ── Register agents ──────────────────────────────────────────────────────
    from agents.sysadmin_agent import register as reg_sysadmin
    reg_sysadmin()

    from agents.memory_agent import register as reg_memory
    memory = reg_memory()

    from agents.web_agent import register as reg_web
    reg_web()

    from agents.hands_agent import register as reg_hands
    reg_hands()

    from agents.eyes_agent import register as reg_eyes
    reg_eyes()

    if not args.cli:
        from agents.voice_agent import register as reg_voice
        reg_voice()

    # ── Wire brain ───────────────────────────────────────────────────────────
    from core.brain import brain
    brain.inject_memory_agent(memory)

    # ── Start kernel listener (background task) ───────────────────────────────
    from kernel.kernel_listener import start as start_kernel
    kernel_task = asyncio.create_task(start_kernel(brain))
    log.info("Kernel listener started ✓")

    # ── Print banner ─────────────────────────────────────────────────────────
    backend = os.environ.get("LLM_BACKEND", "groq")
    model   = os.environ.get("LLM_MODEL", "llama-3.1-8b-instant")
    print()
    print("╔══════════════════════════════════════════╗")
    print("║        MR MEESEEKS — OS COMPANION        ║")
    print(f"║  backend: {backend:<10}  model: {model:<14}║")
    print("║  Type your request. Ctrl+C to exit.      ║")
    print("╚══════════════════════════════════════════╝")
    print()

    if args.cli:
        # 1. Preload CUDA libraries
        preload_cuda_libs()

        # 2. Initialize Kokoro TTS GPU and CPU sessions
        project_root = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(project_root, "models", "kokoro-v1.0.fp16.onnx")
        voices_path = os.path.join(project_root, "models", "voices-v1.0.bin")

        if not os.path.exists(model_path) or not os.path.exists(voices_path):
            log.error(f"Kokoro files not found at {model_path} or {voices_path}")
            sys.exit(1)

        log.info("Loading Kokoro TTS Engine on GPU (with 1 GB VRAM Arena Limit)...")
        try:
            import onnxruntime as ort
            from kokoro_onnx import Kokoro

            gpu_opts = {
                "device_id": 0,
                "gpu_mem_limit": 1024 * 1024 * 1024,
                "cudnn_conv_algo_search": "HEURISTIC",
                "arena_extend_strategy": "kSameAsRequested"
            }
            gpu_sess = ort.InferenceSession(model_path, providers=[("CUDAExecutionProvider", gpu_opts), "CPUExecutionProvider"])
            kokoro_gpu = Kokoro.from_session(gpu_sess, voices_path)
            log.info("Kokoro GPU Session ready ✓")

            kokoro_cpu = None
            use_dynamic_cap = not args.no_dynamic_cap
            if use_dynamic_cap:
                log.info("Loading Kokoro TTS Engine on CPU (for compute capping fallback)...")
                cpu_sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
                kokoro_cpu = Kokoro.from_session(cpu_sess, voices_path)
                log.info("Kokoro CPU Session ready ✓")
        except Exception as e:
            log.error(f"Failed to initialize Kokoro engine: {e}")
            sys.exit(1)

        # ── Voice Input Manager ──────────────────────────────────────────────────
        from core.voice_input import VoiceInputManager
        voice_input_mgr = VoiceInputManager()
        voice_input_mgr.load_model()

        # ── REPL ─────────────────────────────────────────────────────────────────
        interrupted_by_voice = False
        shared_audio_data = []
        vi_thread = None
        stop_listening_event = None
        audio_lock = threading.Lock()
        voice_interrupt_enabled = not args.no_voice_interrupt

        try:
            while True:
                used_voice = False
                try:
                    if not interrupted_by_voice:
                        user_input = (await async_input("You (Press Enter to speak, or type): ")).strip()
                        if not user_input:
                            used_voice = True
                            user_input = voice_input_mgr.record_and_transcribe()
                            if not user_input:
                                print("[Meeseeks] No voice input detected.")
                                continue
                            print(f"You (voice): {user_input}")
                    else:
                        # We were interrupted by voice, so we are already recording!
                        print("\n🎙️  [Meeseeks] Listening... Press ENTER to stop recording.")
                        await async_input("")  # Non-blocking wait for Enter
                        
                        if stop_listening_event:
                            stop_listening_event.set()
                        if vi_thread:
                            vi_thread.join(timeout=2.0)
                            
                        with audio_lock:
                            if shared_audio_data:
                                audio = np.concatenate(shared_audio_data, axis=0).flatten()
                                print("[System] Transcribing your full input...")
                                segments, _ = voice_input_mgr.model.transcribe(audio, beam_size=1)
                                user_text = " ".join([seg.text for seg in segments]).strip()
                            else:
                                user_text = ""
                        
                        interrupted_by_voice = False
                        shared_audio_data = []
                        vi_thread = None
                        stop_listening_event = None
                        
                        if not user_text:
                            print("[System] No speech detected. Try again.")
                            continue
                        print(f"\nYou (Voice): {user_text}")
                        user_input = user_text
                        used_voice = True
                except (EOFError, KeyboardInterrupt):
                    print("\n[Meeseeks] Goodbye.")
                    break

                if user_input.lower() in {"exit", "quit", "bye"}:
                    print("[Meeseeks] Goodbye.")
                    break

                print("[Meeseeks] Thinking...", flush=True)

                # Reset timing & sentence variables
                global start_time, first_token_time, first_audio_played_time, sentence_count
                first_token_time = None
                first_audio_played_time = None
                start_time = time.time()
                sentence_count = 0

                header_printed = False

                # Create session-specific queues
                session_text_queue = queue.Queue()
                session_audio_queue = queue.Queue(maxsize=3)
                playback_stream_ref = [None]

                # Start worker threads if voice is used
                s_thread = None
                p_thread = None
                if used_voice:
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

                def on_chunk(chunk: str):
                    nonlocal header_printed
                    if not header_printed:
                        print("\n[Meeseeks] ", end="", flush=True)
                        header_printed = True
                    print(chunk, end="", flush=True)

                def on_sentence(sentence: str):
                    if used_voice:
                        session_text_queue.put(sentence)

                # Run brain.process inside an asyncio task
                brain_task = asyncio.create_task(
                    brain.process(
                        user_input,
                        on_chunk=on_chunk,
                        on_sentence=on_sentence
                    )
                )

                interrupted = False
                sent_none = False
                try:
                    if used_voice:
                        # Wait loop that monitors for interrupts while brain task runs
                        print(f"--- {'Press ENTER or speak 3+ words to interrupt' if voice_interrupt_enabled else 'Press ENTER to interrupt'} ---")
                        if voice_interrupt_enabled:
                            print("[System] Voice interrupt active. (Use headphones to prevent speaker feedback!)")

                        while True:
                            key_interrupt = check_stdin_interrupt()
                            voice_interrupt = voice_interrupt_flag.is_set() if voice_interrupt_enabled else False

                            if key_interrupt or voice_interrupt:
                                interrupted = True
                                
                                # Stop audio playback immediately
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

                                # Cancel the brain task
                                brain_task.cancel()

                                if voice_interrupt:
                                    interrupted_by_voice = True
                                    print("\n🛑 [Voice Interrupt] Stopping Meeseeks speech. Continue speaking...")
                                else:
                                    interrupted_by_voice = False
                                    print("\n🛑 [Manual Interrupt] Stopping Meeseeks speech...")
                                break

                            # If brain task is done and we haven't sent the None token yet, send it
                            if brain_task.done() and not sent_none:
                                try:
                                    session_text_queue.put_nowait(None)
                                except queue.Full:
                                    pass
                                sent_none = True

                            # Exit wait loop if brain is done and threads have finished
                            if brain_task.done() and not s_thread.is_alive() and not p_thread.is_alive():
                                break

                            await asyncio.sleep(0.05)
                    else:
                        # Text mode: just await brain task
                        await brain_task
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    log.exception(f"Brain.process raised: {e}")
                    print(f"\n[Meeseeks] Internal error: {e}\n")
                finally:
                    # Signal synthesis thread that we are done generating text
                    if used_voice:
                        try:
                            session_text_queue.put_nowait(None)
                        except queue.Full:
                            pass
                            
                    # Always ensure the background listener is stopped, unless it's a voice interrupt
                    if not interrupted_by_voice:
                        if stop_listening_event:
                            stop_listening_event.set()

                # Cleanup threads if voice was used and not interrupted
                if used_voice and not interrupted:
                    s_thread.join(timeout=1.0)
                    p_thread.join(timeout=1.0)

                print()
                
                # Clear last interaction state
                brain.last_interaction = None
        finally:
            # Clean shutdown — cancel background task
            kernel_task.cancel()
            try:
                await kernel_task
            except asyncio.CancelledError:
                pass
            log.info("Kernel listener stopped.")
    else:
        # ── GUI Mode ─────────────────────────────────────────────────────────────
        from PyQt6.QtWidgets import QApplication
        from core.ui.overlay import UIOverlay
        from core.ui.tray import TrayController
        from core.ui.panel import ControlPanel
        from core.ui.hotkey import GlobalHotkeyMonitor
        from core.ipc_bus import bus

        overlay = UIOverlay()

        panel = ControlPanel(hotkey_label="Ctrl+Alt+Space")
        panel.quit_requested.connect(QApplication.instance().quit)

        tray = TrayController()
        tray.open_panel_requested.connect(panel.show_near_cursor)
        tray.quit_requested.connect(QApplication.instance().quit)
        tray.show()

        # Voice recording and toggle logic
        voice_stop_event = None
        voice_recording_task = None

        async def handle_submit(prompt: str):
            full_text = ""
            def on_chunk(chunk: str):
                nonlocal full_text
                full_text += chunk

            def on_sentence(sentence: str):
                asyncio.create_task(bus.dispatch("speak", {"text": sentence}))

            try:
                await brain.process(prompt, on_chunk=on_chunk, on_sentence=on_sentence)
            except Exception as e:
                log.error(f"Brain process error: {e}")

        async def toggle_voice():
            nonlocal voice_stop_event, voice_recording_task
            from core.state_machine import State
            
            current_state = brain.state_machine.current
            
            if current_state == State.IDLE:
                voice_stop_event = asyncio.Event()
                await brain.state_machine.transition(State.LISTENING)
                
                from core.voice_input import VoiceInputManager
                voice_mgr = VoiceInputManager()
                
                async def record_task():
                    try:
                        text = await voice_mgr.record_and_transcribe_async(voice_stop_event)
                        if text and text.strip():
                            log.info(f"Voice query: {text}")
                            await handle_submit(text)
                        else:
                            await brain.state_machine.transition(State.IDLE)
                    except Exception as ex:
                        log.error(f"Voice task error: {ex}")
                        await brain.state_machine.transition(State.IDLE)
                
                voice_recording_task = asyncio.create_task(record_task())
            elif current_state == State.LISTENING:
                if voice_stop_event:
                    voice_stop_event.set()
            else:
                # Cancel thinking/acting/speaking and go back to Idle
                if voice_recording_task and not voice_recording_task.done():
                    voice_recording_task.cancel()
                overlay.dismiss()
                await brain.state_machine.transition(State.IDLE)

        # Wire global summons hotkey to toggle voice conversation directly
        hotkey_monitor = GlobalHotkeyMonitor("<ctrl>+<alt>+<space>")
        hotkey_monitor.triggered.connect(lambda: asyncio.create_task(toggle_voice()))
        hotkey_monitor.start()

        # Left-click on status pill toggles voice recording (routed via D-Bus)
        overlay.clicked.connect(lambda: asyncio.create_task(toggle_voice()))

        # Right-click panel positioning is deprecated for the native GNOME Top Bar extension.
        # The control panel can be toggled via the global hotkey or the system tray menu.
        # def show_panel_below_pill():
        #     pass

        # Wire state machine changes to UI status pill updates
        brain.state_machine.listeners.append(overlay.update_state)

        # Wait until Qt application quits
        shutdown_event = asyncio.Event()
        QApplication.instance().aboutToQuit.connect(shutdown_event.set)
        await shutdown_event.wait()

        # Clean shutdown
        if voice_recording_task and not voice_recording_task.done():
            voice_recording_task.cancel()
        hotkey_monitor.stop()
        overlay.close()
        panel.close()
        kernel_task.cancel()
        try:
            await kernel_task
        except asyncio.CancelledError:
            pass
        log.info("Shutdown completed.")


if __name__ == "__main__":
    is_cli = "--cli" in sys.argv
    if is_cli:
        try:
            asyncio.run(main())
        except KeyboardInterrupt:
            print("\n[Meeseeks] Interrupted.")
    else:
        from PyQt6.QtWidgets import QApplication
        import qasync
        
        app = QApplication(sys.argv)
        app.setQuitOnLastWindowClosed(False)
        app.setApplicationName("Mr Meeseeks")
        
        loop = qasync.QEventLoop(app)
        asyncio.set_event_loop(loop)
        
        try:
            loop.run_until_complete(main())
        except KeyboardInterrupt:
            print("\n[Meeseeks] Interrupted.")
