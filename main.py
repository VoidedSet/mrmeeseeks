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
import glob

# ── Dynamic LD_LIBRARY_PATH setup for CUDA & cuDNN v9 libraries ────────────────
project_root = os.path.dirname(os.path.abspath(__file__))
site_packages = os.path.join(project_root, "venv", "lib", f"python{sys.version_info.major}.{sys.version_info.minor}", "site-packages")
nvidia_dirs = glob.glob(os.path.join(site_packages, "nvidia", "*", "lib"))
ort_capi = os.path.join(site_packages, "onnxruntime", "capi")
if os.path.exists(ort_capi):
    nvidia_dirs.append(ort_capi)

if nvidia_dirs:
    curr_ld = os.environ.get("LD_LIBRARY_PATH", "")
    new_ld = ":".join(nvidia_dirs + ([curr_ld] if curr_ld else []))
    if os.environ.get("_MEESEEKS_LD_SET") != "1":
        os.environ["LD_LIBRARY_PATH"] = new_ld
        os.environ["_MEESEEKS_LD_SET"] = "1"
        os.execv(sys.executable, [sys.executable] + sys.argv)

# Disable TensorRT provider probing to prevent harmless C++ warnings
os.environ["ORT_DISABLE_TENSORRT"] = "1"
try:
    import onnxruntime as ort
    ort.set_default_logger_severity(3)  # 3 = ERROR / FATAL only
except Exception:
    pass

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
parser.add_argument("--voice", default="am_onyx", help="Kokoro voice (default: am_onyx)")
parser.add_argument("--speed", type=float, default=1.1, help="Speech speed multiplier (default: 1.1)")
parser.add_argument("--backend", choices=["groq", "ollama"], default=None,
                    help="LLM backend choice: groq or ollama")
parser.add_argument("--model", default=None,
                    help="LLM model name")
parser.add_argument("--dynamic-cap", action="store_true",
                    help="Enable dynamic CPU fallback compute cap for TTS after 2 sentences")
parser.add_argument("--no-voice-interrupt", action="store_true",
                    help="Disable voice-activated barge-in / interrupt")
parser.add_argument("--no-brief", action="store_true", help="Skip the morning brief")
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
logging.getLogger("faster_whisper").setLevel(logging.WARNING)

# ── Log output to file too ────────────────────────────────────────────────────
import os as _os
_os.makedirs("logs/outputs", exist_ok=True)
from datetime import datetime
_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
_fh = logging.FileHandler(f"logs/outputs/run_{_ts}.txt")
_fh.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(message)s"))
logging.getLogger().addHandler(_fh)
log.info(f"Logging to logs/outputs/run_{_ts}.txt")


import queue
import threading
import time
import numpy as np
import asyncio


from utils.audio import (
    preload_cuda_libs,
    clean_text_for_tts,
    playback_worker,
    synthesis_worker,
    voice_interrupt_worker,
    check_stdin_interrupt,
    async_input
)

from core.brain import SentenceStreamer, brain
from core.state_machine import State
import core.llm_provider as llm_mod


def start_global_quit_listener():
    """Starts a global keyboard listener to gracefully quit Meeseeks on Ctrl+Alt+Escape or Ctrl+Alt+Q."""
    try:
        from pynput import keyboard
        
        COMBINATIONS = [
            {keyboard.Key.ctrl, keyboard.Key.alt, keyboard.Key.esc},
            {keyboard.Key.ctrl, keyboard.Key.alt, keyboard.KeyCode(char='q')},
            {keyboard.Key.ctrl, keyboard.Key.alt, keyboard.KeyCode(char='Q')}
        ]
        current_keys = set()

        def on_press(key):
            current_keys.add(key)
            for combo in COMBINATIONS:
                if all(k in current_keys for k in combo):
                    print("\n[Meeseeks] Global exit hotkey detected! Exiting gracefully...")
                    from PyQt6.QtWidgets import QApplication
                    app = QApplication.instance()
                    if app:
                        app.quit()
                    else:
                        import os
                        import signal
                        handle_signal_quit(signal.SIGINT, None)
                    break

        def on_release(key):
            try:
                current_keys.remove(key)
            except KeyError:
                pass

        listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        listener.daemon = True
        listener.start()
        print("[System] Global quit hotkey listener active (Ctrl+Alt+Esc or Ctrl+Alt+Q) ✓")
    except Exception as e:
        print(f"[Warning] Failed to start global keyboard listener: {e}")



async def main():
    from core.service_manager import service_manager
    await service_manager.start_services()
    try:
        await _main_impl()
    finally:
        # Run consolidation if memory agent was injected
        from core.brain import brain
        if brain.memory_agent:
            log.info("Consolidating session memories...")
            try:
                from core.chroma_store import chroma_store
                await asyncio.to_thread(chroma_store.consolidate_session, brain.memory_agent.session_id)
            except Exception as e:
                log.error(f"Error during memory consolidation: {e}")
        
        # Save conversation history to a text file
        if brain.history:
            log.info("Saving conversation history...")
            try:
                filepath = brain.history.save_to_text_file()
                if filepath:
                    print(f"\n[Meeseeks] Conversation log saved to: {filepath}\n")
            except Exception as e:
                log.error(f"Error saving conversation log: {e}")

        await asyncio.shield(service_manager.stop_services())


async def run_morning_brief_if_needed():
    """Checks session monitor and runs morning brief on session open."""
    try:
        from kernel.session_monitor import SessionMonitor
        from subsystems.morning.briefing import run_morning_brief
        monitor = SessionMonitor()
        if await monitor.should_run_brief():
            log.info("[Morning Brief] Session resume / laptop open detected — running brief!")
            await run_morning_brief(user_name="Kshayik")
    except Exception as e:
        log.warning(f"[Morning Brief] Error running brief: {e}")

async def _main_impl():
    # Start the global quit hotkey listener (Ctrl+Alt+Esc or Ctrl+Alt+Q)
    start_global_quit_listener()

    # Propagate CLI arguments directly to environment variables before initializing the provider
    if args.backend:
        os.environ["LLM_BACKEND"] = args.backend
    if args.model:
        os.environ["LLM_MODEL"] = args.model
    if args.voice:
        os.environ["KOKORO_VOICE"] = args.voice
    if args.speed:
        os.environ["KOKORO_SPEED"] = str(args.speed)

    # ── Init LLM provider ────────────────────────────────────────────────────
    from core.llm_provider import init_provider
    try:
        provider = init_provider()
        log.info(f"LLM backend: {provider.name}")
    except ValueError as e:
        log.error(f"Failed to initialize LLM provider: {e}")
        sys.exit(1)

    # ── Register core agents ──────────────────────────────────────────────────
    from agents.memory_agent import register as reg_memory
    memory = reg_memory()

    from agents.web_agent import register as reg_web
    reg_web()

    from agents.email_agent import register as reg_email
    reg_email()

    from agents.voice_agent import register as reg_voice
    reg_voice()

    # ── Wire brain ───────────────────────────────────────────────────────────
    from core.brain import brain
    brain.inject_memory_agent(memory)

    # ── Start kernel listener (background task) ───────────────────────────────
    from kernel.kernel_listener import start as start_kernel
    kernel_task = asyncio.create_task(start_kernel(brain))
    log.info("Kernel listener started ✓")

    # ── Session monitor + Morning Brief ──────────────────────────────
    if not args.no_brief:
        await run_morning_brief_if_needed()


    # ── Print banner ─────────────────────────────────────────────────────────
    backend = os.environ.get("LLM_BACKEND", "groq")
    model   = os.environ.get("LLM_MODEL", "llama-3.1-8b-instant")
    print()
    print("╔══════════════════════════════════════════╗")
    print("║      F.R.I.D.A.Y. — AI OS COMPANION       ║")
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
                "gpu_mem_limit": 1 * 1024 * 1024 * 1024,  # Cap VRAM at 1 GB (1024 MB)
                "cudnn_conv_algo_search": "HEURISTIC",
                "arena_extend_strategy": "kSameAsRequested"
            }
            gpu_sess = ort.InferenceSession(model_path, providers=[("CUDAExecutionProvider", gpu_opts)])
            kokoro_gpu = Kokoro.from_session(gpu_sess, voices_path)
            from agents.voice_agent import set_kokoro_engine
            set_kokoro_engine(kokoro_gpu)
            kokoro_cpu = None
            use_dynamic_cap = False
            log.info("Kokoro GPU Session ready (GPU-ONLY shared session) ✓")
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
                timing_ref = {'first_audio_played_time': None, 'start_time': time.time()}
                sentence_count_ref = [0]

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
                        args=(kokoro_gpu, kokoro_cpu, use_dynamic_cap, args.voice, args.speed, session_text_queue, session_audio_queue, sentence_count_ref),
                        daemon=True
                    )
                    p_thread = threading.Thread(
                        target=playback_worker,
                        args=(session_audio_queue, playback_stream_ref, timing_ref),
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
        from core.ipc_bus import bus

        overlay = UIOverlay()

        # Initialize voice input manager once for the GUI session
        from core.voice_input import VoiceInputManager
        voice_input_mgr = VoiceInputManager()
        voice_input_mgr.load_model()

        # Voice recording and toggle logic
        voice_stop_event = None
        voice_recording_task = None
        brain_process_task = None

        interrupted_by_voice = False
        shared_audio_data = []
        vi_thread = None
        stop_listening_event = None
        audio_lock = threading.Lock()
        voice_interrupt_flag = threading.Event()

        async def handle_submit(prompt: str):
            nonlocal brain_process_task
            nonlocal interrupted_by_voice, shared_audio_data, vi_thread, stop_listening_event, voice_interrupt_flag
            
            # Start background interrupt worker if voice interrupts are enabled
            voice_interrupt_enabled = not args.no_voice_interrupt
            if voice_interrupt_enabled:
                voice_input_mgr.load_model()
                
                voice_interrupt_flag.clear()
                stop_listening_event = threading.Event()
                shared_audio_data = []
                vi_thread = threading.Thread(
                    target=voice_interrupt_worker,
                    args=(voice_input_mgr.model, voice_interrupt_flag, stop_listening_event, shared_audio_data, audio_lock),
                    daemon=True
                )
                vi_thread.start()
            
            full_text = ""
            speech_sent = False

            def on_chunk(chunk: str):
                nonlocal full_text
                full_text += chunk

            def on_sentence(sentence: str):
                nonlocal speech_sent
                speech_sent = True
                asyncio.create_task(bus.dispatch("speak", {"text": sentence}))

            try:
                await brain.state_machine.transition(State.THINKING)
                brain_process_task = asyncio.create_task(
                    brain.process(prompt, on_chunk=on_chunk, on_sentence=on_sentence)
                )
                
                if voice_interrupt_enabled:
                    while not brain_process_task.done():
                        if voice_interrupt_flag.is_set():
                            # Stop speaking immediately
                            await bus.dispatch("stop_speak", {})
                            # Cancel the brain process
                            brain_process_task.cancel()
                            # Transition to LISTENING state
                            interrupted_by_voice = True
                            await brain.state_machine.transition(State.LISTENING)
                            log.info("Voice barge-in triggered! Listening...")
                            return
                        await asyncio.sleep(0.05)
                
                await brain_process_task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                log.error(f"Brain process error: {e}")
            finally:
                # Always ensure the background listener is stopped if we weren't interrupted by voice
                if not interrupted_by_voice:
                    if not speech_sent:
                        await brain.state_machine.transition(State.IDLE)
                    if stop_listening_event:
                        stop_listening_event.set()
                    if vi_thread:
                        await asyncio.to_thread(vi_thread.join, timeout=1.0)
                    vi_thread = None
                    stop_listening_event = None

        async def toggle_voice():
            nonlocal voice_stop_event, voice_recording_task, brain_process_task
            nonlocal interrupted_by_voice, shared_audio_data, vi_thread, stop_listening_event, voice_interrupt_flag
            from core.state_machine import State
            
            current_state = brain.state_machine.current
            
            if current_state == State.IDLE:
                interrupted_by_voice = False
                voice_stop_event = asyncio.Event()
                await brain.state_machine.transition(State.LISTENING)
                
                async def record_task():
                    try:
                        text = await voice_input_mgr.record_and_transcribe_async(voice_stop_event)
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
                if interrupted_by_voice:
                    if stop_listening_event:
                        stop_listening_event.set()
                    if vi_thread:
                        await asyncio.to_thread(vi_thread.join, timeout=2.0)
                    
                    voice_input_mgr.load_model()
                    
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
                    voice_interrupt_flag.clear()
                    
                    if user_text:
                        log.info(f"Voice query (interrupted): {user_text}")
                        await handle_submit(user_text)
                    else:
                        await brain.state_machine.transition(State.IDLE)
                else:
                    if voice_stop_event:
                        voice_stop_event.set()
            else:
                # Cancel thinking/acting/speaking and go back to Idle
                if voice_recording_task and not voice_recording_task.done():
                    voice_recording_task.cancel()
                if brain_process_task and not brain_process_task.done():
                    brain_process_task.cancel()
                await bus.dispatch("stop_speak", {})
                if vi_thread:
                    if stop_listening_event:
                        stop_listening_event.set()
                    await asyncio.to_thread(vi_thread.join, timeout=1.0)
                interrupted_by_voice = False
                shared_audio_data = []
                vi_thread = None
                stop_listening_event = None
                voice_interrupt_flag.clear()
                overlay.dismiss()
                await brain.state_machine.transition(State.IDLE)

        # Left-click on status pill toggles voice recording (routed via D-Bus)
        overlay.clicked.connect(lambda: asyncio.create_task(toggle_voice()))

        # Wire state machine changes to UI status pill updates
        brain.state_machine.listeners.append(overlay.update_state)

        # Wait until Qt application quits
        shutdown_event = asyncio.Event()
        QApplication.instance().aboutToQuit.connect(shutdown_event.set)
        await shutdown_event.wait()

        # Clean shutdown
        if voice_recording_task and not voice_recording_task.done():
            voice_recording_task.cancel()
        overlay.close()
        kernel_task.cancel()
        try:
            await kernel_task
        except asyncio.CancelledError:
            pass
        log.info("Shutdown completed.")


def handle_signal_quit(sig, frame):
    print(f"\n[Meeseeks] Signal received ({sig}). Shutting down gracefully...", flush=True)
    try:
        from core.brain import brain
        if brain.memory_agent:
            print("[Meeseeks] Consolidating session memories...", flush=True)
            from core.chroma_store import chroma_store
            chroma_store.consolidate_session(brain.memory_agent.session_id)
    except Exception as e:
        print(f"[Warning] Failed to consolidate memories on quit: {e}", flush=True)

    try:
        from core.brain import brain
        if brain.history:
            print("[Meeseeks] Saving conversation log...", flush=True)
            filepath = brain.history.save_to_text_file()
            if filepath:
                print(f"[Meeseeks] Conversation log saved to: {filepath}", flush=True)
    except Exception as e:
        print(f"[Warning] Failed to save conversation log on quit: {e}", flush=True)

    try:
        from core.service_manager import service_manager
        service_manager.stop_services_sync()
    except Exception:
        pass
    
    import subprocess
    for proc_name in ["meeseeks_service.py"]:
        try:
            subprocess.run(["pkill", "-9", "-f", proc_name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass
            
    print("[Meeseeks] Shutdown completed. Exiting.", flush=True)
    import sys
    sys.stdout.flush()
    sys.stderr.flush()
    import os
    os._exit(0)


if __name__ == "__main__":
    import signal
    signal.signal(signal.SIGINT, handle_signal_quit)
    signal.signal(signal.SIGTERM, handle_signal_quit)

    is_cli = "--cli" in sys.argv
    if is_cli:
        try:
            asyncio.run(main())
        except KeyboardInterrupt:
            handle_signal_quit(signal.SIGINT, None)
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
            handle_signal_quit(signal.SIGINT, None)
