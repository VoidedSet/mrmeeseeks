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
_input_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

def _blocking_input(prompt: str) -> str:
    return input(prompt)

async def async_input(prompt: str) -> str:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_input_executor, _blocking_input, prompt)


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
        # ── Voice Input Manager ──────────────────────────────────────────────────
        from core.voice_input import VoiceInputManager
        voice_input_mgr = VoiceInputManager()

        # ── REPL ─────────────────────────────────────────────────────────────────
        try:
            while True:
                used_voice = False
                try:
                    user_input = (await async_input("You (Press Enter to speak, or type): ")).strip()
                except (EOFError, KeyboardInterrupt):
                    print("\n[Meeseeks] Goodbye.")
                    break

                if not user_input:
                    used_voice = True
                    user_input = voice_input_mgr.record_and_transcribe()
                    if not user_input:
                        print("[Meeseeks] No voice input detected.")
                        continue
                    print(f"You (voice): {user_input}")

                if user_input.lower() in {"exit", "quit", "bye"}:
                    print("[Meeseeks] Goodbye.")
                    break

                print("[Meeseeks] Thinking...", flush=True)

                header_printed = False

                def on_chunk(chunk: str):
                    nonlocal header_printed
                    if not header_printed:
                        print("\n[Meeseeks] ", end="", flush=True)
                        header_printed = True
                    print(chunk, end="", flush=True)

                def on_sentence(sentence: str):
                    if used_voice:
                        from core.ipc_bus import bus
                        asyncio.create_task(bus.dispatch("speak", {"text": sentence}))

                try:
                    response = await brain.process(
                        user_input,
                        on_chunk=on_chunk,
                        on_sentence=on_sentence
                    )
                    print()
                    
                    # Feedback loop
                    if brain.last_interaction:
                        ans = (await async_input("Did it fulfill the request? (y/n) [y]: ")).strip().lower()
                        user_success = ans != 'n'
                        score_str = (await async_input("Quality score (1-5) [5]: ")).strip()
                        try:
                            score = int(score_str) if score_str else 5
                        except ValueError:
                            score = 5
                        score = max(1, min(5, score))
                        brain.log_finetune_sample(user_success, score)
                        print()
                except Exception as e:
                    log.exception(f"Brain.process raised: {e}")
                    print(f"\n[Meeseeks] Internal error: {e}\n")
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
