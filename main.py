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

    # ── Voice Input Manager ──────────────────────────────────────────────────
    from core.voice_input import VoiceInputManager
    voice_input_mgr = VoiceInputManager()

    # ── REPL ─────────────────────────────────────────────────────────────────
    try:
        while True:
            used_voice = False
            try:
                user_input = input("You (Press Enter to speak, or type): ").strip()
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

            try:
                response = await brain.process(user_input)
                print(f"\n[Meeseeks] {response}\n")
                
                if used_voice:
                    from core.ipc_bus import bus
                    await bus.dispatch("speak", {"text": response})
                
                # Feedback loop
                if brain.last_interaction:
                    ans = input("Did it fulfill the request? (y/n) [y]: ").strip().lower()
                    user_success = ans != 'n'
                    score_str = input("Quality score (1-5) [5]: ").strip()
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


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[Meeseeks] Interrupted.")
