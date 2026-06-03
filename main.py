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

    # ── Wire brain ───────────────────────────────────────────────────────────
    from core.brain import brain
    brain.inject_memory_agent(memory)

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

    # ── REPL ─────────────────────────────────────────────────────────────────
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[Meeseeks] Goodbye.")
            break

        if not user_input:
            continue

        if user_input.lower() in {"exit", "quit", "bye"}:
            print("[Meeseeks] Goodbye.")
            break

        print("[Meeseeks] Thinking...", flush=True)

        try:
            response = await brain.process(user_input)
            print(f"\n[Meeseeks] {response}\n")
        except Exception as e:
            log.exception(f"Brain.process raised: {e}")
            print(f"\n[Meeseeks] Internal error: {e}\n")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[Meeseeks] Interrupted.")
