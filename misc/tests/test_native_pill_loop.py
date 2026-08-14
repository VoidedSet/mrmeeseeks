import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import asyncio
from PyQt6.QtWidgets import QApplication
from core.ui.overlay import UIOverlay
from core.state_machine import State

async def run_state_loop(overlay: UIOverlay):
    states = [
        (State.IDLE, "Idle (Slate gray, static dots)"),
        (State.LISTENING, "Listening (Cyan aura, capsule waveform)"),
        (State.THINKING, "Thinking (Purple aura, sweep, dots waving)"),
        (State.ACTING, "Acting/Working (Orange aura, sweep, dots rotating)"),
        (State.SPEAKING, "Speaking (Emerald aura, capsule waveform)")
    ]
    
    print("Starting infinite status pill animation test loop. Kill the task or process to stop.")
    try:
        while True:
            for state, desc in states:
                print(f"Switching state to: {desc}", flush=True)
                overlay.update_state(state)
                await asyncio.sleep(4)
    except asyncio.CancelledError:
        print("Test loop cancelled.")
    finally:
        overlay.update_state(State.IDLE)
        QApplication.quit()

def on_clicked():
    print("D-Bus Clicked signal received by UIOverlay! ✓", flush=True)

def main():
    app = QApplication(sys.argv)
    overlay = UIOverlay()
    overlay.clicked.connect(on_clicked)
    
    import qasync
    loop = qasync.QEventLoop(app)
    asyncio.set_event_loop(loop)
    
    try:
        loop.run_until_complete(run_state_loop(overlay))
    except (KeyboardInterrupt, SystemExit):
        print("Exiting loop...")

if __name__ == "__main__":
    main()
