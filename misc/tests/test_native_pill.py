import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import asyncio
from PyQt6.QtWidgets import QApplication
from core.ui.overlay import UIOverlay
from core.state_machine import State

async def test_flow(overlay: UIOverlay):
    # Test setting state to LISTENING
    print("Setting state to LISTENING (Cyan aura, capsule waveform)...")
    overlay.update_state(State.LISTENING)
    await asyncio.sleep(3)
    
    # Test setting state to THINKING
    print("Setting state to THINKING (Purple aura, sweep, dots waving)...")
    overlay.update_state(State.THINKING)
    await asyncio.sleep(3)
    
    # Test setting state to ACTING
    print("Setting state to ACTING (Orange aura, sweep, dots rotating in circle)...")
    overlay.update_state(State.ACTING)
    await asyncio.sleep(3)
    
    # Test setting state to SPEAKING
    print("Setting state to SPEAKING (Emerald aura, capsule waveform)...")
    overlay.update_state(State.SPEAKING)
    await asyncio.sleep(3)
    
    # Test setting state to IDLE
    print("Setting state to IDLE (Slate gray, static dots)...")
    overlay.update_state(State.IDLE)
    await asyncio.sleep(2)
    
    print("Test flow completed successfully! ✓")
    QApplication.quit()

def on_clicked():
    print("D-Bus Clicked signal received by UIOverlay! ✓")

def main():
    app = QApplication(sys.argv)
    overlay = UIOverlay()
    overlay.clicked.connect(on_clicked)
    
    import qasync
    loop = qasync.QEventLoop(app)
    asyncio.set_event_loop(loop)
    
    print("Starting native status pill test flow...")
    loop.run_until_complete(test_flow(overlay))

if __name__ == "__main__":
    main()
