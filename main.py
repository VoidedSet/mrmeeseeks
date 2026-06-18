import sys
import os
import asyncio

# Set up python search path to resolve subsystem imports
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import dependencies from brain's main.py
from subsystems.brain.main import main, handle_signal_quit

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
