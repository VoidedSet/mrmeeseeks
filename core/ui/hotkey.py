from __future__ import annotations

import logging

from PyQt6.QtCore import QObject, pyqtSignal
from pynput import keyboard


logger = logging.getLogger("ui.hotkey")


class GlobalHotkeyMonitor(QObject):
    """Listens for a global hotkey on a background thread and emits a Qt signal.

    pynput runs its keyboard listener in a separate OS thread; pyqtSignal is the
    safe way to hop back onto the Qt main thread so UI code can react.
    """

    triggered = pyqtSignal()

    def __init__(self, hotkey_combo: str, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._hotkey_combo = hotkey_combo
        self._listener: keyboard.GlobalHotKeys | None = None

    def start(self) -> None:
        if self._listener is not None:
            return

        def _on_triggered() -> None:
            self.triggered.emit()

        try:
            self._listener = keyboard.GlobalHotKeys({self._hotkey_combo: _on_triggered})
            self._listener.start()
            logger.info("global hotkey registered: %s", self._hotkey_combo)
        except Exception:
            logger.exception("failed to register global hotkey %r", self._hotkey_combo)
            raise

    def stop(self) -> None:
        if self._listener is None:
            return
        self._listener.stop()
        self._listener = None
