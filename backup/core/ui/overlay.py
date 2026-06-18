from __future__ import annotations

import logging
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot
from PyQt6.QtDBus import QDBusConnection, QDBusInterface

from core.state_machine import State

log = logging.getLogger("ui_overlay")


class UIOverlay(QObject):
    """Coordinator for the top-bar status segment.
    
    Exposes topbar state transitions over DBus and deprecates/hides cursor tracking.
    """

    prompt_submitted = pyqtSignal(str)
    overlay_dismissed = pyqtSignal()
    clicked = pyqtSignal()

    def __init__(self) -> None:
        super().__init__()

        self._bus = None
        self._interface = None
        
        # Connect to DBus session bus in the background once the event loop is active
        import asyncio
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(0, lambda: asyncio.create_task(self.connect_dbus_bg()))

        # Initial state
        self.update_state(State.IDLE)

    async def connect_dbus_bg(self) -> None:
        log.info("Connecting to GNOME status pill D-Bus in background...")
        try:
            self._bus = QDBusConnection.sessionBus()
            if self._bus.isConnected():
                self._interface = QDBusInterface("org.meeseeks.Pill", "/org/meeseeks/Pill", "org.meeseeks.Pill", self._bus)
                # Connect to Clicked signal from any sender
                connected = self._bus.connect(
                    None,  # service (None means any sender is matched, which is robust)
                    "/org/meeseeks/Pill",
                    "org.meeseeks.Pill",
                    "Clicked",
                    self.on_dbus_clicked
                )
                if connected:
                    log.info("UIOverlay connected to GNOME Pill Clicked D-Bus signal in background ✓")
                else:
                    log.warning("UIOverlay failed to connect to GNOME Pill Clicked D-Bus signal in background!")
            else:
                log.warning("D-Bus not connected in background — GNOME extension integration disabled")
        except Exception as e:
            log.exception(f"Error connecting to D-Bus in background: {e}")

    @pyqtSlot()
    def on_dbus_clicked(self) -> None:
        log.info("GNOME Pill Clicked signal received over D-Bus!")
        self.clicked.emit()

    def update_state(self, state: State) -> None:
        """Called when the brain's state machine changes state."""
        state_str = state.value.lower()
        # Map state names to the values expected by GNOME extension
        if state == State.ACTING:
            state_str = "acting"  # Map ACTING -> acting (Working)
            
        if self._interface and self._interface.isValid():
            self._interface.call("SetState", state_str)
        else:
            log.debug(f"Could not update GNOME Pill state over DBus: Interface not connected yet (State: {state_str})")

    def show_input_near_cursor(self) -> None:
        # Text input card is removed
        pass

    def show_bubble_near_cursor(self) -> None:
        # Text response bubble is removed
        pass

    def update_bubble_text(self, text: str) -> None:
        # Text response bubble is removed
        pass

    def show_cursor_at(self, global_x: int, global_y: int) -> None:
        pass

    def fly_cursor_to(self, global_x: int, global_y: int) -> None:
        pass

    def hide_input(self) -> None:
        pass

    def dismiss(self) -> None:
        self.overlay_dismissed.emit()
        
    def close(self) -> None:
        # Reset to idle on close
        if self._interface and self._interface.isValid():
            self._interface.call("SetState", "idle")
