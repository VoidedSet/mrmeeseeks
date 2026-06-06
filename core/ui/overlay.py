from __future__ import annotations

from PyQt6.QtCore import QObject, Qt, pyqtSignal
from PyQt6.QtGui import QCursor

from core.ui.cursor import BlueCursor
from core.ui.top_bar_pill import TopBarPill
from core.state_machine import State


class UIOverlay(QObject):
    """Coordinator for the top-bar status segment.
    
    Exposes topbar state transitions and deprecates/hides cursor tracking.
    """

    prompt_submitted = pyqtSignal(str)
    overlay_dismissed = pyqtSignal()

    def __init__(self) -> None:
        super().__init__()

        # Deprecate but keep cursor code intact (hidden and disabled)
        self._cursor = BlueCursor()
        self._cursor.hide()

        self._pill = TopBarPill()
        self._pill.show()

        # Initial state
        self._pill.update_state(State.IDLE)

    @property
    def pill(self) -> TopBarPill:
        return self._pill

    def update_state(self, state: State) -> None:
        """Called when the brain's state machine changes state."""
        self._pill.update_state(state)

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
        # Deprecated: show cursor only if explicitly called
        self._cursor.show_at(global_x, global_y)

    def fly_cursor_to(self, global_x: int, global_y: int) -> None:
        # Deprecated: fly cursor only if explicitly called
        self._cursor.show()
        self._cursor.fly_to(global_x, global_y)

    def hide_input(self) -> None:
        pass

    def dismiss(self) -> None:
        self.overlay_dismissed.emit()
        
    def close(self) -> None:
        self._cursor.close()
        self._pill.close()

    def _handle_input_submit(self, prompt: str) -> None:
        self.prompt_submitted.emit(prompt)
