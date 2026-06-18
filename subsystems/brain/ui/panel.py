from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QCursor
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from subsystems.brain.ui.design import Colors, Radius, Sizes
from subsystems.brain.ui.effects import DEFAULT_SHADOW_MARGIN, apply_drop_shadow, fade_in


class ControlPanel(QWidget):
    """Small popup shown when the tray status icon is clicked."""

    quit_requested = pyqtSignal()

    def __init__(self, hotkey_label: str) -> None:
        super().__init__(
            None,
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowStaysOnTopHint,
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFixedWidth(Sizes.PANEL_WIDTH + 2 * DEFAULT_SHADOW_MARGIN)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(
            DEFAULT_SHADOW_MARGIN, DEFAULT_SHADOW_MARGIN, DEFAULT_SHADOW_MARGIN, DEFAULT_SHADOW_MARGIN
        )

        self._card = QFrame(self)
        self._card.setObjectName("card")
        self._card.setFixedWidth(Sizes.PANEL_WIDTH)
        self._card.setStyleSheet(self._stylesheet())
        outer.addWidget(self._card)

        apply_drop_shadow(self._card)

        card_root = QHBoxLayout(self._card)
        card_root.setContentsMargins(0, 0, 0, 0)
        card_root.setSpacing(0)

        stripe = QFrame(self._card)
        stripe.setObjectName("stripe")
        stripe.setFixedWidth(Sizes.ACCENT_STRIPE_WIDTH)
        card_root.addWidget(stripe)

        content = QWidget(self._card)
        card_root.addWidget(content, stretch=1)

        layout = QVBoxLayout(content)
        layout.setContentsMargins(18, 16, 16, 16)
        layout.setSpacing(10)

        title_row = QHBoxLayout()
        title_row.setSpacing(8)
        dot = QLabel("●")
        dot.setObjectName("statusDot")
        title_row.addWidget(dot)
        title = QLabel("Mr Meeseeks")
        title.setObjectName("title")
        title_row.addWidget(title)
        title_row.addStretch(1)
        layout.addLayout(title_row)

        section = QLabel("WAKE OR HOTKEY")
        section.setObjectName("section")
        layout.addWidget(section)

        body = QLabel(f"Say <b>'Hey Meeseeks'</b> or press <b>{hotkey_label}</b> to summon your OS companion.")
        body.setWordWrap(True)
        body.setObjectName("body")
        layout.addWidget(body)

        hint = QLabel("Enter sends · Esc closes")
        hint.setObjectName("hint")
        layout.addWidget(hint)

        footer = QHBoxLayout()
        footer.addStretch(1)
        quit_button = QPushButton("Quit Meeseeks")
        quit_button.setObjectName("quit")
        quit_button.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        quit_button.clicked.connect(self.quit_requested.emit)
        footer.addWidget(quit_button)
        layout.addLayout(footer)

    def _stylesheet(self) -> str:
        return f"""
            QFrame#card {{
                background: qlineargradient(
                    x1:0, y1:0, x2:0, y2:1,
                    stop:0 {Colors.BACKGROUND_PANEL_TOP},
                    stop:1 {Colors.BACKGROUND_PANEL_BOTTOM}
                );
                border: 1px solid {Colors.BORDER_SUBTLE};
                border-top: 1px solid {Colors.INNER_HIGHLIGHT};
                border-radius: {Radius.PANEL}px;
            }}
            QFrame#stripe {{
                background-color: {Colors.ACCENT_BLUE};
                border-top-left-radius: {Radius.PANEL}px;
                border-bottom-left-radius: {Radius.PANEL}px;
            }}
            QLabel {{
                background: transparent;
            }}
            QLabel#statusDot {{
                color: #00d2ff;
                font-size: 12px;
            }}
            QLabel#title {{
                color: {Colors.TEXT_PRIMARY};
                font-size: 15px;
                font-weight: 700;
            }}
            QLabel#section {{
                color: {Colors.TEXT_SECONDARY};
                font-size: 10px;
                font-weight: 700;
                letter-spacing: 1.4px;
            }}
            QLabel#body {{
                color: {Colors.TEXT_PRIMARY};
                font-size: 13px;
            }}
            QLabel#hint {{
                color: {Colors.TEXT_MUTED};
                font-size: 11px;
            }}
            QPushButton#quit {{
                background-color: transparent;
                color: {Colors.TEXT_SECONDARY};
                border: 1px solid {Colors.BORDER_SUBTLE};
                border-radius: {Radius.BUTTON}px;
                padding: 6px 12px;
                font-size: 12px;
                font-weight: 500;
            }}
            QPushButton#quit:hover {{
                color: {Colors.TEXT_PRIMARY};
                border-color: rgba(255, 82, 82, 160);
                background: rgba(255, 82, 82, 32);
            }}
        """

    def show_near_cursor(self) -> None:
        cursor_pos = QCursor.pos()
        self.move(cursor_pos.x() - self.width() // 2, cursor_pos.y() + 12)
        self.show()
        self.raise_()
        self.activateWindow()
        fade_in(self)
