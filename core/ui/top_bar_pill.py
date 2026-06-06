from __future__ import annotations

import math
from PyQt6.QtCore import QSize, QRectF, QTimer, Qt, pyqtSignal, pyqtProperty, QPropertyAnimation, QEasingCurve
from PyQt6.QtGui import QColor, QPainter, QPen, QLinearGradient, QPainterPath
from PyQt6.QtWidgets import QWidget, QHBoxLayout, QLabel

from core.state_machine import State


class AudioWaveBars(QWidget):
    """Sleek audio visualizer wave bars on the right side of the pill."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedSize(32, 16)
        self._state = State.IDLE
        self._phase = 0.0
        self._timer = QTimer(self)
        self._timer.setInterval(16)  # ~60fps
        self._timer.timeout.connect(self._on_tick)
        self._timer.start()

    def set_state(self, state: State) -> None:
        self._state = state
        self.update()

    def _on_tick(self) -> None:
        self._phase += 0.12
        if self._phase > 2.0 * math.pi * 100:
            self._phase = 0.0
        if self._state in (State.LISTENING, State.SPEAKING):
            self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)

        # Set color based on state
        if self._state == State.LISTENING:
            color = QColor("#00d2ff")  # Cyan for user speaking
        elif self._state == State.SPEAKING:
            color = QColor("#10b981")  # Emerald for agent speaking
        else:
            color = QColor("#64748b")  # Dim slate grey for other states
            color.setAlpha(80)

        painter.setBrush(color)

        w = self.width()
        h = self.height()
        cy = h / 2.0

        bar_w = 2.0
        bar_gap = 3.0
        num_bars = 4

        # Center the bars horizontally
        total_w = num_bars * bar_w + (num_bars - 1) * bar_gap
        start_x = (w - total_w) / 2.0

        for i in range(num_bars):
            if self._state in (State.LISTENING, State.SPEAKING):
                # Animate bar heights using phase-shifted sine waves
                offset = self._phase * 1.5 + i * 1.0
                bar_h = 3.0 + 9.0 * abs(math.sin(offset))
            elif self._state in (State.THINKING, State.ACTING):
                # Static minimal height
                bar_h = 3.0
            else:
                # Idle - hide visualizer
                continue

            x_pos = start_x + i * (bar_w + bar_gap)
            y_pos = cy - bar_h / 2.0
            painter.drawRoundedRect(QRectF(x_pos, y_pos, bar_w, bar_h), 1.0, 1.0)


class TopBarPill(QWidget):
    """The status indicator segment designed for the GNOME top bar."""

    clicked = pyqtSignal()
    right_clicked = pyqtSignal()

    def __init__(self) -> None:
        super().__init__(
            None,
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.BypassWindowManagerHint,
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)

        # QHBoxLayout for horizontal flow
        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(14, 0, 14, 0)
        self._layout.setSpacing(8)

        # Label (Left)
        self._label = QLabel("meeseeks", self)
        self._label.setStyleSheet("""
            color: rgba(248, 250, 252, 100);
            font-size: 11px;
            font-family: 'Inter', 'Outfit', 'Roboto', 'Segoe UI', sans-serif;
            font-weight: 600;
            background: transparent;
        """)
        self._layout.addWidget(self._label)

        # Wavebars (Right)
        self._wave = AudioWaveBars(self)
        self._layout.addWidget(self._wave)

        self._state = State.IDLE
        self._bg_color = QColor(100, 116, 139, 12)  # Very translucent idle grey
        self._color_anim = None

        self.setFixedSize(self.sizeHint())

        # Determine dynamic x position at index 1 on left side via AT-SPI2
        self.reposition_to_index_1()

    def reposition_to_index_1(self) -> None:
        """Find the right edge of GNOME's Activities / index 0 button via AT-SPI2."""
        fallback_x = 83
        target_x = fallback_x

        try:
            import pyatspi
            desktop = pyatspi.Registry.getDesktop(0)
            for app in desktop:
                if app and (app.name or "").lower() == "gnome-shell":
                    def find_index_0_right(node):
                        try:
                            role = node.getRoleName()
                            name = node.name or ""
                            ext = node.get_extents(pyatspi.DESKTOP_COORDS)
                            if ext.y == 0 and ext.height > 0 and ext.x == 0:
                                if role in ("toggle button", "panel") or "activities" in name.lower():
                                    return ext.x + ext.width
                            for i in range(node.getChildCount()):
                                val = find_index_0_right(node.getChildAtIndex(i))
                                if val is not None:
                                    return val
                        except:
                            pass
                        return None

                    res = find_index_0_right(app)
                    if res is not None:
                        target_x = res
                        break
        except Exception:
            pass

        self._x = target_x
        self._y = 0
        self.move(self._x, self._y)

    def sizeHint(self) -> QSize:
        hint = super().sizeHint()
        # Full height of the GNOME top bar (40px)
        return QSize(max(hint.width() + 12, 100), 40)

    @pyqtProperty(QColor)
    def bgColor(self) -> QColor:
        return self._bg_color

    @bgColor.setter
    def bgColor(self, color: QColor):
        self._bg_color = color
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)

        # Translucent solid color overlay covering the entire height of the top bar
        painter.setBrush(self._bg_color)
        painter.drawRect(self.rect())

    def _get_color_for_state(self, state: State) -> QColor:
        if state == State.IDLE:
            return QColor(100, 116, 139, 12)  # Dim slate grey
        elif state == State.LISTENING:
            return QColor(0, 210, 255, 35)   # Translucent Cyan
        elif state == State.THINKING:
            return QColor(139, 92, 246, 35)  # Translucent Purple
        elif state == State.ACTING:
            return QColor(245, 158, 11, 35)   # Translucent Orange
        elif state == State.SPEAKING:
            return QColor(16, 185, 129, 35)   # Translucent Emerald Green
        return QColor(100, 116, 139, 12)

    def update_state(self, state: State) -> None:
        self._state = state
        self._wave.set_state(state)

        # Set status text in lowercase
        if state == State.IDLE:
            self._label.setText("meeseeks")
            self._label.setStyleSheet("color: rgba(248, 250, 252, 100); font-size: 11px; font-weight: 600; background: transparent;")
        elif state == State.LISTENING:
            self._label.setText("listening")
            self._label.setStyleSheet("color: #f8fafc; font-size: 11px; font-weight: 600; background: transparent;")
        elif state == State.THINKING:
            self._label.setText("thinking...")
            self._label.setStyleSheet("color: #f8fafc; font-size: 11px; font-weight: 600; background: transparent;")
        elif state == State.ACTING:
            self._label.setText("working...")
            self._label.setStyleSheet("color: #f8fafc; font-size: 11px; font-weight: 600; background: transparent;")
        elif state == State.SPEAKING:
            self._label.setText("speaking")
            self._label.setStyleSheet("color: #f8fafc; font-size: 11px; font-weight: 600; background: transparent;")

        # Animate background color transition
        target_color = self._get_color_for_state(state)
        
        if self._color_anim:
            self._color_anim.stop()

        self._color_anim = QPropertyAnimation(self, b"bgColor")
        self._color_anim.setDuration(300)
        self._color_anim.setStartValue(self._bg_color)
        self._color_anim.setEndValue(target_color)
        self._color_anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._color_anim.start()

        # Dynamically resize the pill window to fit contents
        self.adjustSize()
        hint = self.layout().sizeHint()
        w = max(hint.width() + 16, 100)
        self.setFixedSize(w, 40)
        self.update()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
            event.accept()
        elif event.button() == Qt.MouseButton.RightButton:
            self.right_clicked.emit()
            event.accept()
        else:
            super().mousePressEvent(event)
