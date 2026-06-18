from __future__ import annotations

from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtGui import QAction, QIcon, QPainter, QPixmap, QColor, QPen
from PyQt6.QtWidgets import QMenu, QSystemTrayIcon


def _build_default_icon(size: int = 22) -> QIcon:
    """Generate a small Meeseeks-blue circular status blob icon."""
    pixmap = QPixmap(size, size)
    pixmap.fill(QColor(0, 0, 0, 0))
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setBrush(QColor("#00d2ff"))  # Vibrant Meeseeks cyan-blue
    painter.setPen(QPen(QColor("#0b465c"), 1.2))  # Dark contrast border
    painter.drawEllipse(int(size * 0.2), int(size * 0.2), int(size * 0.6), int(size * 0.6))
    painter.end()
    return QIcon(pixmap)


class TrayController(QObject):
    """Wraps QSystemTrayIcon. Emits high-level signals the app can wire to actions."""

    open_panel_requested = pyqtSignal()
    quit_requested = pyqtSignal()

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._tray = QSystemTrayIcon(_build_default_icon(), parent=self)
        self._tray.setToolTip("Mr Meeseeks")
        self._tray.activated.connect(self._on_activated)

        menu = QMenu()
        open_action = QAction("Open Mr Meeseeks", self)
        open_action.triggered.connect(self.open_panel_requested.emit)
        menu.addAction(open_action)

        menu.addSeparator()

        quit_action = QAction("Quit", self)
        quit_action.triggered.connect(self.quit_requested.emit)
        menu.addAction(quit_action)

        self._tray.setContextMenu(menu)

    def show(self) -> None:
        self._tray.show()

    def hide(self) -> None:
        self._tray.hide()

    def _on_activated(self, reason: QSystemTrayIcon.ActivationReason) -> None:
        if reason == QSystemTrayIcon.ActivationReason.Trigger:
            self.open_panel_requested.emit()
