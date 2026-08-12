"""
email_card.py — Native Glassmorphic GNOME Email Viewer Overlay
Displays full email details (From, Subject, Date, Body) in a sleek dark-mode
PyQt6 card matching the Meeseeks / Athena UI design language.
"""

import sys
import logging
from typing import Optional
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QTextEdit, QPushButton, QGraphicsDropShadowEffect, QFrame
)
from PyQt6.QtGui import QColor, QFont, QIcon

log = logging.getLogger("email_card")

class EmailCardOverlay(QWidget):
    """
    Sleek dark-mode glassmorphic overlay for displaying email details.
    """
    def __init__(self, sender: str, subject: str, date: str, body: str):
        super().__init__()
        self.sender_str = sender
        self.subject_str = subject
        self.date_str = date
        self.body_str = body
        self._init_ui()

    def _init_ui(self):
        # Window Flags: Frameless, Always on Top, Tool window (no taskbar clutter)
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFixedSize(520, 480)

        # Position on top-right of primary screen
        screen = QApplication.primaryScreen()
        if screen:
            geom = screen.availableGeometry()
            x = geom.width() - 550
            y = 70
            self.move(x, y)

        # Main Container Frame with Glassmorphism CSS
        container = QFrame(self)
        container.setObjectName("container")
        container.setFixedSize(500, 460)
        container.setStyleSheet("""
            QFrame#container {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                            stop:0 rgba(22, 33, 44, 245),
                                            stop:1 rgba(10, 16, 22, 245));
                border: 1px solid rgba(0, 210, 255, 60);
                border-radius: 16px;
            }
        """)

        # Drop shadow effect
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(30)
        shadow.setColor(QColor(0, 210, 255, 40))
        shadow.setOffset(0, 6)
        container.setGraphicsEffect(shadow)

        layout = QVBoxLayout(container)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(10)

        # Header Row (Title + Close Button)
        header_layout = QHBoxLayout()
        title_label = QLabel("⚡ ATHENA — EMAIL PREVIEW", container)
        title_label.setStyleSheet("color: #00d2ff; font-weight: bold; font-size: 13px; letter-spacing: 1px;")

        close_btn = QPushButton("✕", container)
        close_btn.setFixedSize(24, 24)
        close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        close_btn.setStyleSheet("""
            QPushButton {
                background: rgba(255, 255, 255, 15);
                border: none;
                border-radius: 12px;
                color: #8da2b5;
                font-weight: bold;
                font-size: 13px;
            }
            QPushButton:hover {
                background: rgba(255, 60, 60, 180);
                color: #ffffff;
            }
        """)
        close_btn.clicked.connect(self.close)

        header_layout.addWidget(title_label)
        header_layout.addStretch()
        header_layout.addWidget(close_btn)
        layout.addLayout(header_layout)

        # Separator line
        sep = QFrame(container)
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("background-color: rgba(0, 210, 255, 30); max-height: 1px;")
        layout.addWidget(sep)

        # Email Headers (From, Subject, Date)
        headers_frame = QFrame(container)
        headers_frame.setStyleSheet("background: rgba(25, 38, 50, 180); border-radius: 10px; padding: 8px;")
        h_layout = QVBoxLayout(headers_frame)
        h_layout.setContentsMargins(10, 8, 10, 8)
        h_layout.setSpacing(4)

        from_label = QLabel(f"<b>From:</b> {self.sender_str}", headers_frame)
        from_label.setStyleSheet("color: #f0f8ff; font-size: 13px;")
        
        subj_label = QLabel(f"<b>Subject:</b> {self.subject_str}", headers_frame)
        subj_label.setStyleSheet("color: #66e5ff; font-size: 13px; font-weight: 600;")
        subj_label.setWordWrap(True)

        date_label = QLabel(f"<b>Date:</b> {self.date_str}", headers_frame)
        date_label.setStyleSheet("color: #8da2b5; font-size: 11px;")

        h_layout.addWidget(from_label)
        h_layout.addWidget(subj_label)
        h_layout.addWidget(date_label)
        layout.addWidget(headers_frame)

        # Email Body Scroll Area
        body_text = QTextEdit(container)
        body_text.setReadOnly(True)
        body_text.setPlainText(self.body_str)
        body_text.setStyleSheet("""
            QTextEdit {
                background: rgba(15, 23, 30, 160);
                border: 1px solid rgba(0, 210, 255, 25);
                border-radius: 10px;
                color: #e0f2fe;
                font-size: 13px;
                line-height: 1.5;
                padding: 10px;
            }
            QScrollBar:vertical {
                background: rgba(0, 0, 0, 30);
                width: 8px;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical {
                background: rgba(0, 210, 255, 100);
                border-radius: 4px;
            }
        """)
        layout.addWidget(body_text)

        # Top-level window layout alignment
        win_layout = QVBoxLayout(self)
        win_layout.setContentsMargins(10, 10, 10, 10)
        win_layout.addWidget(container)


def show_email_card(sender: str, subject: str, date: str, body: str):
    """
    Launches the EmailCardOverlay in a separate non-blocking Qt application process or thread.
    """
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    
    # Store reference on app to prevent garbage collection
    if not hasattr(app, "_email_card_instances"):
        app._email_card_instances = []
        
    card = EmailCardOverlay(sender, subject, date, body)
    app._email_card_instances.append(card)
    card.show()
    return card


if __name__ == "__main__":
    app = QApplication(sys.argv)
    card = EmailCardOverlay(
        sender="Siemens Careers <no-reply@siemens.com>",
        subject="Senior Software Engineer (C++ / Linux) Opportunity",
        date="2026-08-10",
        body="Dear Candidate,\n\nWe were impressed by your profile and would like to invite you to discuss the Senior Software Engineer position at Siemens.\n\nKey requirements:\n- C++ 17/20 expertise\n- Linux system programming\n- High-performance multithreading\n\nBest regards,\nSiemens Talent Acquisition"
    )
    card.show()
    sys.exit(app.exec())
