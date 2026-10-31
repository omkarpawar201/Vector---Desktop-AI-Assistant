"""
Command input bar widget for Vector Desktop AI Assistant.
Features cyberpunk / Fluent dark glowing dock styling, QLineEdit input,
paper airplane send button, circular mic toggle button, and stop generation control.
"""

from __future__ import annotations

from typing import Optional
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QWidget,
)

from app.ui.theme import (
    ACCENT_CYAN,
    ACCENT_PINK,
    ACCENT_PURPLE,
    BG_CARD,
    BG_INPUT,
    BORDER_ACTIVE,
    BORDER_GLOW_CYAN,
    BORDER_SUBTLE,
    TEXT_MUTED,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
)


class CommandInputWidget(QFrame):
    """
    Glowing Cyberpunk / Fluent Composer Dock containing text input,
    pill Send button with airplane icon ✈, voice mic toggle 🎤, and Stop button.
    """

    command_submitted = Signal(str)
    voice_toggled = Signal(bool)
    #: Emitted when the user asks to stop the in-progress generation.  The window
    #: owns the cancellation token and decides what (if anything) is cancellable.
    stop_requested = Signal()

    default_mic_text = "🎤 Voice"

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("composerDock")
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet("""
            QFrame#composerDock {
                background-color: #10131d;
                border: 1px solid rgba(56, 189, 248, 0.42);
                border-radius: 20px;
                padding: 2px 4px;
            }
            QFrame#composerDock:hover {
                border-color: rgba(56, 189, 248, 0.75);
            }
            QLineEdit {
                background-color: transparent;
                border: none;
                color: #f8fafc;
                font-family: 'Segoe UI Variable', 'Segoe UI', -apple-system, sans-serif;
                font-size: 13.5px;
                padding: 0px 14px;
                min-height: 40px;
                max-height: 40px;
            }
            QLineEdit:focus {
                color: #ffffff;
            }
            QLineEdit::placeholder {
                color: #64748b;
                font-size: 13px;
            }
            QPushButton#send_btn {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #2563eb, stop:1 #7c3aed);
                border: 1px solid rgba(255, 255, 255, 0.16);
                border-radius: 17px;
                color: #ffffff;
                font-family: 'Segoe UI Variable', 'Segoe UI', -apple-system, sans-serif;
                font-weight: 700;
                font-size: 13px;
                padding: 0px 16px;
                min-height: 34px;
                max-height: 34px;
            }
            QPushButton#send_btn:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #3b82f6, stop:1 #8b5cf6);
                border: 1px solid #38bdf8;
                color: #ffffff;
            }
            QPushButton#send_btn:pressed {
                background: #1d4ed8;
            }
            QPushButton#send_btn:disabled {
                background-color: #161925;
                color: #475569;
                border: 1px solid #1e2235;
            }
            QPushButton#mic_btn {
                background-color: #161925;
                color: #cbd5e1;
                border: 1px solid #232738;
                border-radius: 17px;
                padding: 0px 14px;
                min-height: 34px;
                max-height: 34px;
                font-family: 'Segoe UI Variable', 'Segoe UI', -apple-system, sans-serif;
                font-size: 12.5px;
                font-weight: 600;
            }
            QPushButton#mic_btn:hover {
                background-color: #1e2235;
                border-color: #38bdf8;
                color: #38bdf8;
            }
            QPushButton#mic_btn:checked {
                background-color: rgba(244, 63, 94, 0.22);
                color: #f43f5e;
                border: 1px solid #f43f5e;
            }
            QPushButton#stop_btn {
                background-color: rgba(244, 63, 94, 0.18);
                color: #f43f5e;
                border: 1px solid rgba(244, 63, 94, 0.45);
                border-radius: 17px;
                font-family: 'Segoe UI Variable', 'Segoe UI', -apple-system, sans-serif;
                font-weight: 600;
                font-size: 12.5px;
                padding: 0px 14px;
                min-height: 34px;
                max-height: 34px;
            }
            QPushButton#stop_btn:hover {
                background-color: rgba(244, 63, 94, 0.35);
                border-color: #f43f5e;
            }
        """)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(8)

        self.input_edit = QLineEdit(self)
        self.input_edit.setPlaceholderText("Message Vector or type a command...")
        self.input_edit.returnPressed.connect(self._on_submit)

        # Stop is shown only while a cancellable generation is running; it is
        # hidden by default and toggled by the owning window.
        self.btn_stop = QPushButton("■ Stop", self)
        self.btn_stop.setObjectName("stop_btn")
        self.btn_stop.setToolTip("Stop generation")
        self.btn_stop.clicked.connect(lambda: self.stop_requested.emit())
        self.btn_stop.hide()

        # Voice toggle button
        self.btn_mic = QPushButton(self.default_mic_text, self)
        self.btn_mic.setObjectName("mic_btn")
        self.btn_mic.setToolTip("Toggle microphone voice input")
        self.btn_mic.setCheckable(True)
        self.btn_mic.toggled.connect(self._on_voice_toggle)

        # Send button with paper airplane icon ✈
        self.btn_send = QPushButton("✈  Send", self)
        self.btn_send.setObjectName("send_btn")
        self.btn_send.setToolTip("Send message (Enter)")
        self.btn_send.clicked.connect(self._on_submit)

        layout.addWidget(self.input_edit, stretch=1)
        layout.addWidget(self.btn_stop)
        layout.addWidget(self.btn_mic)
        layout.addWidget(self.btn_send)

    def set_stop_visible(self, visible: bool) -> None:
        """Shows or hides the Stop control for a running generation."""
        self.btn_stop.setVisible(bool(visible))

    def set_text(self, text: str) -> None:
        """Sets the input text and focuses the field."""
        self.input_edit.setText(text)
        self.input_edit.setFocus()

    def clear(self) -> None:
        """Clears the input field."""
        self.input_edit.clear()

    def text(self) -> str:
        """Returns the current input text."""
        return self.input_edit.text()

    def _on_submit(self) -> None:
        text = self.input_edit.text().strip()
        if text:
            self.input_edit.clear()
            self.command_submitted.emit(text)

    def _on_voice_toggle(self, checked: bool) -> None:
        self.voice_toggled.emit(checked)
