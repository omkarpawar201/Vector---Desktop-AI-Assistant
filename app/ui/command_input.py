"""
Command input bar widget with text submission and mic toggle button.
"""

from typing import Optional
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QWidget,
)


class CommandInputWidget(QFrame):
    """
    Prompt input widget containing a text line edit, send button, and voice toggle button.
    """

    command_submitted = Signal(str)
    voice_toggled = Signal(bool)
    #: Emitted when the user asks to stop the in-progress generation.  The window
    #: owns the cancellation token and decides what (if anything) is cancellable.
    stop_requested = Signal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet("""
            CommandInputWidget {
                background-color: #161624;
                border-top: 1px solid #28283d;
            }
            QLineEdit {
                background-color: #11111b;
                border: 1px solid #313244;
                border-radius: 12px;
                color: #cdd6f4;
                font-family: 'Segoe UI', -apple-system, sans-serif;
                font-size: 13.5px;
                padding: 0px 16px;
                min-height: 42px;
                max-height: 42px;
            }
            QLineEdit:focus {
                border: 1px solid #89b4fa;
                background-color: #131320;
            }
            QLineEdit::placeholder {
                color: #6c7086;
            }
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #89b4fa, stop:1 #74a0e8);
                border: none;
                border-radius: 12px;
                color: #11111b;
                font-family: 'Segoe UI', -apple-system, sans-serif;
                font-weight: bold;
                font-size: 13px;
                padding: 0px 20px;
                min-height: 42px;
                max-height: 42px;
            }
            QPushButton:hover {
                background: #b4befe;
            }
            QPushButton:disabled {
                background-color: #313244;
                color: #585b70;
            }
            QPushButton#mic_btn {
                background-color: #242438;
                color: #cdd6f4;
                border: 1px solid #3b3d56;
                border-radius: 12px;
                padding: 0px 16px;
                min-height: 42px;
                max-height: 42px;
                font-weight: 600;
            }
            QPushButton#mic_btn:hover {
                background-color: #313248;
                border-color: #585b70;
            }
            QPushButton#mic_btn:checked {
                background-color: #f38ba8;
                color: #11111b;
                border: none;
            }
            QPushButton#stop_btn {
                background-color: #f38ba8;
                color: #11111b;
                border-radius: 12px;
                min-height: 42px;
                max-height: 42px;
            }
            QPushButton#stop_btn:hover {
                background-color: #eba0ac;
            }
        """)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 10, 16, 10)
        layout.setSpacing(10)

        self.input_edit = QLineEdit(self)
        self.input_edit.setPlaceholderText("Message Vector...")
        self.input_edit.returnPressed.connect(self._on_submit)

        self.btn_send = QPushButton("Send", self)
        self.btn_send.clicked.connect(self._on_submit)

        # Stop is shown only while a cancellable generation is running; it is
        # hidden by default and toggled by the owning window.
        self.btn_stop = QPushButton("■ Stop", self)
        self.btn_stop.setObjectName("stop_btn")
        self.btn_stop.clicked.connect(lambda: self.stop_requested.emit())
        self.btn_stop.hide()

        self.btn_mic = QPushButton("🎤 Voice", self)
        self.btn_mic.setObjectName("mic_btn")
        self.btn_mic.setCheckable(True)
        self.btn_mic.toggled.connect(self._on_voice_toggle)

        layout.addWidget(self.input_edit)
        layout.addWidget(self.btn_stop)
        layout.addWidget(self.btn_send)
        layout.addWidget(self.btn_mic)

    def set_stop_visible(self, visible: bool) -> None:
        """Shows or hides the Stop control for a running generation."""
        self.btn_stop.setVisible(bool(visible))

    def _on_submit(self) -> None:
        text = self.input_edit.text().strip()
        if text:
            self.input_edit.clear()
            self.command_submitted.emit(text)

    def _on_voice_toggle(self, checked: bool) -> None:
        self.voice_toggled.emit(checked)
