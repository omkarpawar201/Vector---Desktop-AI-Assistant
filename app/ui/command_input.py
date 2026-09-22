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

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet("""
            CommandInputWidget {
                background-color: #1e1e2e;
                border-top: 1px solid #313244;
                padding: 6px;
            }
            QLineEdit {
                background-color: #313244;
                border: 1px solid #45475a;
                border-radius: 6px;
                color: #cdd6f4;
                font-family: 'Segoe UI', sans-serif;
                font-size: 13px;
                padding: 8px 12px;
            }
            QLineEdit:focus {
                border: 1px solid #89b4fa;
            }
            QPushButton {
                background-color: #89b4fa;
                border: none;
                border-radius: 6px;
                color: #11111b;
                font-weight: bold;
                font-size: 13px;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: #b4befe;
            }
            QPushButton#mic_btn {
                background-color: #45475a;
                color: #cdd6f4;
            }
            QPushButton#mic_btn:checked {
                background-color: #f38ba8;
                color: #11111b;
            }
        """)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)

        self.input_edit = QLineEdit(self)
        self.input_edit.setPlaceholderText("Ask Vector to control your PC (e.g., 'set volume to 40', 'open chrome', 'cpu usage')...")
        self.input_edit.returnPressed.connect(self._on_submit)

        self.btn_send = QPushButton("Send", self)
        self.btn_send.clicked.connect(self._on_submit)

        self.btn_mic = QPushButton("🎤 Voice", self)
        self.btn_mic.setObjectName("mic_btn")
        self.btn_mic.setCheckable(True)
        self.btn_mic.toggled.connect(self._on_voice_toggle)

        layout.addWidget(self.input_edit)
        layout.addWidget(self.btn_send)
        layout.addWidget(self.btn_mic)

    def _on_submit(self) -> None:
        text = self.input_edit.text().strip()
        if text:
            self.input_edit.clear()
            self.command_submitted.emit(text)

    def _on_voice_toggle(self, checked: bool) -> None:
        self.voice_toggled.emit(checked)
