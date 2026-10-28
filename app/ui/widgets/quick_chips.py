"""
Quick Prompt Chips Widget for Vector Desktop AI Assistant.
Horizontal row of 1-click suggestion chips displayed beneath the command input.
"""

from __future__ import annotations
from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QHBoxLayout, QPushButton, QWidget

from app.ui.theme import CHIP_BUTTON_STYLE


class QuickChipsWidget(QWidget):
    """
    Horizontal bar of quick command suggestion chips.
    """

    chip_clicked = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setStyleSheet(CHIP_BUTTON_STYLE)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 4)
        layout.setSpacing(6)

        self._add_chip(layout, "🔊  Mute Audio", "mute my audio")
        self._add_chip(layout, "🔊  Set Volume", "set volume to 50")
        self._add_chip(layout, "🔍  Open Chrome", "open chrome")
        self._add_chip(layout, "📁  Search Files", "search files")
        self._add_chip(layout, "📊  CPU Usage", "get cpu usage")

        # More dropdown / action chip
        btn_more = QPushButton("More  ⌄", self)
        btn_more.clicked.connect(lambda: self.chip_clicked.emit("get system stats"))
        layout.addWidget(btn_more)

        layout.addStretch()

    def _add_chip(self, layout: QHBoxLayout, label: str, command: str) -> None:
        btn = QPushButton(label, self)
        btn.clicked.connect(lambda: self.chip_clicked.emit(command))
        layout.addWidget(btn)
