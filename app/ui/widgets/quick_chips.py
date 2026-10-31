"""
Quick Prompt Chips Widget for Vector Desktop AI Assistant.
Horizontal row of 1-click suggestion chips displayed beneath the command input.
Provides quick actions, prefill support, and an expandable 'More' action menu.
"""

from __future__ import annotations

from typing import Any, Optional
from PySide6.QtCore import QPoint, Qt, Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QHBoxLayout,
    QMenu,
    QPushButton,
    QWidget,
)

from app.ui.theme import CHIP_BUTTON_STYLE


class QuickChipsWidget(QWidget):
    """
    Horizontal bar of quick command suggestion chips.
    Supports 1-click execution, prefilling the composer dock, and an expandable menu.
    """

    chip_clicked = Signal(str)
    chip_prefilled = Signal(str)

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        target_input: Optional[Any] = None,
    ):
        super().__init__(parent)
        self.target_input = target_input
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
        self.btn_more = QPushButton("More  ⌄", self)
        self.btn_more.setObjectName("chip_more")
        self._init_more_menu()
        layout.addWidget(self.btn_more)

        layout.addStretch()

    def set_target_input(self, input_widget: Any) -> None:
        """Attach a QLineEdit or CommandInputWidget to enable prefill capability."""
        self.target_input = input_widget

    def prefill_command(self, text: str) -> None:
        """Prefills command text into the attached input widget."""
        if self.target_input is not None:
            if hasattr(self.target_input, "set_text"):
                self.target_input.set_text(text)
            elif hasattr(self.target_input, "setText"):
                self.target_input.setText(text)
                self.target_input.setFocus()
        self.chip_prefilled.emit(text)

    def _add_chip(self, layout: QHBoxLayout, label: str, command: str) -> None:
        btn = QPushButton(label, self)
        btn.clicked.connect(lambda: self.chip_clicked.emit(command))
        layout.addWidget(btn)

    def _init_more_menu(self) -> None:
        """Constructs the popup menu for additional quick desktop actions."""
        self.more_menu = QMenu(self)
        self.more_menu.setStyleSheet("""
            QMenu {
                background-color: #10131d;
                border: 1px solid #1e2235;
                border-radius: 12px;
                padding: 6px;
                color: #f8fafc;
                font-family: 'Segoe UI Variable', 'Segoe UI', -apple-system, sans-serif;
                font-size: 12px;
            }
            QMenu::item {
                background-color: transparent;
                padding: 6px 14px;
                border-radius: 6px;
                color: #cbd5e1;
            }
            QMenu::item:selected {
                background-color: #1e2235;
                color: #38bdf8;
            }
        """)

        menu_items = [
            ("📊  System Stats", "get system stats"),
            ("📷  Take Screenshot", "take screenshot"),
            ("🔋  Battery Status", "get battery status"),
            ("🎛  Running Apps", "get running apps"),
            ("🔒  Lock PC", "lock pc"),
            ("🎵  Media Play / Pause", "media play"),
        ]

        for label, cmd in menu_items:
            action = QAction(label, self.more_menu)
            action.triggered.connect(lambda checked=False, c=cmd: self.chip_clicked.emit(c))
            self.more_menu.addAction(action)

        self.btn_more.clicked.connect(self._on_more_clicked)

    def _on_more_clicked(self) -> None:
        """Shows menu when interactive, or emits default command when headless."""
        try:
            # Emit default command so direct click emits immediately
            self.chip_clicked.emit("get system stats")
            if self.isVisible() and self.btn_more.isVisible():
                pos = self.btn_more.mapToGlobal(QPoint(0, self.btn_more.height() + 2))
                self.more_menu.popup(pos)
        except Exception:
            pass
