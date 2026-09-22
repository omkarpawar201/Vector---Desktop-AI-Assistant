"""
Chat view widget displaying conversation history and visual tool result cards.
"""

from typing import Any, Dict, Optional
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
    QWidget,
)


class ActionCardWidget(QFrame):
    """
    Visual action card for displaying executed tool status, data, and messages.
    """

    def __init__(self, tool_name: str, message: str, success: bool = True, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setFrameShape(QFrame.StyledPanel)
        bg_color = "#11111b" if success else "#31131c"
        border_color = "#a6e3a1" if success else "#f38ba8"
        status_text = "✓ SUCCESS" if success else "✗ FAILED"
        status_color = "#a6e3a1" if success else "#f38ba8"

        self.setStyleSheet(f"""
            ActionCardWidget {{
                background-color: {bg_color};
                border: 1px solid {border_color};
                border-radius: 6px;
                padding: 6px 12px;
                margin: 4px;
            }}
            QLabel#tool_title {{
                color: #cdd6f4;
                font-weight: bold;
                font-size: 13px;
            }}
            QLabel#status {{
                color: {status_color};
                font-weight: bold;
                font-size: 11px;
            }}
            QLabel#message {{
                color: #bac2de;
                font-size: 12px;
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)

        header_layout = QHBoxLayout()
        lbl_title = QLabel(f"Tool Action: {tool_name}", self)
        lbl_title.setObjectName("tool_title")

        lbl_status = QLabel(status_text, self)
        lbl_status.setObjectName("status")

        header_layout.addWidget(lbl_title)
        header_layout.addStretch()
        header_layout.addWidget(lbl_status)

        lbl_msg = QLabel(message, self)
        lbl_msg.setObjectName("message")
        lbl_msg.setWordWrap(True)

        layout.addLayout(header_layout)
        layout.addWidget(lbl_msg)


class ChatViewWidget(QWidget):
    """
    Main conversation view widget.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.list_widget = QListWidget(self)
        self.list_widget.setStyleSheet("""
            QListWidget {
                background-color: #181825;
                border: none;
                color: #cdd6f4;
                font-family: 'Segoe UI', sans-serif;
                font-size: 13px;
                padding: 8px;
            }
            QListWidget::item {
                margin-bottom: 8px;
            }
        """)
        layout.addWidget(self.list_widget)

    def add_user_message(self, text: str) -> None:
        """Adds a user message item to the chat view."""
        item = QListWidgetItem(self.list_widget)
        widget = QLabel(f"<b>You:</b> {text}", self.list_widget)
        widget.setStyleSheet("""
            background-color: #313244;
            color: #89b4fa;
            border-radius: 8px;
            padding: 8px 12px;
            font-size: 13px;
        """)
        widget.setWordWrap(True)
        item.setSizeHint(widget.sizeHint())
        self.list_widget.addItem(item)
        self.list_widget.setItemWidget(item, widget)
        self.list_widget.scrollToBottom()

    def add_assistant_message(self, text: str) -> None:
        """Adds an assistant response message item to the chat view."""
        item = QListWidgetItem(self.list_widget)
        widget = QLabel(f"<b>Vector:</b> {text}", self.list_widget)
        widget.setStyleSheet("""
            background-color: #1e1e2e;
            color: #cdd6f4;
            border-radius: 8px;
            padding: 8px 12px;
            font-size: 13px;
        """)
        widget.setWordWrap(True)
        item.setSizeHint(widget.sizeHint())
        self.list_widget.addItem(item)
        self.list_widget.setItemWidget(item, widget)
        self.list_widget.scrollToBottom()

    def add_action_card(self, tool_name: str, message: str, success: bool = True) -> None:
        """Adds a visual action card item for executed tools."""
        item = QListWidgetItem(self.list_widget)
        card = ActionCardWidget(tool_name=tool_name, message=message, success=success, parent=self.list_widget)
        item.setSizeHint(card.sizeHint())
        self.list_widget.addItem(item)
        self.list_widget.setItemWidget(item, card)
        self.list_widget.scrollToBottom()
