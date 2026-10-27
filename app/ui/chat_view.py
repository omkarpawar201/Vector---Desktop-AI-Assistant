"""
Chat view widget displaying conversation history and visual tool result cards.

The view is the only place that touches Qt widgets for conversation rendering.
Streaming text arrives on the Qt main thread through a signal and is applied
here, so a background worker never mutates a widget directly.
"""

import html
from typing import Any, Dict, Optional
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QSizePolicy,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

#: Styling shared by the final assistant bubble and the in-progress streamed one,
#: so text does not visually shift when a stream is finalized.
_ASSISTANT_BUBBLE_STYLE = """
    background-color: #1e1e2e;
    color: #cdd6f4;
    border-radius: 8px;
    padding: 8px 12px;
    font-size: 13px;
"""


class StreamHandle:
    """
    A handle to an in-progress streamed assistant bubble.

    Returned by :meth:`ChatViewWidget.begin_assistant_stream` and passed back to
    the append/finalize calls so a stale stream (from a request that is no longer
    active) can be ignored by simply not referencing its handle.
    """

    __slots__ = ("item", "label", "text")

    def __init__(self, item: QListWidgetItem, label: QLabel):
        self.item = item
        self.label = label
        self.text = ""



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
        lbl_title = QLabel(f"Tool Action: {html.escape(str(tool_name))}", self)
        lbl_title.setObjectName("tool_title")

        lbl_status = QLabel(status_text, self)
        lbl_status.setObjectName("status")

        header_layout.addWidget(lbl_title)
        header_layout.addStretch()
        header_layout.addWidget(lbl_status)

        lbl_msg = QLabel(html.escape(str(message)), self)
        lbl_msg.setObjectName("message")
        lbl_msg.setWordWrap(True)

        layout.addLayout(header_layout)
        layout.addWidget(lbl_msg)


class ChatViewWidget(QWidget):
    """
    Main conversation view widget.
    """

    #: Emitted when the user asks to load older messages of the shown
    #: conversation.  The window owns pagination and answers the request.
    load_earlier_requested = Signal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.btn_load_earlier = QPushButton("↑ Load earlier messages", self)
        self.btn_load_earlier.setStyleSheet("""
            QPushButton {
                background-color: #313244;
                color: #cdd6f4;
                border: none;
                border-radius: 6px;
                padding: 6px 12px;
                margin: 6px;
            }
            QPushButton:hover { background-color: #45475a; }
        """)
        self.btn_load_earlier.clicked.connect(self.load_earlier_requested.emit)
        self.btn_load_earlier.hide()
        layout.addWidget(self.btn_load_earlier)

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


    def _open_link(self, url):
        try:
            from PySide6.QtGui import QDesktopServices
            from PySide6.QtCore import QUrl
            QDesktopServices.openUrl(QUrl(url))
        except Exception:
            pass

    def _make_message_container(self, label: QLabel, alignment: Qt.AlignmentFlag = Qt.AlignLeft) -> QWidget:
        container = QWidget(self.list_widget)
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        if alignment == Qt.AlignRight:
            layout.addStretch()
        label.setWordWrap(True)
        label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        layout.addWidget(label, 0)
        if alignment == Qt.AlignLeft:
            layout.addStretch()
        return container

    def _update_item_size(self, item: QListWidgetItem, container: QWidget) -> None:
        try:
            container.adjustSize()
            hint = container.sizeHint()
            if hint.width() <= 0 and hasattr(self.list_widget, 'viewport'):
                hint = QSize(self.list_widget.viewport().width() - 40, hint.height())
            item.setSizeHint(hint)
            self.list_widget.update()
        except Exception:
            pass


    def clear(self) -> None:
        """Removes every item from the conversation view."""
        self.list_widget.clear()

    def set_load_more_visible(self, visible: bool) -> None:
        """Shows or hides the "load earlier messages" control."""
        self.btn_load_earlier.setVisible(bool(visible))

    def add_user_message(self, text: str) -> None:
        """Adds a user message item to the chat view."""
        item = QListWidgetItem(self.list_widget)
        widget = QLabel(f"<b>You:</b> {html.escape(str(text))}", self.list_widget)
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
        widget = QLabel(f"<b>Vector:</b> {html.escape(str(text))}", self.list_widget)
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

    def begin_assistant_stream(self) -> StreamHandle:
        """
        Open an empty assistant bubble that will be filled incrementally.

        Returns a handle the caller keeps for the duration of the stream; the
        caller must ignore its own stale handles so a superseded stream cannot
        write into the wrong place.
        """
        item = QListWidgetItem(self.list_widget)
        label = QLabel("<b>Vector:</b> ", self.list_widget)
        label.setStyleSheet(_ASSISTANT_BUBBLE_STYLE)
        label.setWordWrap(True)
        item.setSizeHint(label.sizeHint())
        self.list_widget.addItem(item)
        self.list_widget.setItemWidget(item, label)
        self.list_widget.scrollToBottom()
        return StreamHandle(item, label)

    def append_assistant_stream(self, handle: Optional[StreamHandle], text: str) -> None:
        """Append one text delta to an open streamed bubble (no-op if handle gone)."""
        if handle is None or not text:
            return
        handle.text += text
        handle.label.setText(f"<b>Vector:</b> {html.escape(handle.text)}")
        handle.item.setSizeHint(handle.label.sizeHint())
        self.list_widget.scrollToBottom()

    def finalize_assistant_stream(
        self,
        handle: Optional[StreamHandle],
        text: Optional[str] = None,
        suffix: str = "",
    ) -> None:
        """
        Replace a streamed bubble's body with the authoritative final text.

        ``text`` (the text that will also be persisted) overwrites the incrementally
        assembled body so display and storage are guaranteed to agree; ``suffix``
        is a short plain-text marker such as ``" (stopped)"``.
        """
        if handle is None:
            return
        if text is not None:
            handle.text = text
        handle.label.setText(f"<b>Vector:</b> {html.escape(handle.text)}{suffix}")
        handle.item.setSizeHint(handle.label.sizeHint())
        self.list_widget.scrollToBottom()

    def add_action_card(self, tool_name: str, message: str, success: bool = True) -> None:
        """Adds a visual action card item for executed tools."""
        item = QListWidgetItem(self.list_widget)
        card = ActionCardWidget(tool_name=tool_name, message=message, success=success, parent=self.list_widget)
        item.setSizeHint(card.sizeHint())
        self.list_widget.addItem(item)
        self.list_widget.setItemWidget(item, card)
        self.list_widget.scrollToBottom()

    def add_assistant_error(self, summary: str, details: str = "") -> None:
        try:
            item = QListWidgetItem(self.list_widget)
            container = QWidget(self.list_widget)
            layout = QVBoxLayout(container)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(6)
            summary_label = QLabel()
            summary_label.setTextFormat(Qt.RichText)
            summary_label.setText(f"<b>⚠ Error</b><br/>{html.escape(str(summary))}")
            summary_label.setStyleSheet("""
                background-color: #2a1f1f;
                color: #f38ba8;
                border: 1px solid #f38ba8;
                border-radius: 10px;
                padding: 10px 14px;
                font-size: 13px;
                line-height: 1.6;
            """)
            summary_label.setWordWrap(True)
            summary_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            summary_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
            h = QHBoxLayout()
            h.setContentsMargins(0, 0, 0, 0)
            h.addStretch()
            h.addWidget(summary_label)
            h.addStretch()
            layout.addLayout(h)
            if details:
                det = QLabel(html.escape(str(details)))
                det.setStyleSheet("color: #bac2de; font-size: 12px; line-height: 1.5;")
                det.setWordWrap(True)
                det.setTextInteractionFlags(Qt.TextSelectableByMouse)
                det_h = QHBoxLayout()
                det_h.setContentsMargins(0, 0, 0, 0)
                det_h.addStretch()
                det_h.addWidget(det)
                det_h.addStretch()
                layout.addLayout(det_h)
            container.adjustSize()
            item.setSizeHint(container.sizeHint())
            self.list_widget.addItem(item)
            self.list_widget.setItemWidget(item, container)
            self.list_widget.scrollToBottom()
        except Exception:
            # Fallback
            self.add_assistant_message(f"Error: {summary}")

    def add_welcome_message(self, text: str) -> None:
        try:
            item = QListWidgetItem(self.list_widget)
            label = QLabel()
            label.setTextFormat(Qt.RichText)
            label.setOpenExternalLinks(True)
            try:
                label.linkActivated.connect(self._open_link)
            except Exception:
                pass
            label.setText(f"<b>Vector</b><br/>{html.escape(str(text)).replace(chr(10), '<br/>')}")
            label.setStyleSheet("""
                background-color: #181825;
                color: #a6adc8;
                border: 1px solid #313244;
                border-radius: 10px;
                padding: 12px 16px;
                font-size: 13px;
                line-height: 1.6;
            """)
            label.setWordWrap(True)
            label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
            container = self._make_message_container(label, Qt.AlignLeft)
            item.setSizeHint(container.sizeHint())
            self.list_widget.addItem(item)
            self.list_widget.setItemWidget(item, container)
            self.list_widget.scrollToBottom()
        except Exception:
            self.add_assistant_message(str(text))

    def add_message(self, role: str, text: str) -> None:
        if role == "user":
            self.add_user_message(text)
        elif role == "assistant":
            self.add_assistant_message(text)
        else:
            self.add_assistant_message(text)

