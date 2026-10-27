"""
Chat view widget displaying conversation history, rich markdown formatting,
provider error cards, and visual tool result cards.

The view is the only place that touches Qt widgets for conversation rendering.
Streaming text arrives on the Qt main thread through a signal and is applied
here, so a background worker never mutates a widget directly.
"""

from __future__ import annotations

import html
import re
from typing import Any, Dict, List, Optional
from PySide6.QtCore import QSize, Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


def render_safe_markdown(text: str) -> str:
    """
    Renders a safe subset of Markdown into styled HTML.
    Raw HTML is strictly escaped first to prevent script execution or arbitrary UI injection.
    Only http and https links are permitted as active hyperlinks.
    """
    if not text:
        return ""

    # 1. Escape HTML so raw markup cannot execute or inject arbitrary UI
    escaped = html.escape(str(text))

    # 2. Extract and preserve fenced code blocks
    code_blocks: List[str] = []

    def _save_code_block(match: re.Match) -> str:
        lang = match.group(1).strip() if match.group(1) else ""
        code = match.group(2)
        idx = len(code_blocks)
        badge = (
            f"<div style='color: #6c7086; font-size: 11px; font-weight: bold; "
            f"margin-bottom: 4px; text-transform: uppercase;'>{lang}</div>"
            if lang
            else ""
        )
        html_block = (
            f"<div style=\"background-color: #11111b; border: 1px solid #313244; "
            f"border-radius: 8px; padding: 10px 14px; margin: 6px 0; "
            f"font-family: 'Consolas', 'Cascadia Code', monospace; font-size: 12px; "
            f"color: #cdd6f4; white-space: pre-wrap;\">{badge}{code}</div>"
        )
        code_blocks.append(html_block)
        return f"@@CODEBLOCK_{idx}@@"

    escaped = re.sub(
        r"```([a-zA-Z0-9_-]*)\n?(.*?)```",
        _save_code_block,
        escaped,
        flags=re.DOTALL,
    )

    # 3. Inline code (`code`)
    escaped = re.sub(
        r"`([^`\n]+)`",
        r"<code style=\"background-color: #11111b; border: 1px solid #313244; "
        r"border-radius: 4px; padding: 1px 6px; font-family: 'Consolas', monospace; "
        r"font-size: 12px; color: #f38ba8;\">\1</code>",
        escaped,
    )

    # 4. Links: [text](url) -> safe http/https only
    def _format_link(match: re.Match) -> str:
        label = match.group(1)
        url = match.group(2).strip()
        raw_url = html.unescape(url)
        if raw_url.startswith("https://") or raw_url.startswith("http://"):
            return f'<a href="{url}" style="color: #89b4fa; text-decoration: underline;">{label}</a>'
        return f"{label} ({url})"

    escaped = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", _format_link, escaped)

    # 5. Bold (**text**)
    escaped = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", escaped)

    # 6. Italic (*text*)
    escaped = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", escaped)

    # 7. Headings, blockquotes, lists
    lines = escaped.split("\n")
    processed_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("### "):
            processed_lines.append(
                f"<div style='font-size: 14px; font-weight: bold; color: #b4befe; "
                f"margin: 6px 0 2px 0;'>{stripped[4:]}</div>"
            )
        elif stripped.startswith("## "):
            processed_lines.append(
                f"<div style='font-size: 15px; font-weight: bold; color: #89b4fa; "
                f"margin: 8px 0 3px 0;'>{stripped[3:]}</div>"
            )
        elif stripped.startswith("# "):
            processed_lines.append(
                f"<div style='font-size: 16px; font-weight: bold; color: #89b4fa; "
                f"margin: 10px 0 4px 0;'>{stripped[2:]}</div>"
            )
        elif stripped.startswith("&gt; "):
            processed_lines.append(
                f"<div style='border-left: 3px solid #89b4fa; padding-left: 10px; "
                f"margin: 6px 0; color: #a6adc8; font-style: italic;'>{stripped[5:]}</div>"
            )
        elif stripped.startswith("- ") or stripped.startswith("* "):
            processed_lines.append(
                f"<div style='margin-left: 12px; line-height: 1.5;'>• {stripped[2:]}</div>"
            )
        elif re.match(r"^\d+\.\s+", stripped):
            m = re.match(r"^(\d+)\.\s+(.*)", stripped)
            if m:
                processed_lines.append(
                    f"<div style='margin-left: 12px; line-height: 1.5;'>{m.group(1)}. {m.group(2)}</div>"
                )
            else:
                processed_lines.append(line)
        else:
            processed_lines.append(line)

    result = "<br/>".join(processed_lines)

    # 8. Restore code blocks
    for idx, block in enumerate(code_blocks):
        result = result.replace(f"@@CODEBLOCK_{idx}@@", block)

    return result


def is_provider_error(text: str) -> bool:
    """True when text represents a provider failure rather than a normal assistant answer."""
    clean = str(text).strip()
    return (
        clean.startswith("Gemini API execution error:")
        or clean.startswith("Vector: Gemini API execution error:")
        or clean.startswith("Error: Gemini API")
        or clean.startswith("Google GenAI SDK not installed")
    )


class MessageContainer(QWidget):
    """
    Message container wrapper providing horizontal alignment and content-hugging widths.
    Implements text() so test harnesses examining list_widget items find the content.
    """

    def __init__(self, label: QLabel, role: str = "assistant", parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.label = label
        self.role = role
        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 3, 14, 3)
        layout.setSpacing(0)

        if role == "user":
            layout.addStretch(1)
            layout.addWidget(label, 0)
        else:
            layout.addWidget(label, 0)
            layout.addStretch(1)

    def text(self) -> str:
        """Expose label text for test assertions and inspection."""
        return self.label.text()


class ErrorCardWidget(QFrame):
    """
    Compact card displaying provider failures (e.g. Gemini 404, 400).
    Provides a concise human summary and collapsible technical details with credentials redacted.
    """

    def __init__(
        self,
        raw_text: str,
        parent: Optional[QWidget] = None,
        on_resized: Optional[Any] = None,
    ):
        super().__init__(parent)
        self.raw_text = raw_text
        self.on_resized = on_resized
        self._expanded = False

        summary, details = self._parse_error(raw_text)
        self.summary_text = summary
        self.details_text = details

        self.setStyleSheet("""
            ErrorCardWidget {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #25131c, stop:1 #1c0e15);
                border: 1px solid rgba(243, 139, 168, 0.45);
                border-radius: 12px;
                margin: 2px 14px;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 8, 14, 8)
        layout.setSpacing(8)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(8)

        self.lbl_title = QLabel(f"<b>⚠ {html.escape(self.summary_text)}</b>", self)
        self.lbl_title.setStyleSheet("color: #f38ba8; font-size: 13px; font-weight: bold;")
        self.lbl_title.setTextInteractionFlags(Qt.TextSelectableByMouse)
        header.addWidget(self.lbl_title)
        header.addStretch()

        if self.details_text:
            self.btn_toggle = QPushButton("Details ▼", self)
            self.btn_toggle.setStyleSheet("""
                QPushButton {
                    background-color: rgba(243, 139, 168, 0.12);
                    color: #f38ba8;
                    border: 1px solid rgba(243, 139, 168, 0.35);
                    border-radius: 10px;
                    padding: 3px 12px;
                    font-size: 11px;
                    font-weight: 600;
                }
                QPushButton:hover {
                    background-color: rgba(243, 139, 168, 0.22);
                    border-color: #f38ba8;
                }
            """)
            self.btn_toggle.clicked.connect(self._toggle_details)
            header.addWidget(self.btn_toggle)

        layout.addLayout(header)

        self.lbl_details = None
        if self.details_text:
            self.lbl_details = QLabel(html.escape(self.details_text), self)
            self.lbl_details.setStyleSheet("""
                color: #bac2de;
                font-family: 'Consolas', 'Cascadia Code', monospace;
                font-size: 11px;
                background-color: #180f14;
                border: 1px solid #3b222c;
                border-radius: 6px;
                padding: 8px 10px;
                line-height: 1.4;
            """)
            self.lbl_details.setWordWrap(True)
            self.lbl_details.setTextInteractionFlags(Qt.TextSelectableByMouse)
            self.lbl_details.hide()
            layout.addWidget(self.lbl_details)

    def _toggle_details(self) -> None:
        self._expanded = not self._expanded
        if self.lbl_details:
            self.lbl_details.setVisible(self._expanded)
            self.btn_toggle.setText("Details ▲" if self._expanded else "Details ▼")
            self.adjustSize()
            if callable(self.on_resized):
                self.on_resized()

    def text(self) -> str:
        """Expose raw text for test assertions and history compatibility."""
        return f"<b>Vector:</b> {html.escape(self.raw_text)}"

    @staticmethod
    def _parse_error(text: str) -> tuple[str, str]:
        # Redact any API keys, credentials, or secrets
        sanitized = re.sub(r"AIzaSy[A-Za-z0-9_-]{30,35}", "[REDACTED_API_KEY]", text)
        sanitized = re.sub(
            r"(key|token|secret)=['\"][^'\"]+['\"]",
            r"\1=[REDACTED]",
            sanitized,
            flags=re.IGNORECASE,
        )

        if "404" in sanitized or "NOT_FOUND" in sanitized:
            summary = "Gemini Provider Error: Model Not Found (404)"
        elif "400" in sanitized or "INVALID_ARGUMENT" in sanitized:
            summary = "Gemini Provider Error: Invalid Request / Model Name Format (400)"
        elif (
            "403" in sanitized
            or "PERMISSION_DENIED" in sanitized
            or "401" in sanitized
        ):
            summary = "Gemini Provider Error: Authentication / Permission Denied (401/403)"
        elif "429" in sanitized or "RESOURCE_EXHAUSTED" in sanitized:
            summary = "Gemini Provider Error: Rate Limit / Quota Exceeded (429)"
        elif "Google GenAI SDK not installed" in sanitized:
            summary = "Gemini Provider Error: SDK Not Installed"
        else:
            summary = "Provider Execution Error"

        return summary, sanitized


class StreamHandle:
    """
    A handle to an in-progress streamed assistant bubble.
    """

    __slots__ = ("item", "label", "text", "container")

    def __init__(
        self,
        item: QListWidgetItem,
        label: QLabel,
        container: Optional[QWidget] = None,
    ):
        self.item = item
        self.label = label
        self.text = ""
        self.container = container


class ActionCardWidget(QFrame):
    """
    Visual action card for displaying executed tool status, data, and messages.
    """

    def __init__(
        self,
        tool_name: str,
        message: str,
        success: bool = True,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.tool_name = tool_name
        self.message = message
        self.success = success
        self.setFrameShape(QFrame.StyledPanel)
        bg_color = "#11111b" if success else "#31131c"
        border_color = "#a6e3a1" if success else "#f38ba8"
        status_text = "✓ SUCCESS" if success else "✗ FAILED"
        status_color = "#a6e3a1" if success else "#f38ba8"

        self.setStyleSheet(f"""
            ActionCardWidget {{
                background-color: {bg_color};
                border: 1px solid {border_color};
                border-radius: 12px;
                padding: 8px 14px;
                margin: 2px 14px;
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
                line-height: 1.4;
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(6)

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
        lbl_msg.setTextInteractionFlags(Qt.TextSelectableByMouse)

        layout.addLayout(header_layout)
        layout.addWidget(lbl_msg)

    def text(self) -> str:
        """Expose tool message text for test inspection."""
        return f"{self.tool_name}: {self.message}"


class ChatViewWidget(QWidget):
    """
    Main conversation view widget displaying user messages, assistant responses,
    markdown formatting, error cards, and action history.
    """

    load_earlier_requested = Signal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.btn_load_earlier = QPushButton("↑ Load earlier messages", self)
        self.btn_load_earlier.setStyleSheet("""
            QPushButton {
                background-color: #313244;
                color: #cdd6f4;
                border: 1px solid #45475a;
                border-radius: 8px;
                padding: 6px 14px;
                margin: 8px 14px;
                font-size: 12px;
                font-weight: 500;
            }
            QPushButton:hover { background-color: #45475a; }
        """)
        self.btn_load_earlier.clicked.connect(self.load_earlier_requested.emit)
        self.btn_load_earlier.hide()
        layout.addWidget(self.btn_load_earlier)

        self.list_widget = QListWidget(self)
        self.list_widget.setStyleSheet("""
            QListWidget {
                background-color: #14141e;
                border: none;
                color: #cdd6f4;
                font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif;
                font-size: 13px;
                padding: 12px 4px;
            }
            QListWidget::item {
                background: transparent;
                border: none;
                padding: 0px;
                margin: 0px;
            }
            QListWidget::item:selected {
                background: transparent;
            }
            QScrollBar:vertical {
                background: transparent;
                width: 8px;
                margin: 0px 2px 0px 0px;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical {
                background: #313244;
                min-height: 28px;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical:hover {
                background: #45475a;
            }
            QScrollBar::handle:vertical:pressed {
                background: #585b70;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
                background: none;
            }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                background: none;
            }
        """)
        self.list_widget.setSpacing(14)
        self.list_widget.setVerticalScrollMode(QListWidget.ScrollPerPixel)
        self.list_widget.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        layout.addWidget(self.list_widget)

    def _open_link(self, url_str: str) -> None:
        """Safely opens HTTP and HTTPS links in the user's default browser."""
        try:
            qurl = QUrl(url_str)
            if qurl.scheme().lower() in ("http", "https"):
                QDesktopServices.openUrl(qurl)
        except Exception:
            pass

    def _should_auto_scroll(self) -> bool:
        """True if the user is already near the bottom of the conversation."""
        sb = self.list_widget.verticalScrollBar()
        if not sb:
            return True
        return sb.value() >= (sb.maximum() - 40)

    def _scroll_to_bottom_if_appropriate(self, force: bool = False) -> None:
        """Scrolls to bottom if forced (e.g. user submitted) or if already at bottom."""
        if force or self._should_auto_scroll():
            self.list_widget.scrollToBottom()

    def _create_user_label(self, text: str) -> QLabel:
        """Creates a styled user message label with blue-tinted Fluent surface."""
        label = QLabel(self.list_widget)
        label.setTextFormat(Qt.RichText)
        label.setText(f"<b>You:</b> {html.escape(str(text))}")
        label.setStyleSheet("""
            QLabel {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #2563eb, stop:1 #1d4ed8);
                color: #ffffff;
                border: 1px solid rgba(255, 255, 255, 0.18);
                border-radius: 14px;
                border-bottom-right-radius: 4px;
                padding: 10px 16px;
                font-family: 'Segoe UI', -apple-system, sans-serif;
                font-size: 13px;
                line-height: 1.45;
            }
        """)
        label.setWordWrap(True)
        label.setTextInteractionFlags(Qt.TextSelectableByMouse | Qt.LinksAccessibleByMouse)
        label.linkActivated.connect(self._open_link)
        return label

    def _create_assistant_label(self, formatted_content: str) -> QLabel:
        """Creates a styled assistant message label with neutral dark Fluent surface."""
        label = QLabel(self.list_widget)
        label.setTextFormat(Qt.RichText)
        label.setText(f"<b>Vector:</b> {formatted_content}")
        label.setStyleSheet("""
            QLabel {
                background-color: #1e1e2e;
                color: #cdd6f4;
                border: 1px solid #313244;
                border-radius: 14px;
                border-bottom-left-radius: 4px;
                padding: 12px 18px;
                font-family: 'Segoe UI', -apple-system, sans-serif;
                font-size: 13px;
                line-height: 1.5;
            }
        """)
        label.setWordWrap(True)
        label.setTextInteractionFlags(Qt.TextSelectableByMouse | Qt.LinksAccessibleByMouse)
        label.linkActivated.connect(self._open_link)
        return label

    def _update_item_size(self, item: QListWidgetItem, widget: QWidget) -> None:
        """Accurately calculates and updates the item geometry to prevent clipping."""
        try:
            viewport_w = self.list_widget.viewport().width()
            if viewport_w <= 0:
                viewport_w = max(400, self.list_widget.width() - 24)

            if isinstance(widget, MessageContainer):
                factor = 0.78 if widget.role == "user" else 0.86
                max_w = max(120, int(viewport_w * factor))
                widget.label.setMaximumWidth(max_w)
                widget.adjustSize()
                h = max(38, widget.sizeHint().height())
                item.setSizeHint(QSize(viewport_w, h))
            elif isinstance(widget, (ActionCardWidget, ErrorCardWidget)):
                widget.adjustSize()
                item.setSizeHint(QSize(viewport_w, max(42, widget.sizeHint().height())))
            else:
                widget.adjustSize()
                item.setSizeHint(QSize(viewport_w, max(36, widget.sizeHint().height())))
        except Exception:
            pass

    def resizeEvent(self, event: Any) -> None:
        """Dynamically recalculates all message heights on window resize to prevent clipping."""
        super().resizeEvent(event)
        self._recalculate_all_item_sizes()

    def _recalculate_all_item_sizes(self) -> None:
        """Updates sizeHint for every item according to the new viewport width."""
        viewport_w = self.list_widget.viewport().width()
        if viewport_w <= 0:
            return
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            widget = self.list_widget.itemWidget(item)
            if widget is not None:
                self._update_item_size(item, widget)

    def clear(self) -> None:
        """Removes every item from the conversation view."""
        self.list_widget.clear()

    def set_load_more_visible(self, visible: bool) -> None:
        """Shows or hides the load earlier messages control."""
        self.btn_load_earlier.setVisible(bool(visible))

    def add_user_message(self, text: str) -> None:
        """Adds a user message item to the chat view."""
        item = QListWidgetItem(self.list_widget)
        label = self._create_user_label(text)
        container = MessageContainer(label, role="user", parent=self.list_widget)
        self._update_item_size(item, container)
        self.list_widget.addItem(item)
        self.list_widget.setItemWidget(item, container)
        self._scroll_to_bottom_if_appropriate(force=True)

    def add_assistant_message(self, text: str) -> None:
        """Adds an assistant response or provider error card item to the chat view."""
        raw_str = str(text)

        # Detect provider errors and render as compact ErrorCardWidget
        if is_provider_error(raw_str):
            item = QListWidgetItem(self.list_widget)
            card = ErrorCardWidget(
                raw_str,
                parent=self.list_widget,
                on_resized=lambda: self._update_item_size(item, card),
            )
            self._update_item_size(item, card)
            self.list_widget.addItem(item)
            self.list_widget.setItemWidget(item, card)
            self._scroll_to_bottom_if_appropriate()
            return

        item = QListWidgetItem(self.list_widget)
        formatted = render_safe_markdown(raw_str)
        label = self._create_assistant_label(formatted)
        container = MessageContainer(label, role="assistant", parent=self.list_widget)
        self._update_item_size(item, container)
        self.list_widget.addItem(item)
        self.list_widget.setItemWidget(item, container)
        self._scroll_to_bottom_if_appropriate()

    def begin_assistant_stream(self) -> StreamHandle:
        """
        Open an empty assistant bubble that will be filled incrementally.
        Returns a handle the caller keeps for the duration of the stream.
        """
        item = QListWidgetItem(self.list_widget)
        label = self._create_assistant_label("")
        container = MessageContainer(label, role="assistant", parent=self.list_widget)
        self._update_item_size(item, container)
        self.list_widget.addItem(item)
        self.list_widget.setItemWidget(item, container)
        self._scroll_to_bottom_if_appropriate(force=True)
        return StreamHandle(item, label, container)

    def append_assistant_stream(self, handle: Optional[StreamHandle], text: str) -> None:
        """Append one text delta to an open streamed bubble."""
        if handle is None or not text:
            return
        handle.text += text
        formatted = render_safe_markdown(handle.text)
        handle.label.setText(f"<b>Vector:</b> {formatted}")
        self._update_item_size(handle.item, handle.container or handle.label)
        self._scroll_to_bottom_if_appropriate()

    def finalize_assistant_stream(
        self,
        handle: Optional[StreamHandle],
        text: Optional[str] = None,
        suffix: str = "",
    ) -> None:
        """Replace a streamed bubble's body with the authoritative final text."""
        if handle is None:
            return
        if text is not None:
            handle.text = text

        # Check if final text turned out to be a provider error
        if is_provider_error(handle.text):
            card = ErrorCardWidget(
                handle.text,
                parent=self.list_widget,
                on_resized=lambda: self._update_item_size(handle.item, card),
            )
            self._update_item_size(handle.item, card)
            self.list_widget.setItemWidget(handle.item, card)
            handle.container = card
            self._scroll_to_bottom_if_appropriate()
            return

        formatted = render_safe_markdown(handle.text)
        escaped_suffix = html.escape(str(suffix))
        handle.label.setText(f"<b>Vector:</b> {formatted}{escaped_suffix}")
        self._update_item_size(handle.item, handle.container or handle.label)
        self._scroll_to_bottom_if_appropriate()

    def add_action_card(self, tool_name: str, message: str, success: bool = True) -> None:
        """Adds a visual action card item for executed tools."""
        item = QListWidgetItem(self.list_widget)
        card = ActionCardWidget(
            tool_name=tool_name,
            message=message,
            success=success,
            parent=self.list_widget,
        )
        self._update_item_size(item, card)
        self.list_widget.addItem(item)
        self.list_widget.setItemWidget(item, card)
        self._scroll_to_bottom_if_appropriate()

    def add_assistant_error(self, summary: str, details: str = "") -> None:
        """Renders an error presentation card."""
        item = QListWidgetItem(self.list_widget)
        full_text = f"{summary}: {details}" if details else summary
        card = ErrorCardWidget(
            full_text,
            parent=self.list_widget,
            on_resized=lambda: self._update_item_size(item, card),
        )
        self._update_item_size(item, card)
        self.list_widget.addItem(item)
        self.list_widget.setItemWidget(item, card)
        self._scroll_to_bottom_if_appropriate()

    def add_welcome_message(self, text: str) -> None:
        """Adds the welcome message to an empty conversation view."""
        self.add_assistant_message(text)

    def add_message(self, role: str, text: str) -> None:
        """Generic dispatcher routing message by role."""
        if role == "user":
            self.add_user_message(text)
        else:
            self.add_assistant_message(text)
