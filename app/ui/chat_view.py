"""
Chat view widget displaying conversation history, rich markdown formatting,
cyberpunk / Fluent dark styled avatar cards, provider error cards, and visual tool result cards.

The view is the only place that touches Qt widgets for conversation rendering.
Streaming text arrives on the Qt main thread through a signal and is applied
here, so a background worker never mutates a widget directly.
"""

from __future__ import annotations

from datetime import datetime
import html
import re
from typing import Any, Dict, List, Optional, Tuple

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

from app.ui.theme import (
    ACCENT_CYAN,
    ACCENT_GREEN,
    ACCENT_GREEN_BRIGHT,
    ACCENT_PINK,
    ACCENT_PURPLE,
    ACCENT_RED,
    BG_APP,
    BG_CARD,
    BG_PANEL,
    BORDER_ACTIVE,
    BORDER_SUBTLE,
    TEXT_MUTED,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
)
from app.ui.widgets.avatars import AssistantAvatar, StatusPillBadge, UserAvatar


def format_timestamp(ts: Optional[str] = None) -> str:
    """
    Format a timestamp into a clean, human-readable time like '7:34 PM'.
    Gracefully handles ISO 8601 strings, pre-formatted strings, or None.
    """
    if not ts:
        now = datetime.now()
        s = now.strftime("%I:%M %p")
        return s.lstrip("0") if s.startswith("0") else s

    ts_str = str(ts).strip()
    if "AM" in ts_str or "PM" in ts_str:
        return ts_str

    try:
        dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        local_dt = dt.astimezone() if dt.tzinfo else dt
        s = local_dt.strftime("%I:%M %p")
        return s.lstrip("0") if s.startswith("0") else s
    except Exception:
        return ts_str


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
            f"<div style='color: #38bdf8; font-size: 11px; font-weight: bold; "
            f"margin-bottom: 4px; text-transform: uppercase;'>{lang}</div>"
            if lang
            else ""
        )
        html_block = (
            f"<div style=\"background-color: #0e111a; border: 1px solid #1e2235; "
            f"border-radius: 8px; padding: 10px 14px; margin: 6px 0; "
            f"font-family: 'Consolas', 'Cascadia Code', monospace; font-size: 12px; "
            f"color: #cbd5e1; white-space: pre-wrap;\">{badge}{code}</div>"
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
        r"<code style=\"background-color: #0e111a; border: 1px solid #1e2235; "
        r"border-radius: 4px; padding: 1px 6px; font-family: 'Consolas', monospace; "
        r"font-size: 12px; color: #f43f5e;\">\1</code>",
        escaped,
    )

    # 4. Links: [text](url) -> safe http/https only
    def _format_link(match: re.Match) -> str:
        label = match.group(1)
        url = match.group(2).strip()
        raw_url = html.unescape(url)
        if raw_url.startswith("https://") or raw_url.startswith("http://"):
            return f'<a href="{url}" style="color: #38bdf8; text-decoration: underline;">{label}</a>'
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
                f"<div style='font-size: 14px; font-weight: bold; color: #c084fc; "
                f"margin: 6px 0 2px 0;'>{stripped[4:]}</div>"
            )
        elif stripped.startswith("## "):
            processed_lines.append(
                f"<div style='font-size: 15px; font-weight: bold; color: #38bdf8; "
                f"margin: 8px 0 3px 0;'>{stripped[3:]}</div>"
            )
        elif stripped.startswith("# "):
            processed_lines.append(
                f"<div style='font-size: 16px; font-weight: bold; color: #38bdf8; "
                f"margin: 10px 0 4px 0;'>{stripped[2:]}</div>"
            )
        elif stripped.startswith("&gt; "):
            processed_lines.append(
                f"<div style='border-left: 3px solid #38bdf8; padding-left: 10px; "
                f"margin: 6px 0; color: #94a3b8; font-style: italic;'>{stripped[5:]}</div>"
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
    Message container wrapper providing avatar positioning, styled card/bubble
    containers, timestamps, and horizontal alignment.
    Maintains compatibility with tests accessing .label and .text().
    """

    def __init__(
        self,
        label: QLabel,
        role: str = "assistant",
        title: Optional[str] = None,
        timestamp: Optional[str] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.label = label
        self.role = role
        self.raw_timestamp = timestamp
        formatted_ts = format_timestamp(timestamp)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 4, 14, 4)
        layout.setSpacing(10)
        layout.setAlignment(Qt.AlignTop)

        if role == "user":
            # Right-aligned user bubble with avatar on far right
            layout.addStretch(1)

            self.card = QFrame(self)
            self.card.setObjectName("userBubble")
            self.card.setStyleSheet("""
                QFrame#userBubble {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #2563eb, stop:1 #7c3aed);
                    border: 1px solid rgba(255, 255, 255, 0.16);
                    border-radius: 16px;
                    border-bottom-right-radius: 4px;
                }
            """)
            card_layout = QVBoxLayout(self.card)
            card_layout.setContentsMargins(16, 10, 16, 8)
            card_layout.setSpacing(4)

            self.lbl_title = None

            # Label setup
            self.label.setParent(self.card)
            self.label.setMinimumHeight(30)
            self.label.setStyleSheet("""
                QLabel {
                    background: transparent;
                    color: #ffffff;
                    font-family: 'Segoe UI Variable', 'Segoe UI', -apple-system, sans-serif;
                    font-size: 13px;
                    line-height: 1.45;
                    min-height: 30px;
                }
            """)
            card_layout.addWidget(self.label)

            # Footer / timestamp (bottom-right aligned)
            footer = QHBoxLayout()
            footer.setContentsMargins(0, 0, 0, 0)
            footer.addStretch(1)
            self.lbl_timestamp = QLabel(formatted_ts, self.card)
            self.lbl_timestamp.setStyleSheet("""
                QLabel {
                    background: transparent;
                    color: rgba(255, 255, 255, 0.72);
                    font-family: 'Segoe UI Variable', 'Segoe UI', -apple-system, sans-serif;
                    font-size: 11px;
                    font-weight: 500;
                }
            """)
            footer.addWidget(self.lbl_timestamp)
            card_layout.addLayout(footer)

            layout.addWidget(self.card, 0, Qt.AlignTop)

            self.avatar = UserAvatar(size=34, parent=self)
            layout.addWidget(self.avatar, 0, Qt.AlignTop)

        else:
            # Left-aligned assistant card with glowing 'V' avatar on left
            self.avatar = AssistantAvatar(size=34, parent=self)
            layout.addWidget(self.avatar, 0, Qt.AlignTop)

            self.card = QFrame(self)
            self.card.setObjectName("assistantCard")
            self.card.setStyleSheet("""
                QFrame#assistantCard {
                    background-color: #161925;
                    border: 1px solid #232738;
                    border-radius: 16px;
                }
            """)
            card_layout = QVBoxLayout(self.card)
            card_layout.setContentsMargins(16, 12, 16, 10)
            card_layout.setSpacing(6)

            card_title = title or "Vector"
            self.lbl_title = QLabel(card_title, self.card)
            self.lbl_title.setObjectName("cardTitle")
            self.lbl_title.setStyleSheet("""
                QLabel#cardTitle {
                    background: transparent;
                    color: #38bdf8;
                    font-family: 'Segoe UI Variable', 'Segoe UI', -apple-system, sans-serif;
                    font-size: 13px;
                    font-weight: 700;
                }
            """)
            card_layout.addWidget(self.lbl_title)

            self.label.setParent(self.card)
            self.label.setMinimumHeight(30)
            self.label.setStyleSheet("""
                QLabel {
                    background: transparent;
                    color: #cbd5e1;
                    font-family: 'Segoe UI Variable', 'Segoe UI', -apple-system, sans-serif;
                    font-size: 13px;
                    line-height: 1.5;
                    min-height: 30px;
                }
            """)
            card_layout.addWidget(self.label)

            # Footer / timestamp (bottom-left aligned)
            footer = QHBoxLayout()
            footer.setContentsMargins(0, 2, 0, 0)
            self.lbl_timestamp = QLabel(formatted_ts, self.card)
            self.lbl_timestamp.setStyleSheet("""
                QLabel {
                    background: transparent;
                    color: #64748b;
                    font-family: 'Segoe UI Variable', 'Segoe UI', -apple-system, sans-serif;
                    font-size: 11px;
                    font-weight: 500;
                }
            """)
            footer.addWidget(self.lbl_timestamp)
            footer.addStretch(1)
            card_layout.addLayout(footer)

            layout.addWidget(self.card, 0, Qt.AlignTop)
            layout.addStretch(1)

    def text(self) -> str:
        """Expose label text for test assertions and inspection."""
        return self.label.text()


class ErrorCardWidget(QFrame):
    """
    Compact card displaying provider failures (e.g. Gemini 404, 400).
    Provides a concise human summary and collapsible technical details with credentials redacted.
    Initial height is guaranteed compact (<= 60px).
    """

    def __init__(
        self,
        raw_text: str,
        timestamp: Optional[str] = None,
        parent: Optional[QWidget] = None,
        on_resized: Optional[Any] = None,
    ):
        super().__init__(parent)
        self.raw_text = raw_text
        self.raw_timestamp = timestamp
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
        layout.setSpacing(6)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(8)

        self.lbl_title = QLabel(f"<b>⚠ {html.escape(self.summary_text)}</b>", self)
        self.lbl_title.setStyleSheet("color: #f38ba8; font-size: 13px; font-weight: bold; background: transparent;")
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
    Visual action card for displaying executed tool status, audio feedback, and messages.
    Supports specialized audio feedback (green accent + speaker icon) and
    general tool execution (green border + gear icon + SUCCESS badge).
    """

    def __init__(
        self,
        tool_name: str,
        message: str,
        success: bool = True,
        timestamp: Optional[str] = None,
        is_audio: Optional[bool] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.tool_name = str(tool_name)
        self.message = str(message)
        self.success = bool(success)
        self.raw_timestamp = timestamp
        formatted_ts = format_timestamp(timestamp)

        # Detect audio feedback
        is_audio_action = (
            is_audio
            if is_audio is not None
            else (
                self.tool_name.lower() in ("mute", "unmute", "set_volume", "get_volume", "audio")
                or "audio" in self.message.lower()
                or "volume" in self.message.lower()
            )
        )
        self.is_audio_action = is_audio_action

        if self.success:
            bg_color = "#0e1a17" if is_audio_action else "#10141f"
            border_color = "rgba(16, 185, 129, 0.45)"
        else:
            bg_color = "#1f1015"
            border_color = "rgba(239, 68, 68, 0.45)"

        self.setStyleSheet(f"""
            ActionCardWidget {{
                background-color: {bg_color};
                border: 1px solid {border_color};
                border-radius: 14px;
                margin: 2px 14px;
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(6)

        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(8)

        escaped_tool = html.escape(self.tool_name)
        escaped_msg = html.escape(self.message)

        if is_audio_action:
            lbl_title = QLabel(f"🔊 {escaped_msg}", self)
            lbl_title.setStyleSheet("QLabel { color: #34d399; font-weight: 700; font-size: 13px; background: transparent; }")
            lbl_title.setWordWrap(True)
            header_layout.addWidget(lbl_title, 1)
        else:
            lbl_title = QLabel(f"⚙ Tool Action: {escaped_tool}", self)
            lbl_title.setStyleSheet("QLabel { color: #f8fafc; font-weight: 700; font-size: 13px; background: transparent; }")
            header_layout.addWidget(lbl_title, 1)

        badge_type = "success" if self.success else "failed"
        badge_text = "✔ SUCCESS" if self.success else "✗ FAILED"
        self.badge = StatusPillBadge(badge_text, badge_type=badge_type, parent=self)
        header_layout.addWidget(self.badge, 0, Qt.AlignRight)

        layout.addLayout(header_layout)

        if not is_audio_action and self.message:
            self.lbl_msg = QLabel(escaped_msg, self)
            self.lbl_msg.setStyleSheet("QLabel { color: #cbd5e1; font-size: 12px; line-height: 1.4; background: transparent; }")
            self.lbl_msg.setWordWrap(True)
            self.lbl_msg.setTextInteractionFlags(Qt.TextSelectableByMouse)
            layout.addWidget(self.lbl_msg)

        footer_layout = QHBoxLayout()
        footer_layout.setContentsMargins(0, 2, 0, 0)
        self.lbl_timestamp = QLabel(formatted_ts, self)
        self.lbl_timestamp.setStyleSheet("QLabel { color: #64748b; font-size: 11px; background: transparent; }")
        footer_layout.addWidget(self.lbl_timestamp)
        footer_layout.addStretch(1)
        layout.addLayout(footer_layout)

    def text(self) -> str:
        """Expose tool message text for test inspection."""
        return f"{self.tool_name}: {self.message}"


class ChatViewWidget(QWidget):
    """
    Main conversation view widget displaying user messages, assistant responses,
    markdown formatting, error cards, and action history in a modern Cyberpunk / Fluent
    avatar card layout.
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
                background-color: #161925;
                color: #cbd5e1;
                border: 1px solid #1e2235;
                border-radius: 8px;
                padding: 6px 14px;
                margin: 8px 14px;
                font-family: 'Segoe UI Variable', 'Segoe UI', -apple-system, sans-serif;
                font-size: 12px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #1e2235;
                border-color: #38bdf8;
                color: #f8fafc;
            }
        """)
        self.btn_load_earlier.clicked.connect(self.load_earlier_requested.emit)
        self.btn_load_earlier.hide()
        layout.addWidget(self.btn_load_earlier)

        self.list_widget = QListWidget(self)
        self.list_widget.setStyleSheet("""
            QListWidget {
                background-color: #0b0d14;
                border: none;
                color: #f8fafc;
                font-family: 'Segoe UI Variable', 'Segoe UI', -apple-system, sans-serif;
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
                width: 6px;
                margin: 0px 2px 0px 0px;
                border-radius: 3px;
            }
            QScrollBar::handle:vertical {
                background: #2b324d;
                min-height: 24px;
                border-radius: 3px;
            }
            QScrollBar::handle:vertical:hover {
                background: #38bdf8;
            }
            QScrollBar::handle:vertical:pressed {
                background: #0284c7;
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
        """Creates a styled user message label with white text."""
        label = QLabel(self.list_widget)
        label.setTextFormat(Qt.RichText)
        label.setText(html.escape(str(text)))
        label.setWordWrap(True)
        label.setTextInteractionFlags(Qt.TextSelectableByMouse | Qt.LinksAccessibleByMouse)
        label.linkActivated.connect(self._open_link)
        return label

    def _create_assistant_label(self, formatted_content: str) -> QLabel:
        """Creates a styled assistant message label with soft grey text."""
        label = QLabel(self.list_widget)
        label.setTextFormat(Qt.RichText)
        label.setText(formatted_content)
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
                factor = 0.76 if widget.role == "user" else 0.84
                max_w = max(160, int(viewport_w * factor))
                if hasattr(widget, "card") and widget.card is not None:
                    widget.card.setMaximumWidth(max_w)
                if hasattr(widget, "label") and widget.label is not None:
                    widget.label.setMaximumWidth(max_w - 32)
                widget.adjustSize()
                h = max(48, widget.sizeHint().height() + 4)
                item.setSizeHint(QSize(viewport_w, h))
            elif isinstance(widget, ActionCardWidget):
                widget.setMaximumWidth(max(200, viewport_w - 28))
                widget.adjustSize()
                item.setSizeHint(QSize(viewport_w, max(48, widget.sizeHint().height() + 4)))
            elif isinstance(widget, ErrorCardWidget):
                widget.setMaximumWidth(max(200, viewport_w - 28))
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

    def add_user_message(self, text: str, timestamp: Optional[str] = None) -> None:
        """Adds a user message item with user avatar to the chat view."""
        item = QListWidgetItem(self.list_widget)
        label = self._create_user_label(text)
        container = MessageContainer(
            label,
            role="user",
            timestamp=timestamp,
            parent=self.list_widget,
        )
        self._update_item_size(item, container)
        self.list_widget.addItem(item)
        self.list_widget.setItemWidget(item, container)
        self._scroll_to_bottom_if_appropriate(force=True)

    def add_assistant_message(
        self,
        text: str,
        timestamp: Optional[str] = None,
        title: Optional[str] = None,
    ) -> None:
        """Adds an assistant response or provider error card item to the chat view."""
        raw_str = str(text)

        # Detect provider errors and render as compact ErrorCardWidget
        if is_provider_error(raw_str):
            item = QListWidgetItem(self.list_widget)
            card = ErrorCardWidget(
                raw_str,
                timestamp=timestamp,
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

        card_title = title
        if card_title is None:
            if raw_str.startswith("Hello!"):
                card_title = "Hello! 👋"
            else:
                card_title = "Vector"

        container = MessageContainer(
            label,
            role="assistant",
            title=card_title,
            timestamp=timestamp,
            parent=self.list_widget,
        )
        self._update_item_size(item, container)
        self.list_widget.addItem(item)
        self.list_widget.setItemWidget(item, container)
        self._scroll_to_bottom_if_appropriate()

    def begin_assistant_stream(self, title: Optional[str] = None) -> StreamHandle:
        """
        Open an empty assistant bubble that will be filled incrementally.
        Returns a handle the caller keeps for the duration of the stream.
        """
        item = QListWidgetItem(self.list_widget)
        label = self._create_assistant_label("")
        container = MessageContainer(
            label,
            role="assistant",
            title=title or "Vector",
            parent=self.list_widget,
        )
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
        handle.label.setText(formatted)
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
        handle.label.setText(f"{formatted}{escaped_suffix}")
        self._update_item_size(handle.item, handle.container or handle.label)
        self._scroll_to_bottom_if_appropriate()

    def add_action_card(
        self,
        tool_name: str,
        message: str,
        success: bool = True,
        timestamp: Optional[str] = None,
    ) -> None:
        """Adds a visual action card item for executed tools or audio feedback."""
        item = QListWidgetItem(self.list_widget)
        card = ActionCardWidget(
            tool_name=tool_name,
            message=message,
            success=success,
            timestamp=timestamp,
            parent=self.list_widget,
        )
        self._update_item_size(item, card)
        self.list_widget.addItem(item)
        self.list_widget.setItemWidget(item, card)
        self._scroll_to_bottom_if_appropriate()

    def add_audio_card(
        self,
        message: str,
        success: bool = True,
        timestamp: Optional[str] = None,
    ) -> None:
        """Convenience method for adding specialized audio feedback cards."""
        self.add_action_card(
            tool_name="audio",
            message=message,
            success=success,
            timestamp=timestamp,
        )

    def add_assistant_error(
        self,
        summary: str,
        details: str = "",
        timestamp: Optional[str] = None,
    ) -> None:
        """Renders an error presentation card."""
        item = QListWidgetItem(self.list_widget)
        full_text = f"{summary}: {details}" if details else summary
        card = ErrorCardWidget(
            full_text,
            timestamp=timestamp,
            parent=self.list_widget,
            on_resized=lambda: self._update_item_size(item, card),
        )
        self._update_item_size(item, card)
        self.list_widget.addItem(item)
        self.list_widget.setItemWidget(item, card)
        self._scroll_to_bottom_if_appropriate()

    def add_welcome_message(self, text: str, timestamp: Optional[str] = None) -> None:
        """Adds the welcome message to an empty conversation view."""
        self.add_assistant_message(text, timestamp=timestamp, title="Hello! 👋")

    def add_message(self, role: str, text: str, timestamp: Optional[str] = None) -> None:
        """Generic dispatcher routing message by role."""
        if role == "user":
            self.add_user_message(text, timestamp=timestamp)
        else:
            self.add_assistant_message(text, timestamp=timestamp)
