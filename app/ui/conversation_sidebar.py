"""
Conversation sidebar for Vector Desktop AI Assistant.

A self-contained widget that lists saved conversations (grouped by recency),
lets the user search, start a new chat, refresh, and request rename/delete.
It reads through the existing :class:`~app.memory.history.HistoryManager`
listing/search APIs -- no SQL lives in the widget -- and never owns the active
conversation: it only *reports* the user's intent through signals so the main
window stays the single source of truth for which conversation is active.

Search and refresh run on a background worker (matching the Action History /
Backup dialogs) so scanning message content never blocks the UI thread.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, Signal, Slot
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

#: Recency buckets, most recent first.  The order doubles as display order.
GROUP_TODAY = "Today"
GROUP_YESTERDAY = "Yesterday"
GROUP_PREVIOUS_7_DAYS = "Previous 7 Days"
GROUP_OLDER = "Older"
GROUP_ORDER: Tuple[str, ...] = (
    GROUP_TODAY,
    GROUP_YESTERDAY,
    GROUP_PREVIOUS_7_DAYS,
    GROUP_OLDER,
)

EMPTY_TEXT = "No conversations yet. Start a new chat to begin."
NO_MATCHES_TEXT = "No conversations match your search."
LOAD_ERROR_TEXT = "Could not load conversations. Please try again."

#: Upper bound on rows fetched for the sidebar in one list/search call.
DEFAULT_LIMIT = 500

_ROLE_CONVERSATION_ID = Qt.UserRole + 1

_ACTIVE_BACKGROUND = QColor("#313244")
_ACTIVE_FOREGROUND = QColor("#89b4fa")
_IDLE_FOREGROUND = QColor("#cdd6f4")
_HEADER_FOREGROUND = QColor("#7f849c")


# --------------------------------------------------------------------------
# Pure grouping helpers (unit-tested independently of Qt)
# --------------------------------------------------------------------------
def _parse_timestamp(value: Any) -> Optional[datetime]:
    """Parse a stored ISO timestamp, returning ``None`` when unusable."""
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _bucket_for(moment: Optional[datetime], now: datetime) -> str:
    """Return the recency bucket for a timestamp relative to ``now``."""
    if moment is None:
        return GROUP_OLDER

    ref_tz = now.tzinfo
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=ref_tz or timezone.utc)
    local = moment.astimezone(ref_tz) if ref_tz else moment

    delta_days = (now.date() - local.date()).days
    if delta_days <= 0:
        return GROUP_TODAY
    if delta_days == 1:
        return GROUP_YESTERDAY
    if delta_days <= 7:
        return GROUP_PREVIOUS_7_DAYS
    return GROUP_OLDER


def group_conversations(
    conversations: Optional[List[Dict[str, Any]]],
    now: Optional[datetime] = None,
) -> List[Tuple[str, List[Dict[str, Any]]]]:
    """
    Split conversation summaries into ordered recency groups.

    ``now`` (an aware datetime) is the reference point; when omitted the current
    time is used.  Empty groups are dropped, and the input order is preserved
    within each group so the caller's recency sort is respected.
    """
    reference = now or datetime.now().astimezone()
    buckets: Dict[str, List[Dict[str, Any]]] = {name: [] for name in GROUP_ORDER}
    for conversation in conversations or []:
        if not isinstance(conversation, dict):
            continue
        moment = _parse_timestamp(conversation.get("last_activity"))
        buckets[_bucket_for(moment, reference)].append(conversation)
    return [(name, buckets[name]) for name in GROUP_ORDER if buckets[name]]


def _activity_text(moment: Optional[datetime], now: datetime) -> str:
    """Compact "time of last activity" label for a conversation row."""
    if moment is None:
        return ""
    ref_tz = now.tzinfo
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=ref_tz or timezone.utc)
    local = moment.astimezone(ref_tz) if ref_tz else moment
    if local.date() == now.date():
        return local.strftime("%H:%M")
    return local.strftime("%Y-%m-%d")


# --------------------------------------------------------------------------
# Background listing worker
# --------------------------------------------------------------------------
class ConversationListSignals(QObject):
    """Signals for the background conversation lister."""

    result = Signal(object)
    error = Signal(str)


class ConversationListWorker(QRunnable):
    """Lists or searches conversations off the Qt main UI thread."""

    def __init__(self, history_manager: Optional[Any], query: str = "", limit: int = DEFAULT_LIMIT):
        super().__init__()
        self.history_manager = history_manager
        self.query = query
        self.limit = limit
        self.signals = ConversationListSignals()

    @Slot()
    def run(self) -> None:
        try:
            if self.history_manager is None:
                self.signals.result.emit([])
                return
            if (self.query or "").strip():
                searcher = getattr(self.history_manager, "search_conversations", None)
                entries = (
                    searcher(self.query, limit=self.limit) if searcher is not None else []
                )
            else:
                lister = getattr(self.history_manager, "list_conversations", None)
                entries = lister(limit=self.limit) if lister is not None else []
            self.signals.result.emit(list(entries or []))
        except Exception:
            # Generic message: raw storage details are never surfaced.
            self.signals.error.emit(LOAD_ERROR_TEXT)


# --------------------------------------------------------------------------
# Sidebar widget
# --------------------------------------------------------------------------
class ConversationSidebar(QWidget):
    """
    Lists saved conversations and reports user intent through signals.

    Signals
    -------
    conversation_selected(str)
        The user picked a conversation (single click / keyboard).
    new_chat_requested()
        The "New Chat" button was pressed.
    rename_requested(str)
        Rename was requested for a conversation id.
    delete_requested(str)
        Delete was requested for a conversation id.
    """

    conversation_selected = Signal(str)
    new_chat_requested = Signal()
    rename_requested = Signal(str)
    delete_requested = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None, limit: int = DEFAULT_LIMIT):
        super().__init__(parent)
        self.history_manager: Optional[Any] = None
        self.limit = limit
        self._active_id: Optional[str] = None
        self._suppress_selection = False

        self._setup_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _setup_ui(self) -> None:
        self.setObjectName("conversationSidebar")
        self.setStyleSheet("""
            QWidget#conversationSidebar {
                background-color: #181825;
                border-right: 1px solid #313244;
            }
            QLabel#sb_heading { color: #89b4fa; font-weight: bold; font-size: 13px; }
            QLabel#sb_empty { color: #7f849c; font-size: 12px; padding: 12px; }
            QLabel#sb_error { color: #f38ba8; font-size: 12px; padding: 12px; }
            QLineEdit {
                background-color: #11111b;
                border: 1px solid #313244;
                border-radius: 6px;
                color: #cdd6f4;
                padding: 5px 8px;
            }
            QPushButton#sb_new {
                background-color: #89b4fa;
                color: #11111b;
                font-weight: bold;
                border: none;
                border-radius: 6px;
                padding: 6px 10px;
            }
            QPushButton#sb_new:hover { background-color: #74a0e8; }
            QPushButton#sb_refresh {
                background-color: #313244;
                color: #cdd6f4;
                border: none;
                border-radius: 6px;
                padding: 6px 10px;
            }
            QPushButton#sb_refresh:hover { background-color: #45475a; }
            QListWidget {
                background-color: #181825;
                border: none;
                color: #cdd6f4;
                font-size: 12px;
            }
            QListWidget::item { padding: 5px 6px; border-radius: 6px; }
            QListWidget::item:selected { background-color: #313244; color: #89b4fa; }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        heading = QLabel("💬 Conversations", self)
        heading.setObjectName("sb_heading")
        layout.addWidget(heading)

        top = QHBoxLayout()
        self.btn_new = QPushButton("＋ New Chat", self)
        self.btn_new.setObjectName("sb_new")
        self.btn_new.clicked.connect(self.new_chat_requested.emit)
        self.btn_refresh = QPushButton("↻", self)
        self.btn_refresh.setObjectName("sb_refresh")
        self.btn_refresh.setToolTip("Refresh conversations")
        self.btn_refresh.clicked.connect(self.refresh)
        top.addWidget(self.btn_new, stretch=1)
        top.addWidget(self.btn_refresh)
        layout.addLayout(top)

        self.search = QLineEdit(self)
        self.search.setPlaceholderText("Search conversations…")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self._on_search_changed)
        layout.addWidget(self.search)

        self.list_widget = QListWidget(self)
        self.list_widget.setContextMenuPolicy(Qt.CustomContextMenu)
        self.list_widget.customContextMenuRequested.connect(self._on_context_menu)
        self.list_widget.currentItemChanged.connect(self._on_current_changed)
        layout.addWidget(self.list_widget, stretch=1)

        self.lbl_empty = QLabel(EMPTY_TEXT, self)
        self.lbl_empty.setObjectName("sb_empty")
        self.lbl_empty.setWordWrap(True)
        self.lbl_empty.hide()
        layout.addWidget(self.lbl_empty)

        self.lbl_error = QLabel("", self)
        self.lbl_error.setObjectName("sb_error")
        self.lbl_error.setWordWrap(True)
        self.lbl_error.hide()
        layout.addWidget(self.lbl_error)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def set_history_manager(self, history_manager: Optional[Any]) -> None:
        self.history_manager = history_manager

    def refresh(self) -> None:
        """Reload the list from the database on a background worker."""
        self.load(self.history_manager, self.search.text())

    def load(self, history_manager: Optional[Any], query: str = "") -> None:
        if history_manager is not None:
            self.history_manager = history_manager
        worker = ConversationListWorker(self.history_manager, query=query, limit=self.limit)
        worker.signals.result.connect(self._on_loaded)
        worker.signals.error.connect(self._on_load_error)
        self._start_worker(worker)

    def _start_worker(self, worker: ConversationListWorker) -> None:
        """Seam so tests can run the listing synchronously."""
        QThreadPool.globalInstance().start(worker)

    def set_conversations(
        self,
        conversations: Optional[List[Dict[str, Any]]],
        now: Optional[datetime] = None,
    ) -> None:
        """Rebuild the list from conversation summaries, grouped by recency."""
        reference = now or datetime.now().astimezone()
        self.lbl_error.hide()
        self._suppress_selection = True
        try:
            self.list_widget.clear()
            groups = group_conversations(conversations, now=reference)
            for header, entries in groups:
                self._add_header(header)
                for conversation in entries:
                    self._add_conversation(conversation, reference)
        finally:
            self._suppress_selection = False

        self._update_empty_state(bool(groups))
        self.set_active(self._active_id)

    def set_active(self, conversation_id: Optional[str]) -> None:
        """Highlight the active conversation without emitting a selection."""
        self._active_id = conversation_id
        self._suppress_selection = True
        try:
            for row in range(self.list_widget.count()):
                item = self.list_widget.item(row)
                conv_id = item.data(_ROLE_CONVERSATION_ID)
                if conv_id is None:
                    continue  # group header
                is_active = conv_id == conversation_id
                font = item.font()
                font.setBold(is_active)
                item.setFont(font)
                item.setBackground(QBrush(_ACTIVE_BACKGROUND if is_active else Qt.transparent))
                item.setForeground(QBrush(_ACTIVE_FOREGROUND if is_active else _IDLE_FOREGROUND))
                if is_active:
                    self.list_widget.setCurrentItem(item)
        finally:
            self._suppress_selection = False

    def current_conversation_id(self) -> Optional[str]:
        """The id of the currently selected row, or ``None``."""
        item = self.list_widget.currentItem()
        return item.data(_ROLE_CONVERSATION_ID) if item is not None else None

    def request_rename(self, conversation_id: str) -> None:
        if conversation_id:
            self.rename_requested.emit(str(conversation_id))

    def request_delete(self, conversation_id: str) -> None:
        if conversation_id:
            self.delete_requested.emit(str(conversation_id))

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _add_header(self, text: str) -> None:
        item = QListWidgetItem(text, self.list_widget)
        font = item.font()
        font.setBold(True)
        font.setPointSize(max(1, font.pointSize() - 1))
        item.setFont(font)
        item.setForeground(QBrush(_HEADER_FOREGROUND))
        item.setFlags(Qt.NoItemFlags)  # headers are not selectable
        self.list_widget.addItem(item)

    def _add_conversation(self, conversation: Dict[str, Any], now: datetime) -> None:
        conv_id = conversation.get("id")
        title = str(conversation.get("title") or "Untitled conversation")
        activity = _activity_text(_parse_timestamp(conversation.get("last_activity")), now)
        count = conversation.get("message_count")

        label = f"{title}"
        if activity:
            label += f"\n{activity}"
        if isinstance(count, int):
            label += f"  ·  {count}"

        item = QListWidgetItem(label, self.list_widget)
        item.setData(_ROLE_CONVERSATION_ID, str(conv_id) if conv_id is not None else None)
        item.setToolTip(title)
        self.list_widget.addItem(item)

    def _update_empty_state(self, has_groups: bool) -> None:
        if has_groups:
            self.lbl_empty.hide()
            return
        self.lbl_empty.setText(
            NO_MATCHES_TEXT if (self.search.text() or "").strip() else EMPTY_TEXT
        )
        self.lbl_empty.show()

    def _on_current_changed(self, current: Optional[QListWidgetItem], _previous: Any) -> None:
        if self._suppress_selection or current is None:
            return
        conv_id = current.data(_ROLE_CONVERSATION_ID)
        if conv_id:
            self.conversation_selected.emit(str(conv_id))

    def _on_search_changed(self, _text: str) -> None:
        self.load(self.history_manager, self.search.text())

    @Slot(object)
    def _on_loaded(self, entries: Any) -> None:
        self.set_conversations(list(entries or []))

    @Slot(str)
    def _on_load_error(self, message: str) -> None:
        self._suppress_selection = True
        try:
            self.list_widget.clear()
        finally:
            self._suppress_selection = False
        self.lbl_empty.hide()
        self.lbl_error.setText(str(message) or LOAD_ERROR_TEXT)
        self.lbl_error.show()

    def _on_context_menu(self, pos: Any) -> None:
        item = self.list_widget.itemAt(pos)
        if item is None:
            return
        conv_id = item.data(_ROLE_CONVERSATION_ID)
        if not conv_id:
            return
        menu = QMenu(self)
        action_rename = menu.addAction("Rename…")
        action_delete = menu.addAction("Delete…")
        chosen = menu.exec(self.list_widget.mapToGlobal(pos))
        if chosen is action_rename:
            self.request_rename(str(conv_id))
        elif chosen is action_delete:
            self.request_delete(str(conv_id))
