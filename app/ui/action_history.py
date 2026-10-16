"""
Action History UI for Vector Desktop AI Assistant.

A read-only view over the existing Action Journal.  It reuses the existing
``HistoryManager.get_recent_tool_calls()`` API and the existing ``tool_calls``
schema -- it does not add another logging system.

Privacy: only the journal's already-sanitized ``summary`` (plus the tool name,
execution status and timestamp) is ever read out of a loaded row and displayed.
Raw tool arguments, secrets, ``ToolResult.data`` and full execution payloads are
never rendered.
"""

from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Qt, Signal, Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.memory.journal import JournalStatus


#: Status -> accent colour for the row's left border and status label.
_STATUS_COLORS: Dict[str, str] = {
    JournalStatus.SUCCESS: "#a6e3a1",
    JournalStatus.FAILED: "#f38ba8",
    JournalStatus.PERMISSION_DENIED: "#f38ba8",
    JournalStatus.CONFIRMATION_REQUIRED: "#f9e2af",
    JournalStatus.INVALID_ARGUMENTS: "#f9e2af",
    JournalStatus.TOOL_NOT_FOUND: "#f9e2af",
    JournalStatus.APPROVAL_REJECTED: "#f9e2af",
    JournalStatus.CANCELLED: "#9399b2",
}

#: The order statuses are offered in the filter, most relevant first.
_FILTER_STATUSES: List[str] = [
    JournalStatus.SUCCESS,
    JournalStatus.FAILED,
    JournalStatus.PERMISSION_DENIED,
    JournalStatus.CONFIRMATION_REQUIRED,
    JournalStatus.INVALID_ARGUMENTS,
    JournalStatus.TOOL_NOT_FOUND,
    JournalStatus.APPROVAL_REJECTED,
    JournalStatus.CANCELLED,
]

ALL_STATUSES_LABEL = "All statuses"

EMPTY_HISTORY_TEXT = "No actions yet. Executed tools will appear here."
EMPTY_FILTER_TEXT = "No actions match this filter."
LOAD_ERROR_TEXT = "Could not load action history. Please try again."


def format_timestamp(value: Any) -> str:
    """Formats a stored ISO timestamp for display, falling back to the raw text."""
    text = str(value or "")
    if not text:
        return ""
    try:
        return datetime.fromisoformat(text).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return text


class HistoryLoadSignals(QObject):
    """Signals for the background history loader."""

    result = Signal(object)
    error = Signal(str)


class HistoryLoadWorker(QRunnable):
    """
    Loads Action Journal entries off the Qt main UI thread.

    Any read failure is reported as a generic, non-sensitive message so raw
    exception details never reach the user.
    """

    def __init__(
        self,
        history_manager: Optional[Any],
        conversation_id: Optional[str] = None,
        limit: int = 100,
    ):
        super().__init__()
        self.history_manager = history_manager
        self.conversation_id = conversation_id
        self.limit = limit
        self.signals = HistoryLoadSignals()

    @Slot()
    def run(self) -> None:
        try:
            loader: Optional[Callable[..., List[Dict[str, Any]]]] = getattr(
                self.history_manager, "get_recent_tool_calls", None
            )
            entries = (
                loader(conversation_id=self.conversation_id, limit=self.limit)
                if loader is not None
                else []
            )
            self.signals.result.emit(list(entries or []))
        except Exception:
            # Deliberately generic: never surface raw exception details.
            self.signals.error.emit(LOAD_ERROR_TEXT)


class ActionHistoryRowWidget(QFrame):
    """One journal entry rendered as a card."""

    def __init__(
        self,
        timestamp: str,
        tool_name: str,
        status: str,
        summary: str,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.setFrameShape(QFrame.StyledPanel)
        status_color = _STATUS_COLORS.get(status, "#cdd6f4")

        self.setStyleSheet(f"""
            ActionHistoryRowWidget {{
                background-color: #181825;
                border: 1px solid #313244;
                border-left: 3px solid {status_color};
                border-radius: 6px;
                margin: 3px;
                padding: 6px 10px;
            }}
            QLabel#hist_tool {{
                color: #cdd6f4;
                font-weight: bold;
                font-size: 13px;
            }}
            QLabel#hist_status {{
                color: {status_color};
                font-weight: bold;
                font-size: 11px;
            }}
            QLabel#hist_time {{
                color: #7f849c;
                font-size: 11px;
            }}
            QLabel#hist_summary {{
                color: #bac2de;
                font-size: 12px;
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(3)

        top = QHBoxLayout()
        lbl_tool = QLabel(tool_name or "(unknown tool)", self)
        lbl_tool.setObjectName("hist_tool")

        lbl_status = QLabel(status or "-", self)
        lbl_status.setObjectName("hist_status")

        lbl_time = QLabel(timestamp or "", self)
        lbl_time.setObjectName("hist_time")

        top.addWidget(lbl_tool)
        top.addStretch()
        top.addWidget(lbl_time)
        top.addWidget(lbl_status)

        lbl_summary = QLabel(summary or "", self)
        lbl_summary.setObjectName("hist_summary")
        lbl_summary.setWordWrap(True)

        layout.addLayout(top)
        layout.addWidget(lbl_summary)


class ActionHistoryDialog(QDialog):
    """
    Modal Action History view.

    Shows recent journal entries (newest first), filters by execution status,
    can scope to the active conversation or show all recent actions, and can be
    refreshed.  Database reads run on a background worker so the UI stays
    responsive; failures degrade to a friendly message instead of a crash.
    """

    def __init__(
        self,
        history_manager: Optional[Any],
        conversation_id: Optional[str] = None,
        parent: Optional[QWidget] = None,
        limit: int = 100,
        auto_load: bool = True,
    ):
        super().__init__(parent)
        self.history_manager = history_manager
        self.conversation_id = conversation_id
        self.limit = limit
        self._entries: List[Dict[str, Any]] = []

        self._setup_ui()

        if auto_load:
            self.load()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _setup_ui(self) -> None:
        self.setWindowTitle("Action History — Vector")
        self.setModal(True)
        self.resize(580, 480)

        self.setStyleSheet("""
            QDialog {
                background-color: #1e1e2e;
                color: #cdd6f4;
                font-family: 'Segoe UI', sans-serif;
            }
            QLabel#heading {
                color: #89b4fa;
                font-size: 15px;
                font-weight: bold;
            }
            QLabel {
                color: #cdd6f4;
                font-size: 13px;
            }
            QLabel#empty {
                color: #7f849c;
                font-size: 13px;
                padding: 24px;
            }
            QLabel#error {
                color: #f38ba8;
                font-size: 13px;
                padding: 24px;
            }
            QComboBox {
                background-color: #313244;
                border: 1px solid #45475a;
                border-radius: 4px;
                color: #cdd6f4;
                padding: 4px 8px;
            }
            QCheckBox {
                color: #cdd6f4;
                spacing: 8px;
            }
            QListWidget {
                background-color: #181825;
                border: 1px solid #313244;
                border-radius: 6px;
                color: #cdd6f4;
            }
            QPushButton#btn_refresh {
                background-color: #313244;
                color: #cdd6f4;
                border: none;
                border-radius: 6px;
                padding: 6px 12px;
            }
            QPushButton#btn_refresh:hover {
                background-color: #45475a;
            }
            QPushButton#btn_close {
                background-color: #89b4fa;
                color: #11111b;
                font-weight: bold;
                border: none;
                border-radius: 6px;
                padding: 6px 14px;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        lbl_heading = QLabel("🕑 Action History", self)
        lbl_heading.setObjectName("heading")
        layout.addWidget(lbl_heading)

        controls = QHBoxLayout()

        lbl_filter = QLabel("Status:", self)
        self.filter_combo = QComboBox(self)
        self.filter_combo.addItem(ALL_STATUSES_LABEL, None)
        for status in _FILTER_STATUSES:
            self.filter_combo.addItem(status, status)
        self.filter_combo.currentIndexChanged.connect(self._render)

        self.chk_scope = QCheckBox("Current conversation only", self)
        self.chk_scope.setChecked(self.conversation_id is not None)
        self.chk_scope.toggled.connect(lambda _checked: self.load())

        self.btn_refresh = QPushButton("↻ Refresh", self)
        self.btn_refresh.setObjectName("btn_refresh")
        self.btn_refresh.clicked.connect(self.refresh)

        controls.addWidget(lbl_filter)
        controls.addWidget(self.filter_combo)
        controls.addWidget(self.chk_scope)
        controls.addStretch()
        controls.addWidget(self.btn_refresh)
        layout.addLayout(controls)

        self.list_widget = QListWidget(self)
        layout.addWidget(self.list_widget, stretch=1)

        self.lbl_empty = QLabel(EMPTY_HISTORY_TEXT, self)
        self.lbl_empty.setObjectName("empty")
        self.lbl_empty.setAlignment(Qt.AlignCenter)
        self.lbl_empty.hide()
        layout.addWidget(self.lbl_empty)

        self.lbl_error = QLabel("", self)
        self.lbl_error.setObjectName("error")
        self.lbl_error.setAlignment(Qt.AlignCenter)
        self.lbl_error.setWordWrap(True)
        self.lbl_error.hide()
        layout.addWidget(self.lbl_error)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_close = QPushButton("Close", self)
        btn_close.setObjectName("btn_close")
        btn_close.clicked.connect(self.reject)
        btn_row.addWidget(btn_close)
        layout.addLayout(btn_row)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def _scope_conversation_id(self) -> Optional[str]:
        """The conversation to scope to, or ``None`` for all recent actions."""
        return self.conversation_id if self.chk_scope.isChecked() else None

    def load(self) -> None:
        """Reload journal entries from the database on a background worker."""
        worker = HistoryLoadWorker(
            self.history_manager,
            conversation_id=self._scope_conversation_id(),
            limit=self.limit,
        )
        worker.signals.result.connect(self._on_loaded)
        worker.signals.error.connect(self._on_error)
        self._start_worker(worker)

    def _start_worker(self, worker: HistoryLoadWorker) -> None:
        """Seam so tests can run the load synchronously."""
        QThreadPool.globalInstance().start(worker)

    def refresh(self) -> None:
        """Refresh action; reloads the current view from the database."""
        self.load()

    @Slot(object)
    def _on_loaded(self, entries: Any) -> None:
        self._entries = list(entries or [])
        self._render()

    @Slot(str)
    def _on_error(self, message: str) -> None:
        self._entries = []
        self.list_widget.clear()
        self.lbl_empty.hide()
        self.lbl_error.setText(str(message) or LOAD_ERROR_TEXT)
        self.lbl_error.show()

    # ------------------------------------------------------------------
    # Rendering / filtering
    # ------------------------------------------------------------------
    def _visible_entries(self) -> List[Dict[str, Any]]:
        """Entries passing the active status filter, newest first (as loaded)."""
        selected = self.filter_combo.currentData()
        if not selected:
            return list(self._entries)
        return [
            entry
            for entry in self._entries
            if entry.get("execution_status") == selected
        ]

    def _render(self) -> None:
        self.lbl_error.hide()
        self.list_widget.clear()

        visible = self._visible_entries()

        if visible:
            self.lbl_empty.hide()
        elif self._entries:
            self.lbl_empty.setText(EMPTY_FILTER_TEXT)
            self.lbl_empty.show()
        else:
            self.lbl_empty.setText(EMPTY_HISTORY_TEXT)
            self.lbl_empty.show()

        for entry in visible:
            self._add_row(entry)

    def _add_row(self, entry: Dict[str, Any]) -> None:
        # Only the four display fields are read; ``arguments`` and any other
        # payload are intentionally never touched.
        row = ActionHistoryRowWidget(
            timestamp=format_timestamp(entry.get("timestamp", "")),
            tool_name=str(entry.get("tool_name", "")),
            status=str(entry.get("execution_status", "")),
            summary=str(entry.get("summary", "")),
            parent=self.list_widget,
        )
        item = QListWidgetItem(self.list_widget)
        item.setSizeHint(row.sizeHint())
        self.list_widget.addItem(item)
        self.list_widget.setItemWidget(item, row)
