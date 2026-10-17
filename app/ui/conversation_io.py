"""
Conversation Export & Import UI for Vector Desktop AI Assistant.

A small, focused dialog that wires the existing conversation backup core
(:mod:`app.memory.conversation_io`) to native file dialogs.  It mirrors the
visual language of the Settings / Health / Diagnostics / Action History dialogs
and runs file and database work on background workers so the window never
blocks.

Export always targets the conversation handed to the dialog (the active one).
Import never overwrites anything: the core assigns a fresh id on collision and
writes the whole conversation transactionally.  Imported content is treated as
inert data -- this dialog never routes it through the assistant.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal, Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.memory.conversation_io import (
    EXPORT_ERROR_TEXT,
    IMPORT_ERROR_TEXT,
    ConversationError,
    export_conversation_to_path,
    import_parsed_conversation,
    load_conversation_file,
)

EXPORT_SUCCESS_TEXT = "Conversation exported successfully."
IMPORT_SUCCESS_TEXT = "Conversation imported successfully."
LOAD_ERROR_TEXT = "Could not load the conversation details. Please try again."
EXPORTING_TEXT = "Exporting…"
IMPORTING_TEXT = "Importing…"
GREETING_TEXT = "Back up a conversation to a file, or restore one from a file."

EXPORT_DIALOG_TITLE = "Export Conversation"
IMPORT_DIALOG_TITLE = "Import Conversation"
JSON_FILTER = "Vector conversation (*.json);;All files (*)"


@dataclass
class ImportOutcome:
    """Result of a successful import."""

    conversation_id: str
    title: str
    message_count: int


class _ConversationSignals(QObject):
    """Signals shared by the export and import workers."""

    result = Signal(object)
    error = Signal(str)


class ConversationExportWorker(QRunnable):
    """Serializes and writes one conversation off the Qt main UI thread."""

    def __init__(self, history_manager: Any, conversation_id: Optional[str], path: str):
        super().__init__()
        self.history_manager = history_manager
        self.conversation_id = conversation_id
        self.path = path
        self.signals = _ConversationSignals()

    @Slot()
    def run(self) -> None:
        try:
            written = export_conversation_to_path(
                self.history_manager, self.conversation_id, self.path
            )
            self.signals.result.emit(written)
        except ConversationError as exc:
            self.signals.error.emit(exc.message)
        except Exception:
            self.signals.error.emit(EXPORT_ERROR_TEXT)


class ConversationImportWorker(QRunnable):
    """Reads, validates and transactionally stores a conversation file."""

    def __init__(self, history_manager: Any, path: str):
        super().__init__()
        self.history_manager = history_manager
        self.path = path
        self.signals = _ConversationSignals()

    @Slot()
    def run(self) -> None:
        try:
            parsed = load_conversation_file(self.path)
            new_id = import_parsed_conversation(self.history_manager, parsed)
            self.signals.result.emit(
                ImportOutcome(
                    conversation_id=new_id,
                    title=parsed.title,
                    message_count=len(parsed.messages),
                )
            )
        except ConversationError as exc:
            self.signals.error.emit(exc.message)
        except Exception:
            self.signals.error.emit(IMPORT_ERROR_TEXT)


class ConversationIODialog(QDialog):
    """
    Modal Conversation Backup dialog.

    Signals
    -------
    conversation_imported(str)
        Emitted with the new conversation id after a successful import, so the
        main window can open the restored conversation through the existing
        conversation workflow.  Only emitted when the "open after import"
        option is enabled.
    """

    conversation_imported = Signal(str)

    def __init__(
        self,
        history_manager: Any,
        conversation_id: Optional[str] = None,
        parent: Optional[QWidget] = None,
        auto_load: bool = True,
    ):
        super().__init__(parent)
        self.history_manager = history_manager
        self.conversation_id = conversation_id
        self.last_imported_id: Optional[str] = None

        self._setup_ui()
        if auto_load:
            self.refresh()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _setup_ui(self) -> None:
        self.setWindowTitle("Conversation Backup — Vector")
        self.setModal(True)
        self.resize(560, 380)

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
            QLabel#section { color: #cdd6f4; font-size: 13px; font-weight: bold; }
            QLabel#note { color: #7f849c; font-size: 11px; }
            QLabel#summary { color: #bac2de; font-size: 12px; }
            QLabel#status { color: #a6adc8; font-size: 12px; }
            QLabel#status[error="true"] { color: #f38ba8; }
            QLabel#status[error="false"] { color: #a6adc8; }
            QCheckBox { color: #cdd6f4; spacing: 8px; }
            QPushButton {
                background-color: #313244;
                color: #cdd6f4;
                border: none;
                border-radius: 6px;
                padding: 6px 12px;
            }
            QPushButton:hover { background-color: #45475a; }
            QPushButton:disabled { color: #6c7086; }
            QPushButton#btn_primary {
                background-color: #89b4fa;
                color: #11111b;
                font-weight: bold;
            }
            QPushButton#btn_primary:hover { background-color: #74a0e8; }
            QWidget#card {
                background-color: #181825;
                border: 1px solid #313244;
                border-radius: 6px;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        lbl_heading = QLabel("💾 Conversation Backup", self)
        lbl_heading.setObjectName("heading")
        layout.addWidget(lbl_heading)

        lbl_greeting = QLabel(GREETING_TEXT, self)
        lbl_greeting.setObjectName("note")
        lbl_greeting.setWordWrap(True)
        layout.addWidget(lbl_greeting)

        # --- Export card -------------------------------------------------
        export_card = QWidget(self)
        export_card.setObjectName("card")
        export_layout = QVBoxLayout(export_card)
        export_layout.setContentsMargins(12, 10, 12, 10)
        export_layout.setSpacing(6)

        lbl_export = QLabel("Export active conversation", export_card)
        lbl_export.setObjectName("section")

        self.lbl_summary = QLabel("", export_card)
        self.lbl_summary.setObjectName("summary")
        self.lbl_summary.setWordWrap(True)

        self.btn_export = QPushButton("⬇ Export…", export_card)
        self.btn_export.clicked.connect(self.export_active)

        export_row = QHBoxLayout()
        export_row.addWidget(self.lbl_summary, stretch=1)
        export_row.addWidget(self.btn_export)

        export_layout.addWidget(lbl_export)
        export_layout.addLayout(export_row)
        layout.addWidget(export_card)

        # --- Import card -------------------------------------------------
        import_card = QWidget(self)
        import_card.setObjectName("card")
        import_layout = QVBoxLayout(import_card)
        import_layout.setContentsMargins(12, 10, 12, 10)
        import_layout.setSpacing(6)

        lbl_import = QLabel("Import conversation from file", import_card)
        lbl_import.setObjectName("section")

        lbl_import_note = QLabel(
            "Imported conversations are added as new conversations. Existing "
            "conversations are never overwritten, and imported text is only "
            "stored — it is never executed.",
            import_card,
        )
        lbl_import_note.setObjectName("note")
        lbl_import_note.setWordWrap(True)

        self.chk_open_imported = QCheckBox("Open the imported conversation", import_card)
        self.chk_open_imported.setChecked(True)

        self.btn_import = QPushButton("⬆ Import…", import_card)
        self.btn_import.clicked.connect(self.import_from_file)

        import_row = QHBoxLayout()
        import_row.addWidget(self.chk_open_imported)
        import_row.addStretch()
        import_row.addWidget(self.btn_import)

        import_layout.addWidget(lbl_import)
        import_layout.addWidget(lbl_import_note)
        import_layout.addLayout(import_row)
        layout.addWidget(import_card)

        layout.addStretch()

        self.lbl_status = QLabel("", self)
        self.lbl_status.setObjectName("status")
        self.lbl_status.setProperty("error", "false")
        self.lbl_status.setWordWrap(True)
        layout.addWidget(self.lbl_status)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_close = QPushButton("Close", self)
        btn_close.setObjectName("btn_primary")
        btn_close.clicked.connect(self.reject)
        btn_row.addWidget(btn_close)
        layout.addLayout(btn_row)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    def refresh(self) -> None:
        """Refresh the export summary from the active conversation."""
        self.btn_export.setEnabled(bool(self.conversation_id))
        if not self.conversation_id:
            self.lbl_summary.setText("No active conversation.")
            return
        try:
            metadata = None
            getter = getattr(self.history_manager, "get_conversation", None)
            if getter is not None:
                metadata = getter(self.conversation_id)
            count = len(
                self.history_manager.get_conversation_messages(self.conversation_id, limit=None)
            )
        except Exception:
            self.lbl_summary.setText(LOAD_ERROR_TEXT)
            return

        title = ""
        if isinstance(metadata, dict):
            title = str(metadata.get("title") or "")
        title = title or "Untitled conversation"
        self.lbl_summary.setText(f"{title} — {count} message(s)")

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    def _choose_export_path(self) -> str:
        """Native save dialog. Seam so tests can bypass the modal dialog."""
        default_name = datetime.now().strftime("vector-conversation-%Y%m%d-%H%M%S.json")
        path, _ = QFileDialog.getSaveFileName(
            self, EXPORT_DIALOG_TITLE, default_name, JSON_FILTER
        )
        return path or ""

    def export_active(self) -> None:
        """Prompt for a destination and export the active conversation."""
        if not self.conversation_id:
            self._set_status("No active conversation to export.", error=True)
            return
        path = self._choose_export_path()
        if not path:
            return  # cancelled
        self._export_to(path)

    def _export_to(self, path: str) -> None:
        self._set_busy(True)
        self._set_status(EXPORTING_TEXT, error=False)
        worker = ConversationExportWorker(self.history_manager, self.conversation_id, path)
        worker.signals.result.connect(self._on_exported)
        worker.signals.error.connect(self._on_error)
        self._start_export_worker(worker)

    def _start_export_worker(self, worker: ConversationExportWorker) -> None:
        """Seam so tests can run the export synchronously."""
        QThreadPool.globalInstance().start(worker)

    @Slot(object)
    def _on_exported(self, path: Any) -> None:
        self._set_busy(False)
        self._set_status(EXPORT_SUCCESS_TEXT, error=False)

    # ------------------------------------------------------------------
    # Import
    # ------------------------------------------------------------------
    def _choose_import_path(self) -> str:
        """Native open dialog. Seam so tests can bypass the modal dialog."""
        path, _ = QFileDialog.getOpenFileName(
            self, IMPORT_DIALOG_TITLE, str(Path.home()), JSON_FILTER
        )
        return path or ""

    def import_from_file(self) -> None:
        """Prompt for a file and import the conversation it contains."""
        path = self._choose_import_path()
        if not path:
            return  # cancelled
        self._import_from(path)

    def _import_from(self, path: str) -> None:
        self._set_busy(True)
        self._set_status(IMPORTING_TEXT, error=False)
        worker = ConversationImportWorker(self.history_manager, path)
        worker.signals.result.connect(self._on_imported)
        worker.signals.error.connect(self._on_error)
        self._start_import_worker(worker)

    def _start_import_worker(self, worker: ConversationImportWorker) -> None:
        """Seam so tests can run the import synchronously."""
        QThreadPool.globalInstance().start(worker)

    @Slot(object)
    def _on_imported(self, outcome: Any) -> None:
        self._set_busy(False)
        if not isinstance(outcome, ImportOutcome):
            self._set_status(IMPORT_ERROR_TEXT, error=True)
            return

        self.last_imported_id = outcome.conversation_id
        self._set_status(
            f"{IMPORT_SUCCESS_TEXT} {outcome.message_count} message(s) restored "
            f"as “{outcome.title}”.",
            error=False,
        )
        if self.chk_open_imported.isChecked():
            self.conversation_imported.emit(outcome.conversation_id)

    @Slot(str)
    def _on_error(self, message: str) -> None:
        self._set_busy(False)
        self._set_status(str(message) or IMPORT_ERROR_TEXT, error=True)

    # ------------------------------------------------------------------
    # Small helpers
    # ------------------------------------------------------------------
    def _set_busy(self, busy: bool) -> None:
        self.btn_export.setEnabled(not busy and bool(self.conversation_id))
        self.btn_import.setEnabled(not busy)

    def _set_status(self, text: str, error: bool = False) -> None:
        self.lbl_status.setProperty("error", "true" if error else "false")
        self.lbl_status.setText(text)
        # Re-polish so the [error=...] selector re-applies.
        self.lbl_status.style().unpolish(self.lbl_status)
        self.lbl_status.style().polish(self.lbl_status)
