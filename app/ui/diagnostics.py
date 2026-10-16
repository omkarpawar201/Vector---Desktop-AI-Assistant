"""
Diagnostics & Troubleshooting for Vector Desktop AI Assistant.

Generates a single, compact, human-readable report that helps a user describe a
problem when filing an issue.  The feature is deliberately conservative:

* **Strictly read-only.**  It never executes tools or shell commands, never
  modifies configuration, never installs anything, never loads an AI model and
  never makes a network request.
* **Reuses existing machinery.**  It calls the existing
  :class:`~app.ui.system_health.SystemHealthCollector` for subsystem statuses and
  the existing :class:`~app.core.privacy.PrivacyFilter` for secret redaction.  It
  does not introduce a parallel diagnostics framework.
* **Privacy first.**  Credentials, tokens, passwords, transcripts, conversation
  history, raw tool arguments and full user-directory paths are never included.
  Every dynamic field is path-sanitized, and the finished report is passed
  through :class:`PrivacyFilter` once more as defense in depth.
* **Fault-isolated.**  One failing check (a missing module, an unreadable
  database, an error source that raises) never prevents the rest of the report
  from being produced.

Recent errors are only reported when a safe, in-memory error source is
available (see :class:`RecentErrorLog`); otherwise the section reads
"Not available" rather than guessing.
"""

from __future__ import annotations

import logging
import platform
import re
import sqlite3
import sys
import threading
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal, Slot
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.config import APP_NAME, APP_VERSION
from app.config.settings import Settings, get_settings
from app.core.privacy import PrivacyFilter
from app.ui.system_health import (
    SystemHealthCollector,
    _offline_stt_files_present,
    module_available,
)

#: Shown when the whole report cannot be produced.
DIAGNOSTICS_ERROR_TEXT = "Could not generate the diagnostics report. Please try again."
SAVE_ERROR_TEXT = "Could not save the report. Please try again."
SAVE_SUCCESS_TEXT = "Report saved successfully."
COPY_SUCCESS_TEXT = "Report copied to clipboard."
COPY_ERROR_TEXT = "Could not copy the report to the clipboard."
GENERATING_TEXT = "Generating report…"

NOT_AVAILABLE = "Not available"

#: Optional runtime dependencies worth surfacing when troubleshooting.
#: ``(import name, human label)`` pairs.
OPTIONAL_DEPENDENCIES: List[Tuple[str, str]] = [
    ("needle", "Local inference engine (needle)"),
    ("sherpa_onnx", "Offline STT engine (sherpa-onnx)"),
    ("faster_whisper", "Offline STT engine (faster-whisper)"),
    ("speech_recognition", "Online STT (SpeechRecognition)"),
    ("pyttsx3", "Text-to-speech (pyttsx3)"),
    ("psutil", "System metrics (psutil)"),
    ("google.genai", "Gemini cloud client (google-genai)"),
    ("numpy", "Numerical support (numpy)"),
]

#: Table names the application's database schema is expected to contain.
_EXPECTED_TABLES = ("conversations", "messages", "tool_calls", "user_preferences")


def sanitize_path(value: Any) -> str:
    """
    Redact user-directory components from a string.

    The home directory is collapsed to ``~`` and any ``\\Users\\<name>`` /
    ``/home/<name>`` / ``/Users/<name>`` segment is collapsed to ``<user>`` so a
    report never leaks a full personal path.
    """
    if value is None:
        return ""
    try:
        text = str(value)
    except Exception:
        return ""
    if not text:
        return ""

    try:
        home = str(Path.home())
    except Exception:
        home = ""

    if home and home not in ("/", "\\"):
        for variant in {home, home.replace("\\", "/"), home.replace("/", "\\")}:
            if variant:
                text = text.replace(variant, "~")

    # Belt-and-braces for paths that are not under *this* user's home.
    text = re.sub(
        r"(?i)([A-Za-z]:[\\/]Users[\\/]|/(?:home|Users)/)[^\\/\s]+",
        r"\1<user>",
        text,
    )
    return text


class RecentErrorLog:
    """
    A tiny, bounded, in-memory sink for recent ERROR/CRITICAL log records.

    This is an *error source*, not a telemetry system: records never leave the
    process, no file is written, no network call is made, and only a short
    sanitized message plus a logger name and level are kept (no tracebacks, no
    arguments).  It is safe to leave attached for the whole application
    lifetime.
    """

    MAX_ENTRIES = 10
    MAX_MESSAGE_LENGTH = 240

    _lock = threading.Lock()
    _entries: "deque[Dict[str, str]]" = deque(maxlen=MAX_ENTRIES)
    _handler: Optional[logging.Handler] = None

    @classmethod
    def install(cls, logger: Optional[logging.Logger] = None) -> None:
        """Attach the in-memory sink to a logger (root by default). Idempotent."""
        target = logger or logging.getLogger()
        with cls._lock:
            if cls._handler is not None:
                return
            handler = _ErrorRingHandler(cls)
            handler.setLevel(logging.ERROR)
            target.addHandler(handler)
            cls._handler = handler

    @classmethod
    def is_available(cls) -> bool:
        """True once the sink has been attached to a logger."""
        return cls._handler is not None

    @classmethod
    def recent(cls) -> List[Dict[str, str]]:
        """A copy of the most recent sanitized error records, newest last."""
        with cls._lock:
            return list(cls._entries)

    @classmethod
    def clear(cls) -> None:
        with cls._lock:
            cls._entries.clear()

    @classmethod
    def _record(cls, record: logging.LogRecord) -> None:
        try:
            raw_message = record.getMessage()
        except Exception:
            raw_message = ""
        # Sanitize and truncate at capture time so a secret can never sit in the
        # buffer, even transiently, and a huge message cannot exhaust memory.
        message = PrivacyFilter.sanitize_text(str(raw_message or ""))
        message = sanitize_path(message)
        if len(message) > cls.MAX_MESSAGE_LENGTH:
            message = message[: cls.MAX_MESSAGE_LENGTH].rstrip() + "…"
        entry = {
            "level": str(record.levelname or "ERROR"),
            "logger": str(record.name or ""),
            "message": message,
        }
        with cls._lock:
            cls._entries.append(entry)


class _ErrorRingHandler(logging.Handler):
    """Logging handler that forwards ERROR+ records to :class:`RecentErrorLog`."""

    def __init__(self, log_cls: type):
        super().__init__(level=logging.ERROR)
        self._log_cls = log_cls

    def emit(self, record: logging.LogRecord) -> None:
        # Never let logging raise: a diagnostics sink must not break the app.
        try:
            self._log_cls._record(record)
        except Exception:
            pass


class DiagnosticsCollector:
    """
    Builds the diagnostic report text from existing sources only.

    All inputs are injectable so the collector can be exercised without a real
    model, network, microphone, database or credentials.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        history_manager: Optional[Any] = None,
        health_collector: Optional[SystemHealthCollector] = None,
        error_provider: Optional[Callable[[], List[Dict[str, str]]]] = None,
    ):
        self.settings = settings or get_settings()
        self.history_manager = history_manager
        self.health_collector = health_collector or SystemHealthCollector(
            settings=self.settings, history_manager=history_manager
        )
        self._error_provider = error_provider

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def collect(self) -> str:
        """Return the finished, sanitized report text."""
        sections: List[Tuple[str, List[str]]] = [
            ("Application", self._application_lines()),
            ("AI and Routing Configuration", self._configuration_lines()),
            ("System Health", self._health_lines()),
            ("Optional Dependencies", self._dependency_lines()),
            ("Model and Offline STT Assets", self._asset_lines()),
            ("Database", self._database_lines()),
            ("Recent Errors", self._error_lines()),
        ]
        return self._format(sections)

    # ------------------------------------------------------------------
    # Section builders (each independently fault-isolated by _safe)
    # ------------------------------------------------------------------
    def _application_lines(self) -> List[str]:
        return self._safe(
            lambda: [
                f"Name: {APP_NAME}",
                f"Version: {APP_VERSION}",
                f"Python: {platform.python_version()}",
                f"Platform: {platform.platform()}",
                f"Architecture: {platform.machine()} ({8 * _pointer_size()}-bit)",
            ],
            "Application information",
        )

    def _configuration_lines(self) -> List[str]:
        s = self.settings

        def build() -> List[str]:
            lines = [
                f"Gemini cloud fallback enabled: {_yes_no(getattr(s, 'enable_gemini', None))}",
                f"Gemini credential configured: {_yes_no(bool(str(getattr(s, 'gemini_api_key', '') or '').strip()))}",
                f"Gemini model: {getattr(s, 'gemini_model', NOT_AVAILABLE)}",
                f"Local LLM (Ollama) enabled: {_yes_no(getattr(s, 'enable_local_llm', None))}",
                f"Voice pipeline enabled: {_yes_no(getattr(s, 'enable_voice', None))}",
                f"STT backend mode: {getattr(s, 'voice_stt_backend', NOT_AVAILABLE)}",
                f"Needle generation: {getattr(s, 'needle_generation', NOT_AVAILABLE)}",
                f"Log level: {getattr(s, 'log_level', NOT_AVAILABLE)}",
                f"Database path: {sanitize_path(getattr(s, 'database_path', ''))}",
            ]
            return lines

        return self._safe(build, "Configuration")

    def _health_lines(self) -> List[str]:
        def build() -> List[str]:
            lines: List[str] = []
            for section in self.health_collector.collect():
                lines.append(f"{section.title}:")
                for item in section.items:
                    detail = f" ({item.detail})" if item.detail else ""
                    lines.append(f"  {item.label}: {item.status}{detail}")
            return lines

        return self._safe(build, "System health")

    def _dependency_lines(self) -> List[str]:
        def build() -> List[str]:
            return [
                f"{label}: {'Available' if module_available(name) else 'Not installed'}"
                for name, label in OPTIONAL_DEPENDENCIES
            ]

        return self._safe(build, "Optional dependencies")

    def _asset_lines(self) -> List[str]:
        s = self.settings

        def build() -> List[str]:
            lines = [
                f"Tier 1 tuned weights: {_file_state(getattr(s, 'needle_tuned_weights_path', ''))}",
                f"Tier 0.5 custom ONNX: {_file_state(getattr(s, 'needle_model_path', ''))}",
                f"Official base weights: {_file_state(getattr(s, 'needle_official_weights_path', ''))}",
            ]
            stt_dir = str(getattr(s, "voice_stt_model_dir", "") or "")
            try:
                layout_ok = bool(stt_dir) and _offline_stt_files_present(Path(stt_dir))
            except Exception:
                layout_ok = False
            lines.append(
                f"Offline STT model files: {'Present' if layout_ok else 'Missing'}"
            )
            return lines

        return self._safe(build, "Model assets")

    def _database_lines(self) -> List[str]:
        status, tables = self._inspect_database()
        lines = [f"Status: {status}"]
        if tables is None:
            lines.append("Tables found: Unknown")
        elif tables:
            lines.append(f"Tables found: {', '.join(tables)}")
        else:
            lines.append("Tables found: none")
        missing = [t for t in _EXPECTED_TABLES if tables is not None and t not in tables]
        if tables is not None:
            lines.append(f"Missing expected tables: {', '.join(missing) if missing else 'none'}")
        return lines

    def _error_lines(self) -> List[str]:
        provider = self._error_provider
        if provider is None:
            if not RecentErrorLog.is_available():
                return [f"{NOT_AVAILABLE} (no error source)"]
            provider = RecentErrorLog.recent

        try:
            entries = provider()
        except Exception:
            return [NOT_AVAILABLE]

        if not entries:
            return ["None recorded this session"]

        lines: List[str] = []
        for entry in entries[: RecentErrorLog.MAX_ENTRIES]:
            try:
                level = str(entry.get("level", "ERROR")).upper()
                logger_name = str(entry.get("logger", "") or "")
                message = sanitize_path(PrivacyFilter.sanitize_text(str(entry.get("message", ""))))
            except Exception:
                continue
            if logger_name:
                lines.append(f"{level} [{logger_name}]: {message}")
            else:
                lines.append(f"{level}: {message}")
        return lines or ["None recorded this session"]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _inspect_database(self) -> Tuple[str, Optional[List[str]]]:
        """
        Return ``(status, table_names_or_None)`` for the history database.

        Prefers the injected manager's own connection so the probe reads the same
        database the app uses; falls back to opening the configured path.
        """
        conn_cm = getattr(getattr(self.history_manager, "db", None), "get_connection", None)
        try:
            if conn_cm is not None:
                with conn_cm() as conn:
                    tables = _read_table_names(conn)
                return "Available (readable)", tables

            path = Path(str(self.settings.database_path))
            if not path.exists():
                return "Not configured (not created yet)", None
            conn = sqlite3.connect(str(path))
            try:
                tables = _read_table_names(conn)
            finally:
                conn.close()
            return "Available (readable)", tables
        except Exception:
            return "Unavailable (read failed)", None

    @staticmethod
    def _safe(builder: Callable[[], List[str]], label: str) -> List[str]:
        """Run a section builder; degrade to a single line on failure."""
        try:
            lines = builder()
            return lines if lines else [NOT_AVAILABLE]
        except Exception:
            return [f"{NOT_AVAILABLE} ({label} could not be collected)"]

    def _format(self, sections: List[Tuple[str, List[str]]]) -> str:
        rule = "=" * 60
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        out: List[str] = [
            rule,
            f" {APP_NAME} Diagnostics Report",
            f" Generated: {timestamp}",
            rule,
            "",
        ]
        for title, body in sections:
            out.append(f"[{title}]")
            out.extend(f"  {line}" for line in (body or [NOT_AVAILABLE]))
            out.append("")

        text = "\n".join(out).rstrip() + "\n"
        # Defense in depth: sanitize personal paths, then any remaining secrets.
        text = sanitize_path(text)
        return PrivacyFilter.sanitize_text(text)


def _yes_no(value: Any) -> str:
    if value is None:
        return NOT_AVAILABLE
    return "yes" if bool(value) else "no"


def _file_state(raw_path: Any) -> str:
    path = str(raw_path or "")
    if not path:
        return "Not configured"
    try:
        return "Present" if Path(path).is_file() else "Missing"
    except Exception:
        return "Unknown"


def _pointer_size() -> int:
    return 8 if sys.maxsize > 2 ** 32 else 4


def _read_table_names(conn: sqlite3.Connection) -> List[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()
    return [str(r[0]) for r in rows]


# --------------------------------------------------------------------------
# Qt workers
# --------------------------------------------------------------------------
class DiagnosticsSignals(QObject):
    """Signals for the background report generator."""

    result = Signal(str)
    error = Signal(str)


class DiagnosticsWorker(QRunnable):
    """Generates the diagnostic report off the Qt main UI thread."""

    def __init__(self, collector: DiagnosticsCollector):
        super().__init__()
        self.collector = collector
        self.signals = DiagnosticsSignals()

    @Slot()
    def run(self) -> None:
        try:
            self.signals.result.emit(self.collector.collect())
        except Exception:
            # Deliberately generic: never surface raw exception details.
            self.signals.error.emit(DIAGNOSTICS_ERROR_TEXT)


class SaveReportSignals(QObject):
    """Signals for the background report writer."""

    result = Signal(str)
    error = Signal(str)


class SaveReportWorker(QRunnable):
    """Writes the finished report to disk off the Qt main UI thread."""

    def __init__(self, path: str, text: str):
        super().__init__()
        self.path = path
        self.text = text
        self.signals = SaveReportSignals()

    @Slot()
    def run(self) -> None:
        try:
            Path(self.path).write_text(self.text, encoding="utf-8")
            self.signals.result.emit(self.path)
        except Exception:
            self.signals.error.emit(SAVE_ERROR_TEXT)


# --------------------------------------------------------------------------
# Dialog
# --------------------------------------------------------------------------
class DiagnosticsDialog(QDialog):
    """
    Modal Diagnostics & Troubleshooting view.

    The user can generate a report, preview it, copy it to the clipboard, and
    save it as a ``.txt`` file.  Generation and file writes run on background
    workers so the window stays responsive; failures surface as friendly,
    non-sensitive messages.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        history_manager: Optional[Any] = None,
        parent: Optional[QWidget] = None,
        auto_generate: bool = True,
    ):
        super().__init__(parent)
        self.collector = DiagnosticsCollector(
            settings=settings, history_manager=history_manager
        )
        self._report: str = ""
        self._setup_ui()

        if auto_generate:
            self.generate()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _setup_ui(self) -> None:
        self.setWindowTitle(f"📋 Diagnostics — {APP_NAME}")
        self.setModal(True)
        self.resize(640, 520)

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
            QLabel#note { color: #7f849c; font-size: 11px; }
            QLabel#status { color: #a6adc8; font-size: 12px; }
            QLabel#status[error="true"] { color: #f38ba8; }
            QLabel#status[error="false"] { color: #a6adc8; }
            QPlainTextEdit {
                background-color: #181825;
                border: 1px solid #313244;
                border-radius: 6px;
                color: #cdd6f4;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 12px;
                padding: 8px;
            }
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
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        lbl_heading = QLabel("📋 Diagnostics & Troubleshooting", self)
        lbl_heading.setObjectName("heading")
        layout.addWidget(lbl_heading)

        lbl_note = QLabel(
            "Read-only report. No credentials, history, transcripts or prompts are included.",
            self,
        )
        lbl_note.setObjectName("note")
        lbl_note.setWordWrap(True)
        layout.addWidget(lbl_note)

        self.preview = QPlainTextEdit(self)
        self.preview.setReadOnly(True)
        self.preview.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.preview.setPlaceholderText(GENERATING_TEXT)
        layout.addWidget(self.preview, stretch=1)

        self.lbl_status = QLabel("", self)
        self.lbl_status.setObjectName("status")
        self.lbl_status.setProperty("error", "false")
        layout.addWidget(self.lbl_status)

        btn_row = QHBoxLayout()
        self.btn_generate = QPushButton("↻ Generate", self)
        self.btn_generate.clicked.connect(self.generate)
        self.btn_copy = QPushButton("📋 Copy", self)
        self.btn_copy.clicked.connect(self.copy_to_clipboard)
        self.btn_save = QPushButton("💾 Save as .txt…", self)
        self.btn_save.clicked.connect(self.save_as)
        btn_close = QPushButton("Close", self)
        btn_close.setObjectName("btn_primary")
        btn_close.clicked.connect(self.reject)

        btn_row.addWidget(self.btn_generate)
        btn_row.addWidget(self.btn_copy)
        btn_row.addWidget(self.btn_save)
        btn_row.addStretch()
        btn_row.addWidget(btn_close)
        layout.addLayout(btn_row)

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    def generate(self) -> None:
        """Generate the report on a background worker."""
        self._set_busy(True)
        self._set_status(GENERATING_TEXT)
        worker = DiagnosticsWorker(self.collector)
        worker.signals.result.connect(self._on_report)
        worker.signals.error.connect(self._on_error)
        self._start_worker(worker)

    def _start_worker(self, worker: DiagnosticsWorker) -> None:
        """Seam so tests can run generation synchronously."""
        QThreadPool.globalInstance().start(worker)

    @Slot(str)
    def _on_report(self, text: str) -> None:
        self._report = str(text or "")
        self.preview.setPlainText(self._report)
        self._set_busy(False)
        self._set_status("")

    @Slot(str)
    def _on_error(self, message: str) -> None:
        self._report = ""
        self.preview.setPlainText("")
        self._set_busy(False)
        self._set_status(str(message) or DIAGNOSTICS_ERROR_TEXT, error=True)

    # ------------------------------------------------------------------
    # Clipboard / save
    # ------------------------------------------------------------------
    def _clipboard(self):
        """Seam so tests can inject a fake clipboard."""
        return QGuiApplication.clipboard()

    def copy_to_clipboard(self) -> None:
        """Copy the current report to the system clipboard."""
        if not self._report:
            self._set_status("Nothing to copy yet.", error=True)
            return
        try:
            self._clipboard().setText(self._report)
            self._set_status(COPY_SUCCESS_TEXT)
        except Exception:
            self._set_status(COPY_ERROR_TEXT, error=True)

    def save_as(self) -> None:
        """Prompt for a path, then write the report on a background worker."""
        if not self._report:
            self._set_status("Nothing to save yet.", error=True)
            return
        default_name = datetime.now().strftime("vector-diagnostics-%Y%m%d-%H%M%S.txt")
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Diagnostics Report",
            default_name,
            "Text files (*.txt);;All files (*)",
        )
        if not path:
            return
        self._save_report_to(path)

    def _save_report_to(self, path: str) -> None:
        """Write the report to ``path`` on a background worker."""
        worker = SaveReportWorker(path, self._report)
        worker.signals.result.connect(self._on_saved)
        worker.signals.error.connect(self._on_save_error)
        self._start_save_worker(worker)

    def _start_save_worker(self, worker: SaveReportWorker) -> None:
        """Seam so tests can run the file write synchronously."""
        QThreadPool.globalInstance().start(worker)

    @Slot(str)
    def _on_saved(self, path: str) -> None:
        self._set_status(SAVE_SUCCESS_TEXT)

    @Slot(str)
    def _on_save_error(self, message: str) -> None:
        self._set_status(str(message) or SAVE_ERROR_TEXT, error=True)

    # ------------------------------------------------------------------
    # Small helpers
    # ------------------------------------------------------------------
    def _set_busy(self, busy: bool) -> None:
        enabled = not busy
        self.btn_generate.setEnabled(enabled)
        self.btn_copy.setEnabled(enabled)
        self.btn_save.setEnabled(enabled)

    def _set_status(self, text: str, error: bool = False) -> None:
        self.lbl_status.setProperty("error", "true" if error else "false")
        self.lbl_status.setText(text)
        # Re-polish so the [error=...] style selector re-applies.
        self.lbl_status.style().unpolish(self.lbl_status)
        self.lbl_status.style().polish(self.lbl_status)
