"""
Settings & Preferences dialog for Vector Desktop AI Assistant.

A tabbed, validated preferences UI over the *existing* configuration system.
It exposes only settings the codebase genuinely supports and reads at runtime,
persists them through :class:`~app.config.store.SettingsStore` (the write half of
the existing ``.env`` + ``pydantic-settings`` mechanism), and is explicit about
which changes take effect immediately versus after a restart.

Security: the dialog cannot weaken any safety control.  Tool permissions,
argument validation and approval binding are enforced in ``app/security``; the
only policy toggle surfaced is ``require_confirmation_for_dangerous`` (which
affects DANGEROUS-level tools only) and it is clearly described.  The Gemini API
key is treated as session-only and is never written to disk.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Qt, Signal, Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from app.config import APP_NAME, APP_VERSION
from app.config.settings import Settings
from app.config.store import (
    SAVE_ERROR_TEXT,
    SESSION_ONLY_FIELDS,
    SettingsStore,
)

#: Success and status messages (module-level so tests can reference them).
SAVE_SUCCESS_TEXT = "Settings saved."
NO_CHANGES_TEXT = "No changes to apply."
RESTART_HINT_TEMPLATE = "Saved. Restart Vector to apply: {fields}."
LOAD_ERROR_TEXT = SAVE_ERROR_TEXT

#: The STT backends supported by ``app/voice/stt.py`` (voice_stt_backend Literal).
_STT_BACKENDS = [
    ("Online (default)", "online"),
    ("Offline (local only)", "offline"),
    ("Auto (prefer offline)", "auto"),
]

#: Logging levels accepted by ``app.main`` via ``getattr(logging, level)``.
_LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

#: Human-readable field names for user-facing messages.
FIELD_LABELS: Dict[str, str] = {
    "enable_gemini": "Gemini cloud fallback",
    "gemini_model": "Gemini model",
    "gemini_api_key": "Gemini API key",
    "enable_voice": "Voice pipeline",
    "voice_stt_backend": "Speech-to-text backend",
    "voice_stt_model_dir": "Offline STT model folder",
    "require_confirmation_for_dangerous": "Dangerous-action confirmation",
    "log_level": "Log level",
    "database_path": "Database path",
}


class _SaveSignals(QObject):
    """Signals for the background settings writer."""

    result = Signal(object)
    error = Signal(str)


class SettingsSaveWorker(QRunnable):
    """Validates and persists settings off the Qt main UI thread."""

    def __init__(self, store: SettingsStore, changes: Dict[str, Any]):
        super().__init__()
        self.store = store
        self.changes = changes
        self.signals = _SaveSignals()

    @Slot()
    def run(self) -> None:
        try:
            self.signals.result.emit(self.store.apply(self.changes))
        except Exception:
            # Deliberately generic: never surface raw exception details.
            self.signals.error.emit(SAVE_ERROR_TEXT)


def _make_field(label: str, control: QWidget, description: str = "") -> QWidget:
    """A labelled control with an optional muted description beneath it."""
    box = QWidget()
    layout = QVBoxLayout(box)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(2)

    lbl = QLabel(label, box)
    lbl.setObjectName("field_label")
    layout.addWidget(lbl)
    layout.addWidget(control)

    if description:
        desc = QLabel(description, box)
        desc.setObjectName("field_desc")
        desc.setWordWrap(True)
        layout.addWidget(desc)
    return box


def _info_label(text: str) -> QLabel:
    """A muted, read-only informational line."""
    lbl = QLabel(text)
    lbl.setObjectName("field_desc")
    lbl.setWordWrap(True)
    return lbl


class SettingsDialog(QDialog):
    """
    Modal Settings & Preferences dialog.

    Organised into AI & Routing, Voice, Privacy & Permissions, Appearance and
    General tabs.  ``Apply`` persists and keeps the dialog open; ``Save``
    persists and closes; ``Cancel`` discards unsaved changes.
    """

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        store: Optional[SettingsStore] = None,
        env_path: Optional[str] = None,
        settings: Optional[Settings] = None,
    ):
        super().__init__(parent)
        self.store = store or SettingsStore(env_path=env_path, settings=settings)
        self.settings = self.store.settings
        self._loading = False
        self._initial: Dict[str, Any] = {}

        self._setup_ui()
        self._resync_from_settings()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _setup_ui(self) -> None:
        self.setWindowTitle(f"⚙ Settings — {APP_NAME}")
        self.setModal(True)
        self.resize(640, 560)
        self.setMinimumSize(520, 420)

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
            QLabel#field_label { color: #cdd6f4; font-size: 13px; font-weight: bold; }
            QLabel#field_desc { color: #7f849c; font-size: 11px; }
            QLabel#status { color: #a6adc8; font-size: 12px; }
            QLabel#status[error="true"] { color: #f38ba8; }
            QLabel#status[error="false"] { color: #a6e3a1; }
            QTabWidget::pane {
                border: 1px solid #313244;
                border-radius: 6px;
                top: -1px;
            }
            QTabBar::tab {
                background: #181825;
                color: #a6adc8;
                padding: 7px 12px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
            }
            QTabBar::tab:selected { background: #313244; color: #cdd6f4; }
            QScrollArea { border: none; background: transparent; }
            QWidget#tab_scroll { background: #1e1e2e; }
            QLineEdit, QComboBox {
                background-color: #313244;
                border: 1px solid #45475a;
                border-radius: 4px;
                color: #cdd6f4;
                padding: 6px;
            }
            QLineEdit:read-only { color: #a6adc8; }
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
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        lbl_heading = QLabel("⚙ Settings & Preferences", self)
        lbl_heading.setObjectName("heading")
        layout.addWidget(lbl_heading)

        lbl_note = QLabel(
            "Preferences are saved to your local .env configuration file. "
            "Some changes take effect immediately; others need a restart.",
            self,
        )
        lbl_note.setObjectName("note")
        lbl_note.setWordWrap(True)
        layout.addWidget(lbl_note)

        self.tabs = QTabWidget(self)
        self.tabs.addTab(self._build_ai_tab(), "AI & Routing")
        self.tabs.addTab(self._build_voice_tab(), "Voice")
        self.tabs.addTab(self._build_privacy_tab(), "Privacy & Permissions")
        self.tabs.addTab(self._build_appearance_tab(), "Appearance")
        self.tabs.addTab(self._build_general_tab(), "General")
        layout.addWidget(self.tabs, stretch=1)

        self.lbl_status = QLabel("", self)
        self.lbl_status.setObjectName("status")
        self.lbl_status.setProperty("error", "false")
        self.lbl_status.setWordWrap(True)
        layout.addWidget(self.lbl_status)

        btn_row = QHBoxLayout()
        self.btn_apply = QPushButton("Apply", self)
        self.btn_apply.clicked.connect(self._on_apply)
        self.btn_save = QPushButton("Save", self)
        self.btn_save.setObjectName("btn_primary")
        self.btn_save.clicked.connect(self._on_save)
        self.btn_cancel = QPushButton("Cancel", self)
        self.btn_cancel.clicked.connect(self.reject)

        btn_row.addStretch()
        btn_row.addWidget(self.btn_apply)
        btn_row.addWidget(self.btn_save)
        btn_row.addWidget(self.btn_cancel)
        layout.addLayout(btn_row)

    def _scrollable(self, body: QWidget) -> QScrollArea:
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        body.setObjectName("tab_scroll")
        scroll.setWidget(body)
        return scroll

    def _tab_body(self):
        """Build a scrollable tab body and return ``(scroll, content_layout)``."""
        body = QWidget()
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(14, 14, 14, 14)
        body_layout.setSpacing(14)
        return self._scrollable(body), body_layout

    def _build_ai_tab(self) -> QWidget:
        scroll, layout = self._tab_body()

        self.chk_gemini = QCheckBox("Enable Gemini cloud fallback (Tier 2)")
        layout.addWidget(_make_field(
            "Cloud fallback",
            self.chk_gemini,
            "Routes unresolved queries to Google Gemini when a local tier cannot. "
            "Requires a configured API key.",
        ))

        self.txt_gemini_model = QLineEdit()
        layout.addWidget(_make_field(
            "Gemini model",
            self.txt_gemini_model,
            "Model identifier used for Tier 2 requests (e.g. gemini-2.5-flash).",
        ))

        self.txt_gemini_key = QLineEdit()
        self.txt_gemini_key.setEchoMode(QLineEdit.Password)
        layout.addWidget(_make_field(
            "Gemini API key",
            self.txt_gemini_key,
            "Session only — never written to disk, because Vector has no secure "
            "credential store. Set GEMINI_API_KEY in .env to persist it yourself.",
        ))

        model_name = Path(str(getattr(self.settings, "needle_tuned_weights_path", ""))).name
        layout.addWidget(_info_label(
            f"Tier 1 routing model (production): {model_name or 'not configured'}. "
            "The experimental V5 archive is not selectable here; it remains an "
            "explicit .env override only."
        ))
        layout.addWidget(_info_label(
            "Local LLM (Ollama) is not implemented in this build; there is no "
            "working backend to configure."
        ))

        layout.addStretch()
        self._connect_dirty(self.chk_gemini, self.txt_gemini_model, self.txt_gemini_key)
        return scroll

    def _build_voice_tab(self) -> QWidget:
        scroll, layout = self._tab_body()

        self.chk_voice = QCheckBox("Enable voice pipeline")
        layout.addWidget(_make_field(
            "Voice input",
            self.chk_voice,
            "Enables microphone capture and spoken responses. Takes effect "
            "immediately.",
        ))

        self.cmb_stt_backend = QComboBox()
        for label, value in _STT_BACKENDS:
            self.cmb_stt_backend.addItem(label, value)
        layout.addWidget(_make_field(
            "Speech-to-text backend",
            self.cmb_stt_backend,
            "Online uses Google SpeechRecognition (network). Offline uses the "
            "local sherpa-onnx model and never touches the network. Auto prefers "
            "offline and falls back to online.",
        ))

        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(6)
        self.txt_stt_dir = QLineEdit()
        self.btn_browse_stt = QPushButton("Browse…")
        self.btn_browse_stt.clicked.connect(self._on_browse_stt_dir)
        row_layout.addWidget(self.txt_stt_dir, stretch=1)
        row_layout.addWidget(self.btn_browse_stt)
        layout.addWidget(_make_field(
            "Offline STT model folder",
            row,
            "Directory containing the local sherpa-onnx model (tokens + "
            "encoder/decoder). Changing this may need a restart if a model is "
            "already loaded.",
        ))

        layout.addStretch()
        self._connect_dirty(self.chk_voice, self.cmb_stt_backend, self.txt_stt_dir)
        return scroll

    def _build_privacy_tab(self) -> QWidget:
        scroll, layout = self._tab_body()

        self.chk_confirm = QCheckBox("Require confirmation for dangerous actions")
        layout.addWidget(_make_field(
            "Dangerous-action confirmation",
            self.chk_confirm,
            "When enabled, DANGEROUS-level tools (e.g. shutdown, restart) always "
            "show a confirmation prompt. Disabling it reduces friction but never "
            "allows a BLOCKED tool, and CONFIRM-level tools always prompt.",
        ))

        layout.addWidget(_info_label(
            "Tool permissions, argument validation, candidate admission and "
            "approval binding are enforced in code and cannot be changed here."
        ))
        layout.addWidget(_info_label(
            "The Gemini API key you enter is kept in memory for this session only "
            "and is never written to disk."
        ))

        layout.addStretch()
        self._connect_dirty(self.chk_confirm)
        return scroll

    def _build_appearance_tab(self) -> QWidget:
        scroll, layout = self._tab_body()

        layout.addWidget(_make_field(
            "Theme",
            _info_label("Fixed dark theme (Catppuccin-inspired)."),
            "Theme switching is not implemented in this build, so no appearance "
            "controls are offered.",
        ))
        layout.addStretch()
        return scroll

    def _build_general_tab(self) -> QWidget:
        scroll, layout = self._tab_body()

        self.cmb_log_level = QComboBox()
        self.cmb_log_level.addItems(_LOG_LEVELS)
        layout.addWidget(_make_field(
            "Log level",
            self.cmb_log_level,
            "Verbosity of application logging. Restart required — the level is "
            "read once at startup.",
        ))

        db_path = str(getattr(self.settings, "database_path", "") or "")
        layout.addWidget(_info_label(f"Application: {APP_NAME} {APP_VERSION}"))
        layout.addWidget(_info_label(f"Database: {db_path} (change via .env; restart required)"))

        layout.addStretch()
        self._connect_dirty(self.cmb_log_level)
        return scroll

    def _connect_dirty(self, *widgets: QWidget) -> None:
        for widget in widgets:
            if isinstance(widget, QCheckBox):
                widget.toggled.connect(self._on_widget_changed)
            elif isinstance(widget, QLineEdit):
                widget.textChanged.connect(self._on_widget_changed)
            elif isinstance(widget, QComboBox):
                widget.currentIndexChanged.connect(self._on_widget_changed)

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------
    def _widget_values(self) -> Dict[str, Any]:
        return {
            "enable_gemini": self.chk_gemini.isChecked(),
            "gemini_model": self.txt_gemini_model.text().strip(),
            "gemini_api_key": self.txt_gemini_key.text().strip(),
            "enable_voice": self.chk_voice.isChecked(),
            "voice_stt_backend": self.cmb_stt_backend.currentData(),
            "voice_stt_model_dir": self.txt_stt_dir.text().strip(),
            "require_confirmation_for_dangerous": self.chk_confirm.isChecked(),
            "log_level": self.cmb_log_level.currentText(),
        }

    @staticmethod
    def _select_combo_data(combo: QComboBox, value: Any) -> None:
        index = combo.findData(value)
        combo.setCurrentIndex(index if index >= 0 else 0)

    @staticmethod
    def _select_combo_text(combo: QComboBox, value: str) -> None:
        index = combo.findText(value)
        combo.setCurrentIndex(index if index >= 0 else 0)

    def _resync_from_settings(self) -> None:
        """Load the live settings into the widgets and reset the dirty baseline."""
        self._loading = True
        s = self.settings
        try:
            self.chk_gemini.setChecked(bool(getattr(s, "enable_gemini", True)))
            self.txt_gemini_model.setText(str(getattr(s, "gemini_model", "") or ""))
            self.txt_gemini_key.setText(str(getattr(s, "gemini_api_key", "") or ""))
            self.chk_voice.setChecked(bool(getattr(s, "enable_voice", False)))
            self._select_combo_data(
                self.cmb_stt_backend, str(getattr(s, "voice_stt_backend", "online") or "online")
            )
            self.txt_stt_dir.setText(str(getattr(s, "voice_stt_model_dir", "") or ""))
            self.chk_confirm.setChecked(
                bool(getattr(s, "require_confirmation_for_dangerous", True))
            )
            self._select_combo_text(
                self.cmb_log_level, str(getattr(s, "log_level", "INFO") or "INFO")
            )
        finally:
            self._loading = False
        self._initial = self._widget_values()
        self._update_dirty()

    def _on_widget_changed(self, *_args: Any) -> None:
        self._update_dirty()

    def _is_dirty(self) -> bool:
        return self._widget_values() != self._initial

    def _update_dirty(self) -> None:
        if self._loading:
            return
        self.btn_apply.setEnabled(self._is_dirty())

    def _collect_changes(self) -> Dict[str, Any]:
        return self._widget_values()

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def _on_browse_stt_dir(self) -> None:
        start = self.txt_stt_dir.text().strip() or str(Path.home())
        chosen = QFileDialog.getExistingDirectory(self, "Select offline STT model folder", start)
        if chosen:
            self.txt_stt_dir.setText(chosen)

    def _on_apply(self) -> None:
        self._persist(close=False)

    def _on_save(self) -> None:
        self._persist(close=True)

    def _persist(self, close: bool) -> None:
        if not self._is_dirty():
            if close:
                self.accept()
            else:
                self._set_status(NO_CHANGES_TEXT, error=False)
            return

        self._set_busy(True)
        self._set_status("Saving…", error=False)
        worker = SettingsSaveWorker(self.store, self._collect_changes())
        worker.signals.result.connect(lambda result: self._on_saved(result, close))
        worker.signals.error.connect(self._on_save_error)
        self._start_save_worker(worker)

    def _start_save_worker(self, worker: SettingsSaveWorker) -> None:
        """Seam so tests can run the save synchronously."""
        QThreadPool.globalInstance().start(worker)

    @Slot(object)
    def _on_saved(self, result: Any, close: bool) -> None:
        self._set_busy(False)
        if result is None or not getattr(result, "ok", False):
            message = getattr(result, "error", "") or SAVE_ERROR_TEXT
            self._set_status(message, error=True)
            return

        self._resync_from_settings()

        restart = list(getattr(result, "restart_required", []) or [])
        if restart:
            names = ", ".join(FIELD_LABELS.get(name, name) for name in restart)
            self._set_status(RESTART_HINT_TEMPLATE.format(fields=names), error=False)
        else:
            self._set_status(SAVE_SUCCESS_TEXT, error=False)

        if close:
            self.accept()

    @Slot(str)
    def _on_save_error(self, message: str) -> None:
        self._set_busy(False)
        self._set_status(str(message) or SAVE_ERROR_TEXT, error=True)

    def _set_busy(self, busy: bool) -> None:
        self.btn_apply.setEnabled(not busy and self._is_dirty())
        self.btn_save.setEnabled(not busy)
        self.btn_cancel.setEnabled(not busy)

    def _set_status(self, message: str, error: bool) -> None:
        self.lbl_status.setText(message)
        self.lbl_status.setProperty("error", "true" if error else "false")
        # Re-polish so the dynamic property restyles the label immediately.
        self.lbl_status.style().unpolish(self.lbl_status)
        self.lbl_status.style().polish(self.lbl_status)
