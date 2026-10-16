"""
Headless unit tests for the Diagnostics & Troubleshooting feature.

The collector and dialog are exercised deterministically: file paths point at
``tmp_path``, optional-dependency checks are monkeypatched, the error source is
injected, the clipboard is faked, and generation / file writes run
synchronously through the project's existing test seams.  Nothing here loads a
model, opens the network, touches a microphone, or needs real credentials.
"""

import os
import platform
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from app.config import APP_NAME, APP_VERSION  # noqa: E402
from app.config.settings import Settings  # noqa: E402
from app.memory.database import DatabaseManager  # noqa: E402
from app.memory.history import HistoryManager  # noqa: E402
from app.ui import diagnostics as diag  # noqa: E402
from app.ui.diagnostics import (  # noqa: E402
    COPY_ERROR_TEXT,
    COPY_SUCCESS_TEXT,
    DIAGNOSTICS_ERROR_TEXT,
    NOT_AVAILABLE,
    OPTIONAL_DEPENDENCIES,
    SAVE_ERROR_TEXT,
    SAVE_SUCCESS_TEXT,
    DiagnosticsCollector,
    DiagnosticsDialog,
    DiagnosticsWorker,
    RecentErrorLog,
    SaveReportWorker,
    sanitize_path,
)


@pytest.fixture(scope="session")
def qapp():
    """Session fixture initializing a headless QApplication."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


#: A fake Google/Gemini-style credential, assembled at runtime so the source
#: never contains a literal that trips secret scanners (GitHub secret scanning
#: flags contiguous ``AIza...`` API-key patterns even in test fixtures).  The
#: resulting value still matches PrivacyFilter's Gemini-key pattern so the
#: redaction tests remain meaningful.
_FAKE_GEMINI_KEY = "AIzaSy" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q"


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _settings(tmp_path, **overrides) -> Settings:
    base = dict(
        gemini_api_key="",
        gemini_model="gemini-2.5-flash",
        enable_gemini=True,
        enable_local_llm=False,
        enable_voice=False,
        voice_stt_backend="online",
        needle_model_path=str(tmp_path / "needle_2_custom.onnx"),
        needle_tuned_weights_path=str(tmp_path / "needle3_vector_v4.cact"),
        needle_official_weights_path=str(tmp_path / "needle3.cact"),
        voice_stt_model_dir=str(tmp_path / "stt"),
        database_path=str(tmp_path / "vector.db"),
        log_level="INFO",
    )
    base.update(overrides)
    return Settings(**base)


class _FakeDb:
    """DatabaseManager stand-in whose connection either yields or raises."""

    def __init__(self, raises=False, tables=None):
        self._raises = raises
        self._tables = list(tables or [])

    def get_connection(self):
        outer = self

        class _CM:
            def __enter__(self_inner):
                if outer._raises:
                    raise RuntimeError("db exploded at C:\\Users\\secret\\vector.db")
                return _FakeConn(outer._tables)

            def __exit__(self_inner, *exc):
                return False

        return _CM()


class _FakeConn:
    def __init__(self, tables):
        self._tables = tables

    def execute(self, _sql):
        return _FakeCursor([(t,) for t in self._tables])

    def close(self):
        pass


class _FakeCursor:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return list(self._rows)


class _FakeHistory:
    """HistoryManager stand-in exposing a ``db`` with ``get_connection``."""

    def __init__(self, db=None):
        self.db = db


class _ErrProvider:
    def __init__(self, entries=None, raises=False):
        self.entries = entries
        self.raises = raises

    def __call__(self):
        if self.raises:
            raise RuntimeError("error source failed at C:\\Users\\secret")
        return self.entries


def _collector(tmp_path, history_manager=None, error_provider=None, **overrides):
    return DiagnosticsCollector(
        settings=_settings(tmp_path, **overrides),
        history_manager=history_manager,
        error_provider=error_provider,
    )


# --------------------------------------------------------------------------
# 1. Complete information
# --------------------------------------------------------------------------

def test_report_contains_complete_information(tmp_path):
    (tmp_path / "needle_2_custom.onnx").write_bytes(b"x")
    (tmp_path / "needle3_vector_v4.cact").write_bytes(b"x")
    (tmp_path / "needle3.cact").write_bytes(b"x")
    stt = tmp_path / "stt"
    stt.mkdir()
    (stt / "tokens.txt").write_text("t")
    (stt / "encoder.onnx").write_bytes(b"e")
    (stt / "decoder.onnx").write_bytes(b"d")

    report = _collector(tmp_path).collect()

    assert f"Name: {APP_NAME}" in report
    assert f"Version: {APP_VERSION}" in report
    assert f"Python: {platform.python_version()}" in report
    assert f"Platform: {platform.platform()}" in report
    for heading in (
        "[Application]",
        "[AI and Routing Configuration]",
        "[System Health]",
        "[Optional Dependencies]",
        "[Model and Offline STT Assets]",
        "[Database]",
        "[Recent Errors]",
    ):
        assert heading in report
    assert "[System Health]" in report and "Gemini cloud fallback" in report
    assert "Tier 1 tuned weights: Present" in report
    assert "Tier 0.5 custom ONNX: Present" in report
    assert "Offline STT model files: Present" in report
    assert "Tables found:" in report


# --------------------------------------------------------------------------
# 2. Missing / unavailable configuration fields
# --------------------------------------------------------------------------

def test_missing_configuration_fields_are_honest(tmp_path):
    report = _collector(
        tmp_path,
        needle_model_path="",
        needle_tuned_weights_path="",
        needle_official_weights_path="",
        voice_stt_model_dir="",
    ).collect()

    # Version is always available even with nothing configured.
    assert f"Version: {APP_VERSION}" in report
    assert "Tier 1 tuned weights: Not configured" in report
    assert "Tier 0.5 custom ONNX: Not configured" in report
    assert "Offline STT model files: Missing" in report


def test_unconfigured_gemini_reports_credential_absent(tmp_path):
    report = _collector(tmp_path, gemini_api_key="").collect()
    assert "Gemini credential configured: no" in report


def test_unknown_fields_use_not_available(tmp_path):
    # A section that cannot be built must degrade to a single honest line.
    collector = _collector(tmp_path)

    def boom():
        raise RuntimeError("nope")

    collector._application_lines = lambda: collector._safe(boom, "Application")
    report = collector.collect()
    assert f"{NOT_AVAILABLE} (Application could not be collected)" in report


# --------------------------------------------------------------------------
# 3. Missing optional dependencies and model assets
# --------------------------------------------------------------------------

def test_missing_optional_dependencies(tmp_path, monkeypatch):
    monkeypatch.setattr(diag, "module_available", lambda name: False)
    report = _collector(tmp_path).collect()
    assert report.count("Not installed") == len(OPTIONAL_DEPENDENCIES)
    assert "Local inference engine (needle): Not installed" in report


def test_present_optional_dependencies(tmp_path, monkeypatch):
    monkeypatch.setattr(diag, "module_available", lambda name: True)
    report = _collector(tmp_path).collect()
    assert "Local inference engine (needle): Available" in report
    assert "Offline STT engine (sherpa-onnx): Available" in report
    assert "Not installed" not in report


def test_missing_model_assets(tmp_path):
    report = _collector(tmp_path).collect()
    assert "Tier 1 tuned weights: Missing" in report
    assert "Offline STT model files: Missing" in report


# --------------------------------------------------------------------------
# 4. Database access failure
# --------------------------------------------------------------------------

def test_database_read_failure_is_reported(tmp_path):
    history = _FakeHistory(db=_FakeDb(raises=True))
    report = _collector(tmp_path, history_manager=history).collect()
    assert "Status: Unavailable (read failed)" in report
    assert "Tables found: Unknown" in report
    # Raw exception text / paths never leak.
    assert "exploded" not in report
    assert "C:\\Users\\secret" not in report


def test_database_not_created_is_not_configured(tmp_path):
    report = _collector(tmp_path).collect()  # tmp vector.db does not exist
    assert "Status: Not configured (not created yet)" in report


def test_database_tables_are_listed_via_history_manager(tmp_path):
    history = _FakeHistory(
        db=_FakeDb(tables=["conversations", "messages", "tool_calls", "user_preferences"])
    )
    report = _collector(tmp_path, history_manager=history).collect()
    assert "Status: Available (readable)" in report
    assert "Tables found: conversations, messages, tool_calls, user_preferences" in report
    assert "Missing expected tables: none" in report


# --------------------------------------------------------------------------
# 5. Privacy sanitization
# --------------------------------------------------------------------------

def test_api_key_and_personal_paths_are_sanitized(tmp_path):
    secret = _FAKE_GEMINI_KEY
    provider = _ErrProvider([
        {
            "level": "ERROR",
            "logger": "app.test",
            "message": (
                f"failed with key {secret} access_token=abc123XYZ at "
                r"C:\Users\bob\private\thing.txt"
            ),
        }
    ])
    report = _collector(
        tmp_path, gemini_api_key=secret, error_provider=provider
    ).collect()

    assert secret not in report
    assert "abc123XYZ" not in report
    assert "C:\\Users\\bob" not in report
    assert "[REDACTED_SECRET]" in report
    assert "<user>" in report
    # The credential's presence is still reported truthfully.
    assert "Gemini credential configured: yes" in report


def test_password_and_generic_token_redacted(tmp_path):
    provider = _ErrProvider([
        {"level": "ERROR", "logger": "x", "message": "password=hunter2 access_token: ZZTOP9"},
    ])
    report = _collector(tmp_path, error_provider=provider).collect()
    assert "hunter2" not in report
    assert "ZZTOP9" not in report
    assert "[REDACTED_SECRET]" in report


def test_sanitize_path_helper():
    assert sanitize_path(r"C:\Users\bob\file.txt") == r"C:\Users\<user>\file.txt"
    assert sanitize_path("/home/bob/file.txt") == "/home/<user>/file.txt"
    assert sanitize_path("") == ""
    assert sanitize_path(None) == ""
    home = str(Path.home())
    assert home not in sanitize_path(home + r"\secret.txt")
    assert sanitize_path(home + r"\secret.txt") == r"~\secret.txt"


# --------------------------------------------------------------------------
# 6. No conversation history / transcripts / tool arguments
# --------------------------------------------------------------------------

def test_no_history_transcripts_or_tool_arguments(tmp_path):
    secret_turn = f"My password is hunter2 and my API key is {_FAKE_GEMINI_KEY}"
    secret_arg = "SECRET_TOOL_ARGUMENT_PAYLOAD"

    db = DatabaseManager(db_path=str(tmp_path / "history.db"))
    hm = HistoryManager(db_manager=db)
    conv = hm.create_conversation("t")
    hm.add_message(conv, "user", secret_turn)
    hm.add_message(conv, "assistant", "ok")
    hm.add_execution("set_volume", {"level": secret_arg}, "SUCCESS", "volume set")

    report = _collector(tmp_path, history_manager=hm).collect()

    assert secret_turn not in report
    assert "hunter2" not in report
    assert _FAKE_GEMINI_KEY not in report
    assert secret_arg not in report
    # But the database is still described structurally.
    assert "Tables found:" in report
    assert "tool_calls" in report


# --------------------------------------------------------------------------
# 7. Safe handling of error-source failures
# --------------------------------------------------------------------------

def test_error_source_failure_does_not_break_report(tmp_path):
    provider = _ErrProvider(raises=True)
    report = _collector(tmp_path, error_provider=provider).collect()

    assert "[Recent Errors]" in report
    # The failure is contained: the rest of the report is still present.
    assert "[Application]" in report
    assert "[Database]" in report
    assert "error source failed" not in report
    assert "C:\\Users\\secret" not in report


def test_error_source_empty_reports_none(tmp_path):
    report = _collector(tmp_path, error_provider=_ErrProvider(entries=[])).collect()
    assert "None recorded this session" in report


def test_error_source_without_provider_or_sink_is_not_available(tmp_path):
    report = _collector(tmp_path).collect()
    assert f"{NOT_AVAILABLE} (no error source)" in report


def test_recent_error_log_records_sanitized_entries():
    RecentErrorLog.clear()
    import logging

    RecentErrorLog.install()
    secret = _FAKE_GEMINI_KEY
    logging.getLogger("app.test.diag").error("boom with key %s", secret)

    entries = RecentErrorLog.recent()
    assert entries, "expected at least one captured entry"
    joined = " ".join(e["message"] for e in entries)
    assert secret not in joined
    assert "boom with key" in joined

    report = DiagnosticsCollector(
        settings=Settings(), error_provider=RecentErrorLog.recent
    ).collect()
    assert secret not in report
    RecentErrorLog.clear()


# --------------------------------------------------------------------------
# 8. Clipboard and export behavior
# --------------------------------------------------------------------------

class _FakeClipboard:
    def __init__(self, raises=False):
        self.text = ""
        self.raises = raises

    def setText(self, text):
        if self.raises:
            raise RuntimeError("clipboard unavailable")
        self.text = text


def test_copy_to_clipboard_success(qapp):
    dlg = DiagnosticsDialog(auto_generate=False)
    dlg._report = "REPORT-BODY"
    fake = _FakeClipboard()
    dlg._clipboard = lambda: fake

    dlg.copy_to_clipboard()

    assert fake.text == "REPORT-BODY"
    assert dlg.lbl_status.text() == COPY_SUCCESS_TEXT


def test_copy_to_clipboard_failure_is_generic(qapp):
    dlg = DiagnosticsDialog(auto_generate=False)
    dlg._report = "REPORT-BODY"
    dlg._clipboard = lambda: _FakeClipboard(raises=True)

    dlg.copy_to_clipboard()  # must not raise

    assert dlg.lbl_status.text() == COPY_ERROR_TEXT
    assert "unavailable" not in dlg.lbl_status.text()


def test_copy_with_empty_report(qapp):
    dlg = DiagnosticsDialog(auto_generate=False)
    dlg._report = ""
    dlg.copy_to_clipboard()
    assert "Nothing to copy" in dlg.lbl_status.text()


def test_save_report_writes_file(qapp, tmp_path):
    dlg = DiagnosticsDialog(auto_generate=False)
    dlg._report = "SAVED-REPORT"
    dlg._start_save_worker = lambda worker: worker.run()

    target = tmp_path / "diagnostics.txt"
    dlg._save_report_to(str(target))

    assert target.read_text(encoding="utf-8") == "SAVED-REPORT"
    assert dlg.lbl_status.text() == SAVE_SUCCESS_TEXT


def test_save_report_write_failure_is_generic(qapp, tmp_path):
    dlg = DiagnosticsDialog(auto_generate=False)
    dlg._report = "SAVED-REPORT"
    dlg._start_save_worker = lambda worker: worker.run()

    # A directory as the target makes write_text raise.
    target = tmp_path / "adir"
    target.mkdir()
    dlg._save_report_to(str(target))  # must not raise

    assert dlg.lbl_status.text() == SAVE_ERROR_TEXT
    assert "adir" not in dlg.lbl_status.text()


def test_save_worker_emits_error_on_failure(qapp, tmp_path):
    errors = []
    worker = SaveReportWorker(str(tmp_path / "missing_dir" / "x.txt"), "x")
    worker.signals.error.connect(errors.append)
    worker.run()
    assert errors == [SAVE_ERROR_TEXT]


def test_save_as_with_no_report(qapp):
    dlg = DiagnosticsDialog(auto_generate=False)
    dlg._report = ""
    dlg.save_as()  # must not open a dialog or raise
    assert "Nothing to save" in dlg.lbl_status.text()


# --------------------------------------------------------------------------
# 9. UI responsiveness and worker error handling
# --------------------------------------------------------------------------

class _StubCollector:
    def __init__(self, text="STUB-REPORT", raises=False):
        self.text = text
        self.raises = raises

    def collect(self):
        if self.raises:
            raise RuntimeError("collector exploded at C:\\Users\\secret")
        return self.text


def test_worker_emits_generic_error_on_collection_failure(qapp):
    results, errors = [], []
    worker = DiagnosticsWorker(_StubCollector(raises=True))
    worker.signals.result.connect(results.append)
    worker.signals.error.connect(errors.append)

    worker.run()

    assert results == []
    assert errors == [DIAGNOSTICS_ERROR_TEXT]
    assert "exploded" not in errors[0]


def test_worker_emits_report_on_success(qapp):
    results = []
    worker = DiagnosticsWorker(_StubCollector(text="HELLO"))
    worker.signals.result.connect(results.append)
    worker.run()
    assert results == ["HELLO"]


def test_dialog_generates_and_previews_report(qapp):
    dlg = DiagnosticsDialog(auto_generate=False)
    dlg.collector = _StubCollector(text="PREVIEWED")
    dlg._start_worker = lambda worker: worker.run()

    dlg.generate()

    assert dlg.preview.toPlainText() == "PREVIEWED"
    assert dlg._report == "PREVIEWED"
    assert dlg.lbl_status.text() == ""


def test_dialog_handles_collection_failure(qapp):
    dlg = DiagnosticsDialog(auto_generate=False)
    dlg.collector = _StubCollector(raises=True)
    dlg._start_worker = lambda worker: worker.run()

    dlg.generate()  # must not raise

    assert dlg.preview.toPlainText() == ""
    assert dlg.lbl_status.text() == DIAGNOSTICS_ERROR_TEXT
    assert "exploded" not in dlg.lbl_status.text()
    # Controls are usable again after the failure.
    assert dlg.btn_generate.isEnabled() is True
    assert dlg.btn_copy.isEnabled() is True
    assert dlg.btn_save.isEnabled() is True


def test_dialog_disables_controls_while_generating(qapp):
    dlg = DiagnosticsDialog(auto_generate=False)
    dlg.collector = _StubCollector()
    dlg._start_worker = lambda worker: None  # never completes

    dlg.generate()

    assert dlg.btn_generate.isEnabled() is False
    assert dlg.btn_copy.isEnabled() is False
    assert dlg.btn_save.isEnabled() is False

    dlg._on_report("done")
    assert dlg.btn_generate.isEnabled() is True


def test_dialog_auto_generates_on_construction(qapp, monkeypatch):
    # Force the thread-pool seam to run synchronously so construction is
    # deterministic and no worker outlives the dialog.
    monkeypatch.setattr(DiagnosticsDialog, "_start_worker", lambda self, worker: worker.run())

    dlg = DiagnosticsDialog(settings=Settings(), auto_generate=True)

    assert dlg.preview.isReadOnly() is True
    assert dlg.preview.toPlainText().strip() != ""
    assert dlg._report.strip() != ""


# --------------------------------------------------------------------------
# 10. Regression compatibility
# --------------------------------------------------------------------------

def test_regression_system_health_collector_still_works(tmp_path):
    from app.ui.system_health import SystemHealthCollector

    sections = SystemHealthCollector(
        settings=_settings(tmp_path),
        resource_provider=lambda: None,
    ).collect()
    assert [s.title for s in sections] == [
        "AI and Routing",
        "Voice",
        "Storage",
        "System Resources",
    ]


def test_regression_action_history_imports_and_constructs(qapp):
    from app.ui.action_history import ActionHistoryDialog

    dlg = ActionHistoryDialog(_FakeHistory(), auto_load=False)
    assert dlg.windowTitle() != ""


def test_regression_main_window_exposes_all_dialogs():
    from app.ui.main_window import MainWindow

    for method in ("_open_settings", "_open_action_history", "_open_system_health", "_open_diagnostics"):
        assert callable(getattr(MainWindow, method, None)), method


def test_regression_status_bar_uses_shared_version(qapp):
    from app.ui.status_bar import StatusBarWidget

    bar = StatusBarWidget()
    assert bar.lbl_brand.text() == f"{APP_NAME.upper()} {APP_VERSION}"
