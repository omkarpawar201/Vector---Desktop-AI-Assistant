"""
Headless unit tests for the Action History UI.

These exercise the dialog and its background loader directly -- no microphone,
network, model files, or real tool execution, and no Qt event loop is spun.
Database reads run synchronously through a small test seam so results are
deterministic.
"""

import os
import tempfile
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QLabel  # noqa: E402

from app.memory.database import DatabaseManager  # noqa: E402
from app.memory.history import HistoryManager  # noqa: E402
from app.memory.journal import JournalStatus  # noqa: E402
from app.ui.action_history import (  # noqa: E402
    ALL_STATUSES_LABEL,
    EMPTY_FILTER_TEXT,
    EMPTY_HISTORY_TEXT,
    LOAD_ERROR_TEXT,
    ActionHistoryDialog,
    HistoryLoadWorker,
    format_timestamp,
)


@pytest.fixture(scope="session")
def qapp():
    """Session fixture initializing a headless QApplication."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


class _FakeHistory:
    """Minimal HistoryManager stand-in for the dialog."""

    def __init__(self, entries=None):
        self.entries = list(entries or [])
        self.calls = []
        self.raise_on_read = False

    def get_recent_tool_calls(self, conversation_id=None, limit=50):
        self.calls.append({"conversation_id": conversation_id, "limit": limit})
        if self.raise_on_read:
            raise RuntimeError("db exploded at C:\\Users\\secret\\vector.db")
        return list(self.entries)


def _entry(status, tool="get_cpu_usage", summary="ok", ts="2026-01-01T10:00:00+00:00", **extra):
    entry = {
        "id": extra.pop("id", "id-" + summary),
        "conversation_id": extra.pop("conversation_id", "c1"),
        "tool_name": tool,
        "arguments": extra.pop("arguments", "{}"),
        "summary": summary,
        "execution_status": status,
        "timestamp": ts,
    }
    entry.update(extra)
    return entry


def _dialog(qapp, history, conversation_id=None):
    """A dialog wired to load synchronously, without auto-loading on construction."""
    dlg = ActionHistoryDialog(history, conversation_id=conversation_id, auto_load=False)
    dlg._start_worker = lambda worker: worker.run()
    return dlg


def _label_texts(widget):
    return [lbl.text() for lbl in widget.findChildren(QLabel)]


# --------------------------------------------------------------------------
# Newest first
# --------------------------------------------------------------------------

def test_entries_are_shown_newest_first(qapp):
    with tempfile.TemporaryDirectory() as tmp:
        db = DatabaseManager(db_path=str(Path(tmp) / "history.db"))
        hm = HistoryManager(db_manager=db)
        hm.add_execution("get_cpu_usage", {}, JournalStatus.SUCCESS, "first")
        hm.add_execution("get_memory_usage", {}, JournalStatus.FAILED, "second")
        hm.add_execution("get_disk_usage", {}, JournalStatus.SUCCESS, "third")

        dlg = _dialog(qapp, hm)
        dlg.load()

        assert dlg.list_widget.count() == 3
        assert [e["summary"] for e in dlg._visible_entries()] == ["third", "second", "first"]


# --------------------------------------------------------------------------
# Status filtering and the all-statuses option
# --------------------------------------------------------------------------

def test_status_filter_and_all_statuses_option(qapp):
    fake = _FakeHistory([
        _entry(JournalStatus.SUCCESS, summary="s1"),
        _entry(JournalStatus.FAILED, summary="f1"),
        _entry(JournalStatus.SUCCESS, summary="s2"),
        _entry(JournalStatus.CANCELLED, summary="c1"),
    ])
    dlg = _dialog(qapp, fake)
    dlg.load()

    # The filter offers "All statuses" first, plus every known status.
    labels = [dlg.filter_combo.itemText(i) for i in range(dlg.filter_combo.count())]
    assert labels[0] == ALL_STATUSES_LABEL
    for status in (
        JournalStatus.SUCCESS,
        JournalStatus.FAILED,
        JournalStatus.PERMISSION_DENIED,
        JournalStatus.CONFIRMATION_REQUIRED,
        JournalStatus.INVALID_ARGUMENTS,
        JournalStatus.TOOL_NOT_FOUND,
        JournalStatus.APPROVAL_REJECTED,
        JournalStatus.CANCELLED,
    ):
        assert status in labels

    # Default ("All statuses") shows everything, newest first.
    assert len(dlg._visible_entries()) == 4

    # Filter to SUCCESS.
    dlg.filter_combo.setCurrentIndex(labels.index(JournalStatus.SUCCESS))
    assert [e["summary"] for e in dlg._visible_entries()] == ["s1", "s2"]
    assert dlg.list_widget.count() == 2

    # Filter to FAILED.
    dlg.filter_combo.setCurrentIndex(labels.index(JournalStatus.FAILED))
    assert [e["summary"] for e in dlg._visible_entries()] == ["f1"]

    # Back to all statuses.
    dlg.filter_combo.setCurrentIndex(0)
    assert len(dlg._visible_entries()) == 4
    assert dlg.list_widget.count() == 4


# --------------------------------------------------------------------------
# Refresh
# --------------------------------------------------------------------------

def test_refresh_retrieves_updated_entries(qapp):
    fake = _FakeHistory([_entry(JournalStatus.SUCCESS, summary="old")])
    dlg = _dialog(qapp, fake)
    dlg.load()
    assert dlg.list_widget.count() == 1

    # New entry appears in the journal after the first load.
    fake.entries = [
        _entry(JournalStatus.SUCCESS, summary="newest"),
        _entry(JournalStatus.FAILED, summary="old"),
    ]
    dlg.refresh()

    assert dlg.list_widget.count() == 2
    assert [e["summary"] for e in dlg._visible_entries()] == ["newest", "old"]
    # Refresh re-read from the manager.
    assert len(fake.calls) == 2


# --------------------------------------------------------------------------
# Empty state
# --------------------------------------------------------------------------

def test_empty_history_shows_clear_empty_state(qapp):
    dlg = _dialog(qapp, _FakeHistory([]))
    dlg.load()

    assert dlg.list_widget.count() == 0
    assert dlg.lbl_empty.isHidden() is False
    assert dlg.lbl_empty.text() == EMPTY_HISTORY_TEXT
    assert dlg.lbl_error.isHidden() is True


def test_filter_with_no_matches_shows_filter_empty_state(qapp):
    fake = _FakeHistory([_entry(JournalStatus.SUCCESS, summary="s1")])
    dlg = _dialog(qapp, fake)
    dlg.load()

    labels = [dlg.filter_combo.itemText(i) for i in range(dlg.filter_combo.count())]
    dlg.filter_combo.setCurrentIndex(labels.index(JournalStatus.FAILED))

    assert dlg.list_widget.count() == 0
    assert dlg.lbl_empty.isHidden() is False
    assert dlg.lbl_empty.text() == EMPTY_FILTER_TEXT


# --------------------------------------------------------------------------
# Error handling
# --------------------------------------------------------------------------

def test_read_error_is_handled_gracefully_without_exposing_details(qapp):
    fake = _FakeHistory()
    fake.raise_on_read = True
    dlg = _dialog(qapp, fake)

    dlg.load()  # must not raise

    assert dlg.list_widget.count() == 0
    assert dlg.lbl_error.isHidden() is False
    assert dlg.lbl_error.text() == LOAD_ERROR_TEXT
    # Raw exception text and paths must never reach the user-facing message.
    assert "exploded" not in dlg.lbl_error.text()
    assert "vector.db" not in dlg.lbl_error.text()
    assert "secret" not in dlg.lbl_error.text().lower()


def test_worker_without_supported_manager_returns_empty(qapp):
    results = []
    worker = HistoryLoadWorker(object())
    worker.signals.result.connect(results.append)
    worker.run()

    assert results == [[]]


def test_worker_emits_generic_error_on_read_failure(qapp):
    errors = []
    fake = _FakeHistory()
    fake.raise_on_read = True
    worker = HistoryLoadWorker(fake)
    worker.signals.error.connect(errors.append)
    worker.run()

    assert errors == [LOAD_ERROR_TEXT]


# --------------------------------------------------------------------------
# Privacy: sensitive data is never displayed
# --------------------------------------------------------------------------

def test_sensitive_arguments_and_payloads_are_not_exposed(qapp):
    secret = "AIzaSySUPERSECRETKEYVALUE1234567890"
    fake = _FakeHistory([
        _entry(
            JournalStatus.SUCCESS,
            tool="journal_secret",
            summary="volume set",
            arguments='{"api_key": "' + secret + '", "path": "C:\\\\Users\\\\me"}',
            data={"secret_payload": "TOP_SECRET_PAYLOAD"},
            error="raw exception: password=hunter2",
        ),
    ])
    dlg = _dialog(qapp, fake)
    dlg.load()

    texts = " ".join(_label_texts(dlg.list_widget))

    # Only the sanitized summary is shown.
    assert "volume set" in texts
    assert "journal_secret" in texts
    # Nothing sensitive from arguments / data / error leaks into the UI.
    assert secret not in texts
    assert "TOP_SECRET_PAYLOAD" not in texts
    assert "hunter2" not in texts
    assert "api_key" not in texts
    assert "C:\\Users\\me" not in texts


# --------------------------------------------------------------------------
# Conversation scoping
# --------------------------------------------------------------------------

def test_conversation_scoping_checkbox(qapp):
    fake = _FakeHistory([_entry(JournalStatus.SUCCESS, summary="s1")])
    dlg = _dialog(qapp, fake, conversation_id="conv-123")

    # Defaults to the active conversation.
    assert dlg.chk_scope.isChecked() is True
    dlg.load()
    assert fake.calls[-1]["conversation_id"] == "conv-123"
    assert dlg._scope_conversation_id() == "conv-123"

    # Unchecking widens to all recent actions.
    dlg.chk_scope.setChecked(False)
    assert fake.calls[-1]["conversation_id"] is None
    assert dlg._scope_conversation_id() is None


def test_no_conversation_means_all_actions_by_default(qapp):
    fake = _FakeHistory([_entry(JournalStatus.SUCCESS, summary="s1")])
    dlg = _dialog(qapp, fake, conversation_id=None)

    assert dlg.chk_scope.isChecked() is False
    dlg.load()
    assert fake.calls[-1]["conversation_id"] is None


# --------------------------------------------------------------------------
# Timestamp formatting
# --------------------------------------------------------------------------

def test_format_timestamp():
    assert format_timestamp("2026-01-01T10:00:00+00:00") == "2026-01-01 10:00:00"
    assert format_timestamp("not-a-date") == "not-a-date"
    assert format_timestamp("") == ""
