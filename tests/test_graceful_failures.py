"""
Feature 12 — regression tests for graceful failure handling.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from PySide6.QtWidgets import QApplication

from app.config.settings import Settings
from app.memory.database import DatabaseManager, get_db_manager
from app.memory.history import HistoryManager
from app.tools.base import ToolResult

import app.main as main_module
import app.ui.main_window as main_window_module
from app.ui.main_window import MainWindow


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


def _clear_db_cache():
    from app.memory import database
    database._db_manager_cache.clear()
    from app.config import settings
    settings.get_settings.cache_clear()


class _CapturingPool:
    def __init__(self):
        self.workers = []

    def start(self, worker):
        self.workers.append(worker)


class _RecordingTTS:
    def __init__(self):
        self.spoken = []

    def speak(self, text):
        self.spoken.append(text)


def test_invalid_database_path_fails_gracefully_in_main(monkeypatch, qapp, tmp_path):
    _clear_db_cache()
    bad_db = tmp_path / "not_a_directory"
    bad_db.mkdir()
    # Point DB at a directory (unwritable as a sqlite file)
    monkeypatch.setenv("DATABASE_PATH", str(bad_db))
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("PYTHONPATH", r"E:\Projects\Needle_Desktop_Assistant")
    _clear_db_cache()

    messages = []

    def fake_critical(parent, title, text):
        messages.append((title, text))
        return 0

    def fake_warning(parent, title, text):
        messages.append((title, text))
        return 0

    monkeypatch.setattr(main_module.QMessageBox, "critical", fake_critical)
    monkeypatch.setattr(main_module.QMessageBox, "warning", fake_warning)

    # Prevent argparse from parsing pytest args
    monkeypatch.setattr(sys, "argv", [sys.argv[0]])

    # Use existing qapp
    def fake_qapp(args):
        return qapp
    monkeypatch.setattr(main_module.QApplication, "__init__", lambda self, args: None)
    monkeypatch.setattr(main_module.QApplication, "instance", staticmethod(lambda: qapp))
    monkeypatch.setattr(main_module.QApplication, "setStyle", lambda self, style: None)
    monkeypatch.setattr(main_module, "QApplication", lambda args: qapp)

    exit_code = main_module.main()

    assert exit_code == 1
    assert messages
    title, text = messages[0]
    assert "database" in title.lower() or "database" in text.lower()
    assert "traceback" not in text.lower()
    assert "vector" in title.lower() or "vector" in text.lower()


def test_main_window_initialization_does_not_leave_partial_state_on_db_error(monkeypatch, tmp_path):
    _clear_db_cache()

    def raise_db_error(db_path=None):
        raise OSError("unable to open database file")

    monkeypatch.setattr(main_window_module, "HistoryManager", raise_db_error)
    with pytest.raises(OSError):
        MainWindow()
    _clear_db_cache()


def test_missing_pyttsx3_does_not_prevent_tts_module_import(monkeypatch):
    monkeypatch.setitem(sys.modules, "pyttsx3", None)
    # Remove cached modules to force re-import
    for key in list(sys.modules.keys()):
        if key.startswith("app.voice.tts"):
            del sys.modules[key]
    import app.voice.tts as tts_mod
    engine = tts_mod.TTSEngine()
    # Should be disabled but not raise
    assert hasattr(engine, "speak")
    assert hasattr(engine, "is_available")
    assert engine.is_available is False
    engine.speak("hello")
    engine.set_enabled(True)
    assert engine.is_enabled is False


def test_main_window_starts_without_pyttsx3(monkeypatch, qapp, tmp_path):
    _clear_db_cache()
    from app.memory import database
    database._db_manager_cache.clear()

    # Block pyttsx3 import at runtime
    import builtins

    orig_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "pyttsx3" or name.startswith("pyttsx3."):
            raise ImportError("No module named 'pyttsx3'")
        return orig_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    try:
        # Remove cached modules to force re-import
        for key in list(sys.modules.keys()):
            if key.startswith("app.voice.tts"):
                del sys.modules[key]
            if key == "app.ui.main_window":
                del sys.modules[key]

        db = DatabaseManager(db_path=str(tmp_path / "test.db"))
        history = HistoryManager(db_manager=db)
        import app.ui.main_window as mw
        monkeypatch.setattr(mw, "HistoryManager", lambda: history)

        import app.ui.main_window as mw
        win = mw.MainWindow()
        try:
            assert win.tts is not None
            assert hasattr(win.tts, "speak")
            assert win.tts.is_available is False
            # Text functionality should remain available
            win.chat_view.add_assistant_message("hello")
            assert win.chat_view.list_widget.count() >= 1
        finally:
            win.close()
    finally:
        _clear_db_cache()


def test_history_add_message_failure_does_not_crash_present_result(monkeypatch, qapp, tmp_path):
    _clear_db_cache()
    db = DatabaseManager(db_path=str(tmp_path / "test2.db"))
    history = HistoryManager(db_manager=db)
    monkeypatch.setattr(main_window_module, "HistoryManager", lambda: history)

    win = MainWindow()
    try:
        win.settings = Settings(_env_file=None, enable_gemini=False, enable_voice=False)
        win.tts = _RecordingTTS()
        win.thread_pool = _CapturingPool()

        original_add_message = history.add_message
        call_count = {"count": 0}

        def failing_add_message(*args, **kwargs):
            call_count["count"] += 1
            raise OSError("disk full")

        history.add_message = failing_add_message

        tr = ToolResult(tool="set_volume", success=True, message="Volume set to 50")
        # Should not crash
        win._present_result(tr, win.current_conv_id)

        assert call_count["count"] == 1, "history.add_message should be called once, not retried"
        # Action card should still appear (execution status preserved as success)
        texts = []
        for row in range(win.chat_view.list_widget.count()):
            widget = win.chat_view.list_widget.itemWidget(win.chat_view.list_widget.item(row))
            if widget is not None and hasattr(widget, "text"):
                texts.append(widget.text())
        assert any("Volume" in t or "set_volume" in t for t in texts), texts
    finally:
        win.close()
        _clear_db_cache()


def test_history_add_message_failure_does_not_falsify_failed_result(monkeypatch, qapp, tmp_path):
    _clear_db_cache()
    db = DatabaseManager(db_path=str(tmp_path / "test3.db"))
    history = HistoryManager(db_manager=db)
    monkeypatch.setattr(main_window_module, "HistoryManager", lambda: history)

    win = MainWindow()
    try:
        win.settings = Settings(_env_file=None, enable_gemini=False, enable_voice=False)
        win.tts = _RecordingTTS()
        win.thread_pool = _CapturingPool()

        call_count = {"count": 0}

        def failing_add_message(*args, **kwargs):
            call_count["count"] += 1
            raise RuntimeError("write failed")

        history.add_message = failing_add_message

        tr = ToolResult(tool="shutdown_pc", success=False, message="Permission denied", error="BLOCKED")
        win._present_result(tr, win.current_conv_id)

        assert call_count["count"] == 1
        # Verify the result is still presented as failed (no falsification)
        texts = []
        for row in range(win.chat_view.list_widget.count()):
            widget = win.chat_view.list_widget.itemWidget(win.chat_view.list_widget.item(row))
            if widget is not None and hasattr(widget, "text"):
                texts.append(widget.text())
        assert any("Permission denied" in t for t in texts)
    finally:
        win.close()
        _clear_db_cache()
