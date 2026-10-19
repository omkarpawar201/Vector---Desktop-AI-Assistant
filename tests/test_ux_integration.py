"""
Feature 10 -- UX polish and end-to-end integration for Vector.

These tests exercise the seams *between* features (streaming, conversation
switching, persistence, confirmation, shutdown) on a real Qt window with real
widgets and a real SQLite-backed ``HistoryManager``.  They pin four genuine
integration defects that were found during the Feature 10 review, plus a
sidebar delivery smoke test:

* a stale request finishing late must not hide the Stop control of the newer,
  still-streaming generation;
* rebuilding the chat view (conversation switch / "load earlier") must drop the
  handles of destroyed streamed bubbles, so a late chunk can never touch a
  deleted C++ object;
* closing the window during a generation must cancel its token (no unsafe
  follow-up execution) and release live bubbles;
* dynamic text must be HTML-escaped before it reaches a rich-text ``QLabel``
  (user/assistant messages, action cards and the confirmation dialog), so stored
  or imported text can never be rendered as markup.

Everything is offline and deterministic: no microphone, network, model or
credential is touched, and every modal dialog is bypassed through a seam.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from PySide6.QtWidgets import QApplication, QLabel

from app.config.constants import IntentTier
from app.config.settings import Settings
from app.core.router import RouteResult
from app.memory.database import DatabaseManager
from app.memory.history import HistoryManager
from app.tools.base import ToolResult

import app.ui.main_window as main_window_module
from app.ui.main_window import MainWindow


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


# --------------------------------------------------------------------------
# Helpers / fakes
# --------------------------------------------------------------------------
def _history(tmp_path, name="ux_integration.db"):
    db = DatabaseManager(db_path=str(tmp_path / name))
    return HistoryManager(db_manager=db)


class _CapturingPool:
    """Stands in for QThreadPool: records workers instead of scheduling them."""

    def __init__(self):
        self.workers = []

    def start(self, worker):
        self.workers.append(worker)


class _RecordingTTS:
    def __init__(self):
        self.spoken = []

    def speak(self, text):
        self.spoken.append(text)


class _StreamingRouter:
    """
    Drives the router's streaming hooks exactly as the real router would.

    ``emit_stream=False`` models a provider that cannot stream.  The worker is
    run synchronously by the tests (``worker.run()``), so hooks are invoked on
    the calling thread.
    """

    def __init__(self, result, chunks=(), emit_stream=True):
        self.result = result
        self.chunks = list(chunks)
        self.emit_stream = emit_stream
        self.route_calls = []
        self.tokens = []

    def route_and_execute(self, user_query=None, **kwargs):
        self.route_calls.append(user_query)
        self.tokens.append(kwargs.get("cancellation_token"))
        if not self.emit_stream:
            return self.result
        start = kwargs.get("on_stream_start")
        if start is not None:
            start()
        chunk_cb = kwargs.get("on_chunk")
        for chunk in self.chunks:
            if chunk_cb is not None:
                chunk_cb(chunk)
        return self.result

    def execute_approved(self, approval, **kwargs):
        return ToolResult(tool=approval.tool_name, success=True, message="ok")


def _result(
    response_text="Done.",
    tool_result=None,
    requires_confirmation=False,
    tool_name="",
    raw_arguments=None,
    pending_approval=None,
    completed_results=None,
    tier=IntentTier.TIER_2_GEMINI,
    streamed=False,
    cancelled=False,
    partial=False,
):
    return RouteResult(
        tier_used=tier,
        tool_result=tool_result,
        response_text=response_text,
        requires_confirmation=requires_confirmation,
        tool_name=tool_name,
        raw_arguments=raw_arguments if raw_arguments is not None else {},
        pending_approval=pending_approval,
        completed_results=completed_results or [],
        streamed=streamed,
        cancelled=cancelled,
        partial=partial,
    )


def _window(monkeypatch, history, router):
    """A MainWindow with real widgets, a real history manager and a fake pool/router."""
    monkeypatch.setattr(main_window_module, "HistoryManager", lambda: history)
    win = MainWindow()
    win.settings = Settings(enable_gemini=False, enable_voice=False)
    win.router = router
    win.tts = _RecordingTTS()
    win.thread_pool = _CapturingPool()
    return win


def _list_texts(list_widget):
    """The raw ``text()`` of every item widget currently in a QListWidget."""
    texts = []
    for row in range(list_widget.count()):
        widget = list_widget.itemWidget(list_widget.item(row))
        if widget is not None and hasattr(widget, "text"):
            texts.append(widget.text())
    return texts


# ==========================================================================
# 1. HTML escaping of dynamic text (defect #1)
# ==========================================================================
def test_non_streamed_answer_with_html_is_stored_raw_and_shown_escaped(monkeypatch, qapp, tmp_path):
    hm = _history(tmp_path)
    router = _StreamingRouter(_result(response_text="Use <b>care</b> & <i>mind</i>"), emit_stream=False)
    win = _window(monkeypatch, hm, router)
    try:
        win._handle_user_command("go")
        win.thread_pool.workers[0].run()

        # Storage keeps the exact text...
        stored = [m["content"] for m in hm.get_conversation_messages(win.current_conv_id, limit=None)]
        assert "Use <b>care</b> & <i>mind</i>" in stored

        # ...but the bubble renders it as literal text, never as markup.
        texts = _list_texts(win.chat_view.list_widget)
        assert any("&lt;b&gt;care&lt;/b&gt;" in t for t in texts)
        assert all("<b>care</b>" not in t for t in texts)
        assert all("<i>mind</i>" not in t for t in texts)
    finally:
        win.close()


def test_user_message_is_escaped(qapp):
    from app.ui.chat_view import ChatViewWidget

    view = ChatViewWidget()
    view.add_user_message("<script>alert(1)</script>")

    texts = _list_texts(view.list_widget)
    assert any("&lt;script&gt;" in t for t in texts)
    assert all("<script>" not in t for t in texts)


def test_action_card_is_escaped(qapp):
    from app.ui.chat_view import ChatViewWidget

    view = ChatViewWidget()
    view.add_action_card(tool_name="do_it", message="<b>boom</b>", success=True)

    card = view.list_widget.itemWidget(view.list_widget.item(0))
    joined = " ".join(lbl.text() for lbl in card.findChildren(QLabel))
    assert "&lt;b&gt;boom&lt;/b&gt;" in joined
    assert "<b>boom</b>" not in joined


def test_confirmation_dialog_escapes_dynamic_text(qapp):
    from app.ui.confirmation_dialog import ConfirmationDialog

    dlg = ConfirmationDialog(
        tool_name="shutdown_pc",
        message="Confirm <b>now</b>",
        arguments={"path": "<x>"},
    )
    joined = " ".join(lbl.text() for lbl in dlg.findChildren(QLabel))
    assert "&lt;b&gt;now&lt;/b&gt;" in joined
    assert "<b>now</b>" not in joined
    assert "&lt;x&gt;" in joined


# ==========================================================================
# 2. Stop control belongs to the current generation (defect #2)
# ==========================================================================
def test_stale_finish_does_not_hide_current_stop(monkeypatch, qapp, tmp_path):
    router = _StreamingRouter(_result(response_text="x", streamed=True), chunks=[])
    win = _window(monkeypatch, _history(tmp_path), router)
    try:
        win._handle_user_command("first")
        rid1 = win._active_request_id
        win._handle_user_command("second")
        rid2 = win._active_request_id
        cid = win.current_conv_id
        assert rid1 != rid2

        # The newest request starts streaming -> Stop appears.
        win._on_stream_started(rid2, cid)
        assert win.command_input.btn_stop.isHidden() is False

        # The older request finishes late and must not hide the current Stop.
        win._on_route_finished(_result(response_text="first", streamed=True), cid, rid1)
        assert win._active_request_id == rid2
        assert win.command_input.btn_stop.isHidden() is False, (
            "a stale finish must never hide the Stop of the still-streaming generation"
        )

        # The current request finishes -> Stop hides.
        win._on_route_finished(_result(response_text="second", streamed=True), cid, rid2)
        assert win.command_input.btn_stop.isHidden() is True
    finally:
        win.close()


def test_stale_error_does_not_hide_current_stop(monkeypatch, qapp, tmp_path):
    router = _StreamingRouter(_result(response_text="x", streamed=True), chunks=[])
    win = _window(monkeypatch, _history(tmp_path), router)
    try:
        win._handle_user_command("first")
        rid1 = win._active_request_id
        win._handle_user_command("second")
        rid2 = win._active_request_id
        cid = win.current_conv_id

        win._on_stream_started(rid2, cid)
        assert win.command_input.btn_stop.isHidden() is False

        win._on_route_error("boom", cid, rid1)
        assert win.command_input.btn_stop.isHidden() is False, (
            "a stale error must never hide the Stop of the still-streaming generation"
        )
    finally:
        win.close()


# ==========================================================================
# 3. Rebuilding the chat view drops destroyed streamed bubbles (defect #4)
# ==========================================================================
def test_reload_during_stream_discards_stale_bubble(monkeypatch, qapp, tmp_path):
    router = _StreamingRouter(_result(response_text="Hi", streamed=True), chunks=["Hi"])
    win = _window(monkeypatch, _history(tmp_path), router)
    try:
        win._handle_user_command("hello")
        rid = win._active_request_id
        cid = win.current_conv_id

        win._on_stream_started(rid, cid)
        assert rid in win._live_streams

        # A reload (conversation switch / "load earlier") destroys the widgets.
        win._render_loaded_messages()
        assert rid not in win._live_streams, "a reload must drop handles of destroyed widgets"

        # A late chunk must be a clean no-op, never a RuntimeError.
        win._on_chunk(rid, cid, "late")
    finally:
        win.close()


def test_switch_away_and_back_during_stream_is_safe(monkeypatch, qapp, tmp_path):
    hm = _history(tmp_path)
    a = hm.create_conversation("A")
    b = hm.create_conversation("B")
    hm.add_message(b, "user", "b-only")

    router = _StreamingRouter(_result(response_text="Answer A", streamed=True), chunks=["Answer A"])
    win = _window(monkeypatch, hm, router)
    try:
        win.current_conv_id = a
        win._reload_active_conversation()

        win._handle_user_command("question for A")
        rid = win._active_request_id
        win._on_stream_started(rid, a)
        assert rid in win._live_streams

        # Leave A, then come back: the live bubble died with A's view.
        win._on_conversation_selected(b)
        assert win.current_conv_id == b
        assert rid not in win._live_streams
        win._on_conversation_selected(a)
        assert win.current_conv_id == a

        # A late chunk for the still-active request must not touch a dead widget.
        win._on_chunk(rid, a, "late")

        # The final result is still stored in A (and only A).
        win._on_route_finished(_result(response_text="Answer A", streamed=True), a, rid)
        assert "Answer A" in [m["content"] for m in hm.get_conversation_messages(a, limit=None)]
        assert "Answer A" not in [m["content"] for m in hm.get_conversation_messages(b, limit=None)]
    finally:
        win.close()


# ==========================================================================
# 4. Shutdown during generation (defect #3)
# ==========================================================================
def test_close_during_generation_cancels_and_releases(monkeypatch, qapp, tmp_path):
    router = _StreamingRouter(_result(response_text="Hi", streamed=True), chunks=[])
    win = _window(monkeypatch, _history(tmp_path), router)
    try:
        win._handle_user_command("hello")
        token = win._active_generation["token"]
        assert token.is_cancelled() is False

        win._on_stream_started(win._active_request_id, win.current_conv_id)
        assert win._live_streams

        win.close()

        assert token.is_cancelled() is True, "closing the window must stop a live generation"
        assert win._live_streams == {}, "closing must release live stream handles"
    finally:
        win.close()


# ==========================================================================
# 5. Sidebar background delivery (smoke: the list reaches the widget)
# ==========================================================================
def test_sidebar_worker_delivers_conversation_list(qapp):
    from app.ui.conversation_sidebar import ConversationListWorker, ConversationSidebar

    class _HM:
        def list_conversations(self, limit=500):
            return [{"id": "c1", "title": "Conv One", "last_activity": None}]

    sidebar = ConversationSidebar()
    sidebar.set_history_manager(_HM())

    worker = ConversationListWorker(sidebar.history_manager, query="")
    worker.signals.result.connect(sidebar._on_loaded)
    worker.run()  # same-thread emit -> direct call

    assert sidebar.list_widget.count() >= 1
    joined = " ".join(
        sidebar.list_widget.item(i).text() for i in range(sidebar.list_widget.count())
    )
    assert "Conv One" in joined
