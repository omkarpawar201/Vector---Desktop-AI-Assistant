"""
Feature 9 -- streaming responses and stop generation for Vector.

These pin the contract that an answer can be rendered incrementally through an
optional provider callback, that the final rendered text equals the persisted
text, that a Stop request cancels exactly the generation it was shown for
without blocking, and that a cancelled / partial / failed answer is never stored
as a successful complete response.

Everything is offline and deterministic:

* the Gemini loop is exercised through an injected fake SDK (no network),
* the router's streaming seam is driven by a recording fake client,
* the MainWindow is driven with a fake router, a capturing thread pool and a
  monkeypatched chat view -- no worker thread, no real model, no microphone.

No test here opens a socket, launches a program, or reaches a real model.
"""

from __future__ import annotations

import os
import sys
import types as pytypes

os.environ["QT_QPA_PLATFORM"] = "offscreen"

import pytest

from app.config.constants import IntentTier
from app.config.settings import Settings
from app.core.cancellation import CancellationToken
from app.core.router import IntentRouter, RouteResult
from app.gemini.client import GeminiClient
from app.needle.client import NeedleResult
from app.tools.base import ToolResult
from app.tools.executor import PendingApproval, SafeToolExecutor
from app.tools.registry import ToolRegistry

import app.ui.main_window as main_window_module
from app.ui.main_window import MainWindow


@pytest.fixture(scope="session")
def qapp():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


# ==========================================================================
# 1. ChatViewWidget streaming API
# ==========================================================================

def test_stream_bubble_assembles_and_finalizes_text(qapp):
    from app.ui.chat_view import ChatViewWidget

    view = ChatViewWidget()
    handle = view.begin_assistant_stream()
    view.append_assistant_stream(handle, "Hel")
    view.append_assistant_stream(handle, "lo")
    assert handle.text == "Hello"

    view.finalize_assistant_stream(handle, "Hello world", " (stopped)")
    assert handle.text == "Hello world"
    assert "Hello world" in handle.label.text()
    assert "(stopped)" in handle.label.text()


def test_stream_escapes_html_from_provider(qapp):
    from app.ui.chat_view import ChatViewWidget

    view = ChatViewWidget()
    handle = view.begin_assistant_stream()
    view.append_assistant_stream(handle, "<b>hi</b> & <script>")
    # Raw markup must never be interpreted as rich text.
    assert "&lt;b&gt;hi&lt;/b&gt;" in handle.label.text()
    assert "<script>" not in handle.label.text()


def test_finalize_without_text_keeps_assembled_body(qapp):
    from app.ui.chat_view import ChatViewWidget

    view = ChatViewWidget()
    handle = view.begin_assistant_stream()
    view.append_assistant_stream(handle, "partial")
    view.finalize_assistant_stream(handle, None, " (incomplete)")
    assert "partial" in handle.label.text()
    assert "(incomplete)" in handle.label.text()


def test_stream_methods_tolerate_missing_handle(qapp):
    from app.ui.chat_view import ChatViewWidget

    view = ChatViewWidget()
    # A superseded/absent handle must be a clean no-op, never an exception.
    view.append_assistant_stream(None, "x")
    view.finalize_assistant_stream(None, "y", " (stopped)")


# ==========================================================================
# 2. GeminiClient turn transport
# ==========================================================================

class _Chunk:
    def __init__(self, text=None, function_calls=None):
        self.text = text
        self.function_calls = function_calls or []


class _FuncCall:
    def __init__(self, name="journal_ok", args=None):
        self.name = name
        self.args = args or {}


class _StreamChat:
    """Scripted streaming chat: one iterable of chunks per turn."""

    def __init__(self, turns):
        self.turns = list(turns)
        self.i = 0
        self.stream_messages = []

    def send_message_stream(self, message):
        self.stream_messages.append(message)
        turn = self.turns[self.i]
        self.i += 1
        return iter(turn)


class _ToggleToken:
    """A token that reports cancelled after ``after`` polls."""

    def __init__(self, after):
        self.after = after
        self.n = 0

    def is_cancelled(self):
        self.n += 1
        return self.n > self.after


def test_stream_turn_emits_text_and_assembles_answer():
    chat = _StreamChat([[_Chunk("Hel"), _Chunk("lo")]])
    chunks = []

    calls, text, cancelled, emitted = GeminiClient._stream_turn(
        chat, "hi", chunks.append, None
    )

    assert chunks == ["Hel", "lo"]
    assert text == "Hello"
    assert cancelled is False
    assert emitted is True
    assert calls == []


def test_stream_turn_suppresses_tool_call_narration():
    chat = _StreamChat([[_Chunk("let me check", function_calls=[_FuncCall()]), _Chunk("...")]])
    chunks = []

    calls, text, cancelled, emitted = GeminiClient._stream_turn(
        chat, "hi", chunks.append, None
    )

    # A tool-requesting turn is narration, never the user-facing answer.
    assert chunks == []
    assert text == ""
    assert emitted is False
    assert len(calls) == 1 and calls[0].name == "journal_ok"


def test_stream_turn_cancellation_stops_consuming_immediately():
    chat = _StreamChat([[_Chunk("A"), _Chunk("B"), _Chunk("C")]])
    chunks = []

    calls, text, cancelled, emitted = GeminiClient._stream_turn(
        chat, "hi", chunks.append, _ToggleToken(after=1)
    )

    assert cancelled is True
    assert chunks == ["A"], "no chunk after the stop request may be consumed"
    assert text == "A"
    assert emitted is True


def test_stream_turn_already_cancelled_consumes_nothing():
    chat = _StreamChat([[_Chunk("A")]])
    chunks = []

    calls, text, cancelled, emitted = GeminiClient._stream_turn(
        chat, "hi", chunks.append, _ToggleToken(after=0)
    )

    assert cancelled is True
    assert chunks == []
    assert text == ""
    assert emitted is False


# ==========================================================================
# 3. GeminiClient.execute_reasoning_loop streaming / fallback
# ==========================================================================

class _StubSettings:
    is_gemini_available = True
    gemini_api_key = "test-key"
    gemini_model = "test-model"


class _RecordingExecutor:
    def __init__(self):
        self.calls = []

    def execute_tool(self, tool_name, raw_args, journal=None, **kwargs):
        self.calls.append({"tool_name": tool_name, "raw_args": raw_args})
        return ToolResult.ok(tool=tool_name, message=f"{tool_name} done"), None


class _Response:
    def __init__(self, function_calls, text):
        self.function_calls = function_calls
        self.text = text


class _BlockChat:
    """Non-streaming chat that fails loudly if streamed."""

    def __init__(self, response):
        self._response = response
        self.sent = 0
        self.stream_called = False

    def send_message(self, _message):
        self.sent += 1
        return self._response

    def send_message_stream(self, _message):  # pragma: no cover - must not run
        self.stream_called = True
        raise AssertionError("a non-streaming call must never stream")


def _empty_registry():
    reg = ToolRegistry()
    reg.clear()
    return reg


def _install_fake_genai(monkeypatch, chat):
    """Inject a minimal ``google.genai`` whose chats return ``chat``."""

    class _Part:
        @staticmethod
        def from_function_response(name, response):
            return ("function_response", name, response)

        @staticmethod
        def from_text(text):
            return ("text", text)

    class _Content:
        def __init__(self, role=None, parts=None):
            self.role = role
            self.parts = parts

    class _Chats:
        def create(self, **kwargs):
            return chat

    class _Client:
        def __init__(self, api_key=None):
            self.chats = _Chats()

    types_mod = pytypes.SimpleNamespace(
        GenerateContentConfig=lambda **kwargs: kwargs,
        Part=_Part,
        Content=_Content,
    )
    genai_mod = pytypes.SimpleNamespace(Client=_Client, types=types_mod)
    google_pkg = pytypes.SimpleNamespace(genai=genai_mod)

    monkeypatch.setitem(sys.modules, "google", google_pkg)
    monkeypatch.setitem(sys.modules, "google.genai", genai_mod)


def _client():
    return GeminiClient(
        settings=_StubSettings(),
        registry=_empty_registry(),
        executor=_RecordingExecutor(),
    )


def test_execute_reasoning_loop_streams_answer(monkeypatch):
    chat = _StreamChat([[_Chunk("Hel"), _Chunk("lo world")]])
    _install_fake_genai(monkeypatch, chat)

    chunks = []
    starts = []
    out = _client().execute_reasoning_loop(
        user_query="hi",
        on_chunk=chunks.append,
        on_stream_start=lambda: starts.append(True),
    )

    assert chunks == ["Hel", "lo world"]
    assert starts == [True], "streaming must be announced exactly once"
    assert out["response"] == "Hello world"
    assert out["streamed"] is True
    assert out["cancelled"] is False
    assert out["partial"] is False
    assert len(chat.stream_messages) == 1


def test_execute_reasoning_loop_cancelled_stream_reports_partial(monkeypatch):
    chat = _StreamChat([[_Chunk("Hel"), _Chunk("lo")]])
    _install_fake_genai(monkeypatch, chat)

    chunks = []
    out = _client().execute_reasoning_loop(
        user_query="hi",
        on_chunk=chunks.append,
        cancellation_token=_ToggleToken(after=1),
    )

    assert chunks == ["Hel"]
    assert out["response"] == "Hel"
    assert out["streamed"] is True
    assert out["cancelled"] is True
    assert out["partial"] is True, "a stopped answer is partial and must not persist"


def test_execute_reasoning_loop_never_streams_narration(monkeypatch):
    chat = _StreamChat([
        [_Chunk("checking things", function_calls=[_FuncCall("journal_ok")])],
        [_Chunk("Done!")],
    ])
    _install_fake_genai(monkeypatch, chat)

    chunks = []
    out = _client().execute_reasoning_loop(user_query="do it", on_chunk=chunks.append)

    assert chunks == ["Done!"], "a tool-call turn is narration, not streamed"
    assert out["response"] == "Done!"
    assert out["streamed"] is True


def test_execute_reasoning_loop_without_on_chunk_uses_send_message(monkeypatch):
    chat = _BlockChat(_Response([], "plain answer"))
    _install_fake_genai(monkeypatch, chat)

    out = _client().execute_reasoning_loop(user_query="hi")

    assert out["response"] == "plain answer"
    assert chat.stream_called is False, "no on_chunk => the original blocking path"
    assert chat.sent == 1
    # The non-streaming result shape is byte-for-byte the pre-streaming shape.
    assert "streamed" not in out
    assert "cancelled" not in out
    assert "partial" not in out


def test_supports_streaming_is_advertised():
    assert GeminiClient.supports_streaming is True


# ==========================================================================
# 4. Router streaming pass-through
# ==========================================================================

class _FakeNeedleClient:
    def predict_custom_onnx_intent(self, user_input):
        return NeedleResult(reason="no keyword hit")

    def select_candidates(self, user_input):
        return None


class _StreamingGemini:
    supports_streaming = True

    def __init__(self, result=None):
        self.calls = []
        self._result = result or {
            "success": True,
            "response": "hi",
            "tool_calls": [],
            "streamed": True,
            "cancelled": False,
            "partial": False,
        }

    def execute_reasoning_loop(self, user_query=None, history=None, journal=None, **kwargs):
        self.calls.append({"query": user_query, "kwargs": kwargs})
        return dict(self._result)


class _NonStreamingGemini(_StreamingGemini):
    supports_streaming = False

    def __init__(self):
        super().__init__(result={"success": True, "response": "hi", "tool_calls": []})


def _build_router(gemini):
    reg = _empty_registry()
    return IntentRouter(
        settings=Settings(enable_gemini=True, gemini_api_key="test-key"),
        registry=reg,
        executor=SafeToolExecutor(registry=reg),
        needle_client=_FakeNeedleClient(),
        gemini_client=gemini,
    )


def test_router_forwards_stream_hooks_to_streaming_provider():
    gemini = _StreamingGemini()
    router = _build_router(gemini)
    token = CancellationToken()
    chunk_cb = lambda _t: None
    start_cb = lambda: None

    res = router.route_and_execute(
        "hello there",
        on_chunk=chunk_cb,
        on_stream_start=start_cb,
        cancellation_token=token,
        request_id="req-1",
    )

    kwargs = gemini.calls[0]["kwargs"]
    assert kwargs["on_chunk"] is chunk_cb
    assert kwargs["on_stream_start"] is start_cb
    assert kwargs["cancellation_token"] is token
    assert kwargs["request_id"] == "req-1"

    assert res.tier_used == IntentTier.TIER_2_GEMINI
    assert res.streamed is True
    assert res.response_text == "hi"


def test_router_withholds_stream_hooks_from_non_streaming_provider():
    gemini = _NonStreamingGemini()
    router = _build_router(gemini)
    token = CancellationToken()

    res = router.route_and_execute(
        "hello there",
        on_chunk=lambda _t: None,
        on_stream_start=lambda: None,
        cancellation_token=token,
        request_id="req-2",
    )

    kwargs = gemini.calls[0]["kwargs"]
    assert kwargs["on_chunk"] is None, "a non-streaming provider must not receive a chunk hook"
    assert kwargs["on_stream_start"] is None
    assert kwargs["cancellation_token"] is token
    assert kwargs["request_id"] == "req-2"
    assert res.streamed is False


def test_router_cancelled_before_start_skips_provider():
    gemini = _StreamingGemini()
    router = _build_router(gemini)
    token = CancellationToken()
    token.cancel()

    res = router.route_and_execute("hello there", cancellation_token=token)

    assert gemini.calls == [], "a pre-cancelled generation must not open a provider call"
    assert res.tier_used == IntentTier.TIER_2_GEMINI
    assert res.cancelled is True
    assert res.response_text == "Generation stopped."


def test_router_propagates_cancelled_and_partial_flags():
    gemini = _StreamingGemini(result={
        "success": False,
        "response": "Hel",
        "tool_calls": [],
        "streamed": True,
        "cancelled": True,
        "partial": True,
    })
    router = _build_router(gemini)

    res = router.route_and_execute("hello there", on_chunk=lambda _t: None)

    assert res.streamed is True
    assert res.cancelled is True
    assert res.partial is True


# ==========================================================================
# 5. MainWindow streaming + Stop
# ==========================================================================

class _CapturingPool:
    def __init__(self):
        self.workers = []

    def start(self, worker):
        self.workers.append(worker)


class _RecordingHistory:
    def __init__(self):
        self.messages = []

    def create_conversation(self, title="New Session"):
        return "conv-test"

    def add_message(self, conversation_id, role, content):
        self.messages.append((conversation_id, role, content))
        return "msg-test"


class _RecordingTTS:
    def __init__(self):
        self.spoken = []

    def speak(self, text):
        self.spoken.append(text)


class _StreamingRouter:
    """
    Fake router that drives the streaming hooks exactly as the real one would.

    ``emit_stream=False`` models a provider that cannot stream: the router never
    invokes ``on_stream_start``/``on_chunk``.  ``probe`` runs after
    ``on_stream_start`` so a test can observe UI state mid-stream (worker.run()
    is synchronous, so there is no other window).
    """

    def __init__(self, result, chunks=(), probe=None, emit_stream=True):
        self.result = result
        self.chunks = list(chunks)
        self.probe = probe
        self.emit_stream = emit_stream
        self.route_calls = []

    def route_and_execute(self, user_query=None, **kwargs):
        self.route_calls.append(user_query)
        if not self.emit_stream:
            return self.result
        start = kwargs.get("on_stream_start")
        if start is not None:
            start()
        if self.probe is not None:
            self.probe()
        chunk_cb = kwargs.get("on_chunk")
        for chunk in self.chunks:
            if chunk_cb is not None:
                chunk_cb(chunk)
        return self.result

    def execute_approved(self, approval, **kwargs):
        return ToolResult.ok(tool=approval.tool_name, message="ok")


_BEGIN_HANDLE = object()


def _build_window(monkeypatch, router, enable_voice=False):
    monkeypatch.setattr(main_window_module, "HistoryManager", _RecordingHistory)

    win = MainWindow()
    win.settings = Settings(enable_gemini=False, enable_voice=enable_voice)
    win.router = router
    win.tts = _RecordingTTS()

    pool = _CapturingPool()
    win.thread_pool = pool

    events = {"messages": [], "append": [], "finalize": [], "opens": 0}

    def begin():
        events["opens"] += 1
        return _BEGIN_HANDLE

    def append(handle, text):
        events["append"].append((handle, text))

    def finalize(handle, text=None, suffix=""):
        events["finalize"].append((handle, text, suffix))

    win.chat_view.begin_assistant_stream = begin
    win.chat_view.append_assistant_stream = append
    win.chat_view.finalize_assistant_stream = finalize
    win.chat_view.add_assistant_message = lambda text: events["messages"].append(("assistant", text))
    win.chat_view.add_user_message = lambda text: events["messages"].append(("user", text))
    win.chat_view.add_action_card = lambda **kw: events["messages"].append(("action", kw))

    return win, pool, events


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


def test_streamed_answer_is_rendered_and_persisted_once(monkeypatch, qapp):
    router = _StreamingRouter(_result(response_text="Hi there", streamed=True), chunks=["Hi", " there"])
    win, pool, events = _build_window(monkeypatch, router)
    try:
        win._handle_user_command("hello")
        assert len(pool.workers) == 1
        pool.workers[0].run()

        assert events["opens"] == 1
        assert events["append"] == [(_BEGIN_HANDLE, "Hi"), (_BEGIN_HANDLE, " there")]
        assert events["finalize"] == [(_BEGIN_HANDLE, "Hi there", "")], (
            "the bubble must be finalized with the exact persisted text"
        )
        assert ("conv-test", "user", "hello") in win.history.messages
        assert ("conv-test", "assistant", "Hi there") in win.history.messages
        # No duplicate rendering: a streamed answer is never also added as a message.
        assert ("assistant", "Hi there") not in events["messages"]
    finally:
        win.close()


def test_stop_is_visible_only_during_streaming(monkeypatch, qapp):
    observed = {}
    router = _StreamingRouter(
        _result(response_text="Hi", streamed=True),
        chunks=["Hi"],
    )
    win, pool, events = _build_window(monkeypatch, router)
    router.probe = lambda: observed.setdefault("stop_hidden_during", win.command_input.btn_stop.isHidden())
    try:
        assert win.command_input.btn_stop.isHidden() is True
        win._handle_user_command("hello")
        assert win.command_input.btn_stop.isHidden() is True, "hidden before the stream starts"

        pool.workers[0].run()

        assert observed["stop_hidden_during"] is False, "Stop must show while generating"
        assert win.command_input.btn_stop.isHidden() is True, "Stop must hide when finished"
    finally:
        win.close()


def test_non_streaming_answer_path_is_unchanged(monkeypatch, qapp):
    router = _StreamingRouter(_result(response_text="Plain"), emit_stream=False)
    win, pool, events = _build_window(monkeypatch, router)
    try:
        win._handle_user_command("hello")
        pool.workers[0].run()

        assert events["opens"] == 0
        assert events["append"] == []
        assert events["finalize"] == []
        assert ("assistant", "Plain") in events["messages"]
        assert ("conv-test", "assistant", "Plain") in win.history.messages
        assert win.command_input.btn_stop.isHidden() is True
    finally:
        win.close()


def test_cancelled_stream_shows_stopped_and_is_not_persisted(monkeypatch, qapp):
    router = _StreamingRouter(
        _result(response_text="Hel", streamed=True, cancelled=True, partial=True),
        chunks=["Hel"],
    )
    win, pool, events = _build_window(monkeypatch, router)
    try:
        win._handle_user_command("hello")
        pool.workers[0].run()

        assert events["finalize"] == [(_BEGIN_HANDLE, "Hel", " (stopped)")]
        assert not any(
            role == "assistant" for _cid, role, _content in win.history.messages
        ), "a stopped answer must never be stored as a complete response"
        assert ("conv-test", "user", "hello") in win.history.messages
    finally:
        win.close()


def test_stop_cancels_the_captured_generation_not_a_later_one(monkeypatch, qapp):
    router = _StreamingRouter(_result(response_text="Hi", streamed=True), chunks=["Hi"])
    win, pool, events = _build_window(monkeypatch, router)
    try:
        win._handle_user_command("first")
        token1 = pool.workers[0].cancellation_token
        win._handle_user_command("second")
        token2 = pool.workers[1].cancellation_token

        assert token1 is not token2
        assert token1.is_cancelled() is False

        win.command_input.stop_requested.emit()

        assert token2.is_cancelled() is True, "Stop cancels the current generation"
        assert token1.is_cancelled() is False, "Stop must never cancel an older generation"
    finally:
        win.close()


def test_stale_chunks_after_a_newer_request_are_ignored(monkeypatch, qapp):
    router = _StreamingRouter(_result(response_text="Hi", streamed=True), chunks=["Hi"])
    win, pool, events = _build_window(monkeypatch, router)
    try:
        win._handle_user_command("first")
        rid1 = win._active_request_id
        win._handle_user_command("second")
        rid2 = win._active_request_id
        assert rid1 != rid2

        win._on_stream_started(rid2, "conv-test")
        before = len(events["append"])

        win._on_chunk(rid1, "conv-test", "stale")
        assert len(events["append"]) == before, "a superseded request's chunk must be dropped"

        win._on_chunk(rid2, "conv-test", "fresh")
        assert events["append"][-1] == (_BEGIN_HANDLE, "fresh")
    finally:
        win.close()


def test_result_for_another_conversation_is_persisted_but_not_rendered(monkeypatch, qapp):
    router = _StreamingRouter(_result(response_text="Hi", streamed=True), chunks=["Hi"])
    win, pool, events = _build_window(monkeypatch, router)
    try:
        win._handle_user_command("hello")
        worker = pool.workers[0]

        # The user switches away before the answer arrives.
        win.current_conv_id = "other-conv"
        worker.run()

        assert ("assistant", "Hi") not in events["messages"], "no rendering into the wrong chat"
        assert events["append"] == [], "no streamed chunk may land in the wrong chat"
        assert ("conv-test", "assistant", "Hi") in win.history.messages, "still persisted to its own conversation"
    finally:
        win.close()


def test_duplicate_completion_persists_exactly_once(monkeypatch, qapp):
    router = _StreamingRouter(_result(response_text="Hi", streamed=True), chunks=["Hi"])
    win, pool, events = _build_window(monkeypatch, router)
    try:
        win._handle_user_command("hello")
        rid = win._active_request_id
        res = _result(response_text="Hi", streamed=True)

        win._on_route_finished(res, "conv-test", rid)
        win._on_route_finished(res, "conv-test", rid)  # a late duplicate

        assistant = [m for m in win.history.messages if m[1] == "assistant"]
        assert assistant == [("conv-test", "assistant", "Hi")]
    finally:
        win.close()


def test_provider_failure_after_partial_shows_failed_and_stores_nothing(monkeypatch, qapp):
    router = _StreamingRouter(_result(response_text="Hi", streamed=True), chunks=["Hi"])
    win, pool, events = _build_window(monkeypatch, router)
    try:
        win._handle_user_command("hello")
        rid = win._active_request_id
        win._on_stream_started(rid, "conv-test")
        win._on_chunk(rid, "conv-test", "par")

        win._on_route_error("boom", "conv-test", rid)

        assert events["finalize"] == [(_BEGIN_HANDLE, None, " (failed)")]
        assert not any(
            role == "assistant" for _cid, role, _content in win.history.messages
        ), "partial output must never be stored"
        assert ("assistant", "Error: boom") in events["messages"]
        assert win.command_input.btn_stop.isHidden() is True
    finally:
        win.close()


def test_confirmation_stream_bubble_is_closed_and_not_persisted(monkeypatch, qapp):
    created = []

    class _FakeDialog:
        def __init__(self, tool_name, message, arguments=None, parent=None):
            self.tool_name = tool_name
            self.message = message
            created.append(self)

        def exec(self):
            return False

    monkeypatch.setattr(main_window_module, "ConfirmationDialog", _FakeDialog)

    router = _StreamingRouter(
        _result(
            response_text="Confirm shutdown",
            requires_confirmation=True,
            tool_name="shutdown_pc",
            pending_approval=PendingApproval("shutdown_pc", {}),
            streamed=True,
        )
    )
    win, pool, events = _build_window(monkeypatch, router)
    try:
        win._handle_user_command("shut it down")
        pool.workers[0].run()

        assert len(created) == 1, "the confirmation must still be shown"
        assert events["finalize"] == [(_BEGIN_HANDLE, "Confirm shutdown", "")]
        assert not any(
            _content == "Confirm shutdown" for _cid, _role, _content in win.history.messages
        ), "confirmation text is not a stored response"
        assert ("assistant", "Action cancelled by user.") in events["messages"]
    finally:
        win.close()


# ==========================================================================
# 6. Safety invariant: a stopped generation never bypasses confirmation
# ==========================================================================

def test_stopped_confirmation_stream_still_gates_the_action(monkeypatch, qapp):
    created = []

    class _FakeDialog:
        def __init__(self, tool_name, message, arguments=None, parent=None):
            self.tool_name = tool_name
            created.append(self)

        def exec(self):
            return False

    monkeypatch.setattr(main_window_module, "ConfirmationDialog", _FakeDialog)

    pending = PendingApproval("shutdown_pc", {})
    router = _StreamingRouter(
        _result(
            response_text="Confirm shutdown",
            requires_confirmation=True,
            tool_name="shutdown_pc",
            pending_approval=pending,
            streamed=True,
        )
    )
    win, pool, events = _build_window(monkeypatch, router)
    try:
        win._handle_user_command("shut it down")
        worker = pool.workers[0]

        # The user stops generation while the gated action is pending.
        win.command_input.stop_requested.emit()
        worker.run()

        assert len(created) == 1, "the confirmation prompt must still appear"
        assert len(pool.workers) == 1, "a declined confirmation executes nothing"
        assert not any(kind == "action" for kind, *_ in events["messages"])
    finally:
        win.close()
