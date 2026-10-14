"""
MainWindow integration tests for the voice-to-command and confirmation UI seam.

These tests drive ``MainWindow._on_transcribe_finished``, ``_handle_user_command``
and ``_on_route_finished`` directly with a fake router, a capturing thread pool, a
recording history/TTS, and a monkeypatched ``ConfirmationDialog``.

Nothing here opens a microphone, touches the network, or loads the Needle /
Gemini / Whisper models: the router is replaced before any worker is run, and
``MainWindow.__init__`` only *constructs* lazy clients (no model, no call).

The confirmation contract under test is action-bound: approving a modal runs the
exact (tool, arguments) that was displayed, once, without re-resolving the
original query -- and a consumed/absent approval executes nothing.
"""

import os

os.environ["QT_QPA_PLATFORM"] = "offscreen"

import pytest

from app.config.constants import IntentTier
from app.config.settings import Settings
from app.core.router import RouteResult
from app.tools.base import ToolResult
from app.tools.executor import PendingApproval

import app.ui.main_window as main_window_module
from app.ui.main_window import ApprovalWorker, MainWindow, RouteWorker


# --------------------------------------------------------------------------
# Fixtures and fakes
# --------------------------------------------------------------------------

@pytest.fixture(scope="session")
def qapp():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


class _CapturingPool:
    """Stands in for QThreadPool: records workers instead of scheduling them."""

    def __init__(self):
        self.workers = []

    def start(self, worker):
        self.workers.append(worker)


class _FakeRouter:
    """Canned routing result plus a recorded, action-bound approval path.

    ``execute_approved`` never re-runs ``route_and_execute``: the two are
    recorded separately so a test can prove the original query was not rerouted.
    """

    def __init__(self, respond, approve=None):
        self._respond = respond
        self._approve = approve or (
            lambda approval: ToolResult.ok(
                tool=approval.tool_name, message=f"Approved {approval.tool_name}"
            )
        )
        self.route_calls = []
        self.approval_calls = []

    def route_and_execute(self, user_query=None, **kwargs):
        self.route_calls.append(user_query)
        return self._respond(user_query)

    def execute_approved(self, approval, **kwargs):
        self.approval_calls.append(approval)
        return self._approve(approval)


class _RaisingRouter:
    """Raises on every call, to exercise the worker error path."""

    def __init__(self, exc):
        self.exc = exc
        self.route_calls = []
        self.approval_calls = []

    def route_and_execute(self, user_query=None, **kwargs):
        self.route_calls.append(user_query)
        raise self.exc

    def execute_approved(self, approval, **kwargs):
        self.approval_calls.append(approval)
        raise self.exc


class _RecordingHistory:
    """In-memory stand-in for HistoryManager; never touches the SQLite database."""

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


def _install_dialog(monkeypatch, accept):
    """Replace ConfirmationDialog so exec() returns ``accept`` without a modal."""
    created = []

    class _FakeDialog:
        def __init__(self, tool_name, message, arguments=None, parent=None):
            self.tool_name = tool_name
            self.message = message
            self.arguments = arguments
            created.append(self)

        def exec(self):
            return accept

    monkeypatch.setattr(main_window_module, "ConfirmationDialog", _FakeDialog)
    return created


def _build_window(monkeypatch, router, enable_voice=True):
    """A MainWindow wired to fakes; returns (window, pool, recorded messages)."""
    monkeypatch.setattr(main_window_module, "HistoryManager", _RecordingHistory)

    win = MainWindow()
    win.settings = Settings(enable_gemini=False, enable_voice=enable_voice)
    win.router = router
    win.tts = _RecordingTTS()

    pool = _CapturingPool()
    win.thread_pool = pool

    messages = []
    win.chat_view.add_assistant_message = lambda text: messages.append(("assistant", text))
    win.chat_view.add_user_message = lambda text: messages.append(("user", text))
    win.chat_view.add_action_card = lambda **kw: messages.append(("action", kw))

    return win, pool, messages


def _pending(tool_name, arguments=None):
    return PendingApproval(tool_name=tool_name, arguments=arguments or {})


def _result(
    response_text="Done.",
    tool_result=None,
    requires_confirmation=False,
    tool_name="",
    raw_arguments=None,
    pending_approval=None,
    completed_results=None,
    tier=IntentTier.TIER_0_DIRECT,
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
    )


# --------------------------------------------------------------------------
# Empty transcript must not reach the router at all
# --------------------------------------------------------------------------

@pytest.mark.parametrize("text", ["", "   ", "\n\t"])
def test_empty_transcript_does_not_route_or_execute(monkeypatch, qapp, text):
    router = _FakeRouter(lambda q: _result())
    win, pool, messages = _build_window(monkeypatch, router)
    try:
        win._on_transcribe_finished(text)

        assert pool.workers == [], "an empty transcript must not start a routing worker"
        assert router.route_calls == [], "an empty transcript must never reach the router"
        assert (
            "assistant",
            "No speech detected. Please speak clearly into your microphone.",
        ) in messages
    finally:
        win.close()


# --------------------------------------------------------------------------
# Non-empty transcript is routed and its result is rendered
# --------------------------------------------------------------------------

def test_transcript_is_routed_and_result_rendered(monkeypatch, qapp):
    tool_result = ToolResult.ok(tool="set_volume", message="Volume set to 30")
    router = _FakeRouter(lambda q: _result(tool_result=tool_result))
    win, pool, messages = _build_window(monkeypatch, router)
    try:
        win._on_transcribe_finished("  set volume to 30  ")

        assert len(pool.workers) == 1
        worker = pool.workers[0]
        assert isinstance(worker, RouteWorker)
        assert worker.user_query == "set volume to 30", "transcript must be stripped before routing"

        worker.run()  # fake router -> result signal -> _on_route_finished

        assert router.route_calls == ["set volume to 30"]
        assert ("user", "set volume to 30") in messages
        assert ("assistant", "Volume set to 30") in messages
        assert any(kind == "action" for kind, _ in messages), "tool execution must render an action card"
    finally:
        win.close()


# --------------------------------------------------------------------------
# Typed command enters the same handler as a voice transcript
# --------------------------------------------------------------------------

def test_typed_command_uses_the_same_handler(monkeypatch, qapp):
    router = _FakeRouter(lambda q: _result(tool_result=ToolResult.ok(tool="launch_app", message="Opened.")))
    win, pool, messages = _build_window(monkeypatch, router)
    try:
        win.command_input.command_submitted.emit("open chrome")

        assert len(pool.workers) == 1
        assert isinstance(pool.workers[0], RouteWorker)
        assert pool.workers[0].user_query == "open chrome"
        assert ("user", "open chrome") in messages
    finally:
        win.close()


# --------------------------------------------------------------------------
# Confirmation accepted: the displayed action is executed verbatim, once
# --------------------------------------------------------------------------

def test_confirmation_accepted_executes_bound_action(monkeypatch, qapp):
    created = _install_dialog(monkeypatch, accept=True)
    pending = _pending("shutdown_pc", {})

    router = _FakeRouter(
        lambda q: _result(
            response_text="Confirm shutdown",
            requires_confirmation=True,
            tool_name="shutdown_pc",
            raw_arguments={},
            pending_approval=pending,
        )
    )
    win, pool, messages = _build_window(monkeypatch, router)
    try:
        win._on_transcribe_finished("shut down the pc")
        assert len(pool.workers) == 1
        pool.workers[0].run()  # -> confirmation required -> dialog accepted

        assert len(created) == 1, "the confirmation modal must be shown exactly once"
        assert created[0].tool_name == "shutdown_pc"

        assert len(pool.workers) == 2, "accepting must start an approval worker"
        assert isinstance(pool.workers[1], ApprovalWorker)
        assert pool.workers[1].approval is pending, "the worker must carry the bound pending approval"

        pool.workers[1].run()  # executes the approved action

        assert router.route_calls == ["shut down the pc"], "the original query must not be rerouted on approval"
        assert router.approval_calls == [pending]
        assert ("assistant", "Approved shutdown_pc") in messages
        assert any(
            kind == "action" and kw.get("tool_name") == "shutdown_pc"
            for kind, kw in messages
        )
    finally:
        win.close()


# --------------------------------------------------------------------------
# Confirmation cancelled: reported, nothing executes
# --------------------------------------------------------------------------

def test_confirmation_cancelled_reports_and_executes_nothing(monkeypatch, qapp):
    created = _install_dialog(monkeypatch, accept=False)

    router = _FakeRouter(
        lambda q: _result(
            response_text="Confirm shutdown",
            requires_confirmation=True,
            tool_name="shutdown_pc",
            pending_approval=_pending("shutdown_pc", {}),
        )
    )
    win, pool, messages = _build_window(monkeypatch, router)
    try:
        win._on_transcribe_finished("shut down the pc")
        pool.workers[0].run()

        assert len(created) == 1
        assert len(pool.workers) == 1, "cancel must not start an approval worker"
        assert router.route_calls == ["shut down the pc"]
        assert router.approval_calls == [], "cancel must not execute anything"
        assert ("assistant", "Action cancelled by user.") in messages
    finally:
        win.close()


# --------------------------------------------------------------------------
# A missing / consumed approval executes nothing
# --------------------------------------------------------------------------

def test_missing_pending_approval_executes_nothing(monkeypatch, qapp):
    created = _install_dialog(monkeypatch, accept=True)

    router = _FakeRouter(
        lambda q: _result(
            response_text="Confirm shutdown",
            requires_confirmation=True,
            tool_name="shutdown_pc",
            pending_approval=None,
        )
    )
    win, pool, messages = _build_window(monkeypatch, router)
    try:
        win._on_transcribe_finished("shut down the pc")
        pool.workers[0].run()

        assert len(created) == 1
        assert len(pool.workers) == 1, "no approval token => no approval worker"
        assert router.approval_calls == []
        assert (
            "assistant",
            "This action could not be confirmed safely. Please try again.",
        ) in messages
    finally:
        win.close()


# --------------------------------------------------------------------------
# Binding: an approving pass cannot substitute a different action
# --------------------------------------------------------------------------

def test_confirmation_is_bound_to_the_displayed_action(monkeypatch, qapp):
    """
    Regression for the pre-Milestone-B defect.

    The modal displays ``shutdown_pc``.  A hypothetical re-resolution (e.g. the
    Needle / Gemini tier) would now yield ``restart_pc``.  Because approval is
    bound to the displayed action and never reroutes, only ``shutdown_pc`` may
    execute.
    """
    created = _install_dialog(monkeypatch, accept=True)
    pending = _pending("shutdown_pc", {})

    def respond(query):
        # If the window ever rerouted on approval it would get restart_pc.
        return _result(
            response_text="Confirm shutdown",
            requires_confirmation=True,
            tool_name="shutdown_pc",
            pending_approval=pending,
        )

    def approve(approval):
        # A re-resolution would have produced restart_pc; assert it never does.
        assert approval.tool_name == "shutdown_pc"
        return ToolResult.ok(tool=approval.tool_name, message="Shutting down.")

    router = _FakeRouter(respond, approve=approve)
    win, pool, messages = _build_window(monkeypatch, router)
    try:
        win._on_transcribe_finished("shut down the pc")
        pool.workers[0].run()

        assert created[0].tool_name == "shutdown_pc"

        pool.workers[1].run()

        assert router.route_calls == ["shut down the pc"], "approval must not reroute the query"
        assert router.approval_calls == [pending]
        assert pending.tool_name == "shutdown_pc"
        assert not any(
            kind == "action" and kw.get("tool_name") == "restart_pc"
            for kind, kw in messages
        ), "an action other than the approved one must never execute"
    finally:
        win.close()


# --------------------------------------------------------------------------
# Already-completed steps are reported but never repeated on approval
# --------------------------------------------------------------------------

def test_completed_steps_are_reported_and_not_repeated(monkeypatch, qapp):
    created = _install_dialog(monkeypatch, accept=True)
    done = ToolResult.ok(tool="set_volume", message="Volume set to 30")
    pending = _pending("shutdown_pc", {})

    router = _FakeRouter(
        lambda q: _result(
            response_text="Confirm shutdown",
            requires_confirmation=True,
            tool_name="shutdown_pc",
            pending_approval=pending,
            completed_results=[done],
        )
    )
    win, pool, messages = _build_window(monkeypatch, router)
    try:
        win._on_transcribe_finished("turn it up and shut down")
        pool.workers[0].run()

        # The safe step that already ran is surfaced...
        assert ("assistant", "Volume set to 30") in messages
        assert any(
            kind == "action" and kw.get("tool_name") == "set_volume"
            for kind, kw in messages
        )
        assert len(created) == 1

        pool.workers[1].run()

        # ...and only the single gated action was added by the approval.
        assert router.approval_calls == [pending]
        assert sum(
            1 for kind, kw in messages
            if kind == "action" and kw.get("tool_name") == "set_volume"
        ) == 1, "an already-executed step must not run a second time"
    finally:
        win.close()


# --------------------------------------------------------------------------
# Worker error: surfaced, no execution
# --------------------------------------------------------------------------

def test_worker_error_is_reported_and_does_not_execute(monkeypatch, qapp):
    router = _RaisingRouter(RuntimeError("router exploded"))
    win, pool, messages = _build_window(monkeypatch, router)
    try:
        win._on_transcribe_finished("do something")

        assert len(pool.workers) == 1
        pool.workers[0].run()  # raises internally -> error signal

        assert router.route_calls == ["do something"]
        assert router.approval_calls == []
        assert ("assistant", "Error: router exploded") in messages
        assert not any(kind == "action" for kind, _ in messages), "a failed worker must not execute a tool"
    finally:
        win.close()
