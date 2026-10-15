"""
Action Journal tests for Vector Desktop AI Assistant.

These verify that the *real* outcome of each tool execution is journaled into the
existing ``tool_calls`` SQLite table, that arguments/summaries are sanitized and
bounded before storage, that journaling is strictly fail-safe, and that the
router/executor/Gemini plumbing forwards the journal without changing behaviour.

Everything here is deterministic and side-effect free: tools are test-only fakes,
the database is a throwaway SQLite file, and the Gemini loop is driven through an
injected fake SDK -- nothing launches a program, changes a setting, touches the
network, or reaches a real model.
"""

from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
import types as pytypes
from pathlib import Path

import pytest

from app.config.constants import IntentTier, PermissionLevel
from app.memory.database import DatabaseManager
from app.memory.history import HistoryManager
from app.memory.journal import ExecutionJournal, JournalStatus, MAX_SUMMARY_LENGTH
from app.security.permissions import PermissionEngine
from app.tools.base import BaseTool, ToolResult
from app.tools.executor import PendingApproval, SafeToolExecutor
from app.tools.registry import ToolRegistry


# --------------------------------------------------------------------------
# Test-only fake tools (no real side effects)
# --------------------------------------------------------------------------

class _OkTool(BaseTool):
    name = "journal_ok"
    description = "Test-only tool that always succeeds."
    parameters = {"type": "object", "properties": {}, "required": []}
    permission_level = PermissionLevel.SAFE

    def execute(self, **kwargs):
        return ToolResult.ok(tool=self.name, message="everything fine")


class _FailingTool(BaseTool):
    name = "journal_fail"
    description = "Test-only tool that reports failure."
    parameters = {"type": "object", "properties": {}, "required": []}
    permission_level = PermissionLevel.SAFE

    def execute(self, **kwargs):
        return ToolResult.fail(tool=self.name, error="BOOM", message="tool reported boom")


class _RaisingTool(BaseTool):
    name = "journal_raise"
    description = "Test-only tool that raises."
    parameters = {"type": "object", "properties": {}, "required": []}
    permission_level = PermissionLevel.SAFE

    def execute(self, **kwargs):
        raise RuntimeError("kaboom")


class _NeedsArgTool(BaseTool):
    name = "journal_needs_arg"
    description = "Test-only tool with one required argument."
    parameters = {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    }
    permission_level = PermissionLevel.SAFE

    def execute(self, **kwargs):  # pragma: no cover - rejected before execution
        raise AssertionError("a tool with missing required args was executed")


class _BlockedTool(BaseTool):
    name = "journal_blocked"
    description = "Test-only tool pinned to BLOCKED."
    parameters = {"type": "object", "properties": {}, "required": []}
    permission_level = PermissionLevel.BLOCKED

    def execute(self, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("a BLOCKED tool was executed")


class _ConfirmTool(BaseTool):
    name = "journal_confirm"
    description = "Test-only tool that always requires confirmation."
    parameters = {
        "type": "object",
        "properties": {"target": {"type": "string"}},
        "required": ["target"],
    }
    permission_level = PermissionLevel.CONFIRM

    def execute(self, **kwargs):
        return ToolResult.ok(tool=self.name, message=f"confirmed {kwargs.get('target')}")


class _SecretTool(BaseTool):
    name = "journal_secret"
    description = "Test-only tool used to check argument/summary sanitization."
    parameters = {
        "type": "object",
        "properties": {
            "api_key": {"type": "string"},
            "note": {"type": "string"},
        },
        "required": [],
    }
    permission_level = PermissionLevel.SAFE

    def execute(self, **kwargs):
        note = kwargs.get("note", "")
        return ToolResult.ok(tool=self.name, message=f"done with password=topsecret123 note={note}")


class _DataTool(BaseTool):
    name = "journal_data"
    description = "Test-only tool returning a secret payload in ToolResult.data."
    parameters = {"type": "object", "properties": {}, "required": []}
    permission_level = PermissionLevel.SAFE

    def execute(self, **kwargs):
        return ToolResult.ok(
            tool=self.name,
            message="short summary",
            data={"secret_payload": "TOP_SECRET_PAYLOAD"},
        )


class _LongMessageTool(BaseTool):
    name = "journal_long"
    description = "Test-only tool returning an over-long message."
    parameters = {"type": "object", "properties": {}, "required": []}
    permission_level = PermissionLevel.SAFE

    def execute(self, **kwargs):
        return ToolResult.ok(tool=self.name, message="A" * (MAX_SUMMARY_LENGTH + 250))


_ALL_FAKES = [
    _OkTool, _FailingTool, _RaisingTool, _NeedsArgTool,
    _BlockedTool, _ConfirmTool, _SecretTool, _DataTool, _LongMessageTool,
]


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

@pytest.fixture()
def history():
    """A HistoryManager over a throwaway SQLite file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db = DatabaseManager(db_path=str(Path(tmpdir) / "journal.db"))
        yield HistoryManager(db_manager=db)


@pytest.fixture()
def registry():
    """The shared registry with the test-only fakes added, then removed."""
    reg = ToolRegistry()
    added = []
    for tool_cls in _ALL_FAKES:
        tool = tool_cls()
        reg.register(tool)
        added.append(tool.name)
    try:
        yield reg
    finally:
        for name in added:
            reg._tools.pop(name, None)


@pytest.fixture()
def executor(registry):
    return SafeToolExecutor(registry=registry, permission_engine=PermissionEngine())


def _rows(history, conversation_id=None):
    return history.get_recent_tool_calls(conversation_id=conversation_id, limit=100)


def _journal(history, conversation_id="conv-1"):
    return ExecutionJournal(history, conversation_id)


# --------------------------------------------------------------------------
# Status mapping: every real outcome is journaled with the right status
# --------------------------------------------------------------------------

def test_successful_execution_is_journaled(executor, history):
    j = _journal(history)
    res, _sm = executor.execute_tool("journal_ok", {}, journal=j)
    assert res.success is True

    rows = _rows(history, "conv-1")
    assert len(rows) == 1
    assert rows[0]["tool_name"] == "journal_ok"
    assert rows[0]["execution_status"] == JournalStatus.SUCCESS
    assert rows[0]["conversation_id"] == "conv-1"
    assert json.loads(rows[0]["arguments"]) == {}


def test_tool_reported_failure_is_journaled_as_failed(executor, history):
    j = _journal(history)
    res, _sm = executor.execute_tool("journal_fail", {}, journal=j)
    assert res.success is False

    rows = _rows(history)
    assert len(rows) == 1
    assert rows[0]["execution_status"] == JournalStatus.FAILED
    assert "boom" in rows[0]["summary"].lower()


def test_unhandled_exception_is_journaled_as_failed(executor, history):
    j = _journal(history)
    res, _sm = executor.execute_tool("journal_raise", {}, journal=j)
    assert res.success is False
    assert res.error == "EXECUTION_EXCEPTION"

    rows = _rows(history)
    assert len(rows) == 1
    assert rows[0]["execution_status"] == JournalStatus.FAILED
    assert "kaboom" in rows[0]["summary"]


def test_unknown_tool_is_journaled_as_tool_not_found(executor, history):
    _res, _sm = executor.execute_tool("no_such_tool_xyz", {}, journal=_journal(history))
    rows = _rows(history)
    assert len(rows) == 1
    assert rows[0]["execution_status"] == JournalStatus.TOOL_NOT_FOUND
    assert rows[0]["tool_name"] == "no_such_tool_xyz"


def test_invalid_arguments_are_journaled(executor, history):
    _res, _sm = executor.execute_tool("journal_needs_arg", {}, journal=_journal(history))
    rows = _rows(history)
    assert len(rows) == 1
    assert rows[0]["execution_status"] == JournalStatus.INVALID_ARGUMENTS
    assert "query" in rows[0]["summary"]


def test_blocked_tool_is_journaled_as_permission_denied(executor, history):
    _res, _sm = executor.execute_tool("journal_blocked", {}, journal=_journal(history))
    rows = _rows(history)
    assert len(rows) == 1
    assert rows[0]["execution_status"] == JournalStatus.PERMISSION_DENIED


def test_confirmation_request_is_journaled_and_never_success(executor, history):
    _res, _sm = executor.execute_tool(
        "journal_confirm", {"target": "disk"}, journal=_journal(history)
    )
    rows = _rows(history)
    assert len(rows) == 1
    assert rows[0]["execution_status"] == JournalStatus.CONFIRMATION_REQUIRED
    assert JournalStatus.SUCCESS not in {r["execution_status"] for r in rows}


def test_approval_mismatch_and_reuse_are_journaled_as_rejected(executor, history):
    j = _journal(history)

    # A token that authorizes a different action must be refused.
    mismatch = PendingApproval("journal_confirm", {"target": "other"})
    _res, _sm = executor.execute_tool(
        "journal_confirm", {"target": "disk"}, approval=mismatch, journal=j
    )
    assert mismatch.consumed is False

    # A genuine approval runs once; a second spend is refused.
    good = PendingApproval("journal_confirm", {"target": "disk"})
    ok_res, _sm = executor.execute_tool(
        "journal_confirm", {"target": "disk"}, approval=good, journal=j
    )
    assert ok_res.success is True
    reuse_res, _sm = executor.execute_tool(
        "journal_confirm", {"target": "disk"}, approval=good, journal=j
    )
    assert reuse_res.success is False
    assert reuse_res.error == "APPROVAL_CONSUMED"

    statuses = [r["execution_status"] for r in _rows(history)]
    assert statuses.count(JournalStatus.APPROVAL_REJECTED) == 2
    assert statuses.count(JournalStatus.SUCCESS) == 1


def test_cancelled_action_is_journaled_as_cancelled(history):
    j = _journal(history)
    entry_id = j.record(
        tool_name="journal_confirm",
        arguments={"target": "disk"},
        status=JournalStatus.CANCELLED,
        summary="Action cancelled by user.",
    )
    assert entry_id is not None

    rows = _rows(history)
    assert len(rows) == 1
    assert rows[0]["execution_status"] == JournalStatus.CANCELLED


# --------------------------------------------------------------------------
# Privacy: sanitization, truncation, and no ToolResult.data persistence
# --------------------------------------------------------------------------

def test_journal_sanitizes_secrets_in_arguments_and_summary(executor, history):
    raw_key = "AIzaSy" + ("a" * 34)
    j = _journal(history)
    executor.execute_tool(
        "journal_secret",
        {"api_key": raw_key, "note": "hello"},
        journal=j,
    )

    row = _rows(history)[0]
    stored_args = json.loads(row["arguments"])

    # A secret-looking argument value must be redacted, not stored verbatim.
    assert stored_args["api_key"] == "[REDACTED_SECRET]"
    assert raw_key not in row["arguments"]
    # ``note`` is not sensitive and is preserved.
    assert stored_args["note"] == "hello"
    # A secret embedded in the tool message must be redacted in the summary.
    assert "topsecret123" not in row["summary"]
    assert "[REDACTED_SECRET]" in row["summary"]


def test_journal_never_persists_tool_result_data(executor, history):
    executor.execute_tool("journal_data", {}, journal=_journal(history))
    row = _rows(history)[0]

    assert row["summary"] == "short summary"
    combined = " ".join(str(v) for v in row.values())
    assert "TOP_SECRET_PAYLOAD" not in combined
    assert "secret_payload" not in combined


def test_over_long_summary_is_truncated(executor, history):
    executor.execute_tool("journal_long", {}, journal=_journal(history))
    row = _rows(history)[0]
    assert len(row["summary"]) == MAX_SUMMARY_LENGTH


# --------------------------------------------------------------------------
# Fail-safe: journaling never changes the execution outcome
# --------------------------------------------------------------------------

class _ExplodingHistory:
    """A HistoryManager stand-in whose every write raises."""

    def add_execution(self, **kwargs):
        raise RuntimeError("disk full")


class _ExplodingJournal:
    def record(self, **kwargs):
        raise RuntimeError("journal unavailable")


def test_record_returns_none_when_history_is_missing_or_unusable():
    assert ExecutionJournal(None, "c").record("t", {}, "SUCCESS") is None
    assert ExecutionJournal(object(), "c").record("t", {}, "SUCCESS") is None
    assert ExecutionJournal(_ExplodingHistory(), "c").record("t", {}, "SUCCESS") is None


def test_journal_failure_never_changes_execution_outcome(executor, history):
    j = ExecutionJournal(_ExplodingHistory(), "conv-x")
    res, _sm = executor.execute_tool("journal_ok", {}, journal=j)
    # The tool still succeeded and no exception escaped despite the failing journal.
    assert res.success is True

    # A journal whose record() itself raises is also contained by the executor.
    res2, _sm = executor.execute_tool("journal_ok", {}, journal=_ExplodingJournal())
    assert res2.success is True


def test_no_journal_is_a_clean_noop(executor, history):
    res, _sm = executor.execute_tool("journal_ok", {})
    assert res.success is True
    assert _rows(history) == []


# --------------------------------------------------------------------------
# Querying
# --------------------------------------------------------------------------

def test_get_recent_tool_calls_scopes_by_conversation_newest_first(history):
    _journal(history, "conv-a").record("journal_ok", {}, JournalStatus.SUCCESS, "a1")
    _journal(history, "conv-b").record("journal_ok", {}, JournalStatus.SUCCESS, "b1")
    _journal(history, "conv-a").record("journal_ok", {}, JournalStatus.FAILED, "a2")

    conv_a = _rows(history, "conv-a")
    assert [r["summary"] for r in conv_a] == ["a2", "a1"]
    assert all(r["conversation_id"] == "conv-a" for r in conv_a)

    everything = _rows(history)
    assert len(everything) == 3


# --------------------------------------------------------------------------
# Router plumbing (Tier 0 confirmation -> approved execution)
# --------------------------------------------------------------------------

class _StubTier0Matcher:
    """Forces Tier 0 to resolve the query to the confirmation-requiring fake."""

    @staticmethod
    def match(_query):
        from app.core.matcher import MatchResult

        return MatchResult(
            matched=True,
            tool_name="journal_confirm",
            arguments={"target": "disk"},
            confidence=1.0,
            reason="stub",
        )


def test_router_journals_confirmation_then_approved_execution(monkeypatch, registry, executor, history):
    import app.core.router as router_mod
    from app.config.settings import Settings

    monkeypatch.setattr(router_mod, "Tier0Matcher", _StubTier0Matcher)

    router = router_mod.IntentRouter(
        settings=Settings(enable_gemini=False),
        registry=registry,
        executor=executor,
        history_manager=history,
    )

    route_res = router.route_and_execute("journal test", conversation_id="conv-r")
    assert route_res.requires_confirmation is True
    assert route_res.pending_approval is not None

    # The request itself is journaled as pending, never as success.
    pending = _rows(history, "conv-r")
    assert [r["execution_status"] for r in pending] == [JournalStatus.CONFIRMATION_REQUIRED]

    approved = router.execute_approved(route_res.pending_approval, conversation_id="conv-r")
    assert approved.success is True

    statuses = [r["execution_status"] for r in _rows(history, "conv-r")]
    assert statuses == [JournalStatus.SUCCESS, JournalStatus.CONFIRMATION_REQUIRED]
    assert all(r["conversation_id"] == "conv-r" for r in _rows(history, "conv-r"))


def test_router_without_history_manager_does_not_journal(monkeypatch, registry, executor):
    import app.core.router as router_mod
    from app.config.settings import Settings

    monkeypatch.setattr(router_mod, "Tier0Matcher", _StubTier0Matcher)

    router = router_mod.IntentRouter(
        settings=Settings(enable_gemini=False),
        registry=registry,
        executor=executor,
        history_manager=None,
    )
    route_res = router.route_and_execute("journal test", conversation_id="conv-r")
    # No history manager -> no journal; behaviour is unchanged.
    assert route_res.requires_confirmation is True


# --------------------------------------------------------------------------
# UI cancellation boundary
# --------------------------------------------------------------------------

def test_ui_cancellation_path_journals_cancelled(history):
    from app.core.router import RouteResult
    from app.ui.main_window import MainWindow

    stub_self = pytypes.SimpleNamespace(history=history, current_conv_id="conv-ui")
    route_res = RouteResult(
        tier_used=IntentTier.TIER_0_DIRECT,
        tool_name="journal_confirm",
        raw_arguments={"target": "disk"},
    )

    MainWindow._journal_cancellation(stub_self, route_res)

    rows = _rows(history, "conv-ui")
    assert len(rows) == 1
    assert rows[0]["execution_status"] == JournalStatus.CANCELLED
    assert rows[0]["tool_name"] == "journal_confirm"


# --------------------------------------------------------------------------
# Gemini loop forwarding
# --------------------------------------------------------------------------

class _RecordingExecutor:
    def __init__(self):
        self.calls = []

    def execute_tool(self, tool_name, raw_args, journal=None, **kwargs):
        self.calls.append({"tool_name": tool_name, "raw_args": raw_args, "journal": journal})
        return ToolResult.ok(tool=tool_name, message="done"), None


class _StubSettings:
    is_gemini_available = True
    gemini_api_key = "test-key"
    gemini_model = "test-model"


def _install_fake_genai(monkeypatch):
    """Injects a minimal fake ``google.genai`` that yields exactly one tool call."""

    class _Response:
        def __init__(self, function_calls, text):
            self.function_calls = function_calls
            self.text = text

    class _FuncCall:
        def __init__(self, name, args):
            self.name = name
            self.args = args

    calls = {"sent": 0}

    class _Chat:
        def send_message(self, _content):
            calls["sent"] += 1
            if calls["sent"] == 1:
                return _Response([_FuncCall("journal_ok", {})], None)
            return _Response([], "all done")

    class _Chats:
        def create(self, **kwargs):
            return _Chat()

    class _Client:
        def __init__(self, api_key=None):
            self.chats = _Chats()

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

    types_mod = pytypes.SimpleNamespace(
        GenerateContentConfig=lambda **kwargs: kwargs,
        Part=_Part,
        Content=_Content,
    )
    genai_mod = pytypes.SimpleNamespace(Client=_Client, types=types_mod)
    google_pkg = pytypes.SimpleNamespace(genai=genai_mod)

    monkeypatch.setitem(sys.modules, "google", google_pkg)
    monkeypatch.setitem(sys.modules, "google.genai", genai_mod)


def test_gemini_loop_forwards_journal_to_executor(monkeypatch, registry):
    from app.gemini.client import GeminiClient

    _install_fake_genai(monkeypatch)

    recording_executor = _RecordingExecutor()
    client = GeminiClient(
        settings=_StubSettings(),
        registry=registry,
        executor=recording_executor,
    )

    sentinel = object()
    out = client.execute_reasoning_loop(user_query="do the thing", journal=sentinel)

    assert out["success"] is True
    assert len(recording_executor.calls) == 1
    assert recording_executor.calls[0]["tool_name"] == "journal_ok"
    # The very same journal object flows through to the executor.
    assert recording_executor.calls[0]["journal"] is sentinel


# --------------------------------------------------------------------------
# Schema migration
# --------------------------------------------------------------------------

_OLD_SCHEMA = """
CREATE TABLE conversations (
    id TEXT PRIMARY KEY, created_at TEXT NOT NULL, title TEXT NOT NULL
);
CREATE TABLE messages (
    id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL, role TEXT NOT NULL,
    content TEXT NOT NULL, timestamp TEXT NOT NULL,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
);
CREATE TABLE tool_calls (
    id TEXT PRIMARY KEY,
    message_id TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    arguments TEXT NOT NULL,
    confidence REAL NOT NULL,
    execution_status TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    FOREIGN KEY (message_id) REFERENCES messages(id) ON DELETE CASCADE
);
INSERT INTO conversations VALUES ('c1', 't', 'legacy');
INSERT INTO messages VALUES ('m1', 'c1', 'user', 'hi', 't');
INSERT INTO tool_calls VALUES ('tc1', 'm1', 'set_volume', '{"level": 30}', 1.0, 'SUCCESS', 't');
"""


def test_old_tool_calls_schema_is_migrated_idempotently():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = str(Path(tmpdir) / "legacy.db")

        conn = sqlite3.connect(db_path)
        conn.executescript(_OLD_SCHEMA)
        conn.commit()
        conn.close()

        # Opening with DatabaseManager triggers the migration.
        db = DatabaseManager(db_path=db_path)

        with db.get_connection() as c:
            cols = {r["name"]: r for r in c.execute("PRAGMA table_info(tool_calls);").fetchall()}
            assert "conversation_id" in cols
            assert "summary" in cols
            assert cols["message_id"]["notnull"] == 0

            legacy = c.execute("SELECT * FROM tool_calls WHERE id = 'tc1';").fetchone()
            assert legacy["tool_name"] == "set_volume"
            assert legacy["conversation_id"] is None
            assert legacy["summary"] is None

        # Re-opening must be a no-op (idempotent) and preserve the legacy row.
        DatabaseManager(db_path=db_path)
        with db.get_connection() as c:
            count = c.execute("SELECT COUNT(*) AS n FROM tool_calls;").fetchone()["n"]
            assert count == 1

        # And the migrated table accepts a message-less execution.
        hm = HistoryManager(db_manager=db)
        entry_id = hm.add_execution("get_cpu_usage", {}, "SUCCESS", "ok", conversation_id="c2")
        assert entry_id is not None
