"""
Focused tests for the Conversation Manager (Feature 8).

Everything is offline and deterministic: temporary SQLite databases hold real
``HistoryManager`` state, the Qt windows run headless (``offscreen``), and every
modal dialog (delete confirmation, rename prompt, error box) is bypassed through
a seam.  No microphone, network, model or credential is ever touched.
"""

import os
from datetime import datetime, timedelta, timezone

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from app.config.constants import IntentTier  # noqa: E402
from app.config.settings import Settings  # noqa: E402
from app.core.router import RouteResult  # noqa: E402
from app.memory import conversation_io as cio  # noqa: E402
from app.memory.database import DatabaseManager  # noqa: E402
from app.memory.history import (  # noqa: E402
    DEFAULT_CONVERSATION_TITLE,
    HistoryManager,
    normalize_title,
)
from app.tools.executor import PendingApproval  # noqa: E402
from app.ui.conversation_sidebar import (  # noqa: E402
    EMPTY_TEXT,
    GROUP_OLDER,
    GROUP_PREVIOUS_7_DAYS,
    GROUP_TODAY,
    GROUP_YESTERDAY,
    ConversationSidebar,
    group_conversations,
)

import app.ui.main_window as main_window_module  # noqa: E402
from app.ui.main_window import MainWindow, RouteWorker  # noqa: E402

_ROLE_CONVERSATION_ID = Qt.UserRole + 1


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


# --------------------------------------------------------------------------
# Helpers / fakes
# --------------------------------------------------------------------------
def make_history(tmp_path, name="cm.db"):
    db = DatabaseManager(db_path=str(tmp_path / name))
    return HistoryManager(db_manager=db)


class _CapturingPool:
    """Stands in for QThreadPool: records workers instead of scheduling them."""

    def __init__(self):
        self.workers = []

    def start(self, worker):
        self.workers.append(worker)


class _FakeRouter:
    def __init__(self, respond):
        self._respond = respond
        self.route_calls = []

    def route_and_execute(self, user_query=None, **kwargs):
        self.route_calls.append(user_query)
        return self._respond(user_query)


def _result(response_text="Done.", **kwargs):
    return RouteResult(
        tier_used=kwargs.pop("tier", IntentTier.TIER_0_DIRECT),
        tool_result=kwargs.pop("tool_result", None),
        response_text=response_text,
        requires_confirmation=kwargs.pop("requires_confirmation", False),
        tool_name=kwargs.pop("tool_name", ""),
        raw_arguments=kwargs.pop("raw_arguments", {}),
        pending_approval=kwargs.pop("pending_approval", None),
        completed_results=kwargs.pop("completed_results", []),
    )


def _make_window(monkeypatch, history):
    """Construct a MainWindow bound to ``history`` without touching a modal."""
    monkeypatch.setattr(main_window_module, "HistoryManager", lambda: history)
    win = MainWindow()
    win.settings = Settings(enable_gemini=False, enable_voice=False)
    win.thread_pool = _CapturingPool()
    return win


def _item_texts(list_widget):
    texts = []
    for row in range(list_widget.count()):
        widget = list_widget.itemWidget(list_widget.item(row))
        if widget is not None and hasattr(widget, "text"):
            texts.append(widget.text())
    return texts


# ==========================================================================
# HistoryManager: listing / counting / paging
# ==========================================================================
def test_list_conversations_orders_by_recent_activity(tmp_path):
    hm = make_history(tmp_path)
    a = hm.create_conversation("A")
    b = hm.create_conversation("B")
    hm.add_message(a, "user", "older")
    hm.add_message(b, "user", "newer")

    summaries = hm.list_conversations()

    assert [s["id"] for s in summaries] == [b, a]
    assert summaries[0]["message_count"] == 1
    assert summaries[0]["last_activity"]
    assert summaries[0]["title"] == "B"


def test_count_and_page_paginate_without_truncation(tmp_path):
    hm = make_history(tmp_path)
    conv = hm.create_conversation("Long")
    for i in range(120):
        hm.add_message(conv, "user" if i % 2 == 0 else "assistant", f"msg-{i:03d}")

    assert hm.count_conversation_messages(conv) == 120

    page0 = hm.get_conversation_page(conv, limit=50, offset=0)
    page1 = hm.get_conversation_page(conv, limit=50, offset=50)
    page2 = hm.get_conversation_page(conv, limit=50, offset=100)

    assert [m["content"] for m in page0][0] == "msg-070"
    assert [m["content"] for m in page0][-1] == "msg-119"
    assert [m["content"] for m in page1][0] == "msg-020"
    assert [m["content"] for m in page2][0] == "msg-000"
    assert len(page2) == 20

    combined = page2 + page1 + page0
    assert [m["content"] for m in combined] == [f"msg-{i:03d}" for i in range(120)]


def test_page_preserves_chronological_order_with_ties(tmp_path):
    hm = make_history(tmp_path)
    stamp = "2026-01-01T00:00:00+00:00"
    hm.import_conversation(
        "ties-conv",
        "Ties",
        stamp,
        [{"role": "user", "content": f"m{i}", "timestamp": stamp} for i in range(5)],
    )

    page = hm.get_conversation_page("ties-conv", limit=10, offset=0)

    assert [m["content"] for m in page] == ["m0", "m1", "m2", "m3", "m4"]


# ==========================================================================
# HistoryManager: search
# ==========================================================================
def test_search_matches_title_and_content(tmp_path):
    hm = make_history(tmp_path)
    a = hm.create_conversation("Groceries")
    b = hm.create_conversation("Work")
    hm.add_message(b, "user", "remember the secret token word")

    assert [s["id"] for s in hm.search_conversations("groc")] == [a]
    assert [s["id"] for s in hm.search_conversations("token word")] == [b]
    assert {s["id"] for s in hm.search_conversations("e")} >= {a, b}
    assert hm.search_conversations("nothing-here") == []
    # Blank query lists everything (browse behaviour).
    assert {s["id"] for s in hm.search_conversations("   ")} == {a, b}


def test_search_escapes_like_wildcards(tmp_path):
    hm = make_history(tmp_path)
    a = hm.create_conversation("Percent")
    b = hm.create_conversation("Plain")
    hm.add_message(a, "user", "battery at 100% now")
    hm.add_message(b, "user", "no special chars")

    # A literal '%' must match only the conversation that literally contains it.
    assert [s["id"] for s in hm.search_conversations("%")] == [a]


# ==========================================================================
# HistoryManager: create / rename / delete
# ==========================================================================
def test_create_and_normalize_title(tmp_path):
    hm = make_history(tmp_path)
    assert normalize_title("   ") == DEFAULT_CONVERSATION_TITLE
    assert normalize_title("  Hello  ") == "Hello"

    blank = hm.create_conversation("   ")
    assert hm.get_conversation(blank)["title"] == DEFAULT_CONVERSATION_TITLE


def test_rename_conversation_normalizes_and_reports(tmp_path):
    hm = make_history(tmp_path)
    conv = hm.create_conversation("Old")

    assert hm.rename_conversation(conv, "  New Name  ") is True
    assert hm.get_conversation(conv)["title"] == "New Name"

    assert hm.rename_conversation(conv, "   ") is True
    assert hm.get_conversation(conv)["title"] == DEFAULT_CONVERSATION_TITLE

    assert hm.rename_conversation("missing-id", "x") is False


def test_delete_conversation_scoped_messages_and_journal(tmp_path):
    hm = make_history(tmp_path)
    a = hm.create_conversation("A")
    b = hm.create_conversation("B")
    hm.add_message(a, "user", "in-a")
    hm.add_message(b, "user", "in-b")
    hm.add_execution(
        tool_name="set_volume", arguments={"level": 30}, status="SUCCESS",
        summary="a journal", conversation_id=a,
    )
    hm.add_execution(
        tool_name="mute", arguments={}, status="SUCCESS",
        summary="b journal", conversation_id=b,
    )

    assert hm.delete_conversation(a) is True

    assert hm.get_conversation(a) is None
    assert hm.count_conversation_messages(a) == 0
    assert hm.get_recent_tool_calls(conversation_id=a) == []

    # Unrelated conversation is untouched.
    assert hm.get_conversation(b) is not None
    assert [m["content"] for m in hm.get_conversation_messages(b, limit=None)] == ["in-b"]
    assert len(hm.get_recent_tool_calls(conversation_id=b)) == 1

    assert hm.delete_conversation(a) is False


# ==========================================================================
# Grouping helper (pure)
# ==========================================================================
def test_group_conversations_buckets_by_recency():
    now = datetime(2026, 10, 17, 12, 0, tzinfo=timezone.utc)

    def conv(cid, when):
        return {"id": cid, "title": cid, "last_activity": when}

    conversations = [
        conv("today", "2026-10-17T09:00:00+00:00"),
        conv("yesterday", "2026-10-16T09:00:00+00:00"),
        conv("three-days", "2026-10-14T09:00:00+00:00"),
        conv("ten-days", "2026-10-07T09:00:00+00:00"),
    ]

    groups = group_conversations(conversations, now=now)

    assert [name for name, _ in groups] == [
        GROUP_TODAY, GROUP_YESTERDAY, GROUP_PREVIOUS_7_DAYS, GROUP_OLDER,
    ]
    mapping = {name: [c["id"] for c in items] for name, items in groups}
    assert mapping[GROUP_TODAY] == ["today"]
    assert mapping[GROUP_YESTERDAY] == ["yesterday"]
    assert mapping[GROUP_PREVIOUS_7_DAYS] == ["three-days"]
    assert mapping[GROUP_OLDER] == ["ten-days"]


def test_group_conversations_treats_bad_timestamps_as_older():
    now = datetime(2026, 10, 17, 12, 0, tzinfo=timezone.utc)
    groups = group_conversations(
        [{"id": "x", "last_activity": "not-a-date"}, {"id": "y", "last_activity": None}],
        now=now,
    )
    assert [name for name, _ in groups] == [GROUP_OLDER]
    assert {c["id"] for _, items in groups for c in items} == {"x", "y"}


# ==========================================================================
# Sidebar widget
# ==========================================================================
def _sidebar_entries():
    now = datetime(2026, 10, 17, 12, 0, tzinfo=timezone.utc)
    return now, [
        {"id": "t1", "title": "Today chat", "last_activity": "2026-10-17T11:00:00+00:00", "message_count": 2},
        {"id": "y1", "title": "Yesterday chat", "last_activity": "2026-10-16T11:00:00+00:00", "message_count": 5},
    ]


def test_sidebar_groups_and_highlights_active(qapp):
    sidebar = ConversationSidebar()
    now, entries = _sidebar_entries()
    sidebar.set_conversations(entries, now=now)

    texts = [sidebar.list_widget.item(r).text() for r in range(sidebar.list_widget.count())]
    assert "Today" in texts
    assert "Yesterday" in texts
    assert any("Today chat" in t for t in texts)
    assert any("Yesterday chat" in t for t in texts)

    sidebar.set_active("t1")
    active_row = sidebar.list_widget.currentItem()
    assert active_row is not None
    assert active_row.data(_ROLE_CONVERSATION_ID) == "t1"
    assert active_row.font().bold() is True


def test_sidebar_emits_selection(qapp):
    sidebar = ConversationSidebar()
    now, entries = _sidebar_entries()
    sidebar.set_conversations(entries, now=now)

    received = []
    sidebar.conversation_selected.connect(received.append)

    target_row = next(
        r for r in range(sidebar.list_widget.count())
        if sidebar.list_widget.item(r).data(_ROLE_CONVERSATION_ID) == "y1"
    )
    sidebar.list_widget.setCurrentRow(target_row)

    assert received == ["y1"]


def test_sidebar_empty_and_no_match_states(qapp):
    sidebar = ConversationSidebar()

    sidebar.set_conversations([])
    assert sidebar.lbl_empty.text() == EMPTY_TEXT

    sidebar.search.setText("anything")
    sidebar._update_empty_state(False)
    assert sidebar.lbl_empty.text() != EMPTY_TEXT

    now, entries = _sidebar_entries()
    sidebar.search.setText("")
    sidebar.set_conversations(entries, now=now)
    assert sidebar.list_widget.count() > 0


# ==========================================================================
# MainWindow: startup / listing / switching / pagination
# ==========================================================================
def test_startup_creates_new_chat_when_empty(monkeypatch, qapp, tmp_path):
    hm = make_history(tmp_path)
    win = _make_window(monkeypatch, hm)
    try:
        assert win.current_conv_id
        assert hm.get_conversation(win.current_conv_id) is not None
        assert hm.get_conversation(win.current_conv_id)["title"] == DEFAULT_CONVERSATION_TITLE
        # A single empty conversation exists and shows the welcome text.
        assert len(hm.list_conversations()) == 1
        assert win.chat_view.list_widget.count() == 1
    finally:
        win.close()


def test_startup_reopens_most_recent_conversation(monkeypatch, qapp, tmp_path):
    hm = make_history(tmp_path)
    a = hm.create_conversation("A")
    hm.add_message(a, "user", "a1")
    b = hm.create_conversation("B")
    hm.add_message(b, "user", "b1")
    hm.add_message(b, "assistant", "b2")

    win = _make_window(monkeypatch, hm)
    try:
        assert win.current_conv_id == b
        assert win.chat_view.list_widget.count() == 2
    finally:
        win.close()


def test_switch_loads_selected_conversation(monkeypatch, qapp, tmp_path):
    hm = make_history(tmp_path)
    a = hm.create_conversation("A")
    hm.add_message(a, "user", "a1")
    hm.add_message(a, "assistant", "a2")
    b = hm.create_conversation("B")
    for text in ("b1", "b2", "b3"):
        hm.add_message(b, "user", text)

    win = _make_window(monkeypatch, hm)
    try:
        assert win.current_conv_id == b
        win._on_conversation_selected(a)

        assert win.current_conv_id == a
        assert win.chat_view.list_widget.count() == 2
        assert any("a1" in t for t in _item_texts(win.chat_view.list_widget))
    finally:
        win.close()


def test_switch_to_missing_conversation_falls_back(monkeypatch, qapp, tmp_path):
    hm = make_history(tmp_path)
    a = hm.create_conversation("A")
    hm.add_message(a, "user", "a1")

    win = _make_window(monkeypatch, hm)
    try:
        win._on_conversation_selected("does-not-exist")

        assert win.current_conv_id != "does-not-exist"
        assert hm.get_conversation(win.current_conv_id) is not None
    finally:
        win.close()


def test_pagination_loads_more_than_50_messages(monkeypatch, qapp, tmp_path):
    hm = make_history(tmp_path)
    conv = hm.create_conversation("Long")
    for i in range(120):
        hm.add_message(conv, "user" if i % 2 == 0 else "assistant", f"msg-{i:03d}")

    win = _make_window(monkeypatch, hm)
    try:
        # Newest page first.
        assert win.chat_view.list_widget.count() == 50
        assert win._has_older is True
        last = _item_texts(win.chat_view.list_widget)[-1]
        assert "msg-119" in last

        win._on_load_earlier()
        assert win.chat_view.list_widget.count() == 100

        win._on_load_earlier()
        assert win.chat_view.list_widget.count() == 120
        assert win._has_older is False

        # Oldest message is now present and the first item; ordering is intact.
        first = _item_texts(win.chat_view.list_widget)[0]
        assert "msg-000" in first

        # A further request is a no-op.
        win._on_load_earlier()
        assert win.chat_view.list_widget.count() == 120
    finally:
        win.close()


# ==========================================================================
# MainWindow: create / rename
# ==========================================================================
def test_new_chat_creates_and_activates(monkeypatch, qapp, tmp_path):
    hm = make_history(tmp_path)
    first = hm.create_conversation("First")
    hm.add_message(first, "user", "hi")

    win = _make_window(monkeypatch, hm)
    try:
        previous = win.current_conv_id
        win._on_new_chat()

        assert win.current_conv_id != previous
        assert hm.get_conversation(win.current_conv_id)["title"] == DEFAULT_CONVERSATION_TITLE
        assert win.chat_view.list_widget.count() == 1  # welcome
        assert win.sidebar._active_id == win.current_conv_id
    finally:
        win.close()


def test_rename_preserves_title_on_cancel(monkeypatch, qapp, tmp_path):
    hm = make_history(tmp_path)
    conv = hm.create_conversation("Keep Me")
    win = _make_window(monkeypatch, hm)
    try:
        monkeypatch.setattr(win, "_prompt_title", lambda current: (current, False))
        win._on_rename_conversation(conv)
        assert hm.get_conversation(conv)["title"] == "Keep Me"
    finally:
        win.close()


def test_rename_accepts_and_normalizes_empty(monkeypatch, qapp, tmp_path):
    hm = make_history(tmp_path)
    conv = hm.create_conversation("Old")
    win = _make_window(monkeypatch, hm)
    try:
        monkeypatch.setattr(win, "_prompt_title", lambda current: ("Renamed", True))
        win._on_rename_conversation(conv)
        assert hm.get_conversation(conv)["title"] == "Renamed"

        monkeypatch.setattr(win, "_prompt_title", lambda current: ("   ", True))
        win._on_rename_conversation(conv)
        assert hm.get_conversation(conv)["title"] == DEFAULT_CONVERSATION_TITLE
    finally:
        win.close()


# ==========================================================================
# MainWindow: delete
# ==========================================================================
def test_delete_requires_confirmation(monkeypatch, qapp, tmp_path):
    hm = make_history(tmp_path)
    a = hm.create_conversation("A")
    b = hm.create_conversation("B")
    win = _make_window(monkeypatch, hm)
    try:
        monkeypatch.setattr(win, "_confirm_delete", lambda title: False)
        win._on_delete_conversation(a)
        assert hm.get_conversation(a) is not None, "declining confirmation must not delete"

        monkeypatch.setattr(win, "_confirm_delete", lambda title: True)
        win._on_delete_conversation(a)
        assert hm.get_conversation(a) is None
        assert hm.get_conversation(b) is not None, "only the requested conversation is removed"
    finally:
        win.close()


def test_delete_active_switches_to_safe_replacement(monkeypatch, qapp, tmp_path):
    hm = make_history(tmp_path)
    a = hm.create_conversation("A")
    b = hm.create_conversation("B")
    hm.add_message(b, "user", "b1")

    win = _make_window(monkeypatch, hm)
    try:
        assert win.current_conv_id == b
        monkeypatch.setattr(win, "_confirm_delete", lambda title: True)
        win._on_delete_conversation(b)

        assert hm.get_conversation(b) is None
        assert win.current_conv_id == a, "deleting the active chat must activate a safe replacement"
        assert win.current_conv_id not in (None, b)
    finally:
        win.close()


def test_delete_only_active_creates_new_chat(monkeypatch, qapp, tmp_path):
    hm = make_history(tmp_path)
    a = hm.create_conversation("Only")
    win = _make_window(monkeypatch, hm)
    try:
        assert win.current_conv_id == a
        monkeypatch.setattr(win, "_confirm_delete", lambda title: True)
        win._on_delete_conversation(a)

        assert hm.get_conversation(a) is None
        assert win.current_conv_id != a
        assert hm.get_conversation(win.current_conv_id) is not None
    finally:
        win.close()


def test_delete_failure_keeps_ui_consistent(monkeypatch, qapp, tmp_path):
    hm = make_history(tmp_path)
    a = hm.create_conversation("A")
    b = hm.create_conversation("B")

    win = _make_window(monkeypatch, hm)
    try:
        monkeypatch.setattr(win, "_confirm_delete", lambda title: True)
        errors = []
        monkeypatch.setattr(win, "_show_error", errors.append)

        def _boom(_cid):
            raise RuntimeError("db locked")

        monkeypatch.setattr(hm, "delete_conversation", _boom)
        win._on_delete_conversation(a)

        assert errors, "a delete failure must be reported"
        assert hm.get_conversation(a) is not None, "the conversation survives a failed delete"
        # Sidebar still reflects a consistent state (both conversations listed).
        assert {s["id"] for s in hm.list_conversations()} == {a, b}
    finally:
        win.close()


# ==========================================================================
# MainWindow: isolation / stale workers
# ==========================================================================
def test_new_message_and_journal_use_active_conversation(monkeypatch, qapp, tmp_path):
    hm = make_history(tmp_path)
    a = hm.create_conversation("A")
    b = hm.create_conversation("B")

    win = _make_window(monkeypatch, hm)
    try:
        win.current_conv_id = a
        win._reload_active_conversation()

        win._handle_user_command("ping active")

        assert [m["content"] for m in hm.get_conversation_messages(a, limit=None)] == ["ping active"]
        assert hm.get_conversation_messages(b, limit=None) == []

        route_res = _result(tool_name="shutdown_pc", raw_arguments={})
        win._journal_cancellation(route_res, a)

        a_journal = hm.get_recent_tool_calls(conversation_id=a)
        assert len(a_journal) == 1
        assert a_journal[0]["conversation_id"] == a
        assert hm.get_recent_tool_calls(conversation_id=b) == []
    finally:
        win.close()


def test_stale_route_result_is_not_shown_in_new_conversation(monkeypatch, qapp, tmp_path):
    hm = make_history(tmp_path)
    a = hm.create_conversation("A")
    b = hm.create_conversation("B")
    hm.add_message(b, "user", "b-only")

    win = _make_window(monkeypatch, hm)
    try:
        win.current_conv_id = a
        win._reload_active_conversation()
        win.router = _FakeRouter(lambda q: _result(response_text="reply-for-A"))

        win._handle_user_command("question for A")
        assert len(win.thread_pool.workers) == 1
        worker = win.thread_pool.workers[0]
        assert isinstance(worker, RouteWorker)
        assert worker.conversation_id == a

        # User switches away before the background result arrives.
        win._on_conversation_selected(b)
        assert win.current_conv_id == b

        worker.run()  # -> _on_route_finished(..., a)

        # The reply must not appear in B's view...
        assert not any("reply-for-A" in t for t in _item_texts(win.chat_view.list_widget))
        # ...but it is still persisted to the conversation it belongs to.
        assert "reply-for-A" in [m["content"] for m in hm.get_conversation_messages(a, limit=None)]
        assert "reply-for-A" not in [m["content"] for m in hm.get_conversation_messages(b, limit=None)]
        assert win.current_conv_id == b
    finally:
        win.close()


def test_stale_confirmation_is_dropped(tmp_path, monkeypatch, qapp):
    hm = make_history(tmp_path)
    a = hm.create_conversation("A")
    b = hm.create_conversation("B")

    created = []

    class _FakeDialog:
        def __init__(self, tool_name, message, arguments=None, parent=None):
            created.append(tool_name)

        def exec(self):
            return True

    monkeypatch.setattr(main_window_module, "ConfirmationDialog", _FakeDialog)

    win = _make_window(monkeypatch, hm)
    try:
        win.current_conv_id = a
        win._reload_active_conversation()
        win.router = _FakeRouter(
            lambda q: _result(
                response_text="Confirm shutdown",
                requires_confirmation=True,
                tool_name="shutdown_pc",
                pending_approval=PendingApproval(tool_name="shutdown_pc", arguments={}),
            )
        )

        win._handle_user_command("shut down")
        worker = win.thread_pool.workers[0]

        win._on_conversation_selected(b)
        worker.run()

        assert created == [], "a confirmation for a left conversation must not be shown"
        assert len(win.thread_pool.workers) == 1, "no approval worker may start"
    finally:
        win.close()


# ==========================================================================
# Import / export compatibility
# ==========================================================================
def test_imported_conversation_becomes_active_and_is_listed(monkeypatch, qapp, tmp_path):
    hm = make_history(tmp_path)
    # At least one existing conversation so import is not the only row.
    hm.create_conversation("Existing")

    win = _make_window(monkeypatch, hm)
    try:
        source = hm.create_conversation("Imported Source")
        hm.add_message(source, "user", "imported-question")
        hm.add_message(source, "assistant", "imported-answer")

        # Round-trip through the public export/import core.
        parsed = cio.parse_conversation(cio.serialize_conversation(hm, source))
        new_id = cio.import_parsed_conversation(hm, parsed)

        win._on_conversation_imported(new_id)

        assert win.current_conv_id == new_id
        assert win.chat_view.list_widget.count() == 2
        assert any("imported-question" in t for t in _item_texts(win.chat_view.list_widget))

        # The imported conversation is now discoverable through the sidebar APIs.
        assert new_id in {s["id"] for s in hm.list_conversations()}
        # And it can be exported again (round-trip compatibility).
        again = cio.serialize_conversation(hm, win.current_conv_id)
        assert "imported-question" in again
    finally:
        win.close()


# ==========================================================================
# Regression: seam exposure
# ==========================================================================
def test_main_window_exposes_conversation_manager_methods():
    from app.ui.main_window import MainWindow as MW

    for method in (
        "_on_new_chat",
        "_on_conversation_selected",
        "_on_rename_conversation",
        "_on_delete_conversation",
        "_on_load_earlier",
        "_populate_sidebar",
        "_reload_active_conversation",
    ):
        assert callable(getattr(MW, method, None)), method
