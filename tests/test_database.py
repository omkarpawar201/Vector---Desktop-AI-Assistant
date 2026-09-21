"""
Unit tests for DatabaseManager, HistoryManager, and PreferenceManager.
"""

from pathlib import Path
import tempfile
import pytest
from app.memory.database import DatabaseManager
from app.memory.history import HistoryManager
from app.memory.preferences import PreferenceManager


@pytest.fixture
def temp_db():
    """Fixture providing a temporary SQLite database manager."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = str(Path(tmpdir) / "test_vector.db")
        db_mgr = DatabaseManager(db_path=db_path)
        yield db_mgr


def test_database_manager_initialization(temp_db):
    """Verify database manager initializes tables and WAL mode."""
    with temp_db.get_connection() as conn:
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        table_names = [r["name"] for r in tables]

        assert "conversations" in table_names
        assert "messages" in table_names
        assert "tool_calls" in table_names
        assert "user_preferences" in table_names


def test_history_manager(temp_db):
    """Verify conversation, message, and tool call persistence."""
    hm = HistoryManager(db_manager=temp_db)

    # 1. Create conversation
    conv_id = hm.create_conversation(title="Test Session")
    assert bool(conv_id) is True

    # 2. Add message
    msg_id = hm.add_message(conversation_id=conv_id, role="user", content="Set volume to 40")
    assert bool(msg_id) is True

    # 3. Add tool call trace
    call_id = hm.add_tool_call(
        message_id=msg_id,
        tool_name="set_volume",
        arguments={"level": 40},
        confidence=1.0,
        execution_status="SUCCESS"
    )
    assert bool(call_id) is True

    # 4. Fetch messages
    msgs = hm.get_conversation_messages(conversation_id=conv_id)
    assert len(msgs) == 1
    assert msgs[0]["content"] == "Set volume to 40"
    assert msgs[0]["role"] == "user"


def test_preference_manager(temp_db):
    """Verify preference setting, getting, and listing."""
    pm = PreferenceManager(db_manager=temp_db)

    # Default fallback
    assert pm.get_preference("theme", default="dark") == "dark"

    # Set string preference
    pm.set_preference("theme", "light")
    assert pm.get_preference("theme") == "light"

    # Set dict/JSON preference
    pm.set_preference("toggles", {"voice": True, "gemini": False})
    toggles = pm.get_preference("toggles")
    assert isinstance(toggles, dict)
    assert toggles["voice"] is True
    assert toggles["gemini"] is False

    # List all preferences
    all_prefs = pm.list_preferences()
    assert "theme" in all_prefs
    assert "toggles" in all_prefs
