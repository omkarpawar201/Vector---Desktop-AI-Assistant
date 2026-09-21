"""
Conversation History and Tool Execution Trace Manager for Vector Desktop AI Assistant.
"""

from datetime import datetime, timezone
import json
from typing import Any, Dict, List, Optional
from uuid import uuid4

from app.memory.database import DatabaseManager, get_db_manager


class HistoryManager:
    """
    Manages storage and retrieval of conversations, messages, and tool call execution traces.
    """

    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        self.db = db_manager or get_db_manager()

    def create_conversation(self, title: str = "New Session") -> str:
        """
        Creates a new conversation record and returns its unique ID.
        """
        conv_id = str(uuid4())
        now = datetime.now(timezone.utc).isoformat()

        with self.db.get_connection() as conn:
            conn.execute(
                "INSERT INTO conversations (id, created_at, title) VALUES (?, ?, ?)",
                (conv_id, now, title)
            )
            conn.commit()

        return conv_id

    def add_message(self, conversation_id: str, role: str, content: str) -> str:
        """
        Adds a user, assistant, or system message to a conversation.
        Returns the created message ID.
        """
        msg_id = str(uuid4())
        now = datetime.now(timezone.utc).isoformat()

        with self.db.get_connection() as conn:
            conn.execute(
                "INSERT INTO messages (id, conversation_id, role, content, timestamp) VALUES (?, ?, ?, ?, ?)",
                (msg_id, conversation_id, role, content, now)
            )
            conn.commit()

        return msg_id

    def add_tool_call(
        self,
        message_id: str,
        tool_name: str,
        arguments: Dict[str, Any],
        confidence: float,
        execution_status: str
    ) -> str:
        """
        Records a tool execution trace linked to a specific message ID.
        """
        call_id = str(uuid4())
        now = datetime.now(timezone.utc).isoformat()
        args_json = json.dumps(arguments or {})

        with self.db.get_connection() as conn:
            conn.execute(
                """INSERT INTO tool_calls 
                   (id, message_id, tool_name, arguments, confidence, execution_status, timestamp) 
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (call_id, message_id, tool_name, args_json, confidence, execution_status, now)
            )
            conn.commit()

        return call_id

    def get_conversation_messages(self, conversation_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Retrieves messages for a conversation ordered by timestamp.
        """
        with self.db.get_connection() as conn:
            rows = conn.execute(
                "SELECT id, role, content, timestamp FROM messages WHERE conversation_id = ? ORDER BY timestamp ASC LIMIT ?",
                (conversation_id, limit)
            ).fetchall()

            return [dict(row) for row in rows]

    def get_recent_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieves recent messages across all conversations.
        """
        with self.db.get_connection() as conn:
            rows = conn.execute(
                "SELECT id, conversation_id, role, content, timestamp FROM messages ORDER BY timestamp DESC LIMIT ?",
                (limit,)
            ).fetchall()

            return [dict(row) for row in rows]
