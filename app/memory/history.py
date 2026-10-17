"""
Conversation History and Tool Execution Trace Manager for Vector Desktop AI Assistant.
"""

from datetime import datetime, timezone
import json
from typing import Any, Dict, List, Optional
from uuid import uuid4

from app.memory.database import DatabaseManager, get_db_manager

#: Title used whenever a conversation is created or renamed with a blank title.
DEFAULT_CONVERSATION_TITLE = "New Chat"


def normalize_title(title: Optional[str]) -> str:
    """
    Return a always-usable conversation title.

    A missing, empty or whitespace-only title becomes
    :data:`DEFAULT_CONVERSATION_TITLE`; every other title is returned stripped
    of leading/trailing whitespace (interior spacing is preserved verbatim).
    """
    text = (title or "").strip()
    return text or DEFAULT_CONVERSATION_TITLE


class HistoryManager:
    """
    Manages storage and retrieval of conversations, messages, and tool call execution traces.
    """

    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        self.db = db_manager or get_db_manager()

    def create_conversation(self, title: str = "New Session") -> str:
        """
        Creates a new conversation record and returns its unique ID.

        A blank/whitespace-only ``title`` is replaced by a safe default so a
        conversation can never be stored under an unusable name.
        """
        conv_id = str(uuid4())
        now = datetime.now(timezone.utc).isoformat()

        with self.db.get_connection() as conn:
            conn.execute(
                "INSERT INTO conversations (id, created_at, title) VALUES (?, ?, ?)",
                (conv_id, now, normalize_title(title))
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

    def add_execution(
        self,
        tool_name: str,
        arguments: Optional[Dict[str, Any]] = None,
        status: str = "",
        summary: str = "",
        conversation_id: Optional[str] = None,
        message_id: Optional[str] = None,
        confidence: float = 0.0,
    ) -> str:
        """
        Records one tool execution (an Action Journal entry).

        Unlike :meth:`add_tool_call`, ``message_id`` is optional, so an actual
        execution can be journaled without manufacturing a message row.  Storage
        only: callers are responsible for sanitizing ``arguments``/``summary``
        (see :class:`app.memory.journal.ExecutionJournal`).
        """
        call_id = str(uuid4())
        now = datetime.now(timezone.utc).isoformat()
        args_json = json.dumps(arguments or {})

        with self.db.get_connection() as conn:
            conn.execute(
                """INSERT INTO tool_calls
                   (id, message_id, conversation_id, tool_name, arguments, summary, confidence, execution_status, timestamp)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (call_id, message_id, conversation_id, tool_name, args_json,
                 summary, confidence, status, now)
            )
            conn.commit()

        return call_id

    def get_recent_tool_calls(
        self,
        conversation_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Retrieves recent Action Journal entries, newest first.

        With ``conversation_id`` the result is scoped to that conversation.
        """
        with self.db.get_connection() as conn:
            if conversation_id is None:
                rows = conn.execute(
                    "SELECT id, conversation_id, tool_name, arguments, summary, "
                    "execution_status, timestamp FROM tool_calls "
                    "ORDER BY timestamp DESC, rowid DESC LIMIT ?",
                    (limit,)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT id, conversation_id, tool_name, arguments, summary, "
                    "execution_status, timestamp FROM tool_calls "
                    "WHERE conversation_id = ? ORDER BY timestamp DESC, rowid DESC LIMIT ?",
                    (conversation_id, limit)
                ).fetchall()

            return [dict(row) for row in rows]

    def get_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves one conversation's metadata (``id``, ``title``, ``created_at``).

        Returns ``None`` when no such conversation exists.  Used by conversation
        export so the exported header reflects stored metadata rather than a
        guess.
        """
        with self.db.get_connection() as conn:
            row = conn.execute(
                "SELECT id, created_at, title FROM conversations WHERE id = ?",
                (conversation_id,)
            ).fetchone()

            return dict(row) if row is not None else None

    def conversation_exists(self, conversation_id: str) -> bool:
        """True when a conversation with this id already exists."""
        with self.db.get_connection() as conn:
            row = conn.execute(
                "SELECT 1 FROM conversations WHERE id = ?", (conversation_id,)
            ).fetchone()

            return row is not None

    def import_conversation(
        self,
        conversation_id: str,
        title: str,
        created_at: Optional[str],
        messages: List[Dict[str, str]],
    ) -> None:
        """
        Inserts a whole imported conversation in a single transaction.

        The conversation row and every message are written together: if any
        insert fails the transaction is rolled back, so a failed import can never
        leave a partially-imported conversation behind.  The caller supplies a
        ``conversation_id`` already validated to be unique; message ids are
        freshly generated (imported message ids are intentionally not reused).
        """
        now = datetime.now(timezone.utc).isoformat()
        conv_id = str(conversation_id)

        with self.db.get_connection() as conn:
            try:
                conn.execute(
                    "INSERT INTO conversations (id, created_at, title) VALUES (?, ?, ?)",
                    (conv_id, created_at or now, title or "Imported Conversation"),
                )
                for message in messages:
                    conn.execute(
                        "INSERT INTO messages (id, conversation_id, role, content, timestamp) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (
                            str(uuid4()),
                            conv_id,
                            message["role"],
                            message["content"],
                            message["timestamp"],
                        ),
                    )
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    def get_conversation_messages(
        self, conversation_id: str, limit: Optional[int] = 50
    ) -> List[Dict[str, Any]]:
        """
        Retrieves messages for a conversation ordered by timestamp.

        ``rowid`` is used as a tie-break so messages that share a timestamp keep
        their insertion order, which conversation export relies on.  Pass
        ``limit=None`` to retrieve every message.
        """
        with self.db.get_connection() as conn:
            if limit is None:
                rows = conn.execute(
                    "SELECT id, role, content, timestamp FROM messages WHERE conversation_id = ? "
                    "ORDER BY timestamp ASC, rowid ASC",
                    (conversation_id,)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT id, role, content, timestamp FROM messages WHERE conversation_id = ? "
                    "ORDER BY timestamp ASC, rowid ASC LIMIT ?",
                    (conversation_id, limit)
                ).fetchall()

            return [dict(row) for row in rows]

    def get_recent_history(
        self,
        conversation_id: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Retrieves recent messages.

        With ``conversation_id`` the result is scoped to that conversation and
        returned oldest-first (the order a dialogue context must be replayed
        in).  Without it the legacy behaviour is preserved: the newest ``limit``
        rows across every conversation, newest-first.

        ``limit`` always counts individual message rows, not user/assistant
        turns; callers that need complete turns must group the rows themselves.
        """
        with self.db.get_connection() as conn:
            if conversation_id is None:
                rows = conn.execute(
                    "SELECT id, conversation_id, role, content, timestamp FROM messages ORDER BY timestamp DESC LIMIT ?",
                    (limit,)
                ).fetchall()

                return [dict(row) for row in rows]

            # Take the newest rows, then flip to chronological order so the
            # oldest of the retained window is presented first.
            rows = conn.execute(
                "SELECT id, conversation_id, role, content, timestamp FROM messages "
                "WHERE conversation_id = ? ORDER BY timestamp DESC, rowid DESC LIMIT ?",
                (conversation_id, limit)
            ).fetchall()

            return [dict(row) for row in reversed(rows)]

    # ------------------------------------------------------------------
    # Conversation management (listing, search, rename, delete, paging)
    # ------------------------------------------------------------------

    #: Columns shared by every conversation summary query.
    _SUMMARY_COLUMNS = (
        "c.id AS id, c.title AS title, c.created_at AS created_at, "
        "COALESCE(MAX(m.timestamp), c.created_at) AS last_activity, "
        "COUNT(m.id) AS message_count"
    )

    def list_conversations(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Returns every conversation as a summary, most recently active first.

        Each summary carries ``id``, ``title``, ``created_at``, ``last_activity``
        (the newest message time, falling back to ``created_at``) and
        ``message_count``.  Ordering is by ``last_activity`` descending, then
        ``created_at`` descending, so the sidebar's recency grouping is stable.
        """
        sql = (
            f"SELECT {self._SUMMARY_COLUMNS} FROM conversations c "
            "LEFT JOIN messages m ON m.conversation_id = c.id "
            "GROUP BY c.id ORDER BY last_activity DESC, c.created_at DESC"
        )
        params: tuple = ()
        if limit is not None:
            sql += " LIMIT ?"
            params = (limit,)

        with self.db.get_connection() as conn:
            rows = conn.execute(sql, params).fetchall()
            return [dict(row) for row in rows]

    def search_conversations(
        self, query: str, limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Returns conversation summaries whose title or any message matches
        ``query`` (case-insensitive substring), most recently active first.

        A blank query lists everything, so the sidebar can reuse this method for
        both browsing and searching.  ``LIKE`` wildcards in the query are escaped
        so a literal ``%`` or ``_`` is matched rather than acting as a wildcard.
        """
        text = (query or "").strip()
        if not text:
            return self.list_conversations(limit=limit)

        pattern = self._like_pattern(text)
        sql = (
            f"SELECT {self._SUMMARY_COLUMNS} FROM conversations c "
            "LEFT JOIN messages m ON m.conversation_id = c.id "
            "WHERE c.title LIKE ? ESCAPE '\\' "
            "OR c.id IN (SELECT conversation_id FROM messages WHERE content LIKE ? ESCAPE '\\') "
            "GROUP BY c.id ORDER BY last_activity DESC, c.created_at DESC"
        )
        params: tuple = (pattern, pattern)
        if limit is not None:
            sql += " LIMIT ?"
            params = (pattern, pattern, limit)

        with self.db.get_connection() as conn:
            rows = conn.execute(sql, params).fetchall()
            return [dict(row) for row in rows]

    @staticmethod
    def _like_pattern(query: str) -> str:
        escaped = (
            query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        )
        return f"%{escaped}%"

    def count_conversation_messages(self, conversation_id: str) -> int:
        """Returns how many messages a conversation holds (0 when missing)."""
        with self.db.get_connection() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM messages WHERE conversation_id = ?",
                (conversation_id,)
            ).fetchone()
            return int(row["n"]) if row is not None else 0

    def get_conversation_page(
        self, conversation_id: str, limit: int = 50, offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Returns one chronological page of a conversation, counted from the end.

        ``offset == 0`` is the *newest* ``limit`` messages; ``offset == limit``
        is the previous ``limit``, and so on -- so the UI can incrementally load
        older history without ever silently truncating.  Each page is returned in
        chronological (oldest-first) order, with ``rowid`` as a tie-break so
        messages sharing a timestamp keep their insertion order.
        """
        safe_limit = max(0, int(limit))
        safe_offset = max(0, int(offset))
        with self.db.get_connection() as conn:
            rows = conn.execute(
                "SELECT id, role, content, timestamp FROM messages "
                "WHERE conversation_id = ? "
                "ORDER BY timestamp DESC, rowid DESC LIMIT ? OFFSET ?",
                (conversation_id, safe_limit, safe_offset)
            ).fetchall()
            return [dict(row) for row in reversed(rows)]

    def rename_conversation(self, conversation_id: str, title: str) -> bool:
        """
        Renames one conversation.  Returns ``True`` when a row was updated.

        The new title is normalized, so an empty/whitespace-only request falls
        back to a safe default instead of storing an unusable name.
        """
        with self.db.get_connection() as conn:
            cursor = conn.execute(
                "UPDATE conversations SET title = ? WHERE id = ?",
                (normalize_title(title), conversation_id)
            )
            conn.commit()
            return cursor.rowcount > 0

    def delete_conversation(self, conversation_id: str) -> bool:
        """
        Deletes one conversation and everything that belongs to it.

        Returns ``True`` when a conversation row was removed, ``False`` when the
        id did not exist (so a stray delete can never touch an unrelated
        conversation).  All three deletes run in a single transaction:

        * Action Journal entries scoped to the conversation (``tool_calls`` with
          this ``conversation_id``) or attached to one of its messages;
        * the conversation's messages;
        * the conversation row itself.

        The journal rows are removed explicitly rather than relying on the
        message foreign key, because Action Journal entries are stored with
        ``message_id = NULL`` and would otherwise be orphaned.
        """
        with self.db.get_connection() as conn:
            try:
                conn.execute(
                    "DELETE FROM tool_calls WHERE conversation_id = ? "
                    "OR message_id IN (SELECT id FROM messages WHERE conversation_id = ?)",
                    (conversation_id, conversation_id)
                )
                conn.execute(
                    "DELETE FROM messages WHERE conversation_id = ?",
                    (conversation_id,)
                )
                cursor = conn.execute(
                    "DELETE FROM conversations WHERE id = ?",
                    (conversation_id,)
                )
                conn.commit()
                return cursor.rowcount > 0
            except Exception:
                conn.rollback()
                raise
