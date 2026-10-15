"""
SQLite Database Manager for Vector Desktop AI Assistant.
Enables WAL mode and initializes indexed schemas for conversation history, tool calls, and preferences.
"""

from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3
from typing import Generator, Optional
from app.config.settings import Settings, get_settings


class DatabaseManager:
    """
    Manages SQLite connection lifecycle, schema migrations, and WAL mode configuration.
    """

    def __init__(self, db_path: Optional[str] = None):
        settings = get_settings()
        self.db_path = db_path or settings.database_path
        self._ensure_dir_and_init_schema()

    def _ensure_dir_and_init_schema(self) -> None:
        """Ensures database directory exists and creates initial tables/indexes if missing."""
        path_obj = Path(self.db_path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Enable WAL mode for high performance
            cursor.execute("PRAGMA journal_mode=WAL;")
            cursor.execute("PRAGMA synchronous=NORMAL;")

            # 1. Conversations Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    title TEXT NOT NULL
                );
            """)

            # 2. Messages Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                );
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_messages_conv_id ON messages(conversation_id);")

            # 3. Tool Calls / Action Journal Table.
            #    ``message_id`` is nullable so an execution can be journaled even
            #    when it is not tied to a specific message; ``conversation_id``
            #    and ``summary`` carry the action-journal fields.
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS tool_calls (
                    id TEXT PRIMARY KEY,
                    message_id TEXT,
                    conversation_id TEXT,
                    tool_name TEXT NOT NULL,
                    arguments TEXT NOT NULL,
                    summary TEXT,
                    confidence REAL NOT NULL,
                    execution_status TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY (message_id) REFERENCES messages(id) ON DELETE CASCADE
                );
            """)
            self._migrate_tool_calls_schema(cursor)

            # Indexes are created after migration so they never reference a
            # column (conversation_id) that a pre-migration table lacks.
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_tool_calls_msg_id ON tool_calls(message_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_tool_calls_conv_id ON tool_calls(conversation_id);")

            # 4. User Preferences Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_preferences (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
            """)

            conn.commit()

    def _migrate_tool_calls_schema(self, cursor: sqlite3.Cursor) -> None:
        """
        Idempotently upgrades a pre-existing ``tool_calls`` table to the
        action-journal shape.

        The original schema pinned ``message_id`` NOT NULL and had no
        ``conversation_id``/``summary`` columns, which is why
        ``HistoryManager.add_tool_call`` was never wired up: an execution could
        not be recorded without first manufacturing a message row.  Databases
        created before this change are rebuilt once so ``message_id`` is
        nullable and the new columns exist; fresh databases already have the new
        schema and this is a no-op.
        """
        existing = {
            row["name"]: row
            for row in cursor.execute("PRAGMA table_info(tool_calls);").fetchall()
        }
        if not existing:
            return

        message_id_not_null = bool(
            existing.get("message_id") and existing["message_id"]["notnull"]
        )

        if message_id_not_null:
            # Rebuild the table so message_id becomes nullable.  Nothing in the
            # schema references tool_calls, so the rename/split is safe and the
            # child FK (to messages) is preserved for the copied rows.
            cursor.execute("ALTER TABLE tool_calls RENAME TO tool_calls_legacy;")
            cursor.execute("""
                CREATE TABLE tool_calls (
                    id TEXT PRIMARY KEY,
                    message_id TEXT,
                    conversation_id TEXT,
                    tool_name TEXT NOT NULL,
                    arguments TEXT NOT NULL,
                    summary TEXT,
                    confidence REAL NOT NULL,
                    execution_status TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY (message_id) REFERENCES messages(id) ON DELETE CASCADE
                );
            """)
            legacy_cols = {
                row["name"]
                for row in cursor.execute("PRAGMA table_info(tool_calls_legacy);").fetchall()
            }

            def _legacy(name: str) -> str:
                return name if name in legacy_cols else "NULL"

            cursor.execute(f"""
                INSERT INTO tool_calls
                    (id, message_id, conversation_id, tool_name, arguments, summary, confidence, execution_status, timestamp)
                SELECT id, message_id, {_legacy("conversation_id")}, tool_name, arguments,
                       {_legacy("summary")}, confidence, execution_status, timestamp
                FROM tool_calls_legacy;
            """)
            cursor.execute("DROP TABLE tool_calls_legacy;")
            return

        # message_id already nullable: only add any missing columns.
        if "conversation_id" not in existing:
            cursor.execute("ALTER TABLE tool_calls ADD COLUMN conversation_id TEXT;")
        if "summary" not in existing:
            cursor.execute("ALTER TABLE tool_calls ADD COLUMN summary TEXT;")

    @contextmanager
    def get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """
        Yields an open SQLite connection with Row factory enabled.
        Automatically closes connection upon exit.
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        # Enforce foreign key constraints on every connection. PRAGMA foreign_keys
        # is connection-scoped, so it must be set here (not just at schema init)
        # for ON DELETE CASCADE rules to actually take effect.
        conn.execute("PRAGMA foreign_keys = ON;")
        try:
            yield conn
        finally:
            conn.close()


# Singleton database manager cache. Keyed by resolved db path so that a single
# DatabaseManager (and its schema/pools) is shared per database file across the
# app. Test suites that pass an explicit db_path still get isolated instances.
_db_manager_cache: "dict[Optional[str], DatabaseManager]" = {}


def get_db_manager(db_path: Optional[str] = None) -> DatabaseManager:
    cache_key = str(Path(db_path).resolve()) if db_path else None
    if cache_key not in _db_manager_cache:
        _db_manager_cache[cache_key] = DatabaseManager(db_path=db_path)
    return _db_manager_cache[cache_key]
