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

            # 3. Tool Calls Table (Trace Logger)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS tool_calls (
                    id TEXT PRIMARY KEY,
                    message_id TEXT NOT NULL,
                    tool_name TEXT NOT NULL,
                    arguments TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    execution_status TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY (message_id) REFERENCES messages(id) ON DELETE CASCADE
                );
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_tool_calls_msg_id ON tool_calls(message_id);")

            # 4. User Preferences Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_preferences (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
            """)

            conn.commit()

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
