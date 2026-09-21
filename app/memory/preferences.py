"""
User Preference Manager for Vector Desktop AI Assistant.
Persists application settings and custom user key-value preferences in SQLite.
"""

from datetime import datetime, timezone
import json
from typing import Any, Dict, Optional
from app.memory.database import DatabaseManager, get_db_manager


class PreferenceManager:
    """
    Key-value preference store backed by SQLite database.
    """

    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        self.db = db_manager or get_db_manager()

    def set_preference(self, key: str, value: Any) -> None:
        """
        Sets a key-value preference in the user_preferences table.
        Automatically serializes non-string values to JSON.
        """
        now = datetime.now(timezone.utc).isoformat()
        serialized_val = json.dumps(value) if not isinstance(value, str) else value

        with self.db.get_connection() as conn:
            conn.execute(
                """INSERT INTO user_preferences (key, value, updated_at) 
                   VALUES (?, ?, ?) 
                   ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at""",
                (key, serialized_val, now)
            )
            conn.commit()

    def get_preference(self, key: str, default: Any = None) -> Any:
        """
        Retrieves a preference value by key. Deserializes JSON values if applicable.
        Returns default if key does not exist.
        """
        with self.db.get_connection() as conn:
            row = conn.execute(
                "SELECT value FROM user_preferences WHERE key = ?",
                (key,)
            ).fetchone()

            if not row:
                return default

            raw_val = row["value"]
            try:
                return json.loads(raw_val)
            except Exception:
                return raw_val

    def list_preferences(self) -> Dict[str, Any]:
        """
        Returns all stored user preferences as a key-value dictionary.
        """
        with self.db.get_connection() as conn:
            rows = conn.execute("SELECT key, value FROM user_preferences").fetchall()
            result = {}
            for row in rows:
                k = row["key"]
                raw_v = row["value"]
                try:
                    result[k] = json.loads(raw_v)
                except Exception:
                    result[k] = raw_v
            return result
