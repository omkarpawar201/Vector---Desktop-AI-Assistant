"""
Context assembly for Vector Desktop AI Assistant.

``build_system_snapshot`` / ``build_context`` describe the machine.  The
conversation helper below assembles the *recent dialogue* of the active
conversation so Tier 2 can resolve follow-up references ("turn it down a little
more") that the local router deliberately refuses to guess at.

History is untrusted context, never authorization: it is scoped to one
conversation, capped to complete turns, sanitized, and never allowed to carry a
``system`` role that could masquerade as an instruction channel.
"""

import psutil

from typing import Any, Dict, List, Optional

from app.core.privacy import PrivacyFilter


class ContextBuilder:
    """
    Builds system snapshots and recent-dialogue context for Vector.
    """

    #: Complete user-anchored turns handed to Tier 2 by default.
    DEFAULT_TURNS = 5
    #: Roles that may travel as dialogue.  ``system`` (and anything unknown) is
    #: dropped: history is untrusted context, never an instruction channel.
    _DIALOGUE_ROLES = ("user", "assistant")
    #: Upper bound on rows read from storage before turn-grouping.
    _MAX_ROWS = 200

    @staticmethod
    def build_system_snapshot() -> Dict[str, Any]:
        """
        Returns a snapshot of the host system's resource usage.
        """
        try:
            return {
                "cpu_percent": psutil.cpu_percent(interval=None),
                "ram_percent": psutil.virtual_memory().percent,
                "disk_percent": psutil.disk_usage("/").percent
            }
        except Exception:
            return {}

    @classmethod
    def build_context(
        cls,
        user_query: str,
        conversation_history: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Assembles complete context including user query, recent messages, and system metrics.
        """
        recent_history = (conversation_history or [])[-5:]  # Include last 5 turns
        return {
            "query": user_query,
            "system": cls.build_system_snapshot(),
            "history": recent_history
        }

    @classmethod
    def build_conversation_context(
        cls,
        conversation_id: Optional[str],
        history_manager: Optional[Any] = None,
        current_query: Optional[str] = None,
        turns: int = DEFAULT_TURNS
    ) -> List[Dict[str, str]]:
        """
        Returns the recent dialogue of one conversation, oldest first.

        Guarantees:

        * only the named conversation is read;
        * the newest ``turns`` *complete* turns are kept -- a turn is one user
          message plus its replies, so a window never ends mid-exchange even
          when a turn produced several assistant messages;
        * a trailing user message that equals ``current_query`` is dropped, so a
          query already passed separately is never duplicated in the history;
        * every content string passes through the privacy sanitizer;
        * any storage or serialization failure degrades to ``[]`` so a normal
          command stays routable without history.

        The return value is a list of ``{"role": "user"|"assistant",
        "content": str}`` dicts, the shape the Gemini client formats into its
        native message API.
        """
        if not conversation_id or history_manager is None:
            return []

        try:
            rows = history_manager.get_recent_history(
                conversation_id=conversation_id, limit=cls._MAX_ROWS
            )
        except Exception:
            return []

        dialogue = cls._as_dialogue(rows)
        dialogue = cls._drop_current_query(dialogue, current_query)
        dialogue = cls._recent_turns(dialogue, turns)

        return [
            {
                "role": message["role"],
                "content": PrivacyFilter.sanitize_text(message["content"]),
            }
            for message in dialogue
            if message["content"]
        ]

    @classmethod
    def _as_dialogue(cls, rows: Optional[List[Dict[str, Any]]]) -> List[Dict[str, str]]:
        """Keeps only well-formed user/assistant rows with non-empty content."""
        dialogue: List[Dict[str, str]] = []
        for row in rows or []:
            if not isinstance(row, dict):
                continue
            role = row.get("role")
            content = row.get("content")
            if role not in cls._DIALOGUE_ROLES:
                continue
            if not isinstance(content, str) or not content.strip():
                continue
            dialogue.append({"role": role, "content": content})
        return dialogue

    @staticmethod
    def _drop_current_query(
        dialogue: List[Dict[str, str]],
        current_query: Optional[str]
    ) -> List[Dict[str, str]]:
        """Removes a trailing user row that is the current query."""
        if not dialogue or not current_query:
            return dialogue
        last = dialogue[-1]
        if last["role"] == "user" and last["content"].strip() == current_query.strip():
            return dialogue[:-1]
        return dialogue

    @classmethod
    def _group_turns(cls, dialogue: List[Dict[str, str]]) -> List[List[Dict[str, str]]]:
        """Splits a dialogue into user-anchored turns."""
        turns: List[List[Dict[str, str]]] = []
        current: List[Dict[str, str]] = []
        for message in dialogue:
            if message["role"] == "user" and current:
                turns.append(current)
                current = []
            current.append(message)
        if current:
            turns.append(current)
        return turns

    @classmethod
    def _recent_turns(
        cls,
        dialogue: List[Dict[str, str]],
        turns: int
    ) -> List[Dict[str, str]]:
        """Flattens the newest ``turns`` complete turns in chronological order."""
        if turns <= 0:
            return []
        grouped = cls._group_turns(dialogue)
        return [message for turn in grouped[-turns:] for message in turn]