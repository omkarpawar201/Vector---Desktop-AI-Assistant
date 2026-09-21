"""
Context Builder for Vector Desktop AI Assistant.
Assembles minimal required context for simple vs conversational user requests.
"""

from typing import Any, Dict, List, Optional
import psutil


class ContextBuilder:
    """
    Builds context payloads for intent resolution and LLM prompts.
    """

    @staticmethod
    def build_system_snapshot() -> Dict[str, Any]:
        """
        Returns a lightweight snapshot of current system metrics.
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
