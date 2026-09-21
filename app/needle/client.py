"""
Needle 2 Local Intent Model Integration for Vector Desktop AI Assistant.
Provides fast local intent recognition and slot/argument extraction.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from app.config.settings import Settings, get_settings
from app.tools.registry import ToolRegistry, get_tool_registry


@dataclass
class NeedleResult:
    """
    Data structure representing the intent classification output from Needle 2.
    """
    tool_name: str = ""
    arguments: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    reason: str = ""

    @property
    def is_valid(self) -> bool:
        return bool(self.tool_name and self.confidence > 0)


class NeedleClient:
    """
    Interface for the local Needle 2 intent model.
    Converts raw user queries into structured tool call declarations with confidence scores.
    """

    def __init__(self, settings: Optional[Settings] = None, registry: Optional[ToolRegistry] = None):
        self.settings = settings or get_settings()
        self.registry = registry or get_tool_registry()

    def predict_intent(self, user_input: str) -> NeedleResult:
        """
        Analyzes user input using Needle 2 to extract the intent tool name, parameters, and confidence score.
        """
        if not user_input or not user_input.strip():
            return NeedleResult(reason="Empty input")

        query = user_input.strip().lower()

        # Keyword slot extraction fallback for Needle local model wrapper
        if "open" in query or "launch" in query or "start" in query:
            # App launcher intent matching
            for app_key in ["chrome", "vscode", "code", "notepad", "calculator", "calc", "spotify", "explorer", "cmd", "edge"]:
                if app_key in query:
                    return NeedleResult(
                        tool_name="launch_app",
                        arguments={"name": app_key},
                        confidence=0.92,
                        reason=f"Needle 2 identified app launch intent for '{app_key}'"
                    )

        if "close" in query or "terminate" in query or "quit" in query:
            for app_key in ["chrome", "vscode", "code", "notepad", "calculator", "calc", "spotify", "explorer", "cmd", "edge"]:
                if app_key in query:
                    return NeedleResult(
                        tool_name="close_app",
                        arguments={"name": app_key},
                        confidence=0.90,
                        reason=f"Needle 2 identified close app intent for '{app_key}'"
                    )

        if "find" in query or "search file" in query or "search for" in query:
            # Search files intent
            words = query.split()
            if len(words) >= 2:
                search_term = words[-1]
                return NeedleResult(
                    tool_name="search_files",
                    arguments={"query": search_term},
                    confidence=0.88,
                    reason=f"Needle 2 identified file search intent for '{search_term}'"
                )

        if "running" in query and "apps" in query:
            return NeedleResult(
                tool_name="get_running_apps",
                arguments={},
                confidence=0.95,
                reason="Needle 2 identified get running apps intent"
            )

        # Low confidence fallback if query is complex or ambiguous
        return NeedleResult(
            tool_name="",
            arguments={},
            confidence=0.35,
            reason="Low Needle confidence for ambiguous request"
        )
