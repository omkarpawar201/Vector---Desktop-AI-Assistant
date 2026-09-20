"""
Base tool interfaces and standardized execution results for Vector Desktop AI Assistant.
"""

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field

from app.config.constants import PermissionLevel


class ToolResult(BaseModel):
    """
    Standardized data structure returned by every tool execution in Vector.
    Prevents raw Python exceptions from leaking to LLMs.
    """
    success: bool = Field(description="True if the tool executed successfully, False otherwise")
    tool: str = Field(description="Name of the tool that was executed")
    data: Optional[Dict[str, Any]] = Field(default=None, description="Structured data returned by the tool")
    message: str = Field(default="", description="Human-readable summary message")
    error: Optional[str] = Field(default=None, description="Error details or exception type if execution failed")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 UTC timestamp of execution"
    )

    @classmethod
    def ok(cls, tool: str, data: Optional[Dict[str, Any]] = None, message: str = "") -> "ToolResult":
        """Factory method for successful tool execution."""
        return cls(
            success=True,
            tool=tool,
            data=data or {},
            message=message or f"Tool '{tool}' executed successfully."
        )

    @classmethod
    def fail(cls, tool: str, error: str, message: str = "") -> "ToolResult":
        """Factory method for failed tool execution."""
        return cls(
            success=False,
            tool=tool,
            data=None,
            error=error,
            message=message or f"Tool '{tool}' failed: {error}"
        )


class BaseTool(ABC):
    """
    Abstract Base Class for all Vector tools.
    Every desktop capability (volume, app launcher, files, power, system) implements this interface.
    """

    name: str
    description: str
    parameters: Dict[str, Any]
    permission_level: PermissionLevel = PermissionLevel.SAFE

    @abstractmethod
    def execute(self, **kwargs: Any) -> ToolResult:
        """
        Executes the tool with validated arguments and returns a standardized ToolResult.
        Must never raise uncaught exceptions.
        """
        pass

    def to_schema(self) -> Dict[str, Any]:
        """
        Exports the tool definition as a standardized OpenAPI / JSON Schema dictionary
        consumable by both Needle 2 and Gemini API.
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "permission_level": self.permission_level.value
        }
