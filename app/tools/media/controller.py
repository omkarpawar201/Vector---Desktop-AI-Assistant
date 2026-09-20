"""
Media control tools for Vector Desktop AI Assistant.
Triggers Windows system virtual media keys for play, pause, next, and previous track.
"""

from typing import Any
import pyautogui
from app.config.constants import PermissionLevel
from app.tools.base import BaseTool, ToolResult

# PyAutoGUI safety setting: disable fail-safe for media key triggers
pyautogui.FAILSAFE = False


class MediaPlayTool(BaseTool):
    """Triggers global media play command."""
    name = "media_play"
    description = "Resume or start system media playback."
    parameters = {"type": "object", "properties": {}, "required": []}
    permission_level = PermissionLevel.SAFE

    def execute(self, **kwargs: Any) -> ToolResult:
        try:
            pyautogui.press("playpause")
            return ToolResult.ok(tool=self.name, message="Media play command triggered.")
        except Exception as e:
            return ToolResult.fail(tool=self.name, error=str(e))


class MediaPauseTool(BaseTool):
    """Triggers global media pause command."""
    name = "media_pause"
    description = "Pause current system media playback."
    parameters = {"type": "object", "properties": {}, "required": []}
    permission_level = PermissionLevel.SAFE

    def execute(self, **kwargs: Any) -> ToolResult:
        try:
            pyautogui.press("playpause")
            return ToolResult.ok(tool=self.name, message="Media pause command triggered.")
        except Exception as e:
            return ToolResult.fail(tool=self.name, error=str(e))


class MediaNextTool(BaseTool):
    """Skips to the next media track."""
    name = "media_next"
    description = "Skip to the next song or video track."
    parameters = {"type": "object", "properties": {}, "required": []}
    permission_level = PermissionLevel.SAFE

    def execute(self, **kwargs: Any) -> ToolResult:
        try:
            pyautogui.press("nexttrack")
            return ToolResult.ok(tool=self.name, message="Skipped to next media track.")
        except Exception as e:
            return ToolResult.fail(tool=self.name, error=str(e))


class MediaPreviousTool(BaseTool):
    """Reverses to the previous media track."""
    name = "media_previous"
    description = "Return to the previous song or video track."
    parameters = {"type": "object", "properties": {}, "required": []}
    permission_level = PermissionLevel.SAFE

    def execute(self, **kwargs: Any) -> ToolResult:
        try:
            pyautogui.press("prevtrack")
            return ToolResult.ok(tool=self.name, message="Returned to previous media track.")
        except Exception as e:
            return ToolResult.fail(tool=self.name, error=str(e))
