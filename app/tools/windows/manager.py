"""
Window management tools for Vector Desktop AI Assistant.
Provides window inspection, focus, maximize, minimize, close, move, and resize via win32gui.
"""

from typing import Any, Dict
import win32con
import win32gui
import win32process

from app.config.constants import PermissionLevel
from app.tools.base import BaseTool, ToolResult


def _get_active_hwnd() -> int:
    return win32gui.GetForegroundWindow()


class GetActiveWindowTool(BaseTool):
    """Retrieves the title and process details of the currently focused active window."""
    name = "get_active_window"
    description = "Get title, window handle, and process details of the active foreground window."
    parameters = {"type": "object", "properties": {}, "required": []}
    permission_level = PermissionLevel.SAFE

    def execute(self, **kwargs: Any) -> ToolResult:
        try:
            hwnd = _get_active_hwnd()
            if not hwnd:
                return ToolResult.ok(tool=self.name, data={"active": False}, message="No active window focused.")

            title = win32gui.GetWindowText(hwnd)
            _, pid = win32process.GetWindowThreadProcessId(hwnd)
            rect = win32gui.GetWindowRect(hwnd)

            data = {
                "hwnd": hwnd,
                "title": title,
                "pid": pid,
                "bounds": {"left": rect[0], "top": rect[1], "right": rect[2], "bottom": rect[3]}
            }
            return ToolResult.ok(tool=self.name, data=data, message=f"Active window: '{title}' (PID {pid}).")
        except Exception as e:
            return ToolResult.fail(tool=self.name, error=str(e))


class MaximizeWindowTool(BaseTool):
    """Maximizes the currently active or specified window."""
    name = "maximize_window"
    description = "Maximize the active foreground window."
    parameters = {"type": "object", "properties": {}, "required": []}
    permission_level = PermissionLevel.SAFE

    def execute(self, **kwargs: Any) -> ToolResult:
        try:
            hwnd = _get_active_hwnd()
            if not hwnd:
                return ToolResult.fail(tool=self.name, error="NO_ACTIVE_WINDOW", message="No active window to maximize.")

            title = win32gui.GetWindowText(hwnd)
            win32gui.ShowWindow(hwnd, win32con.SW_MAXIMIZE)
            return ToolResult.ok(tool=self.name, message=f"Maximized window '{title}'.")
        except Exception as e:
            return ToolResult.fail(tool=self.name, error=str(e))


class MinimizeWindowTool(BaseTool):
    """Minimizes the currently active window to taskbar."""
    name = "minimize_window"
    description = "Minimize the active foreground window to the taskbar."
    parameters = {"type": "object", "properties": {}, "required": []}
    permission_level = PermissionLevel.SAFE

    def execute(self, **kwargs: Any) -> ToolResult:
        try:
            hwnd = _get_active_hwnd()
            if not hwnd:
                return ToolResult.fail(tool=self.name, error="NO_ACTIVE_WINDOW", message="No active window to minimize.")

            title = win32gui.GetWindowText(hwnd)
            win32gui.ShowWindow(hwnd, win32con.SW_MINIMIZE)
            return ToolResult.ok(tool=self.name, message=f"Minimized window '{title}'.")
        except Exception as e:
            return ToolResult.fail(tool=self.name, error=str(e))


class CloseWindowTool(BaseTool):
    """Closes the currently active window safely."""
    name = "close_window"
    description = "Close the active foreground window."
    parameters = {"type": "object", "properties": {}, "required": []}
    permission_level = PermissionLevel.SAFE

    def execute(self, **kwargs: Any) -> ToolResult:
        try:
            hwnd = _get_active_hwnd()
            if not hwnd:
                return ToolResult.fail(tool=self.name, error="NO_ACTIVE_WINDOW", message="No active window to close.")

            title = win32gui.GetWindowText(hwnd)
            win32gui.PostMessage(hwnd, win32con.WM_CLOSE, 0, 0)
            return ToolResult.ok(tool=self.name, message=f"Closed window '{title}'.")
        except Exception as e:
            return ToolResult.fail(tool=self.name, error=str(e))


class MoveWindowTool(BaseTool):
    """Moves the active window to specified screen coordinates (x, y)."""
    name = "move_window"
    description = "Move the active window to specified X and Y screen coordinates."
    parameters = {
        "type": "object",
        "properties": {
            "x": {"type": "integer", "description": "Screen X position in pixels"},
            "y": {"type": "integer", "description": "Screen Y position in pixels"}
        },
        "required": ["x", "y"]
    }
    permission_level = PermissionLevel.SAFE

    def execute(self, **kwargs: Any) -> ToolResult:
        x = kwargs.get("x")
        y = kwargs.get("y")
        if x is None or y is None:
            return ToolResult.fail(tool=self.name, error="MISSING_COORDINATES", message="X and Y parameters are required.")

        try:
            hwnd = _get_active_hwnd()
            if not hwnd:
                return ToolResult.fail(tool=self.name, error="NO_ACTIVE_WINDOW", message="No active window to move.")

            rect = win32gui.GetWindowRect(hwnd)
            width = rect[2] - rect[0]
            height = rect[3] - rect[1]

            win32gui.MoveWindow(hwnd, int(x), int(y), width, height, True)
            return ToolResult.ok(tool=self.name, data={"x": x, "y": y}, message=f"Moved window to ({x}, {y}).")
        except Exception as e:
            return ToolResult.fail(tool=self.name, error=str(e))


class ResizeWindowTool(BaseTool):
    """Resizes the active window to specified width and height."""
    name = "resize_window"
    description = "Resize the active window to a specific width and height in pixels."
    parameters = {
        "type": "object",
        "properties": {
            "width": {"type": "integer", "minimum": 100, "description": "Window width in pixels"},
            "height": {"type": "integer", "minimum": 100, "description": "Window height in pixels"}
        },
        "required": ["width", "height"]
    }
    permission_level = PermissionLevel.SAFE

    def execute(self, **kwargs: Any) -> ToolResult:
        width = kwargs.get("width")
        height = kwargs.get("height")
        if width is None or height is None:
            return ToolResult.fail(tool=self.name, error="MISSING_DIMENSIONS", message="Width and height parameters are required.")

        try:
            hwnd = _get_active_hwnd()
            if not hwnd:
                return ToolResult.fail(tool=self.name, error="NO_ACTIVE_WINDOW", message="No active window to resize.")

            rect = win32gui.GetWindowRect(hwnd)
            win32gui.MoveWindow(hwnd, rect[0], rect[1], int(width), int(height), True)
            return ToolResult.ok(tool=self.name, data={"width": width, "height": height}, message=f"Resized window to {width}x{height} pixels.")
        except Exception as e:
            return ToolResult.fail(tool=self.name, error=str(e))
