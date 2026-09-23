"""
Application management tools for Vector Desktop AI Assistant.
Provides Launch Application, Close Application, and List Running Apps capabilities.
"""

import os
import subprocess
from typing import Any, Dict, List
import psutil

from app.config.constants import PermissionLevel
from app.tools.base import BaseTool, ToolResult

# Common application name aliases for Windows startup
APP_ALIASES: Dict[str, str] = {
    "chrome": "chrome.exe",
    "google chrome": "chrome.exe",
    "edge": "msedge.exe",
    "msedge": "msedge.exe",
    "notepad": "notepad.exe",
    "calc": "calc.exe",
    "calculator": "calc.exe",
    "code": "code.cmd",
    "vscode": "code.cmd",
    "vs code": "code.cmd",
    "spotify": "spotify.exe",
    "explorer": "explorer.exe",
    "file explorer": "explorer.exe",
    "cmd": "cmd.exe",
    "terminal": "wt.exe",
    "powershell": "powershell.exe"
}


class LaunchAppTool(BaseTool):
    """Launches an application by name or executable path."""
    name = "launch_app"
    description = "Launch an installed desktop application by name (e.g., Chrome, Notepad, VS Code, Calculator)."
    parameters = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Name or alias of the application to open (e.g. 'chrome', 'notepad', 'code')"
            }
        },
        "required": ["name"]
    }
    permission_level = PermissionLevel.SAFE

    def execute(self, **kwargs: Any) -> ToolResult:
        app_name = kwargs.get("name", "").strip().lower()
        if not app_name:
            return ToolResult.fail(tool=self.name, error="MISSING_APP_NAME", message="Application name is required.")

        target = APP_ALIASES.get(app_name, app_name)

        # Security guard: block shell operators/metacharacters in the target to
        # prevent command injection when falling back to the 'start' shell path.
        _BLOCKED_SHELL_CHARS = set("&|;<>`$(){}[]")
        if any(ch in target for ch in _BLOCKED_SHELL_CHARS):
            return ToolResult.fail(
                tool=self.name,
                error="INVALID_APP_NAME",
                message=f"Application name '{app_name}' contains invalid characters and was rejected."
            )

        try:
            # Attempt to launch via Windows start command or subprocess
            if os.path.isabs(target) and os.path.exists(target):
                os.startfile(target)
            else:
                # Use Windows start protocol with explicit quiting for universal path/app launching
                subprocess.Popen(f'start "" "{target}"', shell=True)

            return ToolResult.ok(
                tool=self.name,
                data={"launched": app_name, "target": target},
                message=f"Launched '{app_name}' successfully."
            )
        except Exception as e:
            return ToolResult.fail(tool=self.name, error=str(e))


class CloseAppTool(BaseTool):
    """Terminates running instances of an application by name."""
    name = "close_app"
    description = "Close or terminate running instances of an application by name."
    parameters = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Name of the application process to close (e.g. 'chrome', 'spotify', 'notepad')"
            }
        },
        "required": ["name"]
    }
    permission_level = PermissionLevel.SAFE

    def execute(self, **kwargs: Any) -> ToolResult:
        app_name = kwargs.get("name", "").strip().lower()
        if not app_name:
            return ToolResult.fail(tool=self.name, error="MISSING_APP_NAME", message="Application name is required.")

        target = APP_ALIASES.get(app_name, app_name)
        clean_target = target.replace(".exe", "").replace(".cmd", "").lower()

        closed_pids: List[int] = []
        try:
            for proc in psutil.process_iter(['pid', 'name']):
                try:
                    pname = proc.info['name'].lower()
                    if clean_target in pname:
                        proc.terminate()
                        closed_pids.append(proc.info['pid'])
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

            if closed_pids:
                return ToolResult.ok(
                    tool=self.name,
                    data={"closed_count": len(closed_pids), "pids": closed_pids},
                    message=f"Closed {len(closed_pids)} process instance(s) matching '{app_name}'."
                )
            else:
                return ToolResult.fail(
                    tool=self.name,
                    error="APP_NOT_RUNNING",
                    message=f"No running processes found matching '{app_name}'."
                )
        except Exception as e:
            return ToolResult.fail(tool=self.name, error=str(e))


class GetRunningAppsTool(BaseTool):
    """Lists currently active user-facing applications."""
    name = "get_running_apps"
    description = "Get a list of currently active running desktop applications."
    parameters = {"type": "object", "properties": {}, "required": []}
    permission_level = PermissionLevel.SAFE

    def execute(self, **kwargs: Any) -> ToolResult:
        try:
            running_apps = set()
            for proc in psutil.process_iter(['name']):
                try:
                    name = proc.info['name']
                    if name and name.endswith('.exe') and not name.startswith('svchost'):
                        running_apps.add(name.replace('.exe', ''))
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

            app_list = sorted(list(running_apps))[:20]  # Return top 20 active processes
            return ToolResult.ok(
                tool=self.name,
                data={"running_apps": app_list, "count": len(app_list)},
                message=f"Found {len(app_list)} active desktop applications: {', '.join(app_list[:10])}..."
            )
        except Exception as e:
            return ToolResult.fail(tool=self.name, error=str(e))
