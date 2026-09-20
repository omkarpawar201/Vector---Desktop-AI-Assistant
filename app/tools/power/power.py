"""
Power management tools for Vector Desktop AI Assistant.
Provides Lock, Sleep, Restart, and Shutdown OS capabilities with safety permissions.
"""

import ctypes
import subprocess
from typing import Any
from app.config.constants import PermissionLevel
from app.tools.base import BaseTool, ToolResult


class LockPcTool(BaseTool):
    """Locks the current Windows user session instantly."""
    name = "lock_pc"
    description = "Lock the Windows workstation session."
    parameters = {"type": "object", "properties": {}, "required": []}
    permission_level = PermissionLevel.SAFE

    def execute(self, **kwargs: Any) -> ToolResult:
        try:
            ctypes.windll.user32.LockWorkStation()
            return ToolResult.ok(tool=self.name, message="Workstation locked.")
        except Exception as e:
            return ToolResult.fail(tool=self.name, error=str(e))


class SleepPcTool(BaseTool):
    """Puts the computer into sleep mode."""
    name = "sleep_pc"
    description = "Put the computer into low-power sleep mode."
    parameters = {"type": "object", "properties": {}, "required": []}
    permission_level = PermissionLevel.CONFIRM

    def execute(self, **kwargs: Any) -> ToolResult:
        try:
            # SetSuspendState(bHibernate=False, bForce=False, bWakeupEventsDisabled=False)
            res = ctypes.windll.powrprof.SetSuspendState(0, 1, 0)
            if res == 0:
                # Fallback to Rundll32 if SetSuspendState fails
                subprocess.run(["rundll32.exe", "powrprof.dll,SetSuspendState", "0,1,0"], check=True)
            return ToolResult.ok(tool=self.name, message="System entering sleep mode.")
        except Exception as e:
            return ToolResult.fail(tool=self.name, error=str(e))


class RestartPcTool(BaseTool):
    """Restarts the computer immediately."""
    name = "restart_pc"
    description = "Restart the Windows operating system immediately."
    parameters = {"type": "object", "properties": {}, "required": []}
    permission_level = PermissionLevel.DANGEROUS

    def execute(self, **kwargs: Any) -> ToolResult:
        try:
            subprocess.run(["shutdown", "/r", "/t", "0"], check=True)
            return ToolResult.ok(tool=self.name, message="System restart command issued.")
        except Exception as e:
            return ToolResult.fail(tool=self.name, error=str(e))


class ShutdownPcTool(BaseTool):
    """Shuts down the computer immediately."""
    name = "shutdown_pc"
    description = "Shut down the Windows operating system immediately."
    parameters = {"type": "object", "properties": {}, "required": []}
    permission_level = PermissionLevel.DANGEROUS

    def execute(self, **kwargs: Any) -> ToolResult:
        try:
            subprocess.run(["shutdown", "/s", "/t", "0"], check=True)
            return ToolResult.ok(tool=self.name, message="System shutdown command issued.")
        except Exception as e:
            return ToolResult.fail(tool=self.name, error=str(e))
