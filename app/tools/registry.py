"""
Tool and Capability Registry for Vector Desktop AI Assistant.
Central heart of Vector where all desktop capabilities register.
Neither Needle nor Gemini execute OS actions directly—they query this registry.
"""

from typing import Any, Dict, List, Optional
from app.tools.base import BaseTool


class CapabilityRegistry:
    """
    Registry tracking all capabilities supported locally by Vector.
    Used by the router to instantly distinguish between locally executable tasks
    and complex queries requiring cloud Gemini reasoning.
    """

    def __init__(self):
        self._capabilities: Dict[str, str] = {
            "system_monitoring": "Query CPU, RAM, Disk, and Battery metrics",
            "volume_control": "Get, set, mute, or unmute system volume",
            "power_operations": "Lock, sleep, restart, or shutdown PC",
            "media_control": "Play, pause, skip, or reverse media playback",
            "app_management": "Launch, close, or list running applications",
            "file_operations": "Search files, open files/folders, query file info",
            "window_management": "Maximize, minimize, focus, resize, or move windows",
            "terminal_cmd": "Execute whitelisted terminal CLI commands"
        }

    def register_capability(self, name: str, description: str) -> None:
        """Register a new supported local capability."""
        self._capabilities[name.lower()] = description

    def is_supported(self, capability_name: str) -> bool:
        """Check if a capability is supported locally."""
        return capability_name.lower() in self._capabilities

    def list_capabilities(self) -> Dict[str, str]:
        """Return a dictionary of all supported local capabilities."""
        return self._capabilities.copy()


class ToolRegistry:
    """
    Singleton registry storing instances of all BaseTool capabilities.
    """

    _instance: Optional["ToolRegistry"] = None

    def __new__(cls) -> "ToolRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._tools = {}
            cls._instance.capability_registry = CapabilityRegistry()
        return cls._instance

    def register(self, tool: BaseTool) -> None:
        """Register a tool instance."""
        if not hasattr(tool, "name") or not tool.name:
            raise ValueError("Cannot register tool without a valid 'name' attribute.")
        self._tools[tool.name] = tool

    def get(self, tool_name: str) -> Optional[BaseTool]:
        """Retrieve a registered tool by name."""
        return self._tools.get(tool_name)

    def list_tools(self) -> List[BaseTool]:
        """Return a list of all registered tool instances."""
        return list(self._tools.values())

    def export_schemas(self) -> List[Dict[str, Any]]:
        """
        Export JSON Schemas for all registered tools.
        Consumable by both Needle 2 and Gemini API function declarations.
        """
        return [tool.to_schema() for tool in self._tools.values()]

    def clear(self) -> None:
        """Clear all registered tools (used for testing resets)."""
        self._tools.clear()


# Helper function to get global registry instance
def get_tool_registry() -> ToolRegistry:
    """Returns the global ToolRegistry instance."""
    registry = ToolRegistry()
    if not registry.list_tools():
        register_all_default_tools(registry)
    return registry


def register_all_default_tools(registry: Optional[ToolRegistry] = None) -> ToolRegistry:
    """Registers all system, media, app, file, window, power, and terminal tools into registry."""
    reg = registry or ToolRegistry()
    if reg.list_tools():
        return reg

    from app.tools.applications.launcher import CloseAppTool, GetRunningAppsTool, LaunchAppTool
    from app.tools.files.manager import GetFileInfoTool, OpenFileTool, OpenFolderTool, SearchFilesTool
    from app.tools.media.controller import MediaNextTool, MediaPauseTool, MediaPreviousTool, MediaPlayTool
    from app.tools.power.power import LockPcTool, RestartPcTool, ShutdownPcTool, SleepPcTool
    from app.tools.system.system_info import (
        GetBatteryStatusTool, GetCpuUsageTool, GetDiskUsageTool, GetMemoryUsageTool, GetSystemStatsTool
    )
    from app.tools.system.volume import GetVolumeTool, MuteTool, SetVolumeTool, UnmuteTool
    from app.tools.terminal.executor import ExecuteTerminalCommandTool
    from app.tools.windows.manager import (
        CloseWindowTool, GetActiveWindowTool, MaximizeWindowTool, MinimizeWindowTool, MoveWindowTool, ResizeWindowTool
    )

    tools = [
        GetSystemStatsTool(), GetCpuUsageTool(), GetMemoryUsageTool(), GetDiskUsageTool(), GetBatteryStatusTool(),
        GetVolumeTool(), SetVolumeTool(), MuteTool(), UnmuteTool(),
        LockPcTool(), SleepPcTool(), RestartPcTool(), ShutdownPcTool(),
        MediaPlayTool(), MediaPauseTool(), MediaNextTool(), MediaPreviousTool(),
        LaunchAppTool(), CloseAppTool(), GetRunningAppsTool(),
        SearchFilesTool(), OpenFileTool(), OpenFolderTool(), GetFileInfoTool(),
        GetActiveWindowTool(), MaximizeWindowTool(), MinimizeWindowTool(), CloseWindowTool(), MoveWindowTool(), ResizeWindowTool(),
        ExecuteTerminalCommandTool()
    ]
    for t in tools:
        reg.register(t)
    return reg
