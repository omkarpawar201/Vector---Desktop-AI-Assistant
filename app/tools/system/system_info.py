"""
System stats tools for Vector Desktop AI Assistant.
Provides high-speed monitoring for CPU, Memory, Disk, and Battery metrics.
"""

from typing import Any, Dict
import psutil

from app.config.constants import PermissionLevel
from app.tools.base import BaseTool, ToolResult


class GetSystemStatsTool(BaseTool):
    """Retrieves an overall system health snapshot (CPU, Memory, Disk, Battery)."""
    name = "get_system_stats"
    description = "Get an overall system stats snapshot including CPU, RAM, Disk, and Battery usage."
    parameters = {"type": "object", "properties": {}, "required": []}
    permission_level = PermissionLevel.SAFE

    def execute(self, **kwargs: Any) -> ToolResult:
        try:
            cpu = psutil.cpu_percent(interval=0.1)
            mem = psutil.virtual_memory()
            disk = psutil.disk_usage("/")
            battery = psutil.sensors_battery()

            data = {
                "cpu_percent": cpu,
                "memory_percent": mem.percent,
                "memory_used_gb": round(mem.used / (1024**3), 2),
                "memory_total_gb": round(mem.total / (1024**3), 2),
                "disk_percent": disk.percent,
                "disk_free_gb": round(disk.free / (1024**3), 2),
                "battery_percent": battery.percent if battery else None,
                "power_plugged": battery.power_plugged if battery else None,
            }
            msg = f"CPU: {cpu}% | RAM: {mem.percent}% ({data['memory_used_gb']}/{data['memory_total_gb']} GB) | Disk: {disk.percent}% used"
            return ToolResult.ok(tool=self.name, data=data, message=msg)
        except Exception as e:
            return ToolResult.fail(tool=self.name, error=str(e))


class GetCpuUsageTool(BaseTool):
    """Retrieves current CPU usage percentage."""
    name = "get_cpu_usage"
    description = "Get the current CPU usage percentage across all cores."
    parameters = {"type": "object", "properties": {}, "required": []}
    permission_level = PermissionLevel.SAFE

    def execute(self, **kwargs: Any) -> ToolResult:
        try:
            cpu = psutil.cpu_percent(interval=0.1)
            return ToolResult.ok(
                tool=self.name,
                data={"cpu_percent": cpu},
                message=f"Current CPU usage is {cpu}%."
            )
        except Exception as e:
            return ToolResult.fail(tool=self.name, error=str(e))


class GetMemoryUsageTool(BaseTool):
    """Retrieves current RAM memory usage."""
    name = "get_memory_usage"
    description = "Get current system RAM memory usage, total memory, and free memory."
    parameters = {"type": "object", "properties": {}, "required": []}
    permission_level = PermissionLevel.SAFE

    def execute(self, **kwargs: Any) -> ToolResult:
        try:
            mem = psutil.virtual_memory()
            data = {
                "percent": mem.percent,
                "used_gb": round(mem.used / (1024**3), 2),
                "available_gb": round(mem.available / (1024**3), 2),
                "total_gb": round(mem.total / (1024**3), 2),
            }
            return ToolResult.ok(
                tool=self.name,
                data=data,
                message=f"Memory usage: {mem.percent}% ({data['used_gb']} GB used of {data['total_gb']} GB)."
            )
        except Exception as e:
            return ToolResult.fail(tool=self.name, error=str(e))


class GetDiskUsageTool(BaseTool):
    """Retrieves current Disk space usage."""
    name = "get_disk_usage"
    description = "Get primary disk drive space usage and available free storage."
    parameters = {"type": "object", "properties": {}, "required": []}
    permission_level = PermissionLevel.SAFE

    def execute(self, **kwargs: Any) -> ToolResult:
        try:
            disk = psutil.disk_usage("/")
            data = {
                "percent": disk.percent,
                "used_gb": round(disk.used / (1024**3), 2),
                "free_gb": round(disk.free / (1024**3), 2),
                "total_gb": round(disk.total / (1024**3), 2),
            }
            return ToolResult.ok(
                tool=self.name,
                data=data,
                message=f"Disk usage: {disk.percent}% ({data['free_gb']} GB free of {data['total_gb']} GB)."
            )
        except Exception as e:
            return ToolResult.fail(tool=self.name, error=str(e))


class GetBatteryStatusTool(BaseTool):
    """Retrieves battery charge level and charging state."""
    name = "get_battery_status"
    description = "Get the laptop battery status, percentage, and charging power connection."
    parameters = {"type": "object", "properties": {}, "required": []}
    permission_level = PermissionLevel.SAFE

    def execute(self, **kwargs: Any) -> ToolResult:
        try:
            battery = psutil.sensors_battery()
            if not battery:
                return ToolResult.ok(
                    tool=self.name,
                    data={"has_battery": False},
                    message="No battery detected (Desktop system)."
                )

            data = {
                "has_battery": True,
                "percent": battery.percent,
                "power_plugged": battery.power_plugged,
            }
            state = "Plugged in" if battery.power_plugged else "Discharging"
            return ToolResult.ok(
                tool=self.name,
                data=data,
                message=f"Battery at {battery.percent}% ({state})."
            )
        except Exception as e:
            return ToolResult.fail(tool=self.name, error=str(e))
