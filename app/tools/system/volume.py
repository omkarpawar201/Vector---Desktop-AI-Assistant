"""
Volume control tools for Vector Desktop AI Assistant.
Provides native Windows master audio volume control (get, set, mute, unmute).
"""

from typing import Any, Dict
from app.config.constants import PermissionLevel
from app.tools.base import BaseTool, ToolResult


def _get_audio_endpoint():
    """
    Helper function to get the Windows Default Audio Endpoint via pycaw.
    Supports both legacy and modern pycaw AudioDevice interfaces.
    """
    import comtypes
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

    # Ensure COM is initialized for the current thread
    try:
        comtypes.CoInitialize()
    except Exception:
        pass

    devices = AudioUtilities.GetSpeakers()
    
    # Handle pycaw AudioDevice wrapper compatibility
    if hasattr(devices, "Activate"):
        interface = devices.Activate(IAudioEndpointVolume._iid_, comtypes.CLSCTX_ALL, None)
    elif hasattr(devices, "Endpoint"):
        interface = devices.Endpoint.Activate(IAudioEndpointVolume._iid_, comtypes.CLSCTX_ALL, None)
    elif hasattr(devices, "_dev"):
        interface = devices._dev.Activate(IAudioEndpointVolume._iid_, comtypes.CLSCTX_ALL, None)
    else:
        raise AttributeError("Could not initialize Windows Audio Endpoint interface from pycaw.")

    volume = interface.QueryInterface(IAudioEndpointVolume)
    return volume


class GetVolumeTool(BaseTool):
    """Retrieves current master volume percentage and mute status."""
    name = "get_volume"
    description = "Get current system volume level (0 to 100) and mute status."
    parameters = {"type": "object", "properties": {}, "required": []}
    permission_level = PermissionLevel.SAFE

    def execute(self, **kwargs: Any) -> ToolResult:
        try:
            volume = _get_audio_endpoint()
            current_vol = round(volume.GetMasterVolumeLevelScalar() * 100)
            is_muted = bool(volume.GetMute())
            
            data = {"level": current_vol, "is_muted": is_muted}
            status = "Muted" if is_muted else "Unmuted"
            return ToolResult.ok(
                tool=self.name,
                data=data,
                message=f"Current volume is {current_vol}% ({status})."
            )
        except Exception as e:
            return ToolResult.fail(tool=self.name, error=str(e))


class SetVolumeTool(BaseTool):
    """Sets master volume level to a specific percentage (0 to 100)."""
    name = "set_volume"
    description = "Set system volume to a specified percentage level from 0 to 100."
    parameters = {
        "type": "object",
        "properties": {
            "level": {
                "type": "integer",
                "minimum": 0,
                "maximum": 100,
                "description": "Volume percentage level from 0 to 100"
            }
        },
        "required": ["level"]
    }
    permission_level = PermissionLevel.SAFE

    def execute(self, **kwargs: Any) -> ToolResult:
        level = kwargs.get("level")
        if level is None:
            return ToolResult.fail(tool=self.name, error="MISSING_LEVEL", message="Volume level parameter is required.")

        try:
            level = max(0, min(100, int(level)))
            volume = _get_audio_endpoint()
            volume.SetMasterVolumeLevelScalar(level / 100.0, None)
            
            return ToolResult.ok(
                tool=self.name,
                data={"level": level},
                message=f"Volume set to {level}%."
            )
        except Exception as e:
            return ToolResult.fail(tool=self.name, error=str(e))


class MuteTool(BaseTool):
    """Mutes system audio."""
    name = "mute"
    description = "Mute system audio."
    parameters = {"type": "object", "properties": {}, "required": []}
    permission_level = PermissionLevel.SAFE

    def execute(self, **kwargs: Any) -> ToolResult:
        try:
            volume = _get_audio_endpoint()
            volume.SetMute(1, None)
            return ToolResult.ok(tool=self.name, data={"is_muted": True}, message="System audio muted.")
        except Exception as e:
            return ToolResult.fail(tool=self.name, error=str(e))


class UnmuteTool(BaseTool):
    """Unmutes system audio."""
    name = "unmute"
    description = "Unmute system audio."
    parameters = {"type": "object", "properties": {}, "required": []}
    permission_level = PermissionLevel.SAFE

    def execute(self, **kwargs: Any) -> ToolResult:
        try:
            volume = _get_audio_endpoint()
            volume.SetMute(0, None)
            return ToolResult.ok(tool=self.name, data={"is_muted": False}, message="System audio unmuted.")
        except Exception as e:
            return ToolResult.fail(tool=self.name, error=str(e))
