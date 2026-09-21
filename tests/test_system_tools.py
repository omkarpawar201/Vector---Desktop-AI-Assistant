"""
Unit tests for System Stats, Volume, Power, and Media tools.
"""

from app.config.constants import PermissionLevel
from app.tools.media.controller import MediaNextTool, MediaPauseTool, MediaPlayTool, MediaPreviousTool
from app.tools.power.power import LockPcTool, RestartPcTool, ShutdownPcTool, SleepPcTool
from app.tools.system.system_info import (
    GetBatteryStatusTool,
    GetCpuUsageTool,
    GetDiskUsageTool,
    GetMemoryUsageTool,
    GetSystemStatsTool,
)
from app.tools.system.volume import GetVolumeTool, MuteTool, SetVolumeTool, UnmuteTool


def test_system_info_tools():
    stats_tool = GetSystemStatsTool()
    res = stats_tool.execute()
    assert res.success is True
    assert "cpu_percent" in res.data
    assert "memory_percent" in res.data

    cpu_tool = GetCpuUsageTool()
    res_cpu = cpu_tool.execute()
    assert res_cpu.success is True
    assert "cpu_percent" in res_cpu.data

    mem_tool = GetMemoryUsageTool()
    res_mem = mem_tool.execute()
    assert res_mem.success is True
    assert "used_gb" in res_mem.data

    disk_tool = GetDiskUsageTool()
    res_disk = disk_tool.execute()
    assert res_disk.success is True
    assert "free_gb" in res_disk.data

    batt_tool = GetBatteryStatusTool()
    res_batt = batt_tool.execute()
    assert res_batt.success is True


def test_volume_tools():
    get_vol = GetVolumeTool()
    res_get = get_vol.execute()
    assert res_get.success is True
    assert "level" in res_get.data

    set_vol = SetVolumeTool()
    res_set = set_vol.execute(level=30)
    assert res_set.success is True
    assert res_set.data["level"] == 30

    mute_tool = MuteTool()
    assert mute_tool.execute().success is True

    unmute_tool = UnmuteTool()
    assert unmute_tool.execute().success is True


def test_power_tool_permissions():
    lock_tool = LockPcTool()
    assert lock_tool.permission_level == PermissionLevel.SAFE

    sleep_tool = SleepPcTool()
    assert sleep_tool.permission_level == PermissionLevel.CONFIRM

    restart_tool = RestartPcTool()
    assert restart_tool.permission_level == PermissionLevel.DANGEROUS

    shutdown_tool = ShutdownPcTool()
    assert shutdown_tool.permission_level == PermissionLevel.DANGEROUS


def test_media_tools():
    play = MediaPlayTool()
    assert play.permission_level == PermissionLevel.SAFE
    res_play = play.execute()
    assert res_play.success is True

    pause = MediaPauseTool()
    assert pause.execute().success is True

    nxt = MediaNextTool()
    assert nxt.execute().success is True

    prev = MediaPreviousTool()
    assert prev.execute().success is True
