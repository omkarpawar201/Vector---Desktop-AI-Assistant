"""
Unit tests for Application, File, Window Manager, and Safe Terminal tools.
"""

from pathlib import Path
from app.tools.applications.launcher import CloseAppTool, GetRunningAppsTool, LaunchAppTool
from app.tools.files.manager import GetFileInfoTool, OpenFolderTool, SearchFilesTool
from app.tools.terminal.executor import ExecuteTerminalCommandTool
from app.tools.windows.manager import GetActiveWindowTool, MaximizeWindowTool, MinimizeWindowTool


def test_app_launcher_tools():
    running_tool = GetRunningAppsTool()
    res_running = running_tool.execute()
    assert res_running.success is True
    assert "running_apps" in res_running.data

    launch_tool = LaunchAppTool()
    # Missing app name parameter test
    assert launch_tool.execute().success is False

    close_tool = CloseAppTool()
    assert close_tool.execute().success is False  # missing name


def test_file_manager_tools():
    search_tool = SearchFilesTool()
    res_search = search_tool.execute(query="README", directory=str(Path.cwd()))
    assert res_search.success is True
    assert res_search.data["count"] >= 1

    info_tool = GetFileInfoTool()
    res_info = info_tool.execute(path="README.md")
    assert res_info.success is True
    assert res_info.data["name"] == "README.md"
    assert res_info.data["size_kb"] > 0


def test_window_manager_tools():
    active_tool = GetActiveWindowTool()
    res_active = active_tool.execute()
    assert res_active.success is True

    max_tool = MaximizeWindowTool()
    assert max_tool.execute().success is True

    min_tool = MinimizeWindowTool()
    assert min_tool.execute().success is True


def test_safe_terminal_tool():
    term_tool = ExecuteTerminalCommandTool()

    # 1. Whitelisted command success
    res_python = term_tool.execute(command="python --version")
    assert res_python.success is True
    assert "Python" in res_python.data["stdout"]

    # 2. Non-whitelisted command rejection
    res_curl = term_tool.execute(command="curl https://example.com")
    assert res_curl.success is False
    assert res_curl.error == "COMMAND_NOT_WHITELISTED"

    # 3. Forbidden shell operator rejection
    res_inject = term_tool.execute(command="python --version && echo hacked")
    assert res_inject.success is False
    assert res_inject.error == "SHELL_OPERATOR_PROHIBITED"
