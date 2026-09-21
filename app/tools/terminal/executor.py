"""
Safe Terminal Execution Tool for Vector Desktop AI Assistant.
Provides whitelisted CLI execution with strict shell injection prevention.
"""

import shlex
import subprocess
from typing import Any, List
from app.config.constants import ALLOWED_TERMINAL_COMMANDS, PermissionLevel
from app.tools.base import BaseTool, ToolResult


class ExecuteTerminalCommandTool(BaseTool):
    """Executes safe CLI terminal commands from a strict whitelist."""
    name = "execute_terminal_command"
    description = "Execute a terminal CLI command (whitelisted commands only: git, python, pip, npm, node, docker, systemctl)."
    parameters = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "Terminal command string to execute (e.g., 'git status', 'python --version', 'pip list')"
            }
        },
        "required": ["command"]
    }
    permission_level = PermissionLevel.SAFE

    # Forbidden shell injection operators
    FORBIDDEN_OPERATORS: List[str] = ["&&", "||", ";", "|", ">", "<", "`", "$("]

    def execute(self, **kwargs: Any) -> ToolResult:
        cmd_str = kwargs.get("command", "").strip()
        if not cmd_str:
            return ToolResult.fail(tool=self.name, error="MISSING_COMMAND", message="Terminal command is required.")

        # 1. Security Check: Block dangerous shell chaining operators
        for op in self.FORBIDDEN_OPERATORS:
            if op in cmd_str:
                return ToolResult.fail(
                    tool=self.name,
                    error="SHELL_OPERATOR_PROHIBITED",
                    message=f"Command chaining operator '{op}' is prohibited for safety reasons."
                )

        # 2. Tokenize command safely
        try:
            tokens = shlex.split(cmd_str)
        except Exception as e:
            return ToolResult.fail(tool=self.name, error="INVALID_COMMAND_SYNTAX", message=f"Failed to parse command tokens: {e}")

        if not tokens:
            return ToolResult.fail(tool=self.name, error="EMPTY_COMMAND", message="Command string was empty.")

        # 3. Whitelist check on primary executable binary
        binary = tokens[0].lower()
        if binary not in ALLOWED_TERMINAL_COMMANDS:
            return ToolResult.fail(
                tool=self.name,
                error="COMMAND_NOT_WHITELISTED",
                message=f"Command '{binary}' is not whitelisted. Allowed commands: {', '.join(sorted(ALLOWED_TERMINAL_COMMANDS))}"
            )

        # 4. Safe Subprocess Execution (No shell=True)
        try:
            res = subprocess.run(tokens, capture_output=True, text=True, timeout=30, check=False)
            data = {
                "command": cmd_str,
                "exit_code": res.returncode,
                "stdout": res.stdout.strip(),
                "stderr": res.stderr.strip()
            }

            if res.returncode == 0:
                output_preview = res.stdout.strip()[:300] or "Executed cleanly with no output."
                return ToolResult.ok(
                    tool=self.name,
                    data=data,
                    message=f"Command '{cmd_str}' succeeded (exit code 0):\n{output_preview}"
                )
            else:
                err_preview = res.stderr.strip()[:300] or res.stdout.strip()[:300] or "Non-zero exit code"
                return ToolResult.fail(
                    tool=self.name,
                    error=f"EXIT_CODE_{res.returncode}",
                    message=f"Command '{cmd_str}' failed with exit code {res.returncode}:\n{err_preview}"
                )
        except subprocess.TimeoutExpired:
            return ToolResult.fail(tool=self.name, error="COMMAND_TIMEOUT", message=f"Command '{cmd_str}' timed out after 30 seconds.")
        except Exception as e:
            return ToolResult.fail(tool=self.name, error=str(e))
