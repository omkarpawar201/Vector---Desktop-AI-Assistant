"""
File management tools for Vector Desktop AI Assistant.
Provides Search Files, Open File, Open Folder, and Get File Info capabilities.
"""

from datetime import datetime, timezone
import os
from pathlib import Path
from typing import Any, List

from app.config.constants import PermissionLevel
from app.tools.base import BaseTool, ToolResult


class SearchFilesTool(BaseTool):
    """Searches for files matching a keyword query."""
    name = "search_files"
    description = "Search for files or folders matching a query keyword in a directory."
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "File name or keyword search query"
            },
            "directory": {
                "type": "string",
                "description": "Directory path to search in (defaults to User Home directory)"
            }
        },
        "required": ["query"]
    }
    permission_level = PermissionLevel.SAFE

    def execute(self, **kwargs: Any) -> ToolResult:
        query = kwargs.get("query", "").strip().lower()
        search_dir = kwargs.get("directory") or str(Path.home())

        if not query:
            return ToolResult.fail(tool=self.name, error="MISSING_QUERY", message="Search query is required.")

        search_path = Path(search_dir)
        if not search_path.exists():
            return ToolResult.fail(tool=self.name, error="PATH_NOT_FOUND", message=f"Directory '{search_dir}' not found.")

        matches: List[str] = []
        max_matches = 20

        try:
            # Walk directory with depth limit to ensure fast response
            for root, dirs, files in os.walk(search_path):
                # Ignore system / heavy hidden folders
                dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ('AppData', 'node_modules', 'venv', '__pycache__')]

                for name in files + dirs:
                    if query in name.lower():
                        full_path = str(Path(root) / name)
                        matches.append(full_path)
                        if len(matches) >= max_matches:
                            break
                if len(matches) >= max_matches:
                    break

            return ToolResult.ok(
                tool=self.name,
                data={"matches": matches, "count": len(matches), "search_directory": search_dir},
                message=f"Found {len(matches)} matching file(s) for '{query}' in {search_dir}."
            )
        except Exception as e:
            return ToolResult.fail(tool=self.name, error=str(e))


class OpenFileTool(BaseTool):
    """Opens a file using the default Windows system application."""
    name = "open_file"
    description = "Open a file at a specific path using the default associated application."
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Absolute or relative path to the file to open"
            }
        },
        "required": ["path"]
    }
    permission_level = PermissionLevel.SAFE

    def execute(self, **kwargs: Any) -> ToolResult:
        path_str = kwargs.get("path", "").strip()
        if not path_str:
            return ToolResult.fail(tool=self.name, error="MISSING_PATH", message="File path is required.")

        file_path = Path(path_str).resolve()
        if not file_path.exists() or not file_path.is_file():
            return ToolResult.fail(tool=self.name, error="FILE_NOT_FOUND", message=f"File '{path_str}' does not exist.")

        try:
            os.startfile(str(file_path))
            return ToolResult.ok(
                tool=self.name,
                data={"path": str(file_path)},
                message=f"Opened file '{file_path.name}'."
            )
        except Exception as e:
            return ToolResult.fail(tool=self.name, error=str(e))


class OpenFolderTool(BaseTool):
    """Opens a folder path in Windows File Explorer."""
    name = "open_folder"
    description = "Open a folder directory in Windows File Explorer."
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Absolute or relative directory path to open"
            }
        },
        "required": ["path"]
    }
    permission_level = PermissionLevel.SAFE

    def execute(self, **kwargs: Any) -> ToolResult:
        path_str = kwargs.get("path", "").strip()
        if not path_str:
            return ToolResult.fail(tool=self.name, error="MISSING_PATH", message="Folder path is required.")

        folder_path = Path(path_str).resolve()
        if not folder_path.exists() or not folder_path.is_dir():
            return ToolResult.fail(tool=self.name, error="FOLDER_NOT_FOUND", message=f"Directory '{path_str}' does not exist.")

        try:
            os.startfile(str(folder_path))
            return ToolResult.ok(
                tool=self.name,
                data={"path": str(folder_path)},
                message=f"Opened folder '{folder_path.name}' in File Explorer."
            )
        except Exception as e:
            return ToolResult.fail(tool=self.name, error=str(e))


class GetFileInfoTool(BaseTool):
    """Retrieves file metadata (size, extension, modified time)."""
    name = "get_file_info"
    description = "Get file details including size, extension, created time, and modified time."
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Absolute or relative path to the target file"
            }
        },
        "required": ["path"]
    }
    permission_level = PermissionLevel.SAFE

    def execute(self, **kwargs: Any) -> ToolResult:
        path_str = kwargs.get("path", "").strip()
        if not path_str:
            return ToolResult.fail(tool=self.name, error="MISSING_PATH", message="File path is required.")

        file_path = Path(path_str).resolve()
        if not file_path.exists():
            return ToolResult.fail(tool=self.name, error="PATH_NOT_FOUND", message=f"Path '{path_str}' does not exist.")

        try:
            stat = file_path.stat()
            size_kb = round(stat.st_size / 1024, 2)
            mtime = datetime.fromtimestamp(stat.st_mtime, timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

            data = {
                "name": file_path.name,
                "path": str(file_path),
                "is_dir": file_path.is_dir(),
                "size_kb": size_kb,
                "extension": file_path.suffix,
                "last_modified": mtime
            }
            return ToolResult.ok(
                tool=self.name,
                data=data,
                message=f"File '{file_path.name}': {size_kb} KB, modified {mtime}."
            )
        except Exception as e:
            return ToolResult.fail(tool=self.name, error=str(e))
