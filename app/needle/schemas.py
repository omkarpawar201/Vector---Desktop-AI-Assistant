"""
Intent tool declarations for Cactus Needle 2 native C++ decode engine.
Provides tool functions registered with Needle for grammar-constrained decoding.
"""


def launch_app(name: str):
    """Launch a desktop application by name (e.g. chrome, notepad, vscode, calculator, spotify)."""
    pass


def close_app(name: str):
    """Close a running desktop application by name."""
    pass


def search_files(query: str):
    """Search for files or documents on disk matching a query."""
    pass


def get_running_apps():
    """List all currently active running applications on the system."""
    pass


def get_cactus_tools():
    """Returns the list of tool functions for Cactus Needle decoding."""
    return [launch_app, close_app, search_files, get_running_apps]
