"""
Headless unit tests for PySide6 UI components.
"""

import os
import pytest
from PySide6.QtWidgets import QApplication

# Set headless Qt offscreen platform for test environments
os.environ["QT_QPA_PLATFORM"] = "offscreen"


@pytest.fixture(scope="session")
def qapp():
    """Session fixture initializing QApplication."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


def test_status_bar_widget_init(qapp):
    """Verify StatusBarWidget initialization."""
    from app.ui.status_bar import StatusBarWidget
    widget = StatusBarWidget()
    assert widget.lbl_brand.text() == "VECTOR 1.0"
    assert "CPU:" in widget.lbl_cpu.text()


def test_command_input_widget_init(qapp):
    """Verify CommandInputWidget initialization."""
    from app.ui.command_input import CommandInputWidget
    widget = CommandInputWidget()
    assert widget.input_edit.placeholderText() != ""


def test_confirmation_dialog_init(qapp):
    """Verify ConfirmationDialog initialization."""
    from app.ui.confirmation_dialog import ConfirmationDialog
    dlg = ConfirmationDialog(tool_name="shutdown_pc", message="Confirm shutdown")
    assert "shutdown_pc" in dlg.windowTitle() or "shutdown_pc" in dlg.findChild(object, "heading").text()
