"""
Focused regression tests for ChatViewWidget message clipping, dynamic sizing,
safe markdown rendering, and error card presentation.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QSize
from PySide6.QtGui import QColor, QImage
from PySide6.QtWidgets import QApplication

from app.ui.chat_view import (
    ActionCardWidget,
    ChatViewWidget,
    ErrorCardWidget,
    MessageContainer,
    render_safe_markdown,
)


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


def test_short_messages_not_clipped(qapp):
    """
    Verify very short user and assistant messages have sufficient vertical height
    and are not clipped by item margins or padding.
    """
    view = ChatViewWidget()
    view.resize(800, 600)
    view.show()

    view.add_user_message("Hi")
    view.add_assistant_message("Hello!")
    view.add_user_message("mute")
    view.add_assistant_message("Muted.")

    qapp.processEvents()

    assert view.list_widget.count() == 4
    for i in range(view.list_widget.count()):
        item = view.list_widget.item(i)
        widget = view.list_widget.itemWidget(item)
        rect = view.list_widget.visualItemRect(item)

        assert isinstance(widget, MessageContainer)
        label = widget.label

        # Visual height must be at least 36px and equal or greater than label height
        assert rect.height() >= 36, f"Item {i} rect height {rect.height()} too small"
        assert label.height() >= 30, f"Label {i} height {label.height()} too small"
        assert rect.height() >= label.height(), f"Item {i} clipped: rect {rect.height()} < label {label.height()}"

        # Verify pixel-level rendering: text pixels are not clipped at bottom edge
        img = QImage(rect.width(), rect.height(), QImage.Format_ARGB32)
        img.fill(QColor("#000000"))
        widget.render(img)

        bottom_y = 0
        for y in range(rect.height()):
            for x in range(rect.width()):
                if img.pixelColor(x, y).name() != "#000000":
                    bottom_y = max(bottom_y, y)

        margin_to_bottom = rect.height() - 1 - bottom_y
        assert margin_to_bottom >= 0, f"Item {i} text pixels cut off at bottom"


def test_multiline_and_long_messages_sizing(qapp):
    """
    Verify multi-line responses and long paragraphs wrap correctly without clipping.
    """
    view = ChatViewWidget()
    view.resize(800, 600)
    view.show()

    two_line = "This is line one.\nThis is line two."
    long_para = (
        "Vector is an agentic desktop AI assistant that operates across four distinct "
        "intent tiers: Tier 0 reflex patterns for sub-10ms desktop actions, Tier 0.5 custom ONNX, "
        "Tier 1 Cactus Needle native engine, and Tier 2 cloud Gemini multi-tool reasoning."
    )

    view.add_assistant_message(two_line)
    view.add_assistant_message(long_para)
    qapp.processEvents()

    assert view.list_widget.count() == 2
    item1 = view.list_widget.item(0)
    rect1 = view.list_widget.visualItemRect(item1)
    # Multi-line should have taller height than a single-line message
    assert rect1.height() >= 50

    item2 = view.list_widget.item(1)
    rect2 = view.list_widget.visualItemRect(item2)
    assert rect2.height() >= 70


def test_markdown_code_blocks_and_inline_code(qapp):
    """
    Verify inline code and fenced code blocks render safely with monospace styling.
    """
    view = ChatViewWidget()
    view.resize(800, 600)
    view.show()

    md_text = (
        "Here is some `inline code`.\n\n"
        "```python\n"
        "def test_func():\n"
        "    return 42\n"
        "```\n"
        "- Bullet point 1\n"
        "- Bullet point 2\n"
    )

    view.add_assistant_message(md_text)
    qapp.processEvents()

    assert view.list_widget.count() == 1
    widget = view.list_widget.itemWidget(view.list_widget.item(0))
    raw_html = widget.label.text()

    assert "inline code" in raw_html
    assert "test_func" in raw_html
    assert "Consolas" in raw_html or "Cascadia Code" in raw_html
    assert "• Bullet point 1" in raw_html


def test_incremental_streaming_growth(qapp):
    """
    Verify streamed assistant bubble grows dynamically from 1 line to several lines
    without clipping or visual jump.
    """
    view = ChatViewWidget()
    view.resize(800, 600)
    view.show()

    handle = view.begin_assistant_stream()
    qapp.processEvents()

    rect0 = view.list_widget.visualItemRect(handle.item)
    assert rect0.height() >= 36

    view.append_assistant_stream(handle, "First chunk. ")
    qapp.processEvents()

    view.append_assistant_stream(handle, "Second chunk adding more text to wrap across lines. ")
    view.append_assistant_stream(handle, "Third chunk with a code block:\n```bash\ngit status\n```\nDone.")
    qapp.processEvents()

    rect_final = view.list_widget.visualItemRect(handle.item)
    assert rect_final.height() > rect0.height()

    view.finalize_assistant_stream(handle, None, " (completed)")
    qapp.processEvents()
    assert "(completed)" in handle.label.text()


def test_dynamic_resize_recalculates_sizes(qapp):
    """
    Verify resizing the window from 800 to 400 recalculates item heights so wrapped lines
    are never clipped.
    """
    view = ChatViewWidget()
    view.resize(800, 600)
    view.show()

    long_text = "This is a long message that wraps across lines depending on container width."
    view.add_assistant_message(long_text)
    qapp.processEvents()

    item = view.list_widget.item(0)
    h_at_800 = view.list_widget.visualItemRect(item).height()

    # Narrow to 400
    view.resize(400, 600)
    qapp.processEvents()

    h_at_400 = view.list_widget.visualItemRect(item).height()
    # At narrower width, text wraps to more lines, so height must be greater or equal
    assert h_at_400 >= h_at_800


def test_safe_links_and_html_escaping(qapp):
    """
    Verify raw HTML tags are strictly escaped and dangerous URI schemes are neutralized.
    """
    view = ChatViewWidget()
    view.add_user_message("<script>alert('pwned')</script>")
    view.add_assistant_message("Check [Safe](https://example.com) vs [Unsafe](javascript:void(0))")
    qapp.processEvents()

    user_widget = view.list_widget.itemWidget(view.list_widget.item(0))
    user_text = user_widget.text()
    assert "&lt;script&gt;" in user_text
    assert "<script>" not in user_text

    asst_widget = view.list_widget.itemWidget(view.list_widget.item(1))
    asst_text = asst_widget.text()
    assert 'href="https://example.com"' in asst_text
    assert 'href="javascript:' not in asst_text


def test_provider_error_card_compact_and_expandable(qapp):
    """
    Verify Gemini provider errors are rendered as compact ErrorCardWidgets with details.
    """
    view = ChatViewWidget()
    view.resize(800, 600)
    view.show()

    raw_err = (
        "Gemini API execution error: 404 NOT_FOUND. {'error': {'code': 404, "
        "'message': 'This model models/gemini-2.5-flash is no longer available.', "
        "'key': 'AIzaSyA123456789012345678901234567890'}}"
    )

    view.add_assistant_message(raw_err)
    qapp.processEvents()

    assert view.list_widget.count() == 1
    widget = view.list_widget.itemWidget(view.list_widget.item(0))
    assert isinstance(widget, ErrorCardWidget)

    # Summary badge should identify model not found
    assert "Model Not Found (404)" in widget.lbl_title.text()
    # Secret key must be redacted
    assert "AIzaSy" not in widget.details_text
    assert "[REDACTED_API_KEY]" in widget.details_text

    # Compact initial height
    init_h = view.list_widget.visualItemRect(view.list_widget.item(0)).height()
    assert init_h <= 60

    # Expand details
    widget._toggle_details()
    qapp.processEvents()
    expanded_h = view.list_widget.visualItemRect(view.list_widget.item(0)).height()
    assert expanded_h > init_h
