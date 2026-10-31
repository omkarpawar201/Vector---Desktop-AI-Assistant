"""
Vector Design Tokens and Theme Module.
Defines color palettes, typography, gradients, and reusable stylesheets for the
cyberpunk / Windows 11 Fluent dark desktop dashboard.
"""

from __future__ import annotations
from PySide6.QtGui import QColor, QFont, QLinearGradient

# --------------------------------------------------------------------------
# Color Palette
# --------------------------------------------------------------------------
BG_APP = "#0b0d14"          # Deep obsidian backdrop
BG_PANEL = "#10131d"        # Primary card and sidebar surface
BG_CARD = "#161925"         # Inner message bubbles and cards
BG_ELEVATED = "#1e2235"     # Elevated hover and active surfaces
BG_INPUT = "#0e111a"        # Deep composer dock input background

BORDER_SUBTLE = "#1e2235"   # Subtle separators and container borders
BORDER_ACTIVE = "#2b324d"   # Active borders
BORDER_GLOW_CYAN = "#38bdf8"# Glowing cyan accent border
BORDER_GLOW_PURPLE = "#c084fc"

# Accents
ACCENT_CYAN = "#38bdf8"     # Neon cyan
ACCENT_PURPLE = "#c084fc"   # Neon purple / magenta
ACCENT_BLUE = "#2563eb"     # Royal blue
ACCENT_GREEN = "#10b981"    # Emerald / success green
ACCENT_GREEN_BRIGHT = "#34d399"
ACCENT_ORANGE = "#fb923c"   # Amber / disk gauge
ACCENT_PINK = "#f43f5e"     # Coral / volume gauge
ACCENT_RED = "#ef4444"      # Error red

# Text
TEXT_PRIMARY = "#f8fafc"    # Crisp white headers and primary labels
TEXT_SECONDARY = "#cbd5e1"  # Soft light grey body text
TEXT_MUTED = "#64748b"      # Muted timestamps and captions
TEXT_DIM = "#475569"        # Dim inactive elements

# QColors for QPainter drawing
QCOLOR_CYAN = QColor("#38bdf8")
QCOLOR_CYAN_DARK = QColor("#0284c7")
QCOLOR_PURPLE = QColor("#c084fc")
QCOLOR_PURPLE_DARK = QColor("#7c3aed")
QCOLOR_ORANGE = QColor("#fb923c")
QCOLOR_ORANGE_DARK = QColor("#ea580c")
QCOLOR_PINK = QColor("#f43f5e")
QCOLOR_PINK_DARK = QColor("#be123c")
QCOLOR_GREEN = QColor("#10b981")
QCOLOR_TRACK = QColor("#1e2235")

# --------------------------------------------------------------------------
# Reusable Stylesheets
# --------------------------------------------------------------------------
SLIM_SCROLLBAR_STYLE = """
QScrollBar:vertical {
    background: transparent;
    width: 6px;
    margin: 0px;
    border-radius: 3px;
}
QScrollBar::handle:vertical {
    background: #2b324d;
    min-height: 24px;
    border-radius: 3px;
}
QScrollBar::handle:vertical:hover {
    background: #38bdf8;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
    background: none;
}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
    background: none;
}
"""

CARD_PANEL_STYLE = """
QFrame#cardPanel {
    background-color: #10131d;
    border: 1px solid #1e2235;
    border-radius: 16px;
}
"""

NAV_BUTTON_STYLE = """
QPushButton {
    background-color: transparent;
    color: #94a3b8;
    border: none;
    border-radius: 12px;
    padding: 10px 14px;
    font-family: 'Segoe UI Variable', 'Segoe UI', -apple-system, sans-serif;
    font-size: 13px;
    font-weight: 500;
    text-align: left;
}
QPushButton:hover {
    background-color: #161925;
    color: #f8fafc;
}
QPushButton:checked, QPushButton#activeNav {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #1d4ed8, stop:1 #6d28d9);
    color: #ffffff;
    font-weight: 600;
    border: 1px solid rgba(56, 189, 248, 0.4);
}
"""

QUICK_ACTION_PILL_STYLE = """
QPushButton {
    background-color: #141724;
    color: #cbd5e1;
    border: 1px solid #1e2235;
    border-radius: 10px;
    padding: 8px 12px;
    font-family: 'Segoe UI Variable', 'Segoe UI', -apple-system, sans-serif;
    font-size: 12px;
    font-weight: 500;
    text-align: left;
}
QPushButton:hover {
    background-color: #1e2235;
    border-color: #38bdf8;
    color: #ffffff;
}
QPushButton:pressed {
    background-color: #0b0d14;
}
"""

CHIP_BUTTON_STYLE = """
QPushButton {
    background-color: #141724;
    color: #cbd5e1;
    border: 1px solid #1e2235;
    border-radius: 14px;
    padding: 6px 14px;
    font-family: 'Segoe UI Variable', 'Segoe UI', -apple-system, sans-serif;
    font-size: 11.5px;
    font-weight: 500;
}
QPushButton:hover {
    background-color: #1e2235;
    border-color: #38bdf8;
    color: #ffffff;
}
"""

from datetime import datetime
from typing import Optional


def format_timestamp(ts: Optional[str] = None) -> str:
    """
    Format a timestamp into a clean, human-readable time like '7:34 PM'.
    Gracefully handles ISO 8601 strings, pre-formatted strings, or None.
    """
    if not ts:
        now = datetime.now()
        s = now.strftime("%I:%M %p")
        return s.lstrip("0") if s.startswith("0") else s

    ts_str = str(ts).strip()
    if "AM" in ts_str or "PM" in ts_str:
        return ts_str

    try:
        dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        local_dt = dt.astimezone() if dt.tzinfo else dt
        s = local_dt.strftime("%I:%M %p")
        return s.lstrip("0") if s.startswith("0") else s
    except Exception:
        return ts_str

