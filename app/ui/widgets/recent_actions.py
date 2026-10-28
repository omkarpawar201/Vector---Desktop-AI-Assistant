"""
Recent Actions Activity Feed Widget for Vector Desktop AI Assistant.
Displays a real-time list of executed tool actions with icons, timestamps,
and SUCCESS / FAILED status badges.
"""

from __future__ import annotations
from datetime import datetime
from typing import Optional

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.ui.theme import (
    BG_CARD,
    BG_PANEL,
    BORDER_SUBTLE,
    SLIM_SCROLLBAR_STYLE,
    TEXT_MUTED,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
)
from app.ui.widgets.avatars import StatusPillBadge

# Friendly display names and icons for tools
TOOL_META = {
    "mute": ("Mute Audio", "🔊"),
    "unmute": ("Unmute Audio", "🔊"),
    "set_volume": ("Set Volume", "🔊"),
    "get_volume": ("Get Volume", "🔊"),
    "get_cpu_usage": ("Get CPU Usage", "⏱"),
    "get_system_stats": ("System Stats", "📊"),
    "get_memory_usage": ("Memory Usage", "📊"),
    "get_disk_usage": ("Disk Usage", "💾"),
    "get_battery_status": ("Battery Status", "🔋"),
    "launch_app": ("Launch App", "🎛"),
    "close_app": ("Close App", "🎛"),
    "get_running_apps": ("Running Apps", "🎛"),
    "open_chrome": ("Open Chrome", "🌐"),
    "search_files": ("Search Files", "🔍"),
    "open_file": ("Open File", "📄"),
    "open_folder": ("Open Folder", "📁"),
    "take_screenshot": ("Take Screenshot", "📷"),
    "media_play": ("Media Play", "▶"),
    "media_pause": ("Media Pause", "⏸"),
    "media_next": ("Media Next", "⏭"),
    "media_previous": ("Media Prev", "⏮"),
    "lock_pc": ("Lock PC", "🔒"),
    "sleep_pc": ("Sleep PC", "💤"),
    "execute_terminal_command": ("Terminal Command", "💻"),
}


class RecentActionItemWidget(QWidget):
    """Single action entry in the activity feed."""

    def __init__(
        self,
        tool_name: str,
        timestamp: str,
        success: bool = True,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        title, icon = TOOL_META.get(tool_name.lower(), (tool_name.replace("_", " ").title(), "⚙"))

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(10)

        # Icon
        lbl_icon = QLabel(icon, self)
        lbl_icon.setStyleSheet("font-size: 15px; color: #38bdf8;")
        lbl_icon.setFixedWidth(22)

        # Title & Timestamp stacked
        col = QVBoxLayout()
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(2)

        lbl_title = QLabel(title, self)
        lbl_title.setStyleSheet(f"""
            color: {TEXT_PRIMARY};
            font-family: 'Segoe UI Variable', 'Segoe UI', sans-serif;
            font-size: 12px;
            font-weight: 600;
        """)

        lbl_time = QLabel(timestamp, self)
        lbl_time.setStyleSheet(f"""
            color: {TEXT_MUTED};
            font-family: 'Segoe UI Variable', 'Segoe UI', sans-serif;
            font-size: 10.5px;
        """)

        col.addWidget(lbl_title)
        col.addWidget(lbl_time)

        # Status badge
        badge = StatusPillBadge(
            "Success" if success else "Failed",
            badge_type="success" if success else "failed",
            parent=self,
        )

        layout.addWidget(lbl_icon)
        layout.addLayout(col, stretch=1)
        layout.addWidget(badge)


class RecentActionsWidget(QFrame):
    """
    Recent Actions Activity Feed Panel.
    Displays live execution events from the assistant.
    """

    clear_requested = Signal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("recentActionsCard")
        self.setStyleSheet(f"""
            QFrame#recentActionsCard {{
                background-color: {BG_CARD};
                border: 1px solid {BORDER_SUBTLE};
                border-radius: 16px;
            }}
            QLabel#recentTitle {{
                color: {TEXT_PRIMARY};
                font-family: 'Segoe UI Variable', 'Segoe UI', sans-serif;
                font-weight: 700;
                font-size: 13.5px;
            }}
            QPushButton#btnClear {{
                background-color: transparent;
                color: {TEXT_MUTED};
                border: 1px solid {BORDER_SUBTLE};
                border-radius: 6px;
                padding: 3px 10px;
                font-size: 11px;
                font-weight: 500;
            }}
            QPushButton#btnClear:hover {{
                background-color: #1e2235;
                color: {TEXT_PRIMARY};
                border-color: #38bdf8;
            }}
            QListWidget {{
                background-color: transparent;
                border: none;
                padding: 2px 0px;
            }}
            QListWidget::item {{
                background-color: #121521;
                border: 1px solid #1a1e2e;
                border-radius: 10px;
                margin: 2px 2px;
            }}
            QListWidget::item:hover {{
                background-color: #181c2d;
                border-color: #2b324d;
            }}
            {SLIM_SCROLLBAR_STYLE}
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(8)

        # Header bar: Icon + Title + [Clear]
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(8)

        lbl_icon = QLabel("🕑", self)
        lbl_icon.setStyleSheet("font-size: 14px; color: #818cf8;")
        lbl_title = QLabel("Recent Actions", self)
        lbl_title.setObjectName("recentTitle")

        btn_clear = QPushButton("Clear", self)
        btn_clear.setObjectName("btnClear")
        btn_clear.clicked.connect(self.clear_actions)

        header.addWidget(lbl_icon)
        header.addWidget(lbl_title)
        header.addStretch()
        header.addWidget(btn_clear)
        layout.addLayout(header)

        # Scrollable list
        self.list_widget = QListWidget(self)
        self.list_widget.setVerticalScrollMode(QListWidget.ScrollPerPixel)
        self.list_widget.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        layout.addWidget(self.list_widget, stretch=1)

    def add_action(
        self,
        tool_name: str,
        message: str = "",
        success: bool = True,
        timestamp: Optional[str] = None,
    ) -> None:
        """Adds a new executed action card to the top of the feed."""
        if not timestamp:
            timestamp = datetime.now().strftime("%I:%M %p")

        widget = RecentActionItemWidget(
            tool_name=tool_name,
            timestamp=timestamp,
            success=success,
            parent=self.list_widget,
        )

        item = QListWidgetItem()
        item.setSizeHint(QSize(240, 48))
        self.list_widget.insertItem(0, item)
        self.list_widget.setItemWidget(item, widget)

        # Keep max 50 items
        while self.list_widget.count() > 50:
            self.list_widget.takeItem(self.list_widget.count() - 1)

    def clear_actions(self) -> None:
        """Clears all actions from the feed."""
        self.list_widget.clear()
        self.clear_requested.emit()
