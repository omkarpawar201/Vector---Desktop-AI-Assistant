"""
Navigation & Quick Actions Sidebar for Vector Desktop AI Assistant.
Left column featuring Vector brand header, primary navigation buttons,
and the Quick Actions shortcut stack.
"""

from __future__ import annotations
from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from app.ui.theme import (
    BG_PANEL,
    BORDER_SUBTLE,
    NAV_BUTTON_STYLE,
    QUICK_ACTION_PILL_STYLE,
    SLIM_SCROLLBAR_STYLE,
    TEXT_MUTED,
    TEXT_PRIMARY,
)
from app.ui.widgets.avatars import AssistantAvatar


class NavSidebarWidget(QFrame):
    """
    Left Navigation Sidebar providing category navigation (Chat, Actions, System Info,
    Media, Apps & Files, Settings) and 1-click Quick Action shortcuts.
    """

    chat_clicked = Signal()
    conversations_clicked = Signal()
    actions_clicked = Signal()
    system_info_clicked = Signal()
    media_clicked = Signal()
    apps_clicked = Signal()
    diagnostics_clicked = Signal()
    settings_clicked = Signal()
    quick_action_triggered = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("navSidebar")
        self.setFixedWidth(240)
        self.setStyleSheet(f"""
            QFrame#navSidebar {{
                background-color: {BG_PANEL};
                border-right: 1px solid {BORDER_SUBTLE};
            }}
            QLabel#brandTitle {{
                color: {TEXT_PRIMARY};
                font-family: 'Segoe UI Variable', 'Segoe UI', sans-serif;
                font-size: 17px;
                font-weight: 700;
                letter-spacing: 0.4px;
            }}
            QLabel#brandSub {{
                color: {TEXT_MUTED};
                font-family: 'Segoe UI Variable', 'Segoe UI', sans-serif;
                font-size: 11px;
            }}
            QLabel#sectionHeader {{
                color: {TEXT_MUTED};
                font-family: 'Segoe UI Variable', 'Segoe UI', sans-serif;
                font-size: 10.5px;
                font-weight: 700;
                letter-spacing: 0.6px;
                padding-top: 8px;
                padding-bottom: 2px;
            }}
            {NAV_BUTTON_STYLE}
            {QUICK_ACTION_PILL_STYLE}
            {SLIM_SCROLLBAR_STYLE}
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 14, 12, 14)
        layout.setSpacing(6)

        # 1. Top Brand Header
        header = QHBoxLayout()
        header.setContentsMargins(4, 2, 4, 8)
        header.setSpacing(10)

        self.logo_avatar = AssistantAvatar(size=40, parent=self)
        col_brand = QVBoxLayout()
        col_brand.setContentsMargins(0, 0, 0, 0)
        col_brand.setSpacing(1)

        lbl_brand = QLabel("Vector", self)
        lbl_brand.setObjectName("brandTitle")
        lbl_sub = QLabel("Desktop AI Assistant", self)
        lbl_sub.setObjectName("brandSub")

        col_brand.addWidget(lbl_brand)
        col_brand.addWidget(lbl_sub)

        header.addWidget(self.logo_avatar)
        header.addLayout(col_brand)
        header.addStretch()
        layout.addLayout(header)

        # 2. Primary Navigation Menu
        self.nav_buttons: list[QPushButton] = []

        self.btn_chat = self._create_nav_btn("💬  Chat", is_active=True)
        self.btn_chat.clicked.connect(self._on_chat_selected)
        layout.addWidget(self.btn_chat)

        self.btn_convs = self._create_nav_btn("📂  Conversations")
        self.btn_convs.clicked.connect(self.conversations_clicked.emit)
        layout.addWidget(self.btn_convs)

        self.btn_actions = self._create_nav_btn("📣  Actions")
        self.btn_actions.clicked.connect(self.actions_clicked.emit)
        layout.addWidget(self.btn_actions)

        self.btn_health = self._create_nav_btn("🖥  System Info")
        self.btn_health.clicked.connect(self.system_info_clicked.emit)
        layout.addWidget(self.btn_health)

        self.btn_media = self._create_nav_btn("🎵  Media Control")
        self.btn_media.clicked.connect(self.media_clicked.emit)
        layout.addWidget(self.btn_media)

        self.btn_apps = self._create_nav_btn("📁  Apps & Files")
        self.btn_apps.clicked.connect(self.apps_clicked.emit)
        layout.addWidget(self.btn_apps)

        self.btn_settings = self._create_nav_btn("⚙  Settings")
        self.btn_settings.clicked.connect(self.settings_clicked.emit)
        layout.addWidget(self.btn_settings)

        # 3. Quick Actions Section
        lbl_qa = QLabel("QUICK ACTIONS", self)
        lbl_qa.setObjectName("sectionHeader")
        layout.addWidget(lbl_qa)

        self.quick_action_buttons: list[QPushButton] = []
        self.btn_qa_mute = self._add_quick_action(layout, "🔊  Mute Audio", "mute my audio")
        self.btn_qa_volume = self._add_quick_action(layout, "🔊  Set Volume", "set volume to 50")
        self.btn_qa_open_app = self._add_quick_action(layout, "🎛  Open App", "open chrome")
        self.btn_qa_search = self._add_quick_action(layout, "🔍  Search Files", "search files")
        self.btn_qa_stats = self._add_quick_action(layout, "📊  System Stats", "get system stats")
        self.btn_qa_screenshot = self._add_quick_action(layout, "📷  Take Screenshot", "take screenshot")

        layout.addStretch()

    def _create_nav_btn(self, text: str, is_active: bool = False) -> QPushButton:
        btn = QPushButton(text, self)
        btn.setCheckable(True)
        btn.setChecked(is_active)
        if is_active:
            btn.setObjectName("activeNav")
        self.nav_buttons.append(btn)
        return btn

    def _on_chat_selected(self) -> None:
        self.set_active_nav("Chat")
        self.chat_clicked.emit()

    def set_active_nav(self, nav_name: str) -> None:
        """Sets active indicator on the matching navigation button."""
        for btn in self.nav_buttons:
            if nav_name.lower() in btn.text().lower():
                btn.setChecked(True)
                btn.setObjectName("activeNav")
            else:
                btn.setChecked(False)
                btn.setObjectName("")
            btn.setStyle(btn.style())  # Force style re-eval

    def _add_quick_action(self, layout: QVBoxLayout, label: str, command: str) -> QPushButton:
        btn = QPushButton(label, self)
        btn.clicked.connect(lambda: self.quick_action_triggered.emit(command))
        self.quick_action_buttons.append(btn)
        layout.addWidget(btn)
        return btn
