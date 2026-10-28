"""
Vector Custom UI Widgets Package.
Exposes CircularGaugeWidget, SystemStatusWidget, HeroBannerWidget, AssistantAvatar,
UserAvatar, StatusPillBadge, NavSidebarWidget, RecentActionsWidget, and QuickChipsWidget.
"""

from app.ui.widgets.avatars import AssistantAvatar, StatusPillBadge, UserAvatar
from app.ui.widgets.circular_gauge import CircularGaugeWidget, SystemStatusWidget
from app.ui.widgets.hero_banner import HeroBannerWidget
from app.ui.widgets.nav_sidebar import NavSidebarWidget
from app.ui.widgets.quick_chips import QuickChipsWidget
from app.ui.widgets.recent_actions import RecentActionsWidget

__all__ = [
    "AssistantAvatar",
    "UserAvatar",
    "StatusPillBadge",
    "CircularGaugeWidget",
    "SystemStatusWidget",
    "HeroBannerWidget",
    "NavSidebarWidget",
    "RecentActionsWidget",
    "QuickChipsWidget",
]
