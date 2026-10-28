"""
Avatar widgets and status pill badges for the Vector desktop interface.
Includes the glowing 'V' assistant avatar, user profile avatar, and status badges.
"""

from __future__ import annotations
from typing import Optional

from PySide6.QtCore import QPointF, QRectF, QSize, Qt
from PySide6.QtGui import (
    QColor,
    QFont,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
)
from PySide6.QtWidgets import QLabel, QWidget

from app.ui.theme import (
    ACCENT_CYAN,
    ACCENT_GREEN_BRIGHT,
    ACCENT_PURPLE,
    BORDER_SUBTLE,
    TEXT_PRIMARY,
)


class AssistantAvatar(QWidget):
    """
    36x36 glowing circular badge featuring the stylized 'V' brand mark.
    Displayed next to assistant messages and audio/action cards.
    """

    def __init__(self, size: int = 36, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._size = size
        self.setFixedSize(size, size)

    def sizeHint(self) -> QSize:
        return QSize(self._size, self._size)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        s = float(self._size)
        rect = QRectF(1.0, 1.0, s - 2.0, s - 2.0)

        # 1. Background gradient (Deep Indigo -> Obsidian)
        bg_grad = QLinearGradient(0, 0, s, s)
        bg_grad.setColorAt(0.0, QColor("#1e1b4b"))
        bg_grad.setColorAt(1.0, QColor("#090b14"))

        painter.setPen(Qt.NoPen)
        painter.setBrush(bg_grad)
        painter.drawEllipse(rect)

        # 2. Glowing Accent Border Ring
        ring_grad = QLinearGradient(0, 0, s, s)
        ring_grad.setColorAt(0.0, QColor(ACCENT_CYAN))
        ring_grad.setColorAt(1.0, QColor(ACCENT_PURPLE))

        border_pen = QPen(ring_grad, 1.5)
        painter.setPen(border_pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(rect)

        # 3. Geometric 'V' Lettermark
        # Drawn with precision lines for a razor-sharp modern icon
        v_grad = QLinearGradient(s * 0.28, s * 0.25, s * 0.72, s * 0.75)
        v_grad.setColorAt(0.0, QColor("#38bdf8"))
        v_grad.setColorAt(1.0, QColor("#c084fc"))

        v_pen = QPen(v_grad, s * 0.12, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        painter.setPen(v_pen)

        path = QPainterPath()
        path.moveTo(s * 0.28, s * 0.30)
        path.lineTo(s * 0.50, s * 0.72)
        path.lineTo(s * 0.72, s * 0.30)
        painter.drawPath(path)


class UserAvatar(QWidget):
    """
    36x36 circular profile avatar badge displayed next to user chat bubbles.
    """

    def __init__(self, size: int = 36, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._size = size
        self.setFixedSize(size, size)

    def sizeHint(self) -> QSize:
        return QSize(self._size, self._size)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        s = float(self._size)
        rect = QRectF(1.0, 1.0, s - 2.0, s - 2.0)

        # Background
        bg_grad = QLinearGradient(0, 0, s, s)
        bg_grad.setColorAt(0.0, QColor("#1e293b"))
        bg_grad.setColorAt(1.0, QColor("#0f172a"))

        painter.setPen(Qt.NoPen)
        painter.setBrush(bg_grad)
        painter.drawEllipse(rect)

        # Border
        painter.setPen(QPen(QColor("#334155"), 1.2))
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(rect)

        # User head and body silhouette
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor("#94a3b8"))

        # Head circle
        head_radius = s * 0.17
        painter.drawEllipse(QPointF(s * 0.50, s * 0.38), head_radius, head_radius)

        # Shoulders arc
        body_path = QPainterPath()
        body_path.moveTo(s * 0.26, s * 0.76)
        body_path.quadTo(s * 0.50, s * 0.58, s * 0.74, s * 0.76)
        body_path.lineTo(s * 0.70, s * 0.82)
        body_path.lineTo(s * 0.30, s * 0.82)
        body_path.closeSubpath()
        painter.drawPath(body_path)


class StatusPillBadge(QLabel):
    """
    Compact pill badge used for tool execution status ('SUCCESS' / 'FAILED')
    and system status indicators ('● Ready', '🔒 Local').
    """

    def __init__(
        self,
        text: str = "SUCCESS",
        badge_type: str = "success",
        parent: Optional[QWidget] = None,
    ):
        super().__init__(text, parent)
        self.badge_type = badge_type
        self.setAlignment(Qt.AlignCenter)
        self._apply_style()

    def _apply_style(self) -> None:
        if self.badge_type == "success":
            bg = "rgba(16, 185, 129, 0.14)"
            fg = "#34d399"
            border = "rgba(16, 185, 129, 0.4)"
        elif self.badge_type == "failed":
            bg = "rgba(239, 68, 68, 0.14)"
            fg = "#f87171"
            border = "rgba(239, 68, 68, 0.4)"
        elif self.badge_type == "ready":
            bg = "rgba(56, 189, 248, 0.12)"
            fg = "#38bdf8"
            border = "rgba(56, 189, 248, 0.3)"
        else:  # muted / neutral
            bg = "rgba(255, 255, 255, 0.05)"
            fg = "#94a3b8"
            border = "rgba(255, 255, 255, 0.1)"

        self.setStyleSheet(f"""
            QLabel {{
                background-color: {bg};
                color: {fg};
                border: 1px solid {border};
                border-radius: 10px;
                padding: 3px 10px;
                font-family: 'Segoe UI Variable', 'Segoe UI', -apple-system, sans-serif;
                font-size: 11px;
                font-weight: 700;
                letter-spacing: 0.4px;
            }}
        """)
