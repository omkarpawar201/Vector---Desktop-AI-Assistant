"""
Hero Banner Widget for Vector Desktop AI Assistant.
Renders the landscape mountain banner artwork, gradient Vector brand title,
subtitle, and key system promises ("Fast. Local. Private. Powerful.").
Supports full panorama mode, compact header mode, and collapsible state.
"""

from __future__ import annotations
import os
from typing import Optional

from PySide6.QtCore import QPointF, QRectF, QSize, Qt
from PySide6.QtGui import (
    QColor,
    QFont,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import QFrame, QWidget

from app.ui.theme import BORDER_SUBTLE


class HeroBannerWidget(QFrame):
    """
    Center hero header banner displaying cyber mountain panorama,
    gradient branding, and product tagline.
    Supports compact and collapsible modes.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("heroBanner")
        self.setFixedHeight(140)
        self.setMinimumWidth(360)
        self._is_compact = False

        # Path to local banner asset
        assets_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")
        self._asset_path = os.path.join(assets_dir, "hero_banner.jpg")
        self._pixmap: Optional[QPixmap] = None
        self._load_pixmap()

    def _load_pixmap(self) -> None:
        if os.path.exists(self._asset_path):
            self._pixmap = QPixmap(self._asset_path)

    def sizeHint(self) -> QSize:
        h = 56 if self._is_compact else 140
        return QSize(600, h)

    def is_compact(self) -> bool:
        return self._is_compact

    def set_compact(self, compact: bool) -> None:
        """Toggle between full 140px banner and compact 56px mode."""
        self._is_compact = bool(compact)
        if self._is_compact:
            self.setFixedHeight(56)
        else:
            self.setFixedHeight(140)
        self.updateGeometry()
        self.update()

    def collapse(self) -> None:
        """Collapse or hide the hero banner to maximize message viewport."""
        self.hide()

    def expand(self) -> None:
        """Restore or show the hero banner."""
        self.show()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)

        w = float(self.width())
        h = float(self.height())
        rect = QRectF(0, 0, w, h)
        radius = 16.0

        # Rounded clip path for the entire banner
        clip_path = QPainterPath()
        clip_path.addRoundedRect(rect, radius, radius)
        painter.setClipPath(clip_path)

        # 1. Background Image or Gradient Fallback
        if self._pixmap and not self._pixmap.isNull():
            scaled = self._pixmap.scaled(
                int(w),
                int(h),
                Qt.KeepAspectRatioByExpanding,
                Qt.SmoothTransformation,
            )
            sx = max(0, (scaled.width() - int(w)) // 2)
            sy = max(0, (scaled.height() - int(h)) // 2)
            painter.drawPixmap(0, 0, scaled, sx, sy, int(w), int(h))
        else:
            fallback_grad = QLinearGradient(0, 0, w, h)
            fallback_grad.setColorAt(0.0, QColor("#090b12"))
            fallback_grad.setColorAt(0.5, QColor("#141829"))
            fallback_grad.setColorAt(1.0, QColor("#090b12"))
            painter.fillRect(rect, fallback_grad)

        # 2. Semi-translucent dark vignette overlay for readability
        overlay_grad = QLinearGradient(0, 0, 0, h)
        overlay_grad.setColorAt(0.0, QColor(11, 13, 20, 140))
        overlay_grad.setColorAt(0.5, QColor(11, 13, 20, 175))
        overlay_grad.setColorAt(1.0, QColor(11, 13, 20, 225))
        painter.fillRect(rect, overlay_grad)

        # 3. Outer border
        painter.setClipping(False)
        border_pen = QPen(QColor(56, 189, 248, 60), 1.0)
        painter.setPen(border_pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawRoundedRect(rect.adjusted(0.5, 0.5, -0.5, -0.5), radius, radius)

        # 4. Typography
        if self._is_compact:
            # Compact mode: Single horizontal line layout
            title_grad = QLinearGradient(20, 0, 120, 0)
            title_grad.setColorAt(0.0, QColor("#38bdf8"))
            title_grad.setColorAt(1.0, QColor("#c084fc"))
            painter.setPen(QPen(title_grad, 1))

            font_title = QFont("Segoe UI Variable", 18, QFont.Bold)
            painter.setFont(font_title)
            painter.drawText(QRectF(20, 12, 100, 32), Qt.AlignVCenter | Qt.AlignLeft, "Vector")

            painter.setPen(QColor("#94a3b8"))
            font_sub = QFont("Segoe UI Variable", 11, QFont.Normal)
            painter.setFont(font_sub)
            painter.drawText(
                QRectF(125, 14, w - 145, 30),
                Qt.AlignVCenter | Qt.AlignLeft,
                "— Desktop AI Assistant  ·  Fast. Local. Private. Powerful.",
            )
        else:
            # Full Panorama mode
            title_grad = QLinearGradient(w / 2.0 - 60, 0, w / 2.0 + 60, 0)
            title_grad.setColorAt(0.0, QColor("#38bdf8"))
            title_grad.setColorAt(0.6, QColor("#818cf8"))
            title_grad.setColorAt(1.0, QColor("#c084fc"))

            pen_title = QPen(title_grad, 1)
            painter.setPen(pen_title)
            font_title = QFont("Segoe UI Variable", 24, QFont.Bold)
            font_title.setLetterSpacing(QFont.AbsoluteSpacing, 0.5)
            painter.setFont(font_title)
            title_rect = QRectF(0, 24, w, 34)
            painter.drawText(title_rect, Qt.AlignCenter, "Vector")

            painter.setPen(QColor("#e2e8f0"))
            font_sub = QFont("Segoe UI Variable", 12, QFont.DemiBold)
            painter.setFont(font_sub)
            sub_rect = QRectF(0, 64, w, 22)
            painter.drawText(sub_rect, Qt.AlignCenter, "Your desktop AI assistant")

            painter.setPen(QColor("#94a3b8"))
            font_tag = QFont("Segoe UI Variable", 10, QFont.Normal)
            font_tag.setLetterSpacing(QFont.AbsoluteSpacing, 0.3)
            painter.setFont(font_tag)
            tag_rect = QRectF(0, 90, w, 20)
            painter.drawText(tag_rect, Qt.AlignCenter, "Fast. Local. Private. Powerful.")
