"""
Circular progress ring gauge widget and SystemStatusWidget panel.
Renders smooth antialiased radial progress meters for CPU, RAM, Disk, and Volume.
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
from PySide6.QtWidgets import QFrame, QGridLayout, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from app.ui.theme import (
    ACCENT_GREEN_BRIGHT,
    BG_CARD,
    BORDER_SUBTLE,
    QCOLOR_CYAN,
    QCOLOR_CYAN_DARK,
    QCOLOR_ORANGE,
    QCOLOR_ORANGE_DARK,
    QCOLOR_PINK,
    QCOLOR_PINK_DARK,
    QCOLOR_PURPLE,
    QCOLOR_PURPLE_DARK,
    QCOLOR_TRACK,
    TEXT_MUTED,
    TEXT_PRIMARY,
)


class CircularGaugeWidget(QWidget):
    """
    Antialiased circular progress ring gauge.
    Renders a background track ring, progress arc, centered percentage or icon,
    and category label.
    """

    def __init__(
        self,
        label: str = "CPU",
        initial_value: float = 0.0,
        color_start: QColor = QCOLOR_CYAN,
        color_end: QColor = QCOLOR_CYAN_DARK,
        line_width: int = 8,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.label = label
        self._value = max(0.0, min(100.0, float(initial_value)))
        self.color_start = color_start
        self.color_end = color_end
        self.track_color = QCOLOR_TRACK
        self.line_width = line_width
        self._is_muted = False
        self.setMinimumSize(84, 84)

    @property
    def value(self) -> float:
        return self._value

    def set_value(self, val: float) -> None:
        clamped = max(0.0, min(100.0, float(val)))
        if self._value != clamped:
            self._value = clamped
            self.update()

    @property
    def is_muted(self) -> bool:
        return self._is_muted

    def set_muted(self, muted: bool) -> None:
        if self._is_muted != muted:
            self._is_muted = bool(muted)
            self.update()

    def sizeHint(self) -> QSize:
        return QSize(96, 96)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w = self.width()
        h = self.height()
        side = min(w, h)
        margin = self.line_width + 4
        gauge_size = side - margin * 2

        if gauge_size <= 0:
            return

        cx = w / 2.0
        cy = h / 2.0
        rect = QRectF(cx - gauge_size / 2.0, cy - gauge_size / 2.0, gauge_size, gauge_size)

        # 1. Background Track Ring
        pen_track = QPen(self.track_color, self.line_width, Qt.SolidLine, Qt.RoundCap)
        painter.setPen(pen_track)
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(rect)

        # 2. Progress Arc
        if self._value > 0 and not self._is_muted:
            # Gradient along the arc
            grad = QLinearGradient(rect.topLeft(), rect.bottomRight())
            grad.setColorAt(0.0, self.color_start)
            grad.setColorAt(1.0, self.color_end)

            pen_progress = QPen(grad, self.line_width, Qt.SolidLine, Qt.RoundCap)
            painter.setPen(pen_progress)

            # Qt drawArc uses 1/16th of a degree. 90*16 is 12 o'clock.
            start_angle = 90 * 16
            span_angle = -int((self._value / 100.0) * 360 * 16)
            painter.drawArc(rect, start_angle, span_angle)

        # 3. Center Text
        # Value string or Muted indicator
        painter.setPen(QColor(TEXT_PRIMARY if not self._is_muted else "#f43f5e"))
        font_val = QFont("Segoe UI Variable", 11, QFont.Bold)
        painter.setFont(font_val)

        if self._is_muted:
            val_text = "Muted"
            sub_rect = QRectF(cx - 36, cy - 14, 72, 16)
            painter.drawText(sub_rect, Qt.AlignCenter, val_text)
        else:
            val_text = f"{self._value:.1f}%" if self._value < 100 else f"{int(self._value)}%"
            val_rect = QRectF(cx - 36, cy - 16, 72, 18)
            painter.drawText(val_rect, Qt.AlignCenter, val_text)

        # Category Label below value
        painter.setPen(QColor(TEXT_MUTED))
        font_lbl = QFont("Segoe UI Variable", 9, QFont.Normal)
        painter.setFont(font_lbl)
        lbl_rect = QRectF(cx - 36, cy + 3, 72, 14)
        painter.drawText(lbl_rect, Qt.AlignCenter, self.label)


class SystemStatusWidget(QFrame):
    """
    Card panel displaying live System Status with 4 circular ring meters:
    CPU, RAM, Disk, and Volume in a 2x2 grid.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("systemStatusCard")
        self.setStyleSheet(f"""
            QFrame#systemStatusCard {{
                background-color: {BG_CARD};
                border: 1px solid {BORDER_SUBTLE};
                border-radius: 16px;
            }}
            QLabel#statusTitle {{
                color: {TEXT_PRIMARY};
                font-family: 'Segoe UI Variable', 'Segoe UI', -apple-system, sans-serif;
                font-weight: 700;
                font-size: 13.5px;
            }}
            QLabel#pulseIcon {{
                color: {ACCENT_GREEN_BRIGHT};
                font-weight: bold;
                font-size: 14px;
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(10)

        # Header: Pulse icon + "System Status"
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(8)

        lbl_pulse = QLabel("📈", self)
        lbl_pulse.setObjectName("pulseIcon")
        lbl_title = QLabel("System Status", self)
        lbl_title.setObjectName("statusTitle")

        header.addWidget(lbl_pulse)
        header.addWidget(lbl_title)
        header.addStretch()
        layout.addLayout(header)

        # 2x2 Grid for 4 Gauges
        grid = QGridLayout()
        grid.setContentsMargins(0, 2, 0, 2)
        grid.setSpacing(6)

        self.gauge_cpu = CircularGaugeWidget(
            label="CPU",
            initial_value=0.0,
            color_start=QCOLOR_CYAN,
            color_end=QCOLOR_CYAN_DARK,
            parent=self,
        )
        self.gauge_ram = CircularGaugeWidget(
            label="RAM",
            initial_value=0.0,
            color_start=QCOLOR_PURPLE,
            color_end=QCOLOR_PURPLE_DARK,
            parent=self,
        )
        self.gauge_disk = CircularGaugeWidget(
            label="Disk",
            initial_value=0.0,
            color_start=QCOLOR_ORANGE,
            color_end=QCOLOR_ORANGE_DARK,
            parent=self,
        )
        self.gauge_vol = CircularGaugeWidget(
            label="Volume",
            initial_value=0.0,
            color_start=QCOLOR_PINK,
            color_end=QCOLOR_PINK_DARK,
            parent=self,
        )

        grid.addWidget(self.gauge_cpu, 0, 0, Qt.AlignCenter)
        grid.addWidget(self.gauge_ram, 0, 1, Qt.AlignCenter)
        grid.addWidget(self.gauge_disk, 1, 0, Qt.AlignCenter)
        grid.addWidget(self.gauge_vol, 1, 1, Qt.AlignCenter)

        layout.addLayout(grid)

    def update_metrics(
        self,
        cpu: float,
        ram: float,
        disk: float,
        volume: float,
        is_muted: bool = False,
    ) -> None:
        """Updates all 4 gauges with real-time system values."""
        self.gauge_cpu.set_value(cpu)
        self.gauge_ram.set_value(ram)
        self.gauge_disk.set_value(disk)
        self.gauge_vol.set_value(volume)
        self.gauge_vol.set_muted(is_muted)
