"""
Status bar widget for Vector Desktop AI Assistant.
Displays real-time system monitoring metrics (CPU, RAM, Disk, Volume).
"""

from PySide6.QtCore import QTimer, Qt, Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QFrame, QWidget
import psutil
from app.config import APP_NAME, APP_VERSION
from app.tools.system.volume import GetVolumeTool


class StatusBarWidget(QFrame):
    """
    Real-time system health and status bar with telemetry emission.
    """

    metrics_updated = Signal(float, float, float, float, bool)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet("""
            StatusBarWidget {
                background-color: #0b0d14;
                border-top: 1px solid #1e2235;
                color: #cdd6f4;
                font-family: 'Segoe UI Variable', 'Segoe UI', -apple-system, sans-serif;
                font-size: 11.5px;
                padding: 4px 16px;
            }
            QLabel {
                color: #64748b;
                padding: 0 4px;
            }
            QLabel#brand {
                color: #38bdf8;
                font-weight: 700;
                letter-spacing: 0.5px;
            }
            QLabel#readyBadge {
                color: #34d399;
                font-weight: 600;
            }
            QLabel#securityBadge {
                color: #94a3b8;
                font-size: 11px;
            }
        """)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 3, 12, 3)
        layout.setSpacing(12)

        # Left: Brand + Ready badge
        self.lbl_brand = QLabel(f"{APP_NAME.upper()} {APP_VERSION}", self)
        self.lbl_brand.setObjectName("brand")

        self.lbl_ready = QLabel("● Ready", self)
        self.lbl_ready.setObjectName("readyBadge")

        # Telemetry labels (retained for test assertions)
        self.lbl_cpu = QLabel("CPU: --%", self)
        self.lbl_ram = QLabel("RAM: --%", self)
        self.lbl_disk = QLabel("Disk: --%", self)
        self.lbl_vol = QLabel("Vol: --%", self)

        # Right: Security badges ("🔒 Local   Private   Fast")
        self.lbl_security = QLabel("🔒 Local   ·   Private   ·   Fast", self)
        self.lbl_security.setObjectName("securityBadge")

        layout.addWidget(self.lbl_brand)
        layout.addWidget(self.lbl_ready)
        layout.addStretch()
        layout.addWidget(self.lbl_cpu)
        layout.addWidget(self.lbl_ram)
        layout.addWidget(self.lbl_disk)
        layout.addWidget(self.lbl_vol)
        layout.addStretch()
        layout.addWidget(self.lbl_security)

        # Refresh timer every 2 seconds
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh_metrics)
        self.timer.start(2000)

        # Initial refresh
        self.refresh_metrics()

    def refresh_metrics(self) -> None:
        """Fetch live system metrics and emit metrics_updated signal."""
        try:
            cpu = float(psutil.cpu_percent(interval=None))
            ram = float(psutil.virtual_memory().percent)
            disk = float(psutil.disk_usage("/").percent)

            vol_val = 0.0
            is_muted = False
            try:
                get_vol = GetVolumeTool()
                res = get_vol.execute()
                if res.success and res.data:
                    vol_val = float(res.data.get("level", 0))
                    is_muted = bool(res.data.get("is_muted", False))
                    status = "Muted" if is_muted else f"{int(vol_val)}%"
                    self.lbl_vol.setText(f"Vol: {status}")
            except Exception:
                self.lbl_vol.setText("Vol: N/A")

            self.lbl_cpu.setText(f"CPU: {cpu:.1f}%")
            self.lbl_ram.setText(f"RAM: {ram:.1f}%")
            self.lbl_disk.setText(f"Disk: {disk:.1f}%")

            # Emit live metrics to circular gauges in SystemStatusWidget
            self.metrics_updated.emit(cpu, ram, disk, vol_val, is_muted)
        except Exception:
            pass
