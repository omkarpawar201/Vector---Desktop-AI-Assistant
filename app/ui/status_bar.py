"""
Status bar widget for Vector Desktop AI Assistant.
Displays real-time system monitoring metrics (CPU, RAM, Disk, Volume).
"""

from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import QHBoxLayout, QLabel, QFrame, QWidget
import psutil
from app.tools.system.volume import GetVolumeTool


class StatusBarWidget(QFrame):
    """
    Real-time system health and status bar.
    """

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet("""
            StatusBarWidget {
                background-color: #1e1e2e;
                border-top: 1px solid #313244;
                color: #cdd6f4;
                font-family: 'Segoe UI', sans-serif;
                font-size: 12px;
                padding: 4px 12px;
            }
            QLabel {
                color: #a6adc8;
                padding: 0 8px;
            }
            QLabel#brand {
                color: #89b4fa;
                font-weight: bold;
            }
        """)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)

        self.lbl_brand = QLabel("VECTOR 1.0", self)
        self.lbl_brand.setObjectName("brand")

        self.lbl_cpu = QLabel("CPU: --%", self)
        self.lbl_ram = QLabel("RAM: --%", self)
        self.lbl_disk = QLabel("Disk: --%", self)
        self.lbl_vol = QLabel("Vol: --%", self)

        layout.addWidget(self.lbl_brand)
        layout.addStretch()
        layout.addWidget(self.lbl_cpu)
        layout.addWidget(self.lbl_ram)
        layout.addWidget(self.lbl_disk)
        layout.addWidget(self.lbl_vol)

        # Refresh timer every 2 seconds
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh_metrics)
        self.timer.start(2000)

        # Initial refresh
        self.refresh_metrics()

    def refresh_metrics(self) -> None:
        """Fetch live system metrics."""
        try:
            cpu = psutil.cpu_percent(interval=None)
            ram = psutil.virtual_memory().percent
            disk = psutil.disk_usage("/").percent

            self.lbl_cpu.setText(f"CPU: {cpu:.1f}%")
            self.lbl_ram.setText(f"RAM: {ram:.1f}%")
            self.lbl_disk.setText(f"Disk: {disk:.1f}%")

            # Try to get volume
            try:
                get_vol = GetVolumeTool()
                res = get_vol.execute()
                if res.success and res.data:
                    vol_val = res.data.get("level", 0)
                    is_muted = res.data.get("is_muted", False)
                    status = "Muted" if is_muted else f"{vol_val}%"
                    self.lbl_vol.setText(f"Vol: {status}")
            except Exception:
                self.lbl_vol.setText("Vol: N/A")

        except Exception:
            pass
