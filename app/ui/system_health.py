"""
System Health Dashboard for Vector Desktop AI Assistant.

A lightweight, strictly read-only view of which subsystems are available,
unavailable, not configured, or unknown.  It reuses existing settings and
utilities and performs only cheap checks:

* file/module existence (``Path`` + ``importlib.util.find_spec``),
* one indexed read of the history database,
* a ``psutil`` resource sample.

It deliberately does **not** load any large model, make network requests, run
tools/commands, or change any runtime configuration.  A setting being present is
reported as "Configured" -- never as proof that a backend actually works --
using honest detail text such as "connectivity not tested" or "initialization
not attempted".
"""

from __future__ import annotations

import importlib.util
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Qt, Signal, Slot
from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from app.config.settings import Settings, get_settings


class HealthStatus:
    """
    The status vocabulary shown on the dashboard.

    ``AVAILABLE`` / ``UNAVAILABLE`` describe a check that actually ran;
    ``CONFIGURED`` describes a setting that is present but was not actively
    verified; ``NOT_CONFIGURED`` means the feature is off or a path is unset;
    ``UNKNOWN`` means the state could not be determined reliably.
    """

    AVAILABLE = "Available"
    UNAVAILABLE = "Unavailable"
    NOT_CONFIGURED = "Not configured"
    CONFIGURED = "Configured"
    UNKNOWN = "Unknown"


_STATUS_COLORS: Dict[str, str] = {
    HealthStatus.AVAILABLE: "#a6e3a1",
    HealthStatus.UNAVAILABLE: "#f38ba8",
    HealthStatus.NOT_CONFIGURED: "#f9e2af",
    HealthStatus.CONFIGURED: "#89b4fa",
    HealthStatus.UNKNOWN: "#9399b2",
}

HEALTH_ERROR_TEXT = "Could not collect system health. Please try again."
EMPTY_HEALTH_TEXT = "No health information available."


@dataclass
class HealthItem:
    """One dashboard row: a label, a status, and optional non-sensitive detail."""

    label: str
    status: str
    detail: str = ""


@dataclass
class HealthSection:
    """A titled group of :class:`HealthItem` rows."""

    title: str
    items: List[HealthItem] = field(default_factory=list)


def module_available(name: str) -> bool:
    """
    True when an importable module can be located, without importing it.

    ``find_spec`` does not execute the module, so this never has the side effect
    of loading a model or opening a device.
    """
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def _path_status(raw_path: Optional[str]) -> tuple:
    """Return ``(status, detail)`` for a configured file path (no loading)."""
    if not raw_path:
        return HealthStatus.NOT_CONFIGURED, "No path configured"
    path = Path(str(raw_path))
    if path.is_file():
        return HealthStatus.AVAILABLE, f"File found: {path.name}"
    return HealthStatus.UNAVAILABLE, "File not found"


def _offline_stt_files_present(model_dir: Path) -> bool:
    """True when a sherpa-onnx offline model layout is present in ``model_dir``."""
    try:
        if not model_dir.is_dir():
            return False
        files = [p.name for p in model_dir.iterdir() if p.is_file()]
    except Exception:
        return False
    has_tokens = any("tokens" in f and f.endswith(".txt") for f in files)
    has_encoder = any("encoder" in f and f.endswith(".onnx") for f in files)
    has_decoder = any("decoder" in f and f.endswith(".onnx") for f in files)
    return has_tokens and has_encoder and has_decoder


class SystemHealthCollector:
    """
    Gathers a read-only health report from existing settings and utilities.

    All inputs are injectable so the collector can be exercised deterministically
    without a real model, network, microphone, or GPU.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        history_manager: Optional[Any] = None,
        models_root: Optional[str] = None,
        resource_provider: Optional[Callable[[], Optional[Dict[str, Any]]]] = None,
    ):
        self.settings = settings or get_settings()
        self.history_manager = history_manager
        self._models_root = models_root
        self._resource_provider = resource_provider

    # ------------------------------------------------------------------
    # Sections
    # ------------------------------------------------------------------
    def collect(self) -> List[HealthSection]:
        return [
            self._ai_section(),
            self._voice_section(),
            self._storage_section(),
            self._resources_section(),
        ]

    def _ai_section(self) -> HealthSection:
        s = self.settings
        items: List[HealthItem] = []

        # Gemini configuration -- configuration only, never a network probe.
        key_present = bool(str(getattr(s, "gemini_api_key", "") or "").strip())
        if not getattr(s, "enable_gemini", False):
            items.append(HealthItem(
                "Gemini cloud fallback",
                HealthStatus.NOT_CONFIGURED,
                "Disabled in settings",
            ))
        elif key_present:
            items.append(HealthItem(
                "Gemini cloud fallback",
                HealthStatus.CONFIGURED,
                "API key present; connectivity not tested",
            ))
        else:
            items.append(HealthItem(
                "Gemini cloud fallback",
                HealthStatus.NOT_CONFIGURED,
                "Enabled but no API key configured",
            ))

        # Routing/fallback mode from existing configuration.
        items.append(HealthItem(
            "Local LLM (Ollama) fallback",
            HealthStatus.CONFIGURED if getattr(s, "enable_local_llm", False) else HealthStatus.NOT_CONFIGURED,
            "Enabled" if getattr(s, "enable_local_llm", False) else "Disabled",
        ))

        # Local inference backend: presence of the optional package only.
        if module_available("needle"):
            items.append(HealthItem(
                "Local inference backend (needle)",
                HealthStatus.AVAILABLE,
                "Package installed; initialization not attempted",
            ))
        else:
            items.append(HealthItem(
                "Local inference backend (needle)",
                HealthStatus.UNAVAILABLE,
                "Package not installed",
            ))

        # Tier 1 tuned model weights (file presence only).
        tuned_status, tuned_detail = _path_status(
            getattr(s, "needle_tuned_weights_path", "")
        )
        generation = getattr(s, "needle_generation", "?")
        items.append(HealthItem(
            "Tier 1 model weights (tuned Needle)",
            tuned_status,
            f"{tuned_detail} (generation {generation})",
        ))

        # Tier 0.5 custom ONNX model (file presence only).
        onnx_status, onnx_detail = _path_status(
            getattr(s, "needle_model_path", "")
        )
        items.append(HealthItem(
            "Tier 0.5 model (custom ONNX)",
            onnx_status,
            onnx_detail,
        ))

        return HealthSection("AI and Routing", items)

    def _voice_section(self) -> HealthSection:
        s = self.settings
        items: List[HealthItem] = []

        voice_enabled = bool(getattr(s, "enable_voice", False))
        items.append(HealthItem(
            "Voice pipeline",
            HealthStatus.CONFIGURED if voice_enabled else HealthStatus.NOT_CONFIGURED,
            "Enabled" if voice_enabled else "Disabled in settings",
        ))

        stt_mode = str(getattr(s, "voice_stt_backend", "online") or "online")
        items.append(HealthItem(
            "STT mode",
            HealthStatus.CONFIGURED,
            f"'{stt_mode}'",
        ))

        # Offline STT backend availability, only meaningful for offline/auto.
        if stt_mode not in ("offline", "auto"):
            items.append(HealthItem(
                "Offline STT backend",
                HealthStatus.NOT_CONFIGURED,
                "Not applicable (online mode)",
            ))
        else:
            model_dir = Path(str(getattr(s, "voice_stt_model_dir", "") or ""))
            package_ok = module_available("sherpa_onnx")
            files_ok = _offline_stt_files_present(model_dir)
            if package_ok and files_ok:
                items.append(HealthItem(
                    "Offline STT backend",
                    HealthStatus.AVAILABLE,
                    "Package and model files found; model not loaded",
                ))
            elif not package_ok:
                items.append(HealthItem(
                    "Offline STT backend",
                    HealthStatus.UNAVAILABLE,
                    "'sherpa_onnx' package not installed",
                ))
            else:
                items.append(HealthItem(
                    "Offline STT backend",
                    HealthStatus.UNAVAILABLE,
                    f"Model files missing in {model_dir}",
                ))

        return HealthSection("Voice", items)

    def _storage_section(self) -> HealthSection:
        s = self.settings
        items: List[HealthItem] = []

        # Database / history availability via a cheap read when possible.
        db_status, db_detail = self._probe_database()
        items.append(HealthItem("History database", db_status, db_detail))

        # Application model directory (cheap existence check).
        models_dir = self._model_dir()
        if models_dir.is_dir():
            items.append(HealthItem(
                "Model directory",
                HealthStatus.AVAILABLE,
                f"Found: {models_dir}",
            ))
        else:
            items.append(HealthItem(
                "Model directory",
                HealthStatus.UNAVAILABLE,
                f"Missing: {models_dir}",
            ))

        return HealthSection("Storage", items)

    def _resources_section(self) -> HealthSection:
        try:
            sample = self._sample_resources()
        except Exception:
            sample = None
        items: List[HealthItem] = []

        if not sample:
            items.append(HealthItem("CPU usage", HealthStatus.UNKNOWN, "Not available"))
            items.append(HealthItem("Memory usage", HealthStatus.UNKNOWN, "Not available"))
            items.append(HealthItem("Disk usage", HealthStatus.UNKNOWN, "Not available"))
            return HealthSection("System Resources", items)

        items.append(HealthItem(
            "CPU usage",
            HealthStatus.AVAILABLE,
            f"{sample['cpu_percent']:.1f}%",
        ))
        items.append(HealthItem(
            "Memory usage",
            HealthStatus.AVAILABLE,
            f"{sample['memory_percent']:.1f}% "
            f"({sample['memory_used_gb']} / {sample['memory_total_gb']} GB)",
        ))
        items.append(HealthItem(
            "Disk usage",
            HealthStatus.AVAILABLE,
            f"{sample['disk_percent']:.1f}% used on {sample['disk_path']} "
            f"({sample['disk_free_gb']} GB free)",
        ))
        return HealthSection("System Resources", items)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _model_dir(self) -> Path:
        if self._models_root:
            return Path(self._models_root)
        configured = getattr(self.settings, "needle_model_path", "") or ""
        if configured:
            return Path(str(configured)).parent
        return Path("models")

    def _disk_anchor(self) -> str:
        try:
            anchor = Path(str(self.settings.database_path)).resolve().anchor
            if anchor:
                return anchor
        except Exception:
            pass
        return "/"

    def _probe_database(self) -> tuple:
        """Cheap read of the history database; never raises."""
        loader = getattr(self.history_manager, "get_recent_tool_calls", None)
        if loader is not None:
            try:
                rows = loader(limit=1)
                return HealthStatus.AVAILABLE, f"Readable ({len(rows)} recent entries)"
            except Exception:
                return HealthStatus.UNAVAILABLE, "Read failed"

        path = Path(str(self.settings.database_path))
        if not path.exists():
            return HealthStatus.NOT_CONFIGURED, "Not created yet"
        try:
            conn = sqlite3.connect(str(path))
            try:
                conn.execute("SELECT 1")
            finally:
                conn.close()
            return HealthStatus.AVAILABLE, "Readable"
        except Exception:
            return HealthStatus.UNAVAILABLE, "Read failed"

    def _sample_resources(self) -> Optional[Dict[str, Any]]:
        """
        A single lightweight ``psutil`` sample, or ``None`` if unavailable.

        Overridable/injectable so tests never depend on real hardware.
        """
        if self._resource_provider is not None:
            return self._resource_provider()
        try:
            import psutil

            cpu = psutil.cpu_percent(interval=0.1)
            mem = psutil.virtual_memory()
            anchor = self._disk_anchor()
            disk = psutil.disk_usage(anchor)
            return {
                "cpu_percent": float(cpu),
                "memory_percent": float(mem.percent),
                "memory_used_gb": round(mem.used / (1024 ** 3), 2),
                "memory_total_gb": round(mem.total / (1024 ** 3), 2),
                "disk_percent": float(disk.percent),
                "disk_free_gb": round(disk.free / (1024 ** 3), 2),
                "disk_total_gb": round(disk.total / (1024 ** 3), 2),
                "disk_path": anchor,
            }
        except Exception:
            return None


class HealthCheckSignals(QObject):
    """Signals for the background health collector."""

    result = Signal(object)
    error = Signal(str)


class HealthCheckWorker(QRunnable):
    """Runs the (cheap) health collection off the Qt main UI thread."""

    def __init__(self, collector: SystemHealthCollector):
        super().__init__()
        self.collector = collector
        self.signals = HealthCheckSignals()

    @Slot()
    def run(self) -> None:
        try:
            self.signals.result.emit(self.collector.collect())
        except Exception:
            # Deliberately generic: never surface raw exception details.
            self.signals.error.emit(HEALTH_ERROR_TEXT)


class HealthItemRow(QFrame):
    """One health row: label + detail on the left, status chip on the right."""

    def __init__(
        self,
        label: str,
        status: str,
        detail: str = "",
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.status_text = status
        status_color = _STATUS_COLORS.get(status, "#cdd6f4")

        self.setStyleSheet(f"""
            HealthItemRow {{
                background-color: #181825;
                border: 1px solid #313244;
                border-radius: 6px;
                margin: 2px 0;
                padding: 6px 10px;
            }}
            QLabel#hl_label {{
                color: #cdd6f4;
                font-size: 13px;
                font-weight: bold;
            }}
            QLabel#hl_detail {{
                color: #7f849c;
                font-size: 11px;
            }}
            QLabel#hl_status {{
                color: {status_color};
                font-size: 12px;
                font-weight: bold;
            }}
        """)

        row = QHBoxLayout(self)
        row.setContentsMargins(6, 4, 6, 4)

        left = QVBoxLayout()
        left.setSpacing(1)
        lbl_label = QLabel(label, self)
        lbl_label.setObjectName("hl_label")
        left.addWidget(lbl_label)
        if detail:
            lbl_detail = QLabel(detail, self)
            lbl_detail.setObjectName("hl_detail")
            lbl_detail.setWordWrap(True)
            left.addWidget(lbl_detail)

        lbl_status = QLabel(status, self)
        lbl_status.setObjectName("hl_status")

        row.addLayout(left, stretch=1)
        row.addWidget(lbl_status, alignment=Qt.AlignVCenter)


class SystemHealthDialog(QDialog):
    """
    Modal System Health dashboard.

    Collection runs on a background worker so a slow disk/db probe never blocks
    the UI; failures degrade to a friendly message instead of a crash.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        history_manager: Optional[Any] = None,
        parent: Optional[QWidget] = None,
        auto_load: bool = True,
    ):
        super().__init__(parent)
        self.collector = SystemHealthCollector(
            settings=settings, history_manager=history_manager
        )
        self._report: List[HealthSection] = []
        self._setup_ui()

        if auto_load:
            self.load()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _setup_ui(self) -> None:
        self.setWindowTitle("System Health — Vector")
        self.setModal(True)
        self.resize(620, 560)

        self.setStyleSheet("""
            QDialog {
                background-color: #1e1e2e;
                color: #cdd6f4;
                font-family: 'Segoe UI', sans-serif;
            }
            QLabel#heading {
                color: #89b4fa;
                font-size: 15px;
                font-weight: bold;
            }
            QLabel#section {
                color: #89b4fa;
                font-size: 13px;
                font-weight: bold;
                padding: 10px 0 2px 0;
            }
            QLabel#empty, QLabel#error {
                font-size: 13px;
                padding: 24px;
            }
            QLabel#empty { color: #7f849c; }
            QLabel#error { color: #f38ba8; }
            QScrollArea {
                background-color: #181825;
                border: 1px solid #313244;
                border-radius: 6px;
            }
            QWidget#content { background-color: #181825; }
            QPushButton#btn_refresh {
                background-color: #313244;
                color: #cdd6f4;
                border: none;
                border-radius: 6px;
                padding: 6px 12px;
            }
            QPushButton#btn_refresh:hover { background-color: #45475a; }
            QPushButton#btn_close {
                background-color: #89b4fa;
                color: #11111b;
                font-weight: bold;
                border: none;
                border-radius: 6px;
                padding: 6px 14px;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        header = QHBoxLayout()
        lbl_heading = QLabel("🩺 System Health", self)
        lbl_heading.setObjectName("heading")
        self.btn_refresh = QPushButton("↻ Refresh", self)
        self.btn_refresh.setObjectName("btn_refresh")
        self.btn_refresh.clicked.connect(self.refresh)
        header.addWidget(lbl_heading)
        header.addStretch()
        header.addWidget(self.btn_refresh)
        layout.addLayout(header)

        lbl_note = QLabel("Read-only diagnostics. No models loaded, no network access.", self)
        lbl_note.setStyleSheet("color: #7f849c; font-size: 11px;")
        layout.addWidget(lbl_note)

        self.scroll = QScrollArea(self)
        self.scroll.setWidgetResizable(True)
        self.content = QWidget(self)
        self.content.setObjectName("content")
        self.content_layout = QVBoxLayout(self.content)
        self.content_layout.setContentsMargins(10, 10, 10, 10)
        self.content_layout.setSpacing(4)
        self.scroll.setWidget(self.content)
        layout.addWidget(self.scroll, stretch=1)

        self.lbl_error = QLabel("", self)
        self.lbl_error.setObjectName("error")
        self.lbl_error.setWordWrap(True)
        self.lbl_error.setAlignment(Qt.AlignCenter)
        self.lbl_error.hide()
        layout.addWidget(self.lbl_error)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_close = QPushButton("Close", self)
        btn_close.setObjectName("btn_close")
        btn_close.clicked.connect(self.reject)
        btn_row.addWidget(btn_close)
        layout.addLayout(btn_row)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def load(self) -> None:
        """Collect health on a background worker and re-render when done."""
        worker = HealthCheckWorker(self.collector)
        worker.signals.result.connect(self._on_report)
        worker.signals.error.connect(self._on_error)
        self._start_worker(worker)

    def _start_worker(self, worker: HealthCheckWorker) -> None:
        """Seam so tests can run the collection synchronously."""
        QThreadPool.globalInstance().start(worker)

    def refresh(self) -> None:
        """Refresh action; re-collects and re-renders the dashboard."""
        self.load()

    @Slot(object)
    def _on_report(self, report: Any) -> None:
        self._report = list(report or [])
        self._render()

    @Slot(str)
    def _on_error(self, message: str) -> None:
        self._report = []
        self._clear_content()
        self.lbl_error.setText(str(message) or HEALTH_ERROR_TEXT)
        self.lbl_error.show()

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def _clear_content(self) -> None:
        while self.content_layout.count():
            item = self.content_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                # Detach immediately so the old rows are gone from the widget
                # tree (not merely queued for deletion).
                widget.setParent(None)
                widget.deleteLater()

    def _render(self) -> None:
        self.lbl_error.hide()
        self._clear_content()

        if not self._report:
            empty = QLabel(EMPTY_HEALTH_TEXT, self.content)
            empty.setObjectName("empty")
            empty.setAlignment(Qt.AlignCenter)
            self.content_layout.addWidget(empty)
        else:
            for section in self._report:
                lbl_section = QLabel(section.title, self.content)
                lbl_section.setObjectName("section")
                self.content_layout.addWidget(lbl_section)
                for item in section.items:
                    self.content_layout.addWidget(
                        HealthItemRow(item.label, item.status, item.detail, self.content)
                    )

        self.content_layout.addStretch()
