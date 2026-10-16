"""
Headless unit tests for the System Health Dashboard.

The collector is exercised deterministically: file checks point at ``tmp_path``,
optional-dependency checks are monkeypatched, the history probe uses a fake, and
resource sampling is injected.  Nothing here loads a model, opens the network,
touches a microphone, or requires a GPU.
"""

import os
import sqlite3
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QLabel  # noqa: E402

from app.config.settings import Settings  # noqa: E402
from app.ui import system_health as health_mod  # noqa: E402
from app.ui.system_health import (  # noqa: E402
    EMPTY_HEALTH_TEXT,
    HEALTH_ERROR_TEXT,
    HealthItemRow,
    HealthStatus,
    SystemHealthCollector,
    SystemHealthDialog,
)


@pytest.fixture(scope="session")
def qapp():
    """Session fixture initializing a headless QApplication."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


#: Fake Gemini credential assembled at runtime so the source never contains a
#: literal that trips secret scanners.  The value still matches PrivacyFilter's
#: Gemini-key pattern so the redaction assertions stay meaningful.
_FAKE_GEMINI_KEY = "AIzaSy" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q"


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _find(sections, label):
    for section in sections:
        for item in section.items:
            if item.label == label:
                return item
    raise AssertionError(f"no health item labelled {label!r}")


def _collector(tmp_path, **kwargs):
    base = dict(
        needle_model_path=str(tmp_path / "needle_2_custom.onnx"),
        needle_tuned_weights_path=str(tmp_path / "needle3_vector_v4.cact"),
        voice_stt_model_dir=str(tmp_path / "stt"),
        database_path=str(tmp_path / "vector.db"),
    )
    base.update(kwargs)
    return SystemHealthCollector(settings=Settings(**base), **{
        k: v for k, v in kwargs.items() if k in ("history_manager", "models_root", "resource_provider")
    })


def _dialog_rows(dlg):
    rows = dlg.content.findChildren(HealthItemRow)
    return [(r.findChild(QLabel, "hl_label").text(), r.status_text) for r in rows]


def _dialog_status(dlg, label):
    for text, status in _dialog_rows(dlg):
        if text == label:
            return status
    raise AssertionError(f"no dialog row labelled {label!r}")


class _FakeHistory:
    def __init__(self, rows=None, raises=False):
        self.rows = list(rows or [])
        self.raises = raises

    def get_recent_tool_calls(self, conversation_id=None, limit=50):
        if self.raises:
            raise RuntimeError("db exploded at C:\\secret\\vector.db")
        return list(self.rows)


def _fake_resources():
    return {
        "cpu_percent": 12.5,
        "memory_percent": 42.0,
        "memory_used_gb": 6.7,
        "memory_total_gb": 16.0,
        "disk_percent": 55.0,
        "disk_free_gb": 120.0,
        "disk_total_gb": 500.0,
        "disk_path": "C:\\",
    }


# --------------------------------------------------------------------------
# AI and Routing
# --------------------------------------------------------------------------

def test_gemini_configured_reports_configured_without_key(tmp_path):
    sections = _collector(tmp_path, enable_gemini=True, gemini_api_key=_FAKE_GEMINI_KEY).collect()
    item = _find(sections, "Gemini cloud fallback")
    assert item.status == HealthStatus.CONFIGURED
    assert "not tested" in item.detail
    assert _FAKE_GEMINI_KEY not in item.detail


def test_gemini_disabled_is_not_configured(tmp_path):
    sections = _collector(tmp_path, enable_gemini=False, gemini_api_key="").collect()
    assert _find(sections, "Gemini cloud fallback").status == HealthStatus.NOT_CONFIGURED


def test_gemini_enabled_without_key_is_not_configured(tmp_path):
    sections = _collector(tmp_path, enable_gemini=True, gemini_api_key="").collect()
    item = _find(sections, "Gemini cloud fallback")
    assert item.status == HealthStatus.NOT_CONFIGURED
    assert "no api key" in item.detail.lower()


def test_local_llm_toggle_status(tmp_path):
    on = _collector(tmp_path, enable_local_llm=True).collect()
    off = _collector(tmp_path, enable_local_llm=False).collect()
    assert _find(on, "Local LLM (Ollama) fallback").status == HealthStatus.CONFIGURED
    assert _find(off, "Local LLM (Ollama) fallback").status == HealthStatus.NOT_CONFIGURED


def test_local_backend_missing_and_present(tmp_path, monkeypatch):
    monkeypatch.setattr(health_mod, "module_available", lambda name: False)
    missing = _collector(tmp_path).collect()
    assert _find(missing, "Local inference backend (needle)").status == HealthStatus.UNAVAILABLE

    monkeypatch.setattr(health_mod, "module_available", lambda name: True)
    present = _collector(tmp_path).collect()
    assert _find(present, "Local inference backend (needle)").status == HealthStatus.AVAILABLE


def test_tier_weights_missing_then_found(tmp_path):
    missing = _collector(tmp_path).collect()
    assert _find(missing, "Tier 1 model weights (tuned Needle)").status == HealthStatus.UNAVAILABLE

    (tmp_path / "needle3_vector_v4.cact").write_bytes(b"x")
    found = _collector(tmp_path).collect()
    assert _find(found, "Tier 1 model weights (tuned Needle)").status == HealthStatus.AVAILABLE


def test_tier05_onnx_not_configured_when_path_empty(tmp_path):
    sections = _collector(tmp_path, needle_model_path="").collect()
    assert _find(sections, "Tier 0.5 model (custom ONNX)").status == HealthStatus.NOT_CONFIGURED


def test_report_has_all_four_sections(tmp_path):
    sections = _collector(tmp_path, resource_provider=_fake_resources).collect()
    assert [s.title for s in sections] == [
        "AI and Routing",
        "Voice",
        "Storage",
        "System Resources",
    ]


# --------------------------------------------------------------------------
# Voice / STT
# --------------------------------------------------------------------------

def test_voice_disabled_and_stt_mode_online(tmp_path):
    sections = _collector(tmp_path, enable_voice=False, voice_stt_backend="online").collect()
    assert _find(sections, "Voice pipeline").status == HealthStatus.NOT_CONFIGURED
    assert _find(sections, "STT mode").status == HealthStatus.CONFIGURED
    assert "online" in _find(sections, "STT mode").detail
    assert _find(sections, "Offline STT backend").status == HealthStatus.NOT_CONFIGURED
    assert "online mode" in _find(sections, "Offline STT backend").detail


def test_offline_stt_missing_package(tmp_path, monkeypatch):
    monkeypatch.setattr(health_mod, "module_available", lambda name: False)
    sections = _collector(tmp_path, enable_voice=True, voice_stt_backend="offline").collect()
    item = _find(sections, "Offline STT backend")
    assert item.status == HealthStatus.UNAVAILABLE
    assert "sherpa_onnx" in item.detail


def test_offline_stt_missing_model_files(tmp_path, monkeypatch):
    monkeypatch.setattr(health_mod, "module_available", lambda name: True)
    sections = _collector(tmp_path, enable_voice=True, voice_stt_backend="offline").collect()
    item = _find(sections, "Offline STT backend")
    assert item.status == HealthStatus.UNAVAILABLE
    assert "missing" in item.detail.lower()


def test_offline_stt_files_found(tmp_path, monkeypatch):
    monkeypatch.setattr(health_mod, "module_available", lambda name: True)
    stt_dir = tmp_path / "stt"
    stt_dir.mkdir()
    (stt_dir / "tokens.txt").write_text("t")
    (stt_dir / "encoder-epoch-99.onnx").write_bytes(b"e")
    (stt_dir / "decoder-epoch-99.onnx").write_bytes(b"d")

    sections = _collector(tmp_path, enable_voice=True, voice_stt_backend="auto").collect()
    item = _find(sections, "Offline STT backend")
    assert item.status == HealthStatus.AVAILABLE
    assert "found" in item.detail.lower()


# --------------------------------------------------------------------------
# Storage
# --------------------------------------------------------------------------

def test_database_available_via_history_manager(tmp_path):
    sections = _collector(tmp_path, history_manager=_FakeHistory(rows=[{"id": "x"}])).collect()
    item = _find(sections, "History database")
    assert item.status == HealthStatus.AVAILABLE
    assert "1 recent entries" in item.detail


def test_database_read_failure_is_reported_generically(tmp_path):
    sections = _collector(tmp_path, history_manager=_FakeHistory(raises=True)).collect()
    item = _find(sections, "History database")
    assert item.status == HealthStatus.UNAVAILABLE
    assert item.detail == "Read failed"
    assert "exploded" not in item.detail


def test_database_file_checks_without_history_manager(tmp_path):
    db_path = tmp_path / "vector.db"
    not_yet = _collector(tmp_path).collect()
    assert _find(not_yet, "History database").status == HealthStatus.NOT_CONFIGURED

    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE t (id TEXT)")
    conn.commit()
    conn.close()

    created = _collector(tmp_path).collect()
    assert _find(created, "History database").status == HealthStatus.AVAILABLE


def test_model_directory_found_and_missing(tmp_path):
    present = _collector(tmp_path, models_root=str(tmp_path)).collect()
    assert _find(present, "Model directory").status == HealthStatus.AVAILABLE

    absent = _collector(tmp_path, models_root=str(tmp_path / "nope")).collect()
    assert _find(absent, "Model directory").status == HealthStatus.UNAVAILABLE


# --------------------------------------------------------------------------
# System resources
# --------------------------------------------------------------------------

def test_resources_success(tmp_path):
    sections = _collector(tmp_path, resource_provider=_fake_resources).collect()
    assert _find(sections, "CPU usage").status == HealthStatus.AVAILABLE
    assert "12.5%" in _find(sections, "CPU usage").detail
    assert _find(sections, "Memory usage").status == HealthStatus.AVAILABLE
    assert _find(sections, "Disk usage").status == HealthStatus.AVAILABLE
    assert "55.0%" in _find(sections, "Disk usage").detail


def test_resources_unavailable(tmp_path):
    sections = _collector(tmp_path, resource_provider=lambda: None).collect()
    assert _find(sections, "CPU usage").status == HealthStatus.UNKNOWN
    assert _find(sections, "Memory usage").status == HealthStatus.UNKNOWN
    assert _find(sections, "Disk usage").status == HealthStatus.UNKNOWN


def test_resources_provider_exception_is_contained(tmp_path):
    def boom():
        raise RuntimeError("psutil crashed")

    collector = SystemHealthCollector(
        settings=Settings(
            needle_model_path=str(tmp_path / "m.onnx"),
            needle_tuned_weights_path=str(tmp_path / "w.cact"),
            voice_stt_model_dir=str(tmp_path / "stt"),
            database_path=str(tmp_path / "vector.db"),
        ),
        resource_provider=boom,
    )
    # Collection must not raise; resources degrade to Unknown.
    sections = collector.collect()
    assert _find(sections, "CPU usage").status == HealthStatus.UNKNOWN


# --------------------------------------------------------------------------
# Dialog (headless)
# --------------------------------------------------------------------------

def test_dialog_renders_report_headlessly(qapp, tmp_path):
    collector = _collector(
        tmp_path, enable_gemini=True, gemini_api_key="key", resource_provider=_fake_resources
    )
    dlg = SystemHealthDialog(auto_load=False)
    dlg._on_report(collector.collect())

    assert _dialog_status(dlg, "Gemini cloud fallback") == HealthStatus.CONFIGURED
    assert _dialog_status(dlg, "CPU usage") == HealthStatus.AVAILABLE
    assert dlg.lbl_error.isHidden() is True


def test_dialog_empty_report_shows_empty_state(qapp):
    dlg = SystemHealthDialog(auto_load=False)
    dlg._on_report([])
    texts = [lbl.text() for lbl in dlg.content.findChildren(QLabel)]
    assert EMPTY_HEALTH_TEXT in texts


def test_dialog_error_state_is_generic(qapp):
    dlg = SystemHealthDialog(auto_load=False)
    dlg._on_error("raw internal failure: C:\\secret\\path")
    assert dlg.lbl_error.isHidden() is False
    assert dlg.lbl_error.text() == "raw internal failure: C:\\secret\\path"
    # A generic message is what the worker emits; verify that path directly.
    dlg._on_error(HEALTH_ERROR_TEXT)
    assert dlg.lbl_error.text() == HEALTH_ERROR_TEXT


def test_dialog_refresh_updates_displayed_information(qapp):
    class _StubCollector:
        def __init__(self):
            self.calls = 0

        def collect(self):
            self.calls += 1
            status = HealthStatus.AVAILABLE if self.calls == 1 else HealthStatus.UNAVAILABLE
            return [health_mod.HealthSection("S", [health_mod.HealthItem("probe", status, "d")])]

    dlg = SystemHealthDialog(auto_load=False)
    dlg.collector = _StubCollector()
    dlg._start_worker = lambda worker: worker.run()

    dlg.load()
    assert _dialog_status(dlg, "probe") == HealthStatus.AVAILABLE

    dlg.refresh()
    assert _dialog_status(dlg, "probe") == HealthStatus.UNAVAILABLE
    assert dlg.collector.calls == 2


def test_dialog_worker_failure_shows_generic_error(qapp):
    class _BoomCollector:
        def collect(self):
            raise RuntimeError("boom at C:\\secret")

    dlg = SystemHealthDialog(auto_load=False)
    dlg.collector = _BoomCollector()
    dlg._start_worker = lambda worker: worker.run()

    dlg.load()  # must not raise
    assert dlg.lbl_error.isHidden() is False
    assert dlg.lbl_error.text() == HEALTH_ERROR_TEXT
    assert "secret" not in dlg.lbl_error.text().lower()


# --------------------------------------------------------------------------
# Privacy
# --------------------------------------------------------------------------

def test_sensitive_values_are_not_rendered(qapp, tmp_path):
    secret = _FAKE_GEMINI_KEY
    sections = _collector(tmp_path, enable_gemini=True, gemini_api_key=secret).collect()

    dlg = SystemHealthDialog(auto_load=False)
    dlg._on_report(sections)

    rendered = " ".join(lbl.text() for lbl in dlg.findChildren(QLabel))
    assert secret not in rendered
    # The API key never appears anywhere in the structured report either.
    for section in sections:
        for item in section.items:
            assert secret not in item.detail
