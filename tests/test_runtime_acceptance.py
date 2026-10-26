"""
Feature 11 -- Vector release readiness: real startup and runtime acceptance.

These tests are the *automated* half of the acceptance matrix.  They exercise
the application as a complete desktop program rather than as isolated units:

* the real entry point (``app.main.main``) starts a process that stays alive --
  an incomplete installation must not kill the window during startup;
* a degraded install -- no tuned V4 weights, no cloud credential, no Google
  GenAI SDK -- escalates honestly to a local answer and never fabricates a
  Tier 1 decision;
* a background approval that fails is surfaced in the chat and executes nothing.

Everything is offline and deterministic: no microphone, no network, no model is
loaded, no credential is read, and every write lands under ``tmp_path``.
Flows that need a human, a real device or a network service belong in
``docs/manual_acceptance.md`` instead of here.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from PySide6.QtWidgets import QApplication

from app.config.constants import IntentTier
from app.config.settings import Settings
from app.core.router import IntentRouter
from app.memory.database import DatabaseManager
from app.memory.history import HistoryManager
from app.needle.client import NeedleClient
from app.tools.executor import PendingApproval
from app.tools.registry import ToolRegistry
from app.tools.system.system_info import GetSystemStatsTool

import app.ui.main_window as main_window_module
from app.ui.chat_view import ActionCardWidget
from app.ui.main_window import MainWindow


#: Repository root, used only by the real-process startup test.
REPO_ROOT = Path(__file__).resolve().parents[1]
#: How long the real entry point must stay up before it counts as started.
STARTUP_GRACE_SECONDS = 6


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _settings(tmp_path: Path, **overrides) -> Settings:
    """Degraded-install settings: no credential, no tuned weights, no .env.

    ``_env_file=None`` keeps a real ``.env`` out of the test, and the explicit
    empty ``gemini_api_key`` always wins over any ``GEMINI_API_KEY`` present in
    the process environment, so no test can ever reach the network.
    """
    values = dict(
        _env_file=None,
        enable_gemini=True,
        gemini_api_key="",
        enable_voice=False,
        needle_tuned_weights_path=str(tmp_path / "missing_v4.cact"),
        database_path=str(tmp_path / "runtime.db"),
    )
    values.update(overrides)
    return Settings(**values)


def _registry() -> ToolRegistry:
    """A minimal registry holding only a read-only reporting tool.

    ``get_system_stats`` reports what the machine is doing and changes
    nothing, so the local path can be proven without touching volume, media,
    power, apps or the shell.
    """
    registry = ToolRegistry()
    registry.clear()
    registry.register(GetSystemStatsTool())
    return registry


def _router(tmp_path: Path) -> IntentRouter:
    history = HistoryManager(
        db_manager=DatabaseManager(db_path=str(tmp_path / "router.db"))
    )
    return IntentRouter(
        settings=_settings(tmp_path),
        registry=_registry(),
        history_manager=history,
    )


def _list_texts(list_widget) -> list:
    texts = []
    for row in range(list_widget.count()):
        widget = list_widget.itemWidget(list_widget.item(row))
        if widget is not None and hasattr(widget, "text"):
            texts.append(widget.text())
    return texts


class _CapturingPool:
    """Stands in for QThreadPool: records workers instead of scheduling them."""

    def __init__(self):
        self.workers = []

    def start(self, worker):
        self.workers.append(worker)


class _RecordingTTS:
    def __init__(self):
        self.spoken = []

    def speak(self, text):
        self.spoken.append(text)


class _RaisingApprovalRouter:
    """An approval backend that fails after the user has already approved."""

    def execute_approved(self, approval, **kwargs):
        raise RuntimeError("approval backend exploded")


# ==========================================================================
# 1. Real entry point: the process starts and stays up
# ==========================================================================
def test_real_entrypoint_process_starts_and_stays_alive(tmp_path):
    """
    Launch ``app.main.main()`` in its own interpreter and require it to still
    be running after a grace period.

    This is the only test that covers the actual production entry point: Qt
    application creation, settings load, logging setup, window construction and
    the event loop.  A crash anywhere in that path shows up here as a non-zero
    exit with a traceback, which is exactly what a user would see.

    The child runs with ``cwd`` in ``tmp_path`` and ``PYTHONPATH`` pointing at
    the repository, so it cannot read the project ``.env`` and cannot write
    ``data/vector.db``; ``DATABASE_PATH`` is redirected into ``tmp_path``.
    """
    env = os.environ.copy()
    env["QT_QPA_PLATFORM"] = "offscreen"
    env["DATABASE_PATH"] = str(tmp_path / "entrypoint.db")
    env["GEMINI_API_KEY"] = ""
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")

    proc = subprocess.Popen(
        [sys.executable, "-c", "from app.main import main; main()"],
        cwd=str(tmp_path),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    failure = None
    try:
        try:
            out, err = proc.communicate(timeout=STARTUP_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            # Still running: the window came up and the event loop is live.
            pass
        else:
            failure = (
                f"entry point exited with code {proc.returncode} inside the "
                f"grace period instead of staying up.\n"
                f"stderr:\n{err}\nstdout:\n{out}"
            )
    finally:
        if proc.poll() is None:
            proc.kill()
        try:
            proc.wait(timeout=10)
        except Exception:
            pass

    assert failure is None, failure


# ==========================================================================
# 2. Degraded install: local work still happens, cloud escalation stays honest
# ==========================================================================
def test_degraded_install_serves_local_command(tmp_path):
    """A read-only local command must work with no weights and no credential."""
    router = _router(tmp_path)

    result = router.route_and_execute("system stats")

    assert result.tier_used == IntentTier.TIER_0_DIRECT
    assert result.tool_result is not None, "the local tier must return a result"
    assert result.tool_result.success is True, result.tool_result.message
    assert "Gemini unavailable" not in result.response_text


def test_degraded_install_escalates_to_honest_offline_fallback(tmp_path):
    """A request nothing local can answer falls back instead of crashing."""
    router = _router(tmp_path)

    result = router.route_and_execute(
        "Summarize the complex quantum computing paper"
    )

    assert "Gemini unavailable" in result.response_text
    assert result.execution_trace, "routing must record the steps it considered"
    assert all(
        step.get("matched") is not True for step in result.execution_trace
    ), f"no tier may claim a decision with no model and no cloud: {result.execution_trace}"


def test_missing_tuned_weights_never_fabricate_a_tier1_hit(tmp_path):
    """Tier 1 is refused, with a reason, when the tuned archive is absent."""
    client = NeedleClient(settings=_settings(tmp_path), registry=_registry())

    result = client.predict_cactus_needle_intent("set volume to 40")

    assert result.admitted is False, "an absent model must never be admitted"
    assert result.is_valid is False
    assert "absent" in result.reason, result.reason


# ==========================================================================
# 3. Missing optional dependency: the cloud SDK
# ==========================================================================
def test_missing_genai_sdk_is_reported_without_crashing(tmp_path, monkeypatch):
    """A cloud path taken without the SDK installed yields a readable answer."""
    settings = _settings(
        tmp_path, gemini_api_key="placeholder-key-for-missing-sdk-test"
    )
    router = IntentRouter(settings=settings, registry=_registry())

    # ``None`` in sys.modules makes the import raise ImportError, which is how
    # an installation without ``google-genai`` behaves.
    monkeypatch.setitem(sys.modules, "google", None)
    monkeypatch.setitem(sys.modules, "google.genai", None)

    result = router.route_and_execute(
        "Summarize the complex quantum computing paper"
    )

    assert "SDK not installed" in result.response_text, result.response_text


# ==========================================================================
# 4. A failing background approval reaches the user and runs nothing
# ==========================================================================
def test_failing_approval_is_surfaced_and_executes_nothing(
    monkeypatch, qapp, tmp_path
):
    """Approval is the one step a user already said yes to; it must still fail safely."""
    history = HistoryManager(
        db_manager=DatabaseManager(db_path=str(tmp_path / "approval.db"))
    )
    monkeypatch.setattr(main_window_module, "HistoryManager", lambda: history)

    win = MainWindow()
    try:
        win.settings = _settings(tmp_path, enable_gemini=False)
        win.router = _RaisingApprovalRouter()
        win.tts = _RecordingTTS()
        win.thread_pool = _CapturingPool()
        conversation_id = win.current_conv_id

        win._execute_approved(
            PendingApproval(tool_name="set_volume", arguments={"level": 42}),
            conversation_id,
        )
        assert len(win.thread_pool.workers) == 1, "the approval must run off the UI thread"
        win.thread_pool.workers[0].run()

        texts = _list_texts(win.chat_view.list_widget)
        assert any(
            "Error: approval backend exploded" in text for text in texts
        ), f"a failed approval must be reported to the user: {texts}"

        cards = [
            widget
            for widget in (
                win.chat_view.list_widget.itemWidget(win.chat_view.list_widget.item(i))
                for i in range(win.chat_view.list_widget.count())
            )
            if isinstance(widget, ActionCardWidget)
        ]
        assert not cards, "a failed approval must never be presented as an executed action"
        assert not win.tts.spoken, "a failed approval must not be announced as success"
    finally:
        win.close()


# ==========================================================================
# 5. Shutdown is safe to repeat
# ==========================================================================
def test_closing_twice_leaves_no_live_generation(monkeypatch, qapp, tmp_path):
    """Closing the window twice must be a no-op, never an error."""
    history = HistoryManager(
        db_manager=DatabaseManager(db_path=str(tmp_path / "close.db"))
    )
    monkeypatch.setattr(main_window_module, "HistoryManager", lambda: history)

    win = MainWindow()
    try:
        win.settings = _settings(tmp_path, enable_gemini=False)
        win.tts = _RecordingTTS()
        win.thread_pool = _CapturingPool()
        win.close()
        win.close()
        assert win._live_streams == {}
    finally:
        win.close()
