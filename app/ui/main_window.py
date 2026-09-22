"""
Main Application Window and Asynchronous Qt Worker Pipeline for Vector Desktop AI Assistant.
"""

from typing import Any, Dict, Optional
from PySide6.QtCore import QObject, QRunnable, QThreadPool, Qt, Signal, Slot
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.config.settings import Settings, get_settings
from app.core.router import IntentRouter, RouteResult
from app.memory.history import HistoryManager
from app.tools.applications.launcher import CloseAppTool, GetRunningAppsTool, LaunchAppTool
from app.tools.files.manager import GetFileInfoTool, OpenFileTool, OpenFolderTool, SearchFilesTool
from app.tools.media.controller import MediaNextTool, MediaPauseTool, MediaPlayTool, MediaPreviousTool
from app.tools.power.power import LockPcTool, RestartPcTool, ShutdownPcTool, SleepPcTool
from app.tools.registry import ToolRegistry, get_tool_registry
from app.tools.system.system_info import (
    GetBatteryStatusTool,
    GetCpuUsageTool,
    GetDiskUsageTool,
    GetMemoryUsageTool,
    GetSystemStatsTool,
)
from app.tools.system.volume import GetVolumeTool, MuteTool, SetVolumeTool, UnmuteTool
from app.tools.terminal.executor import ExecuteTerminalCommandTool
from app.tools.windows.manager import (
    CloseWindowTool,
    GetActiveWindowTool,
    MaximizeWindowTool,
    MinimizeWindowTool,
    MoveWindowTool,
    ResizeWindowTool,
)
from app.ui.chat_view import ChatViewWidget
from app.ui.command_input import CommandInputWidget
from app.ui.confirmation_dialog import ConfirmationDialog
from app.ui.settings import SettingsDialog
from app.ui.status_bar import StatusBarWidget
from app.voice.listener import VoiceListener
from app.voice.tts import TTSEngine


class WorkerSignals(QObject):
    """Signals for background RouteWorker thread."""
    result = Signal(object)
    error = Signal(str)


class RouteWorker(QRunnable):
    """Background worker executing IntentRouter pipeline off the Qt main UI thread."""

    def __init__(self, router: IntentRouter, user_query: str, confirmed: bool = False):
        super().__init__()
        self.router = router
        self.user_query = user_query
        self.confirmed = confirmed
        self.signals = WorkerSignals()

    @Slot()
    def run(self) -> None:
        try:
            route_res = self.router.route_and_execute(
                user_query=self.user_query,
                confirmed_by_user=self.confirmed
            )
            self.signals.result.emit(route_res)
        except Exception as e:
            self.signals.error.emit(str(e))


class TranscribeWorker(QRunnable):
    """Background worker listening to microphone off the Qt main UI thread."""

    def __init__(self, voice_listener: VoiceListener):
        super().__init__()
        self.voice_listener = voice_listener
        self.signals = WorkerSignals()

    @Slot()
    def run(self) -> None:
        try:
            text = self.voice_listener.listen_from_mic()
            self.signals.result.emit(text)
        except Exception as e:
            self.signals.error.emit(str(e))


class MainWindow(QMainWindow):
    """
    Vector Main Application Window.
    """

    def __init__(self, show_tier_in_terminal: bool = False):
        super().__init__()
        self.show_tier_in_terminal = show_tier_in_terminal
        self.setWindowTitle("Vector — Desktop AI Assistant")
        self.resize(800, 600)
        self.settings = get_settings()

        # Initialize and register all local tools
        self.registry = get_tool_registry()
        self._register_default_tools()

        self.router = IntentRouter(settings=self.settings, registry=self.registry)
        self.history = HistoryManager()
        self.current_conv_id = self.history.create_conversation("Vector Session")

        self.tts = TTSEngine()
        self.voice_listener = VoiceListener()

        self.thread_pool = QThreadPool.globalInstance()

        self._setup_ui()

    def _register_default_tools(self) -> None:
        """Register all local tools into global ToolRegistry."""
        tools = [
            GetSystemStatsTool(), GetCpuUsageTool(), GetMemoryUsageTool(), GetDiskUsageTool(), GetBatteryStatusTool(),
            GetVolumeTool(), SetVolumeTool(), MuteTool(), UnmuteTool(),
            LockPcTool(), SleepPcTool(), RestartPcTool(), ShutdownPcTool(),
            MediaPlayTool(), MediaPauseTool(), MediaNextTool(), MediaPreviousTool(),
            LaunchAppTool(), CloseAppTool(), GetRunningAppsTool(),
            SearchFilesTool(), OpenFileTool(), OpenFolderTool(), GetFileInfoTool(),
            GetActiveWindowTool(), MaximizeWindowTool(), MinimizeWindowTool(), CloseWindowTool(), MoveWindowTool(), ResizeWindowTool(),
            ExecuteTerminalCommandTool()
        ]
        for t in tools:
            self.registry.register(t)

    def _setup_ui(self) -> None:
        self.setStyleSheet("""
            QMainWindow {
                background-color: #11111b;
            }
        """)

        central = QWidget(self)
        self.setCentralWidget(central)

        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Header bar
        header = QWidget(self)
        header.setStyleSheet("background-color: #1e1e2e; border-bottom: 1px solid #313244; padding: 4px;")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(12, 6, 12, 6)

        lbl_title = QLabel("Vector Assistant", header)
        lbl_title.setStyleSheet("color: #89b4fa; font-weight: bold; font-size: 14px;")

        btn_settings = QPushButton("⚙ Settings", header)
        btn_settings.setStyleSheet("background-color: #313244; color: #cdd6f4; border-radius: 4px; padding: 4px 8px;")
        btn_settings.clicked.connect(self._open_settings)

        header_layout.addWidget(lbl_title)
        header_layout.addStretch()
        header_layout.addWidget(btn_settings)

        main_layout.addWidget(header)

        # Chat View
        self.chat_view = ChatViewWidget(self)
        main_layout.addWidget(self.chat_view, stretch=1)

        # Command Input
        self.command_input = CommandInputWidget(self)
        self.command_input.command_submitted.connect(self._handle_user_command)
        self.command_input.voice_toggled.connect(self._handle_voice_toggle)
        main_layout.addWidget(self.command_input)

        # Status Bar
        self.status_bar = StatusBarWidget(self)
        main_layout.addWidget(self.status_bar)

        # Welcome message
        self.chat_view.add_assistant_message("Hello! I am Vector, your desktop AI assistant. How can I help you today?")

    def _open_settings(self) -> None:
        dialog = SettingsDialog(self)
        dialog.exec()

    def _handle_voice_toggle(self, active: bool) -> None:
        if active:
            self.command_input.btn_mic.setText("🔴 Listening...")
            worker = TranscribeWorker(self.voice_listener)
            worker.signals.result.connect(self._on_transcribe_finished)
            worker.signals.error.connect(lambda err: self._reset_mic_btn())
            self.thread_pool.start(worker)
        else:
            self._reset_mic_btn()

    def _reset_mic_btn(self) -> None:
        self.command_input.btn_mic.setChecked(False)
        self.command_input.btn_mic.setText("🎤 Voice")

    def _on_transcribe_finished(self, text: str) -> None:
        self._reset_mic_btn()
        if text and text.strip():
            self._handle_user_command(text.strip())
        else:
            self.chat_view.add_assistant_message("No speech detected. Please speak clearly into your microphone.")

    def _handle_user_command(self, query: str, confirmed: bool = False) -> None:
        if not confirmed:
            self.chat_view.add_user_message(query)
            self.history.add_message(self.current_conv_id, "user", query)

        worker = RouteWorker(self.router, query, confirmed=confirmed)
        worker.signals.result.connect(lambda res: self._on_route_finished(res, query))
        worker.signals.error.connect(lambda err: self.chat_view.add_assistant_message(f"Error: {err}"))
        self.thread_pool.start(worker)

    def _on_route_finished(self, route_res: RouteResult, original_query: str) -> None:
        # Optionally surface the exactly-resolved tier (and its confidence) to the
        # terminal for every command — no UI clutter, just the routing fact.
        if self.show_tier_in_terminal:
            # flush=True is essential: this GUI app never exits, so an unflushed
            # stdout buffer would swallow every [tier] line until the process dies.
            print(
                f"[tier] {route_res.tier_used.value!s:18} "
                f"tool={route_res.tool_name or '-'} "
                f"conf={route_res.confidence if route_res.confidence else '-'}",
                flush=True,
            )
        if route_res.requires_confirmation:
            # Show Security Confirmation Dialog
            dlg = ConfirmationDialog(
                tool_name=route_res.tool_name,
                message=route_res.response_text,
                arguments=route_res.raw_arguments,
                parent=self
            )
            if dlg.exec():
                # Re-submit with confirmed=True
                self._handle_user_command(original_query, confirmed=True)
            else:
                self.chat_view.add_assistant_message("Action cancelled by user.")
            return

        # Add assistant response
        if route_res.response_text:
            self.chat_view.add_assistant_message(route_res.response_text)
            self.history.add_message(self.current_conv_id, "assistant", route_res.response_text)

            # Speak output aloud via Windows TTS if enabled
            if self.settings.enable_voice:
                self.tts.speak(route_res.response_text)

        # Add visual Action Card if tool executed
        if route_res.tool_result:
            tr = route_res.tool_result
            self.chat_view.add_action_card(tool_name=tr.tool, message=tr.message, success=tr.success)
