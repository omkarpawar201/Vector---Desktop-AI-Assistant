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
from app.tools.base import ToolResult
from app.tools.executor import PendingApproval
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
    """Signals for background Qt workers (routing and microphone capture)."""
    result = Signal(object)
    error = Signal(str)
    ready = Signal()


class RouteWorker(QRunnable):
    """Background worker executing IntentRouter pipeline off the Qt main UI thread."""

    def __init__(
        self,
        router: IntentRouter,
        user_query: str,
        conversation_id: Optional[str] = None
    ):
        super().__init__()
        self.router = router
        self.user_query = user_query
        self.conversation_id = conversation_id
        self.signals = WorkerSignals()

    @Slot()
    def run(self) -> None:
        try:
            route_res = self.router.route_and_execute(
                user_query=self.user_query,
                conversation_id=self.conversation_id,
            )
            self.signals.result.emit(route_res)
        except Exception as e:
            self.signals.error.emit(str(e))


class ApprovalWorker(QRunnable):
    """Background worker executing one user-approved action off the UI thread."""

    def __init__(self, router: IntentRouter, approval: PendingApproval):
        super().__init__()
        self.router = router
        self.approval = approval
        self.signals = WorkerSignals()

    @Slot()
    def run(self) -> None:
        try:
            tool_result = self.router.execute_approved(self.approval)
            self.signals.result.emit(tool_result)
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
            text = self.voice_listener.listen_from_mic(on_ready=self.signals.ready.emit)
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

        # The active conversation must exist before the router so Tier 2 can be
        # handed its recent turns as follow-up context.
        self.history = HistoryManager()
        self.current_conv_id = self.history.create_conversation("Vector Session")

        self.router = IntentRouter(
            settings=self.settings,
            registry=self.registry,
            history_manager=self.history,
        )

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
        if not active:
            self._reset_mic_btn()
            return

        # The same privacy flag that gates spoken output also gates microphone
        # capture: never open the mic while the voice pipeline is disabled.
        if not self.settings.enable_voice:
            self._reset_mic_btn()
            self.chat_view.add_assistant_message(
                "Voice input is disabled. Enable the voice pipeline in Settings to use the microphone."
            )
            return

        # The "Listening..." prompt is shown only after ambient calibration has
        # consumed its audio (ready signal), so the user never speaks into the
        # calibration window.
        worker = TranscribeWorker(self.voice_listener)
        worker.signals.ready.connect(lambda: self.command_input.btn_mic.setText("🔴 Listening..."))
        worker.signals.result.connect(self._on_transcribe_finished)
        worker.signals.error.connect(lambda err: self._reset_mic_btn())
        self.thread_pool.start(worker)

    def _reset_mic_btn(self) -> None:
        self.command_input.btn_mic.setChecked(False)
        self.command_input.btn_mic.setText("🎤 Voice")

    def _on_transcribe_finished(self, text: str) -> None:
        self._reset_mic_btn()
        if text and text.strip():
            self._handle_user_command(text.strip())
        else:
            self.chat_view.add_assistant_message("No speech detected. Please speak clearly into your microphone.")

    def _handle_user_command(self, query: str) -> None:
        self.chat_view.add_user_message(query)
        self.history.add_message(self.current_conv_id, "user", query)

        worker = RouteWorker(self.router, query, self.current_conv_id)
        worker.signals.result.connect(self._on_route_finished)
        worker.signals.error.connect(lambda err: self.chat_view.add_assistant_message(f"Error: {err}"))
        self.thread_pool.start(worker)

    def _execute_approved(self, approval: Optional[PendingApproval]) -> None:
        """Run exactly the action the user approved (single-use token)."""
        if approval is None:
            self.chat_view.add_assistant_message(
                "This action could not be confirmed safely. Please try again."
            )
            return

        worker = ApprovalWorker(self.router, approval)
        worker.signals.result.connect(self._on_approved_finished)
        worker.signals.error.connect(lambda err: self.chat_view.add_assistant_message(f"Error: {err}"))
        self.thread_pool.start(worker)

    def _on_approved_finished(self, tool_result: ToolResult) -> None:
        self._present_result(tool_result)

    def _present_result(self, tr: Optional[ToolResult]) -> None:
        """Surface one executed tool result as a message plus an action card."""
        if tr is None:
            return
        if tr.message:
            self.chat_view.add_assistant_message(tr.message)
            self.history.add_message(self.current_conv_id, "assistant", tr.message)
            if self.settings.enable_voice:
                self.tts.speak(tr.message)
        self.chat_view.add_action_card(tool_name=tr.tool, message=tr.message, success=tr.success)

    def _on_route_finished(self, route_res: RouteResult) -> None:
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

        # Report steps that already ran (e.g. Tier 2 safe calls before a gate).
        # They already happened and are never repeated.
        for completed in route_res.completed_results:
            self._present_result(completed)

        if route_res.requires_confirmation:
            # Show Security Confirmation Dialog bound to the displayed action
            dlg = ConfirmationDialog(
                tool_name=route_res.tool_name,
                message=route_res.response_text,
                arguments=route_res.raw_arguments,
                parent=self
            )
            if dlg.exec():
                # Execute exactly the approved action; never re-resolve the query.
                self._execute_approved(route_res.pending_approval)
            else:
                self.chat_view.add_assistant_message("Action cancelled by user.")
            return

        if route_res.tool_result:
            self._present_result(route_res.tool_result)
        elif route_res.response_text:
            self.chat_view.add_assistant_message(route_res.response_text)
            self.history.add_message(self.current_conv_id, "assistant", route_res.response_text)

            # Speak output aloud via Windows TTS if enabled
            if self.settings.enable_voice:
                self.tts.speak(route_res.response_text)
