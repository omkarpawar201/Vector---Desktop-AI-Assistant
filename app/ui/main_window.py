"""
Main Application Window and Asynchronous Qt Worker Pipeline for Vector Desktop AI Assistant.
"""

from typing import Any, Dict, Optional
from uuid import uuid4

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Qt, Signal, Slot
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from app.config.settings import Settings, get_settings
from app.core.cancellation import CancellationToken
from app.core.router import IntentRouter, RouteResult
from app.memory.history import (
    DEFAULT_CONVERSATION_TITLE,
    HistoryManager,
)
from app.memory.journal import ExecutionJournal
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
from app.ui.action_history import ActionHistoryDialog
from app.ui.chat_view import ChatViewWidget
from app.ui.command_input import CommandInputWidget
from app.ui.confirmation_dialog import ConfirmationDialog
from app.ui.conversation_io import ConversationIODialog
from app.ui.conversation_sidebar import ConversationSidebar
from app.ui.diagnostics import DiagnosticsDialog
from app.ui.settings import SettingsDialog
from app.ui.status_bar import StatusBarWidget
from app.ui.system_health import SystemHealthDialog
from app.ui.widgets import (
    HeroBannerWidget,
    NavSidebarWidget,
    QuickChipsWidget,
    RecentActionsWidget,
    SystemStatusWidget,
)
from app.voice.listener import VoiceListener
try:
    from app.voice.tts import TTSEngine
except Exception:
    import logging as _logging

    _logger = _logging.getLogger(__name__)

    class TTSEngine:  # type: ignore
        def __init__(self, *args, **kwargs):
            self._enabled = False
            self._available = False

        def speak(self, text: str) -> None:
            return

        def set_enabled(self, enabled: bool) -> None:
            self._enabled = False

        @property
        def is_enabled(self) -> bool:
            return False

        @property
        def is_available(self) -> bool:
            return False

#: Messages loaded per page when a conversation is opened.  Older pages are
#: fetched on demand ("Load earlier messages") so long conversations are never
#: silently truncated at a fixed limit.
MESSAGE_PAGE_SIZE = 50
#: Shown only for an empty conversation, so switching to a blank chat is not
#: mistaken for a loading failure.
WELCOME_TEXT = "Hello! I am Vector, your desktop AI assistant. How can I help you today?"


class WorkerSignals(QObject):
    """Signals for background Qt workers (routing and microphone capture)."""
    result = Signal(object)
    error = Signal(str)
    ready = Signal()
    #: Emitted when a provider actually begins streaming an answer.  Carries no
    #: payload: the window already bound the request/conversation at dispatch.
    stream_started = Signal()
    #: One incremental text delta of the assistant answer.
    chunk = Signal(str)


class RouteWorker(QRunnable):
    """Background worker executing IntentRouter pipeline off the Qt main UI thread."""

    def __init__(
        self,
        router: IntentRouter,
        user_query: str,
        conversation_id: Optional[str] = None,
        request_id: Optional[str] = None,
        cancellation_token: Optional[CancellationToken] = None,
    ):
        super().__init__()
        self.router = router
        self.user_query = user_query
        self.conversation_id = conversation_id
        self.request_id = request_id
        self.cancellation_token = cancellation_token
        self.signals = WorkerSignals()

    @Slot()
    def run(self) -> None:
        try:
            route_res = self.router.route_and_execute(
                user_query=self.user_query,
                conversation_id=self.conversation_id,
                on_chunk=self.signals.chunk.emit,
                on_stream_start=self.signals.stream_started.emit,
                cancellation_token=self.cancellation_token,
                request_id=self.request_id,
            )
            self.signals.result.emit(route_res)
        except Exception as e:
            self.signals.error.emit(str(e))


class ApprovalWorker(QRunnable):
    """Background worker executing one user-approved action off the UI thread."""

    def __init__(self, router: IntentRouter, approval: PendingApproval, conversation_id: Optional[str] = None):
        super().__init__()
        self.router = router
        self.approval = approval
        self.conversation_id = conversation_id
        self.signals = WorkerSignals()

    @Slot()
    def run(self) -> None:
        try:
            tool_result = self.router.execute_approved(self.approval, conversation_id=self.conversation_id)
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
        self.resize(1180, 760)
        self.setMinimumSize(980, 620)
        self.settings = get_settings()

        # Initialize and register all local tools
        self.registry = get_tool_registry()
        self._register_default_tools()

        # The active conversation must exist before the router so Tier 2 can be
        # handed its recent turns as follow-up context.  When the database
        # already holds conversations the most recently active one is reopened;
        # otherwise a fresh chat is created.
        self.history = HistoryManager()
        self._init_active_conversation()

        self.router = IntentRouter(
            settings=self.settings,
            registry=self.registry,
            history_manager=self.history,
        )

        self.tts = TTSEngine()
        self.voice_listener = VoiceListener()

        self.thread_pool = QThreadPool.globalInstance()

        # ---- Streaming generation tracking ----------------------------------
        # The latest request id whose chunks may still be rendered; a newer
        # request replaces it so stale workers cannot write into the active chat.
        self._active_request_id: Optional[str] = None
        # The generation the Stop control refers to.  Captured at submit time so
        # a stop can only ever cancel the generation it was shown for.
        self._active_generation: Optional[Dict[str, Any]] = None
        # Request ids whose assistant text has already been persisted, so a
        # completion/cancellation race can never store a response twice.
        self._persisted_requests: set = set()
        # In-progress streamed bubbles keyed by the request that opened them, so
        # an overlapping newer request can never make an older result finalize
        # the wrong bubble, and each result closes exactly its own stream.
        self._live_streams: Dict[str, Any] = {}

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

    def closeEvent(self, event: QCloseEvent) -> None:
        """
        Stop a generation cleanly when the window is closed.

        Closing the window must never leave a generation running: a live
        cancellation token is cancelled so a worker stops at its next checkpoint
        and cannot execute a further action after the window is gone.  Live
        stream handles are dropped because their widgets die with the window.
        """
        generation = self._active_generation
        if generation is not None:
            try:
                generation["token"].cancel()
            except Exception:
                pass
        self._live_streams.clear()
        if hasattr(self, "system_status") and hasattr(self.system_status, "stop_polling"):
            self.system_status.stop_polling()
        if hasattr(self, "status_bar") and hasattr(self.status_bar, "timer"):
            self.status_bar.timer.stop()
        super().closeEvent(event)

    def _setup_ui(self) -> None:
        self.setStyleSheet("""
            QMainWindow {
                background-color: #0b0d14;
            }
        """)

        central = QWidget(self)
        self.setCentralWidget(central)

        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(10, 10, 10, 6)
        main_layout.setSpacing(8)

        # ------------------------------------------------------------------
        # 3-Column Dashboard Body
        # ------------------------------------------------------------------
        body = QWidget(central)
        body_layout = QHBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(10)

        # Column 1: Left Navigation & Quick Actions Stack (width: 240)
        self.left_stack = QStackedWidget(body)
        self.left_stack.setFixedWidth(240)

        # Page 0: NavSidebarWidget
        self.nav_sidebar = NavSidebarWidget(self.left_stack)
        self.nav_sidebar.chat_clicked.connect(self._on_nav_chat)
        self.nav_sidebar.conversations_clicked.connect(lambda: self.left_stack.setCurrentIndex(1))
        self.nav_sidebar.actions_clicked.connect(self._open_action_history)
        self.nav_sidebar.system_info_clicked.connect(self._open_system_health)
        self.nav_sidebar.settings_clicked.connect(self._open_settings)
        self.nav_sidebar.media_clicked.connect(lambda: self._handle_user_command("media play"))
        self.nav_sidebar.apps_clicked.connect(lambda: self._handle_user_command("get running apps"))
        self.nav_sidebar.quick_action_triggered.connect(self._handle_user_command)

        # Page 1: Conversation Sessions Manager
        self.sidebar_container = QWidget(self.left_stack)
        sidebar_cont_layout = QVBoxLayout(self.sidebar_container)
        sidebar_cont_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_cont_layout.setSpacing(6)

        btn_back = QPushButton("← Back to Menu", self.sidebar_container)
        btn_back.setStyleSheet("""
            QPushButton {
                background-color: #141724;
                color: #38bdf8;
                border: 1px solid #1e2235;
                border-radius: 8px;
                padding: 6px 12px;
                font-size: 12px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #1e2235;
                color: #ffffff;
            }
        """)
        btn_back.clicked.connect(lambda: self.left_stack.setCurrentIndex(0))
        sidebar_cont_layout.addWidget(btn_back)

        self.sidebar = ConversationSidebar(self.sidebar_container)
        self.sidebar.conversation_selected.connect(self._on_conversation_selected)
        self.sidebar.new_chat_requested.connect(self._on_new_chat)
        self.sidebar.rename_requested.connect(self._on_rename_conversation)
        self.sidebar.delete_requested.connect(self._on_delete_conversation)
        sidebar_cont_layout.addWidget(self.sidebar, stretch=1)

        self.left_stack.addWidget(self.nav_sidebar)       # Page 0
        self.left_stack.addWidget(self.sidebar_container) # Page 1
        body_layout.addWidget(self.left_stack)

        # Column 2: Center Interaction Hub (Hero Banner + Chat + Composer + Chips)
        center = QWidget(body)
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(8)

        # Hero Banner
        self.hero_banner = HeroBannerWidget(center)
        center_layout.addWidget(self.hero_banner)

        # Chat View
        self.chat_view = ChatViewWidget(center)
        self.chat_view.load_earlier_requested.connect(self._on_load_earlier)
        center_layout.addWidget(self.chat_view, stretch=1)

        # Command Input
        self.command_input = CommandInputWidget(center)
        self.command_input.command_submitted.connect(self._handle_user_command)
        self.command_input.voice_toggled.connect(self._handle_voice_toggle)
        self.command_input.stop_requested.connect(self._on_stop_generation)
        center_layout.addWidget(self.command_input)

        # Quick Prompt Chips
        self.quick_chips = QuickChipsWidget(center)
        self.quick_chips.chip_clicked.connect(self._handle_user_command)
        center_layout.addWidget(self.quick_chips)

        body_layout.addWidget(center, stretch=1)

        # Column 3: Right Telemetry & Activity Panel (width: 290)
        right = QWidget(body)
        right.setFixedWidth(290)
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(10)

        # System Status (2x2 circular ring gauges)
        self.system_status = SystemStatusWidget(right)
        right_layout.addWidget(self.system_status)

        # Recent Actions feed
        self.recent_actions = RecentActionsWidget(right)
        self.recent_actions.connect_journal()
        right_layout.addWidget(self.recent_actions, stretch=1)

        body_layout.addWidget(right)

        main_layout.addWidget(body, stretch=1)

        # Bottom Status Bar
        self.status_bar = StatusBarWidget(central)
        self.status_bar.metrics_updated.connect(self.system_status.update_metrics)
        main_layout.addWidget(self.status_bar)

        # Populate the sidebar and render the active conversation (welcome text
        # only appears for an empty conversation).
        self.sidebar.set_history_manager(self.history)
        self._populate_sidebar()
        self._reload_active_conversation()

    def _on_nav_chat(self) -> None:
        self.left_stack.setCurrentIndex(0)

    # ==================================================================
    # Conversation management
    # ==================================================================
    def _init_active_conversation(self) -> None:
        """
        Resolve the active conversation and reset pagination state.

        Reopens the most recently active stored conversation, or creates a fresh
        one when the database is empty.  Only conversation metadata is read here
        (never messages), so it is cheap enough to run synchronously during
        construction.
        """
        self._loaded_messages: list = []
        self._page_offset = 0
        self._total_messages = 0
        self._has_older = False

        conversations = self._safe_list_conversations()
        active = conversations[0].get("id") if conversations else None
        self.current_conv_id = active or self.history.create_conversation(
            DEFAULT_CONVERSATION_TITLE
        )

    def _safe_list_conversations(self) -> list:
        """Conversation summaries via HistoryManager, degrading to ``[]``."""
        lister = getattr(self.history, "list_conversations", None)
        if lister is None:
            return []
        try:
            return list(lister() or [])
        except Exception:
            return []

    def _populate_sidebar(self) -> None:
        """Synchronously refill the sidebar (used after create/rename/delete)."""
        self.sidebar.set_history_manager(self.history)
        self.sidebar.set_conversations(self._safe_list_conversations())
        self.sidebar.set_active(self.current_conv_id)

    def _on_conversation_selected(self, conversation_id: str) -> None:
        if not conversation_id or conversation_id == self.current_conv_id:
            return
        self._switch_to_conversation(conversation_id)

    def _switch_to_conversation(self, conversation_id: str) -> None:
        """Make ``conversation_id`` active, handling a missing/deleted id."""
        if not self._conversation_exists(conversation_id):
            # The id no longer exists (deleted elsewhere): never leave the UI
            # pointing at a dangling conversation.
            self._switch_to_safe_replacement(exclude=conversation_id)
            self._populate_sidebar()
            return

        self.current_conv_id = conversation_id
        self._reload_active_conversation()
        self.sidebar.set_active(conversation_id)

    def _conversation_exists(self, conversation_id: str) -> bool:
        checker = getattr(self.history, "conversation_exists", None)
        if checker is None:
            return True  # nothing to verify against
        try:
            return bool(checker(conversation_id))
        except Exception:
            return True

    def _switch_to_safe_replacement(self, exclude: Optional[str]) -> None:
        """
        Activate another conversation (or a brand-new chat) instead of a
        deleted/absent one, so the active id is always valid.
        """
        replacement = next(
            (
                c.get("id")
                for c in self._safe_list_conversations()
                if c.get("id") and c.get("id") != exclude
            ),
            None,
        )
        if replacement:
            self.current_conv_id = replacement
            self._reload_active_conversation()
            self.sidebar.set_active(replacement)
        else:
            self._on_new_chat()

    def _on_new_chat(self) -> None:
        """Create and switch to a new empty conversation."""
        self.current_conv_id = self.history.create_conversation(
            DEFAULT_CONVERSATION_TITLE
        )
        self._reload_active_conversation()
        self._populate_sidebar()

    def _conversation_title(self, conversation_id: str) -> str:
        """A conversation's stored title, or a neutral placeholder."""
        getter = getattr(self.history, "get_conversation", None)
        if getter is None:
            return "this conversation"
        try:
            metadata = getter(conversation_id)
        except Exception:
            metadata = None
        if isinstance(metadata, dict) and metadata.get("title"):
            return str(metadata["title"])
        return "this conversation"

    def _on_rename_conversation(self, conversation_id: str) -> None:
        if not conversation_id:
            return
        current_title = self._conversation_title(conversation_id)
        new_title, accepted = self._prompt_title(current_title)
        if not accepted:
            return  # no rename requested: preserve the current title
        try:
            self.history.rename_conversation(conversation_id, new_title)
        except Exception:
            self._show_error("Could not rename the conversation. Please try again.")
        self._populate_sidebar()

    def _prompt_title(self, current_title: str):
        """Ask for a new title. Seam so tests can bypass the modal dialog."""
        return QInputDialog.getText(
            self,
            "Rename Conversation",
            "Title:",
            QLineEdit.Normal,
            current_title,
        )

    def _on_delete_conversation(self, conversation_id: str) -> None:
        """
        Delete one conversation after explicit confirmation.

        When the active conversation is deleted a safe replacement (or a new
        chat) is activated *first*, so the active id is never a deleted one.
        """
        if not conversation_id:
            return
        title = self._conversation_title(conversation_id)
        if not self._confirm_delete(title):
            return

        if conversation_id == self.current_conv_id:
            self._switch_to_safe_replacement(exclude=conversation_id)

        try:
            self.history.delete_conversation(conversation_id)
        except Exception:
            self._show_error(
                "Could not delete the conversation. It may still be available."
            )
        self._populate_sidebar()

    def _confirm_delete(self, title: str) -> bool:
        """Explicit confirmation. Seam so tests can bypass the modal dialog."""
        reply = QMessageBox.question(
            self,
            "Delete Conversation",
            f'Delete "{title}"?\n\n'
            "This permanently removes the conversation, its messages and its "
            "action history. This cannot be undone.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        return reply == QMessageBox.Yes

    def _show_error(self, message: str) -> None:
        """Surface a safe, non-fatal error. Seam so tests can capture it."""
        QMessageBox.warning(self, "Conversation", str(message))

    # ==================================================================
    # Message loading / pagination
    # ==================================================================
    def _reload_active_conversation(self) -> None:
        """Render the first page of the active conversation and load recent actions."""
        self._loaded_messages = []
        self._page_offset = 0
        self._total_messages = 0
        self._has_older = False
        self._load_page(0)
        try:
            if hasattr(self, "recent_actions"):
                self.recent_actions.load_history(
                    self.history, conversation_id=self.current_conv_id
                )
        except Exception:
            pass

    def _on_load_earlier(self) -> None:
        """Fetch and prepend the next older page."""
        if not self._has_older:
            return
        self._load_page(self._page_offset)

    def _load_page(self, offset: int) -> None:
        conversation_id = self.current_conv_id
        page = self._safe_get_page(conversation_id, MESSAGE_PAGE_SIZE, offset)

        if offset == 0:
            self._loaded_messages = list(page)
            self._total_messages = self._safe_count_messages(
                conversation_id, fallback=len(page)
            )
        else:
            self._loaded_messages = list(page) + self._loaded_messages

        self._page_offset = len(self._loaded_messages)
        self._has_older = self._page_offset < self._total_messages
        self._render_loaded_messages()

    def _safe_get_page(self, conversation_id: Optional[str], limit: int, offset: int) -> list:
        reader = getattr(self.history, "get_conversation_page", None)
        if reader is None or not conversation_id:
            return []
        try:
            return list(reader(conversation_id, limit=limit, offset=offset) or [])
        except Exception:
            return []

    def _safe_count_messages(self, conversation_id: Optional[str], fallback: int) -> int:
        counter = getattr(self.history, "count_conversation_messages", None)
        if counter is None or not conversation_id:
            return fallback
        try:
            return int(counter(conversation_id))
        except Exception:
            return fallback

    def _render_loaded_messages(self) -> None:
        """Redraw the chat view from the currently loaded messages."""
        self.chat_view.clear()
        # Any in-progress streamed bubble lived in the view we just cleared, so
        # its handle now points at deleted Qt widgets.  Drop them here so a late
        # chunk/finalize can never touch a destroyed C++ object (which would
        # raise "Internal C++ object already deleted").  A subsequent result for
        # a still-active request is rendered as a normal message instead.
        self._live_streams.clear()
        self.chat_view.set_load_more_visible(self._has_older)

        for message in self._loaded_messages:
            if not isinstance(message, dict):
                continue
            role = message.get("role")
            content = str(message.get("content", ""))
            ts = message.get("timestamp")
            if role == "user":
                self.chat_view.add_user_message(content, timestamp=ts)
            elif role == "assistant":
                self.chat_view.add_assistant_message(content, timestamp=ts)
            # Any other role (e.g. an inert imported 'system' row) is never
            # rendered and is never treated as an instruction.

        if not self._loaded_messages and self._total_messages == 0:
            self.chat_view.add_welcome_message(WELCOME_TEXT)

    def _open_settings(self) -> None:
        dialog = SettingsDialog(self)
        dialog.exec()

    def _open_action_history(self) -> None:
        """Open the read-only Action History view for the active conversation."""
        dialog = ActionHistoryDialog(self.history, self.current_conv_id, parent=self)
        dialog.exec()

    def _open_system_health(self) -> None:
        """Open the read-only System Health dashboard."""
        dialog = SystemHealthDialog(self.settings, self.history, parent=self)
        dialog.exec()

    def _open_diagnostics(self) -> None:
        """Open the read-only Diagnostics & Troubleshooting view."""
        dialog = DiagnosticsDialog(self.settings, self.history, parent=self)
        dialog.exec()

    def _open_conversation_io(self) -> None:
        """Open the Conversation Backup dialog (export / import)."""
        dialog = ConversationIODialog(self.history, self.current_conv_id, parent=self)
        dialog.conversation_imported.connect(self._on_conversation_imported)
        dialog.exec()

    def _on_conversation_imported(self, conversation_id: str) -> None:
        """
        Make a freshly imported conversation the active one.

        The imported messages are read back from storage and rendered so the
        restored conversation is immediately usable through the existing
        conversation workflow.  Imported text is only displayed, never routed.
        """
        if not conversation_id:
            return
        self.current_conv_id = conversation_id
        self._reload_active_conversation()
        self._populate_sidebar()

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
        # Bind the request to the conversation that is active *now*.  Every
        # later step (persistence, rendering, journal) is keyed off this id, so
        # a result can never leak into a conversation selected afterwards.
        conversation_id = self.current_conv_id
        # A unique id and a per-request cancellation token isolate this
        # generation from every other one: a stale worker's chunks carry the
        # wrong id and are ignored, and a later Stop can only cancel the token it
        # captured (never a newer generation's).
        request_id = str(uuid4())
        token = CancellationToken()
        self._active_request_id = request_id
        self._active_generation = {
            "request_id": request_id,
            "conversation_id": conversation_id,
            "token": token,
        }

        self.chat_view.add_user_message(query)
        self.history.add_message(conversation_id, "user", query)

        worker = RouteWorker(
            self.router,
            query,
            conversation_id,
            request_id=request_id,
            cancellation_token=token,
        )
        worker.signals.chunk.connect(
            lambda text, rid=request_id, cid=conversation_id: self._on_chunk(rid, cid, text)
        )
        worker.signals.stream_started.connect(
            lambda rid=request_id, cid=conversation_id: self._on_stream_started(rid, cid)
        )
        worker.signals.result.connect(
            lambda route_res, cid=conversation_id, rid=request_id: self._on_route_finished(route_res, cid, rid)
        )
        worker.signals.error.connect(
            lambda err, cid=conversation_id, rid=request_id: self._on_route_error(err, cid, rid)
        )
        self.thread_pool.start(worker)

    # ==================================================================
    # Streaming / Stop generation
    # ==================================================================
    def _on_stream_started(self, request_id: Optional[str], conversation_id: Optional[str]) -> None:
        """
        A provider began streaming.  Show Stop and open the live bubble.

        Only the newest request may drive the UI, so a stream that started after
        the user already sent a newer command is ignored.  The bubble is stored
        under its request id so a superseded stream never finalizes another's.
        """
        if request_id is None or request_id != self._active_request_id:
            return
        self.command_input.set_stop_visible(True)
        if self._is_active_conversation(conversation_id):
            self._live_streams[request_id] = self.chat_view.begin_assistant_stream()

    def _on_chunk(self, request_id: Optional[str], conversation_id: Optional[str], text: str) -> None:
        """Append one text delta, but only for the active request/conversation."""
        if request_id is None or request_id != self._active_request_id:
            return
        if not self._is_active_conversation(conversation_id):
            return
        self.chat_view.append_assistant_stream(self._live_streams.get(request_id), text)

    def _on_stop_generation(self) -> None:
        """
        Handle the Stop control without blocking the UI.

        Cancels the exact generation the control was shown for (captured at
        submit time), so it can never cancel a later generation.  The result
        still arrives normally and is finalized as a cancelled/partial response.
        """
        self.command_input.set_stop_visible(False)
        generation = self._active_generation
        if generation is not None:
            generation["token"].cancel()

    def _hide_stop(self) -> None:
        self.command_input.set_stop_visible(False)

    def _hide_stop_if_current(self, request_id: Optional[str]) -> None:
        """
        Hide the Stop control only for the generation it was shown for.

        A handler that supplies no request id (a direct/manual caller or the
        single-use approval path) keeps the original behaviour.  A *stale*
        request id must never hide the Stop of the newer generation that is
        still streaming.
        """
        if request_id is None or request_id == self._active_request_id:
            self._hide_stop()

    def _finish_generation(self, request_id: Optional[str]) -> None:
        """
        Clear generation tracking once its result/error has been handled.

        A direct/manual caller that supplies no request id changes nothing, so
        it cannot tear down an unrelated in-flight generation.
        """
        if request_id is None:
            return
        if request_id == self._active_request_id:
            self._active_request_id = None
        generation = self._active_generation
        if generation is not None and generation.get("request_id") == request_id:
            self._active_generation = None

    def _streamed_for(self, request_id: Optional[str]) -> bool:
        """True when this request opened an in-progress streamed bubble."""
        return request_id is not None and request_id in self._live_streams

    def _finalize_streamed(
        self, route_res: RouteResult, conversation_id: Optional[str], request_id: Optional[str], active: bool
    ) -> None:
        """
        Close this request's streamed bubble with the authoritative final text.

        The finalized body is exactly ``route_res.response_text`` (what is also
        persisted), plus a short plain-text marker when the output is incomplete.
        The bubble is popped by request id, so only this request can close it.
        """
        handle = self._live_streams.pop(request_id, None) if request_id is not None else None
        suffix = ""
        if getattr(route_res, "cancelled", False):
            suffix = " (stopped)"
        elif getattr(route_res, "partial", False):
            suffix = " (incomplete)"

        if active:
            if handle is not None:
                self.chat_view.finalize_assistant_stream(handle, route_res.response_text, suffix)
            elif route_res.response_text:
                self.chat_view.add_assistant_message(route_res.response_text + suffix)

    def _discard_live_stream(self, request_id: Optional[str]) -> None:
        """Drop this request's live bubble reference without persisting anything."""
        if request_id is not None:
            self._live_streams.pop(request_id, None)

    def _persist_assistant_once(
        self, request_id: Optional[str], conversation_id: Optional[str], text: str
    ) -> None:
        """
        Persist one assistant response, at most once per request id.

        A direct caller that supplies no request id keeps the original
        un-deduplicated behaviour.
        """
        if request_id is not None:
            if request_id in self._persisted_requests:
                return
            self._persisted_requests.add(request_id)
        try:
            self.history.add_message(conversation_id, "assistant", text)
        except Exception:
            pass

    def _is_active_conversation(self, conversation_id: Optional[str]) -> bool:
        """True when a result belongs to the conversation currently on screen."""
        return conversation_id is None or conversation_id == self.current_conv_id

    def _on_route_error(
        self,
        error: str,
        conversation_id: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> None:
        """Surface a routing failure only in the conversation that raised it."""
        self._hide_stop_if_current(request_id)
        # Close any streamed bubble this request opened; never render a stale
        # earlier request's output in the active chat.
        if request_id is not None and request_id in self._live_streams:
            self.chat_view.finalize_assistant_stream(
                self._live_streams.pop(request_id), None, " (failed)"
            )
        if self._is_active_conversation(conversation_id):
            self.chat_view.add_assistant_message(f"Error: {error}")
        self._finish_generation(request_id)

    def _execute_approved(
        self, approval: Optional[PendingApproval], conversation_id: Optional[str] = None
    ) -> None:
        """Run exactly the action the user approved (single-use token)."""
        conversation_id = conversation_id or self.current_conv_id
        if approval is None:
            if self._is_active_conversation(conversation_id):
                self.chat_view.add_assistant_message(
                    "This action could not be confirmed safely. Please try again."
                )
            return

        worker = ApprovalWorker(self.router, approval, conversation_id)
        worker.signals.result.connect(
            lambda tool_result, cid=conversation_id: self._on_approved_finished(tool_result, cid)
        )
        worker.signals.error.connect(
            lambda err, cid=conversation_id: self._on_route_error(err, cid)
        )
        self.thread_pool.start(worker)

    def _on_approved_finished(
        self, tool_result: ToolResult, conversation_id: Optional[str] = None
    ) -> None:
        self._present_result(tool_result, conversation_id)

    def _present_result(
        self, tr: Optional[ToolResult], conversation_id: Optional[str] = None
    ) -> None:
        """
        Surface one executed tool result as a message plus an action card.

        The assistant message is always persisted to the conversation the
        action belongs to; it is only *rendered* when that conversation is the
        one currently displayed.
        """
        if tr is None:
            return
        conversation_id = conversation_id or self.current_conv_id
        active = self._is_active_conversation(conversation_id)

        if tr.message:
            if active:
                try:
                    self.chat_view.add_assistant_message(tr.message)
                except Exception:
                    pass
            try:
                self.history.add_message(conversation_id, "assistant", tr.message)
            except Exception:
                # History persistence failure must never crash the UI callback,
                # repeat side effects, or falsify tool execution status.
                pass
            if active and self.settings.enable_voice:
                try:
                    if getattr(self.tts, "is_available", True):
                        self.tts.speak(tr.message)
                except Exception:
                    pass
        if active:
            try:
                self.chat_view.add_action_card(tool_name=tr.tool, message=tr.message, success=tr.success)
            except Exception:
                pass
            try:
                if hasattr(self, "recent_actions"):
                    self.recent_actions.add_action(tr.tool, tr.message, tr.success)
            except Exception:
                pass

    def _on_route_finished(
        self,
        route_res: RouteResult,
        conversation_id: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> None:
        # ``conversation_id`` is the conversation the request was issued in; it
        # defaults to the active one for direct/manual callers.
        conversation_id = conversation_id or self.current_conv_id
        active = self._is_active_conversation(conversation_id)
        self._hide_stop_if_current(request_id)

        # Optionally surface the exactly-resolved tier (and its confidence) to the
        # terminal for every command — no UI clutter, just the routing fact.
        if self.show_tier_in_terminal:
            # RouteResult does not define a confidence field (tier-level
            # confidence only exists inside some execution_trace steps), so read
            # it defensively rather than assuming the attribute exists.
            confidence = getattr(route_res, "confidence", None)
            # flush=True is essential: this GUI app never exits, so an unflushed
            # stdout buffer would swallow every [tier] line until the process dies.
            print(
                f"[tier] {route_res.tier_used.value!s:18} "
                f"tool={route_res.tool_name or '-'} "
                f"conf={confidence if confidence else '-'}",
                flush=True,
            )

        # Report steps that already ran (e.g. Tier 2 safe calls before a gate).
        # They already happened and are never repeated.
        for completed in route_res.completed_results:
            self._present_result(completed, conversation_id)

        if route_res.requires_confirmation:
            # Close this request's streamed bubble (if any) before prompting, so
            # the pending action never leaves a live bubble behind.  The
            # non-streaming confirmation path opened no bubble and is unchanged.
            if self._streamed_for(request_id):
                self._finalize_streamed(route_res, conversation_id, request_id, active)
            self._finish_generation(request_id)
            if not active:
                # A confirmation for a conversation the user has since left is
                # dropped (fail closed): no modal appears over the wrong chat and
                # nothing is executed.
                return
            # Show Security Confirmation Dialog bound to the displayed action
            dlg = ConfirmationDialog(
                tool_name=route_res.tool_name,
                message=route_res.response_text,
                arguments=route_res.raw_arguments,
                parent=self
            )
            if dlg.exec():
                # Execute exactly the approved action; never re-resolve the query.
                self._execute_approved(route_res.pending_approval, conversation_id)
            else:
                self.chat_view.add_assistant_message("Action cancelled by user.")
                self._journal_cancellation(route_res, conversation_id)
            return

        if route_res.tool_result:
            self._present_result(route_res.tool_result, conversation_id)
            self._finish_generation(request_id)
            return

        if route_res.response_text:
            cancelled = getattr(route_res, "cancelled", False)
            partial = getattr(route_res, "partial", False)
            if self._streamed_for(request_id):
                # A streamed answer is finalized in place (never as a new
                # message), so the visible bubble and the stored row are the same
                # text.  Partial/cancelled output is displayed but not persisted.
                if active:
                    self._finalize_streamed(route_res, conversation_id, request_id, active)
                else:
                    self._discard_live_stream(request_id)
            elif active:
                self.chat_view.add_assistant_message(route_res.response_text)

            if not partial and not cancelled:
                self._persist_assistant_once(request_id, conversation_id, route_res.response_text)

            # Speak output aloud via Windows TTS if enabled (never a stopped or
            # incomplete answer).
            if active and not (cancelled or partial) and self.settings.enable_voice:
                self.tts.speak(route_res.response_text)

        self._finish_generation(request_id)

    def _journal_cancellation(
        self, route_res: Optional[RouteResult], conversation_id: Optional[str] = None
    ) -> None:
        """
        Record a cancelled confirmation as an Action Journal entry.

        Scoped to the conversation the request was issued in.  Best-effort: a
        journaling failure never crashes the UI path.
        """
        if route_res is None:
            return
        conversation_id = conversation_id or self.current_conv_id
        try:
            journal = ExecutionJournal(self.history, conversation_id)
            journal.record(
                tool_name=route_res.tool_name,
                arguments=route_res.raw_arguments,
                status="CANCELLED",
                summary="Action cancelled by user.",
            )
        except Exception:
            pass
