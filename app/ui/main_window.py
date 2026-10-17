"""
Main Application Window and Asynchronous Qt Worker Pipeline for Vector Desktop AI Assistant.
"""

from typing import Any, Dict, Optional
from PySide6.QtCore import QObject, QRunnable, QThreadPool, Qt, Signal, Slot
from PySide6.QtWidgets import (
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.config.settings import Settings, get_settings
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
from app.voice.listener import VoiceListener
from app.voice.tts import TTSEngine

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
        self.resize(800, 600)
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

        btn_history = QPushButton("🕑 History", header)
        btn_history.setStyleSheet("background-color: #313244; color: #cdd6f4; border-radius: 4px; padding: 4px 8px;")
        btn_history.clicked.connect(self._open_action_history)

        btn_health = QPushButton("🩺 Health", header)
        btn_health.setStyleSheet("background-color: #313244; color: #cdd6f4; border-radius: 4px; padding: 4px 8px;")
        btn_health.clicked.connect(self._open_system_health)

        btn_diagnostics = QPushButton("📋 Diagnostics", header)
        btn_diagnostics.setStyleSheet("background-color: #313244; color: #cdd6f4; border-radius: 4px; padding: 4px 8px;")
        btn_diagnostics.clicked.connect(self._open_diagnostics)

        btn_backup = QPushButton("💾 Backup", header)
        btn_backup.setStyleSheet("background-color: #313244; color: #cdd6f4; border-radius: 4px; padding: 4px 8px;")
        btn_backup.clicked.connect(self._open_conversation_io)

        header_layout.addWidget(lbl_title)
        header_layout.addStretch()
        header_layout.addWidget(btn_history)
        header_layout.addWidget(btn_health)
        header_layout.addWidget(btn_diagnostics)
        header_layout.addWidget(btn_backup)
        header_layout.addWidget(btn_settings)

        main_layout.addWidget(header)

        # Body: conversation sidebar on the left, chat on the right.
        body = QWidget(central)
        body_layout = QHBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(0)

        self.sidebar = ConversationSidebar(body)
        self.sidebar.setFixedWidth(240)
        self.sidebar.conversation_selected.connect(self._on_conversation_selected)
        self.sidebar.new_chat_requested.connect(self._on_new_chat)
        self.sidebar.rename_requested.connect(self._on_rename_conversation)
        self.sidebar.delete_requested.connect(self._on_delete_conversation)
        body_layout.addWidget(self.sidebar)

        right = QWidget(body)
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)

        # Chat View
        self.chat_view = ChatViewWidget(right)
        self.chat_view.load_earlier_requested.connect(self._on_load_earlier)
        right_layout.addWidget(self.chat_view, stretch=1)

        # Command Input
        self.command_input = CommandInputWidget(right)
        self.command_input.command_submitted.connect(self._handle_user_command)
        self.command_input.voice_toggled.connect(self._handle_voice_toggle)
        right_layout.addWidget(self.command_input)

        # Status Bar
        self.status_bar = StatusBarWidget(right)
        right_layout.addWidget(self.status_bar)

        body_layout.addWidget(right, stretch=1)
        main_layout.addWidget(body, stretch=1)

        # Populate the sidebar and render the active conversation (welcome text
        # only appears for an empty conversation).
        self.sidebar.set_history_manager(self.history)
        self._populate_sidebar()
        self._reload_active_conversation()

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
        """Render the first page of the active conversation."""
        self._loaded_messages = []
        self._page_offset = 0
        self._total_messages = 0
        self._has_older = False
        self._load_page(0)

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
        self.chat_view.set_load_more_visible(self._has_older)

        for message in self._loaded_messages:
            if not isinstance(message, dict):
                continue
            role = message.get("role")
            content = str(message.get("content", ""))
            if role == "user":
                self.chat_view.add_user_message(content)
            elif role == "assistant":
                self.chat_view.add_assistant_message(content)
            # Any other role (e.g. an inert imported 'system' row) is never
            # rendered and is never treated as an instruction.

        if not self._loaded_messages and self._total_messages == 0:
            self.chat_view.add_assistant_message(WELCOME_TEXT)

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
        self.chat_view.add_user_message(query)
        self.history.add_message(conversation_id, "user", query)

        worker = RouteWorker(self.router, query, conversation_id)
        worker.signals.result.connect(
            lambda route_res, cid=conversation_id: self._on_route_finished(route_res, cid)
        )
        worker.signals.error.connect(
            lambda err, cid=conversation_id: self._on_route_error(err, cid)
        )
        self.thread_pool.start(worker)

    def _is_active_conversation(self, conversation_id: Optional[str]) -> bool:
        """True when a result belongs to the conversation currently on screen."""
        return conversation_id is None or conversation_id == self.current_conv_id

    def _on_route_error(self, error: str, conversation_id: Optional[str] = None) -> None:
        """Surface a routing failure only in the conversation that raised it."""
        if self._is_active_conversation(conversation_id):
            self.chat_view.add_assistant_message(f"Error: {error}")

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
                self.chat_view.add_assistant_message(tr.message)
            self.history.add_message(conversation_id, "assistant", tr.message)
            if active and self.settings.enable_voice:
                self.tts.speak(tr.message)
        if active:
            self.chat_view.add_action_card(tool_name=tr.tool, message=tr.message, success=tr.success)

    def _on_route_finished(
        self, route_res: RouteResult, conversation_id: Optional[str] = None
    ) -> None:
        # ``conversation_id`` is the conversation the request was issued in; it
        # defaults to the active one for direct/manual callers.
        conversation_id = conversation_id or self.current_conv_id
        active = self._is_active_conversation(conversation_id)

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
        elif route_res.response_text:
            if active:
                self.chat_view.add_assistant_message(route_res.response_text)
            self.history.add_message(conversation_id, "assistant", route_res.response_text)

            # Speak output aloud via Windows TTS if enabled
            if active and self.settings.enable_voice:
                self.tts.speak(route_res.response_text)

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
