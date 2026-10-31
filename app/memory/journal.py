"""
Action Journal for Vector Desktop AI Assistant.

Records what Vector actually attempted and executed -- the real outcome of each
tool call, not a model's stated intention -- into the existing ``tool_calls``
SQLite table via :class:`~app.memory.history.HistoryManager`.

Design notes
------------
* Journaling is centralised one level below the entry points: every execution
  funnels through ``SafeToolExecutor.execute_tool``, so typed commands, voice
  commands and Gemini's multi-tool loop are all covered by the same mechanism.
* Recording is strictly best-effort.  A storage failure is swallowed so it can
  never crash the assistant or cause an already-executed tool to run again.
* Arguments and summaries are passed through the existing
  :class:`~app.core.privacy.PrivacyFilter` before storage, and full tool output
  (``ToolResult.data``) is never stored -- only a short, sanitized summary.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from PySide6.QtCore import QCoreApplication, QObject, Signal

from app.core.privacy import PrivacyFilter


class JournalStatus:
    """Status values written to ``tool_calls.execution_status``.

    These distinguish real outcomes; a value is never ``SUCCESS`` unless the
    tool actually completed successfully.
    """

    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    PERMISSION_DENIED = "PERMISSION_DENIED"
    CONFIRMATION_REQUIRED = "CONFIRMATION_REQUIRED"
    INVALID_ARGUMENTS = "INVALID_ARGUMENTS"
    TOOL_NOT_FOUND = "TOOL_NOT_FOUND"
    APPROVAL_REJECTED = "APPROVAL_REJECTED"
    CANCELLED = "CANCELLED"


#: Summaries are truncated so a verbose tool message cannot dump full output.
MAX_SUMMARY_LENGTH = 500


class JournalBridge(QObject):
    """
    Qt Signal Bridge connecting Action Journal executions to UI widgets
    across thread boundaries.
    """

    action_recorded = Signal(str, str, bool, str)  # tool_name, summary, success, timestamp

    _instance: Optional[JournalBridge] = None

    @classmethod
    def instance(cls) -> JournalBridge:
        if cls._instance is None:
            cls._instance = JournalBridge()
        return cls._instance


class ExecutionJournal:
    """
    Sanitizing, fail-safe writer for one conversation's Action Journal.

    The journal holds a ``HistoryManager`` and an optional ``conversation_id``
    and exposes a single :meth:`record` entry point.  Any failure while
    persisting an entry is contained here and reported as ``None``.
    """

    def __init__(
        self,
        history_manager: Optional[Any] = None,
        conversation_id: Optional[str] = None,
        max_summary_length: int = MAX_SUMMARY_LENGTH,
    ):
        self.history = history_manager
        self.conversation_id = conversation_id
        self.max_summary_length = max_summary_length

    def record(
        self,
        tool_name: str,
        arguments: Optional[Dict[str, Any]] = None,
        status: str = "",
        summary: str = "",
        confidence: float = 0.0,
    ) -> Optional[str]:
        """
        Persists one journal entry and returns its id, or ``None`` on failure.

        Never raises: sanitization, serialization and storage errors all degrade
        to ``None`` so the execution path is unaffected.
        """
        if self.history is None:
            return None

        add_execution = getattr(self.history, "add_execution", None)
        if add_execution is None:
            return None

        try:
            safe_arguments = (
                PrivacyFilter.sanitize_context(dict(arguments))
                if isinstance(arguments, dict)
                else {}
            )
            safe_summary = PrivacyFilter.sanitize_text(str(summary or ""))
            if self.max_summary_length and len(safe_summary) > self.max_summary_length:
                safe_summary = safe_summary[: self.max_summary_length]

            rec_id = add_execution(
                tool_name=str(tool_name or ""),
                arguments=safe_arguments,
                status=str(status or ""),
                summary=safe_summary,
                conversation_id=self.conversation_id,
                confidence=float(confidence or 0.0),
            )

            # Signal live telemetry bridge if Qt event loop is active
            try:
                if QCoreApplication.instance() is not None:
                    from app.ui.theme import format_timestamp

                    is_success = (str(status).upper() == JournalStatus.SUCCESS)
                    now_str = format_timestamp()
                    JournalBridge.instance().action_recorded.emit(
                        str(tool_name or ""),
                        safe_summary,
                        is_success,
                        now_str,
                    )
            except Exception:
                pass

            return rec_id
        except Exception:
            return None
