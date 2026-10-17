"""
Conversation Export & Import for Vector Desktop AI Assistant.

This module adds a documented, versioned JSON backup format for a *single*
conversation.  It reuses the existing database layer
(:class:`~app.memory.history.HistoryManager`) and its APIs -- it does not
introduce another persistence system, another schema, or another config format.

Design guarantees
-----------------
* **Scoped.**  Export reads only the named conversation's messages.  Action
  Journal rows (``tool_calls``), application configuration, API keys and every
  other conversation are never included.
* **Faithful.**  Message order, roles, content and timestamps are preserved.
  Ordinary conversation text is written verbatim -- it is *not* silently
  redacted or rewritten on export.
* **Untrusted input.**  Imported content is data, never instructions.  Nothing
  in this module executes a tool, calls Gemini or triggers AI routing; a
  ``system`` role from a file is stored as inert text.
* **Transactional.**  Import inserts the conversation and all of its messages
  in one transaction (see :meth:`HistoryManager.import_conversation`); a failure
  rolls the whole thing back.
* **Safe failures.**  Every failure is reported as a short, non-sensitive
  message; imported content is never echoed into an error string.

JSON format (``version`` 1)
---------------------------
::

    {
      "format": "vector.conversation",
      "version": 1,
      "exported_at": "2026-10-17T12:00:00+00:00",
      "conversation": {
        "id": "<original conversation id>",
        "title": "<title or \\"\\">",
        "created_at": "<ISO timestamp or \\"\\">"
      },
      "messages": [
        {"role": "user|assistant|system", "content": "...", "timestamp": "<ISO>"}
      ]
    }
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from uuid import uuid4

#: Identifier embedded in every export so a foreign JSON file is rejected fast.
FORMAT_NAME = "vector.conversation"
#: The only format version this build can read or write.
FORMAT_VERSION = 1

#: Upper bound on an import file, checked before the file is read into memory.
MAX_FILE_BYTES = 5 * 1024 * 1024
#: Upper bound on the number of messages accepted from one import.
MAX_MESSAGES = 2000
#: Upper bound on a single message's content.
MAX_CONTENT_CHARS = 100_000

#: Roles that may appear in a conversation.  Anything else is rejected so a
#: malformed file cannot smuggle an unrecognised channel into the database.
ALLOWED_ROLES = ("user", "assistant", "system")

#: The title used when an imported file omits one.
DEFAULT_TITLE = "Imported Conversation"

# ---- Safe, user-facing messages (never contain imported content) ----------
EXPORT_ERROR_TEXT = "Could not export the conversation. Please try again."
IMPORT_ERROR_TEXT = "Could not import the conversation file. Please try again."
INVALID_FORMAT_TEXT = "This file is not a valid Vector conversation export."
UNSUPPORTED_VERSION_TEXT = "This conversation export uses an unsupported format version."
EMPTY_FILE_TEXT = "The selected file is empty."
FILE_TOO_LARGE_TEXT = "The selected file is too large to import."
TOO_MANY_MESSAGES_TEXT = "The conversation has too many messages to import."
INVALID_ROLE_TEXT = "The conversation file contains an invalid message role."
INVALID_FIELD_TEXT = "The conversation file contains a field with an invalid type."
INVALID_TIMESTAMP_TEXT = "The conversation file contains an invalid timestamp."
MESSAGE_TOO_LARGE_TEXT = "The conversation file contains an oversized message."
NO_CONVERSATION_TEXT = "There is no active conversation to export."


class ConversationError(Exception):
    """Base error carrying a safe, user-facing ``message``."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class ConversationExportError(ConversationError):
    """Raised when a conversation cannot be exported."""


class ConversationImportError(ConversationError):
    """Raised when an import file is unreadable, malformed or unsupported."""


@dataclass
class ParsedConversation:
    """A validated conversation decoded from an import file."""

    messages: List[Dict[str, str]] = field(default_factory=list)
    title: str = DEFAULT_TITLE
    created_at: Optional[str] = None
    original_id: Optional[str] = None
    version: int = FORMAT_VERSION
    exported_at: Optional[str] = None


# --------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------
def build_export_payload(history_manager: Any, conversation_id: Optional[str]) -> Dict[str, Any]:
    """
    Build the versioned export dictionary for one conversation.

    Reads the conversation's own metadata and messages only.  Raises
    :class:`ConversationExportError` on a storage failure (without exposing the
    underlying exception).
    """
    if not conversation_id:
        raise ConversationExportError(NO_CONVERSATION_TEXT)

    try:
        messages = history_manager.get_conversation_messages(conversation_id, limit=None)
    except Exception:
        raise ConversationExportError(EXPORT_ERROR_TEXT)

    metadata = _load_metadata(history_manager, conversation_id)
    conversation = {
        "id": str(conversation_id),
        "title": str(metadata.get("title") or ""),
        "created_at": str(metadata.get("created_at") or ""),
    }

    exported_messages: List[Dict[str, str]] = []
    for message in messages or []:
        if not isinstance(message, dict):
            continue
        exported_messages.append(
            {
                "role": str(message.get("role", "")),
                "content": str(message.get("content", "")),
                "timestamp": str(message.get("timestamp", "")),
            }
        )

    return {
        "format": FORMAT_NAME,
        "version": FORMAT_VERSION,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "conversation": conversation,
        "messages": exported_messages,
    }


def serialize_conversation(history_manager: Any, conversation_id: Optional[str]) -> str:
    """Return the export document as pretty-printed UTF-8 JSON text."""
    payload = build_export_payload(history_manager, conversation_id)
    return json.dumps(payload, indent=2, ensure_ascii=False)


def export_conversation_to_path(
    history_manager: Any, conversation_id: Optional[str], path: os.PathLike | str
) -> str:
    """
    Serialize the conversation and write it to ``path`` atomically.

    Returns the path written.  The write goes to a temporary file in the
    destination directory and is renamed into place, so an interrupted write
    cannot corrupt or truncate an existing export.
    """
    text = serialize_conversation(history_manager, conversation_id)
    try:
        _atomic_write_text(Path(path), text)
    except OSError:
        raise ConversationExportError(EXPORT_ERROR_TEXT)
    return str(path)


def _load_metadata(history_manager: Any, conversation_id: str) -> Dict[str, Any]:
    getter: Optional[Callable[[str], Optional[Dict[str, Any]]]] = getattr(
        history_manager, "get_conversation", None
    )
    if getter is None:
        return {}
    try:
        return getter(conversation_id) or {}
    except Exception:
        return {}


def _atomic_write_text(path: Path, text: str) -> None:
    directory = path.parent
    if str(directory):
        directory.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=".vector-export.", suffix=".tmp", dir=str(directory) or "."
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


# --------------------------------------------------------------------------
# Import: reading and validation
# --------------------------------------------------------------------------
def load_conversation_file(path: os.PathLike | str) -> ParsedConversation:
    """
    Read and validate an import file, returning a :class:`ParsedConversation`.

    Enforces the file-size limit before reading, decodes UTF-8, then delegates
    structural validation to :func:`parse_conversation`.  All failures raise
    :class:`ConversationImportError` with a safe message.
    """
    file_path = Path(path)
    try:
        size = file_path.stat().st_size
    except OSError:
        raise ConversationImportError(IMPORT_ERROR_TEXT)

    if size == 0:
        raise ConversationImportError(EMPTY_FILE_TEXT)
    if size > MAX_FILE_BYTES:
        raise ConversationImportError(FILE_TOO_LARGE_TEXT)

    try:
        text = file_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        raise ConversationImportError(INVALID_FORMAT_TEXT)

    return parse_conversation(text)


def parse_conversation(text: str) -> ParsedConversation:
    """
    Validate and decode an export document.

    Rejects malformed JSON, a wrong/absent ``format`` marker, unsupported
    versions, invalid roles, invalid field types, invalid timestamps and
    oversized imports.  Imported content is never echoed into an error message.
    """
    if not isinstance(text, str) or not text.strip():
        raise ConversationImportError(EMPTY_FILE_TEXT)

    try:
        data = json.loads(text)
    except (ValueError, TypeError):
        raise ConversationImportError(INVALID_FORMAT_TEXT)

    if not isinstance(data, dict):
        raise ConversationImportError(INVALID_FORMAT_TEXT)
    if data.get("format") != FORMAT_NAME:
        raise ConversationImportError(INVALID_FORMAT_TEXT)

    version = data.get("version")
    if isinstance(version, bool) or not isinstance(version, int):
        raise ConversationImportError(INVALID_FORMAT_TEXT)
    if version != FORMAT_VERSION:
        raise ConversationImportError(UNSUPPORTED_VERSION_TEXT)

    raw_messages = data.get("messages")
    if not isinstance(raw_messages, list):
        raise ConversationImportError(INVALID_FORMAT_TEXT)
    if len(raw_messages) > MAX_MESSAGES:
        raise ConversationImportError(TOO_MANY_MESSAGES_TEXT)

    conversation = data.get("conversation")
    if conversation is None:
        conversation = {}
    if not isinstance(conversation, dict):
        raise ConversationImportError(INVALID_FORMAT_TEXT)

    title = _optional_str(conversation.get("title"))
    original_id = _optional_str(conversation.get("id"))
    created_at = _optional_str(conversation.get("created_at"))
    exported_at = _optional_str(data.get("exported_at"))

    messages = _validate_messages(raw_messages)
    _assign_timestamps(messages, base=_best_base(created_at, exported_at))

    return ParsedConversation(
        messages=messages,
        title=(title or DEFAULT_TITLE),
        created_at=created_at,
        original_id=original_id,
        version=version,
        exported_at=exported_at,
    )


def _optional_str(value: Any) -> Optional[str]:
    """A value that must be a string (or absent); returns a stripped copy."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise ConversationImportError(INVALID_FIELD_TEXT)
    return value


def _validate_messages(raw_messages: List[Any]) -> List[Dict[str, Any]]:
    validated: List[Dict[str, Any]] = []
    for item in raw_messages:
        if not isinstance(item, dict):
            raise ConversationImportError(INVALID_FORMAT_TEXT)

        role = item.get("role")
        if not isinstance(role, str) or role not in ALLOWED_ROLES:
            raise ConversationImportError(INVALID_ROLE_TEXT)

        content = item.get("content")
        if not isinstance(content, str):
            raise ConversationImportError(INVALID_FIELD_TEXT)
        if len(content) > MAX_CONTENT_CHARS:
            raise ConversationImportError(MESSAGE_TOO_LARGE_TEXT)

        timestamp = item.get("timestamp")
        parsed_dt: Optional[datetime] = None
        if timestamp is not None:
            if not isinstance(timestamp, str):
                raise ConversationImportError(INVALID_FIELD_TEXT)
            if timestamp.strip():
                parsed_dt = _parse_iso(timestamp)

        validated.append(
            {
                "role": role,
                "content": content,
                "timestamp": timestamp if isinstance(timestamp, str) and timestamp.strip() else None,
                "_dt": parsed_dt,
            }
        )
    return validated


def _assign_timestamps(messages: List[Dict[str, Any]], base: datetime) -> None:
    """
    Ensure every message carries a usable timestamp, in array order.

    * When every message already has a valid timestamp, the original strings are
      preserved verbatim (no rewriting).
    * When any timestamp is missing, timestamps are synthesized for the whole
      conversation in order from ``base`` so message ordering is still preserved.

    Missing/blank timestamps are the only case that synthesizes values; an
    invalid *value* is rejected earlier rather than silently replaced.
    """
    any_missing = any(message.get("_dt") is None for message in messages)
    for message in messages:
        parsed = message.pop("_dt", None)
        if any_missing or parsed is None:
            message["timestamp"] = base.isoformat()
            base = base + timedelta(microseconds=1)
        else:
            message["timestamp"] = message["timestamp"]


def _best_base(created_at: Optional[str], exported_at: Optional[str]) -> datetime:
    for candidate in (created_at, exported_at):
        if candidate and candidate.strip():
            try:
                return _parse_iso(candidate)
            except ConversationImportError:
                continue
    return datetime.now(timezone.utc)


def _parse_iso(text: str) -> datetime:
    value = text.strip()
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        raise ConversationImportError(INVALID_TIMESTAMP_TEXT)


# --------------------------------------------------------------------------
# Import: writing
# --------------------------------------------------------------------------
def resolve_conversation_id(
    original_id: Optional[str], exists: Callable[[str], bool]
) -> str:
    """
    Choose the id a conversation is imported under.

    The original id is reused when it is a non-empty string that does not
    already exist; otherwise a fresh UUID is generated.  This is what prevents
    an import from overwriting (or colliding with) an existing conversation.
    """
    if isinstance(original_id, str):
        candidate = original_id.strip()
        if candidate and not exists(candidate):
            return candidate
    return str(uuid4())


def import_parsed_conversation(history_manager: Any, parsed: ParsedConversation) -> str:
    """
    Store a validated conversation and return the id it was stored under.

    The id is resolved against the live database so an existing conversation is
    never overwritten.  The write itself is transactional; a storage failure is
    reported as :class:`ConversationImportError` and leaves nothing behind.
    """
    importer = getattr(history_manager, "import_conversation", None)
    if importer is None:
        raise ConversationImportError(IMPORT_ERROR_TEXT)

    try:
        exists = _make_exists(history_manager)
        new_id = resolve_conversation_id(parsed.original_id, exists)
        messages = [
            {
                "role": message["role"],
                "content": message["content"],
                "timestamp": message["timestamp"],
            }
            for message in parsed.messages
        ]
        importer(new_id, parsed.title, parsed.created_at, messages)
        return new_id
    except ConversationImportError:
        raise
    except Exception:
        # Never surface raw storage details (paths, SQL, content).
        raise ConversationImportError(IMPORT_ERROR_TEXT)


def _make_exists(history_manager: Any) -> Callable[[str], bool]:
    checker = getattr(history_manager, "conversation_exists", None)
    if checker is not None:
        return lambda conv_id: bool(checker(conv_id))

    getter = getattr(history_manager, "get_conversation", None)
    if getter is not None:
        return lambda conv_id: getter(conv_id) is not None

    return lambda _conv_id: False
