"""
Settings persistence for Vector Desktop AI Assistant.

Vector already loads configuration with ``pydantic-settings`` from a ``.env``
file (``SettingsConfigDict(env_file=".env")``).  This module adds the missing
*write* half of that mechanism without inventing a second configuration system:

* it reads the existing ``.env`` file line by line and rewrites only the keys
  that actually changed, preserving every other line (comments, blank lines and
  keys this application does not know about) verbatim;
* it validates a proposed change by reconstructing the *existing*
  :class:`~app.config.settings.Settings` model (so the same Pydantic rules that
  guard startup also guard the UI), and never writes an invalid value;
* it applies accepted values to the live, shared ``Settings`` singleton so
  settings that are read dynamically take effect immediately, while reporting
  which fields need a restart.

No secret is ever written to disk: fields listed in ``SESSION_ONLY_FIELDS``
(the Gemini API key) are applied to the running session but excluded from the
``.env`` file, because the project has no secure credential store.
"""

from __future__ import annotations

import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import ValidationError

from app.config.settings import Settings, get_settings

#: Default env file location -- matches ``SettingsConfigDict(env_file=".env")``,
#: which is resolved relative to the process working directory.
DEFAULT_ENV_PATH = Path(".env")

#: Fields that only take effect after Vector is restarted, because the value is
#: read once during startup: the log level is applied by ``app.main`` before the
#: UI exists, and the database path is bound into the singleton
#: ``DatabaseManager`` on first use.
RESTART_REQUIRED_FIELDS = frozenset({"log_level", "database_path"})

#: Fields applied to the running session but never persisted, because the
#: project has no secure credential store.
SESSION_ONLY_FIELDS = frozenset({"gemini_api_key"})

#: Generic, non-sensitive save failure shown to the user.
SAVE_ERROR_TEXT = "Could not save settings. Please check the values and try again."

#: Matches an uncommented ``.env`` assignment (optionally ``export``-prefixed).
_LINE_RE = re.compile(
    r"^(?P<lead>\s*)(?:export\s+)?(?P<key>[A-Za-z_][A-Za-z0-9_]*)"
    r"(?P<sep>\s*=\s*)(?P<value>.*)$"
)

#: Characters that force a value to be quoted in the ``.env`` file.
_UNSAFE_CHARS = set(" \t#\"'")


def env_key(field_name: str) -> str:
    """The ``.env`` variable name for a Settings field (upper-cased)."""
    return field_name.upper()


def _needs_quotes(text: str) -> bool:
    if text == "":
        return False
    if text != text.strip():
        return True
    return any(ch in _UNSAFE_CHARS for ch in text)


def _format_value(value: Any) -> str:
    """Render a Python value as the raw right-hand side of a ``.env`` line."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, int):
        return str(value)
    text = "" if value is None else str(value)
    if _needs_quotes(text):
        escaped = text.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    return text


def _unquote(raw: str) -> str:
    """Best-effort removal of surrounding quotes for comparison purposes."""
    text = raw.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ("'", '"'):
        inner = text[1:-1]
        return inner.replace('\\"', '"').replace("\\\\", "\\")
    return text


def _friendly_validation_error(exc: ValidationError) -> str:
    """Turn a Pydantic error into a short, non-sensitive, user-facing message."""
    parts: List[str] = []
    for err in exc.errors():
        location = ".".join(str(p) for p in err.get("loc", ())) or "value"
        parts.append(f"{location}: {err.get('msg', 'invalid value')}")
    detail = "; ".join(parts) if parts else "invalid values"
    return f"Invalid settings — {detail}"


@dataclass
class ApplyResult:
    """Outcome of a :meth:`SettingsStore.apply` call."""

    ok: bool
    error: str = ""
    #: Field names whose effective value changed.
    changed: List[str] = field(default_factory=list)
    #: Changed fields that need a restart to take effect.
    restart_required: List[str] = field(default_factory=list)
    #: Env variable names written to the ``.env`` file.
    written_keys: List[str] = field(default_factory=list)
    #: Changed fields intentionally kept out of the ``.env`` file.
    session_only: List[str] = field(default_factory=list)


class SettingsStore:
    """
    Reads, validates and persists Vector settings.

    ``env_path`` and ``settings`` are injectable so the store can be exercised
    deterministically against a temporary file and an isolated ``Settings``
    instance, without touching the repository's real ``.env`` or the global
    singleton.
    """

    def __init__(
        self,
        env_path: Optional[os.PathLike | str] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        self.env_path = Path(env_path) if env_path is not None else DEFAULT_ENV_PATH
        self.settings = settings if settings is not None else get_settings()

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------
    def _read_lines(self) -> List[str]:
        try:
            text = self.env_path.read_text(encoding="utf-8")
        except (FileNotFoundError, OSError):
            return []
        return text.splitlines()

    def read_file_values(self) -> Dict[str, str]:
        """Return ``{ENV_KEY: value}`` parsed from the file (empty if missing)."""
        values: Dict[str, str] = {}
        for line in self._read_lines():
            match = _LINE_RE.match(line)
            if match:
                values[match.group("key")] = _unquote(match.group("value"))
        return values

    def current_values(self) -> Dict[str, Any]:
        """Effective field values from the live ``Settings`` instance."""
        try:
            return dict(self.settings.model_dump())
        except Exception:
            return {}

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def validate(
        self, changes: Dict[str, Any]
    ) -> Tuple[bool, str, Optional[Settings]]:
        """
        Validate ``changes`` against the existing model, using the current
        effective values as the baseline for every field that is not changing.

        Returns ``(ok, error_message, candidate)``; ``candidate`` is a fully
        validated :class:`Settings` instance when ``ok`` is true.
        """
        merged = self.current_values()
        merged.update(changes)
        try:
            candidate = Settings(_env_file=None, **merged)
        except ValidationError as exc:
            return False, _friendly_validation_error(exc), None
        except Exception:
            return False, SAVE_ERROR_TEXT, None
        return True, "", candidate

    # ------------------------------------------------------------------
    # Apply / persist
    # ------------------------------------------------------------------
    def apply(self, changes: Dict[str, Any]) -> ApplyResult:
        """
        Validate, persist and apply ``changes``.

        The ``.env`` file is written first; the live ``Settings`` instance is
        mutated only after a successful write, so a disk failure never leaves
        the running configuration out of sync with what is stored.
        """
        ok, error, candidate = self.validate(changes)
        if not ok or candidate is None:
            return ApplyResult(ok=False, error=error)

        current = self.current_values()
        candidate_values = candidate.model_dump()

        changed = [
            name
            for name in changes
            if name in candidate_values and current.get(name) != candidate_values.get(name)
        ]
        if not changed:
            return ApplyResult(ok=True)

        persisted = [name for name in changed if name not in SESSION_ONLY_FIELDS]
        session_only = [name for name in changed if name in SESSION_ONLY_FIELDS]

        updates = {env_key(name): _format_value(candidate_values[name]) for name in persisted}
        try:
            written = self._write_keys(updates)
        except OSError:
            return ApplyResult(ok=False, error=SAVE_ERROR_TEXT)

        # Only mutate the live instance after the file was written.
        for name in changed:
            try:
                setattr(self.settings, name, candidate_values[name])
            except Exception:
                # A pydantic assignment failure must not crash the UI; the value
                # is still persisted for the next start.
                pass

        return ApplyResult(
            ok=True,
            changed=changed,
            restart_required=sorted(set(changed) & RESTART_REQUIRED_FIELDS),
            written_keys=written,
            session_only=session_only,
        )

    def _write_keys(self, updates: Dict[str, str]) -> List[str]:
        """Rewrite only the given keys, preserving every other line verbatim."""
        if not updates:
            return []

        lines = self._read_lines()

        # Locate the last assignment of each updated key; earlier duplicates are
        # dropped so the new value reliably wins (python-dotenv: last wins).
        last_index: Dict[str, int] = {}
        for index, line in enumerate(lines):
            match = _LINE_RE.match(line)
            if match and match.group("key") in updates:
                last_index[match.group("key")] = index

        out: List[str] = []
        for index, line in enumerate(lines):
            match = _LINE_RE.match(line)
            if match and match.group("key") in updates:
                key = match.group("key")
                if last_index.get(key) != index:
                    continue
                out.append(f"{key}={updates[key]}")
            else:
                out.append(line)

        for key, value in updates.items():
            if key not in last_index:
                out.append(f"{key}={value}")

        self._atomic_write(out)
        return list(updates.keys())

    def _atomic_write(self, lines: List[str]) -> None:
        directory = self.env_path.parent
        if str(directory):
            directory.mkdir(parents=True, exist_ok=True)
        content = "\n".join(lines) + "\n"
        fd, tmp_name = tempfile.mkstemp(
            prefix=".env.", suffix=".tmp", dir=str(directory) or "."
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(content)
            os.replace(tmp_name, self.env_path)
        except Exception:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise
