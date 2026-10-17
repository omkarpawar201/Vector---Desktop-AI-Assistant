"""
Headless tests for Vector Settings & Preferences (Feature 6).

Covers the ``app/config/store.py`` persistence layer and the tabbed
``app/ui/settings.py`` dialog.  Everything is exercised against temporary env
files and isolated ``Settings`` instances: no network, no credentials, no real
microphone, no model loading, and no writes to the repository's own ``.env``.
"""

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import (  # noqa: E402
    QApplication,
    QCheckBox,
    QComboBox,
    QLineEdit,
    QWidget,
)

from app.config.constants import PermissionLevel  # noqa: E402
from app.config.settings import Settings  # noqa: E402
from app.config.store import (  # noqa: E402
    RESTART_REQUIRED_FIELDS,
    SESSION_ONLY_FIELDS,
    SettingsStore,
)
from app.ui import settings as settings_mod  # noqa: E402
from app.ui.settings import (  # noqa: E402
    FIELD_LABELS,
    NO_CHANGES_TEXT,
    RESTART_HINT_TEMPLATE,
    SAVE_ERROR_TEXT,
    SAVE_SUCCESS_TEXT,
    SettingsDialog,
)


@pytest.fixture(scope="session")
def qapp():
    """Session fixture initializing a headless QApplication."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _store(tmp_path, file_text=None):
    """A store backed by a temp env file and an isolated Settings instance."""
    env = tmp_path / ".env"
    if file_text is not None:
        env.write_text(file_text, encoding="utf-8")
    settings = Settings(_env_file=env) if file_text is not None else Settings(_env_file=None)
    return SettingsStore(env_path=env, settings=settings), env, settings


def _dialog(store):
    """A dialog with the background save worker forced to run synchronously."""
    dlg = SettingsDialog(store=store)
    dlg._start_save_worker = lambda worker: worker.run()
    return dlg


# --------------------------------------------------------------------------
# 1. Loading valid existing settings
# --------------------------------------------------------------------------

def test_store_reads_existing_file_values(tmp_path):
    store, _env, settings = _store(
        tmp_path,
        "ENABLE_VOICE=true\nVOICE_STT_BACKEND=offline\nLOG_LEVEL=DEBUG\n",
    )
    assert settings.enable_voice is True
    assert settings.voice_stt_backend == "offline"
    values = store.read_file_values()
    assert values["ENABLE_VOICE"] == "true"
    assert values["VOICE_STT_BACKEND"] == "offline"


# --------------------------------------------------------------------------
# 2. Missing configuration and defaults
# --------------------------------------------------------------------------

def test_missing_file_uses_defaults_and_is_created_on_write(tmp_path):
    store, env, settings = _store(tmp_path, None)
    assert env.exists() is False
    assert settings.enable_voice is False
    assert settings.voice_stt_backend == "online"

    result = store.apply({"enable_voice": True})

    assert result.ok is True
    assert env.exists() is True
    assert "ENABLE_VOICE=true" in env.read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# 3. Malformed / unsupported values
# --------------------------------------------------------------------------

def test_malformed_lines_are_preserved_and_ignored(tmp_path):
    store, env, _settings = _store(
        tmp_path,
        "# a comment\n\nthis is not a valid line\n=missingkey\nENABLE_VOICE=true\n",
    )
    # Parsing ignores junk but still finds the real key.
    assert store.read_file_values().get("ENABLE_VOICE") == "true"

    store.apply({"enable_voice": False})

    text = env.read_text(encoding="utf-8")
    assert "# a comment" in text
    assert "this is not a valid line" in text
    assert "=missingkey" in text
    assert "ENABLE_VOICE=false" in text


def test_unsupported_backend_is_rejected(tmp_path):
    store, env, settings = _store(tmp_path, "ENABLE_VOICE=false\n")
    before = env.read_text(encoding="utf-8")

    result = store.apply({"voice_stt_backend": "telepathy"})

    assert result.ok is False
    assert "voice_stt_backend" in result.error
    # Nothing was written and the live instance was not mutated.
    assert env.read_text(encoding="utf-8") == before
    assert settings.voice_stt_backend == "online"


# --------------------------------------------------------------------------
# 4. Validation failures and save errors
# --------------------------------------------------------------------------

def test_validation_failure_reports_friendly_message(tmp_path):
    store, _env, _settings = _store(tmp_path, None)
    ok, error, candidate = store.validate({"voice_stt_backend": "nope"})
    assert ok is False
    assert candidate is None
    assert "voice_stt_backend" in error
    assert "Input should be" in error


def test_write_failure_is_handled_without_mutating(tmp_path, monkeypatch):
    store, _env, settings = _store(tmp_path, None)
    monkeypatch.setattr(store, "_write_keys", lambda updates: (_ for _ in ()).throw(OSError()))

    result = store.apply({"enable_voice": True})

    assert result.ok is False
    assert result.error == SAVE_ERROR_TEXT
    assert settings.enable_voice is False  # not applied when the write fails


# --------------------------------------------------------------------------
# 5. Save / apply / cancel behavior
# --------------------------------------------------------------------------

def test_store_apply_persists_and_updates_in_place(tmp_path):
    store, env, settings = _store(tmp_path, None)
    original_id = id(settings)

    result = store.apply({"enable_voice": True, "voice_stt_backend": "auto"})

    assert result.ok is True
    assert set(result.changed) == {"enable_voice", "voice_stt_backend"}
    assert settings.enable_voice is True
    assert settings.voice_stt_backend == "auto"
    assert id(settings) == original_id  # mutated in place, same object
    text = env.read_text(encoding="utf-8")
    assert "ENABLE_VOICE=true" in text
    assert "VOICE_STT_BACKEND=auto" in text


def test_dialog_apply_keeps_dialog_open_and_saves(tmp_path, qapp):
    store, env, settings = _store(tmp_path, None)
    dlg = _dialog(store)
    accepted = []
    dlg.accepted.connect(lambda: accepted.append(True))

    assert dlg.btn_apply.isEnabled() is False  # clean
    dlg.chk_voice.setChecked(True)
    assert dlg.btn_apply.isEnabled() is True
    dlg._on_apply()

    assert dlg.lbl_status.text() == SAVE_SUCCESS_TEXT
    assert accepted == []  # Apply does not close
    assert settings.enable_voice is True
    assert "ENABLE_VOICE=true" in env.read_text(encoding="utf-8")
    assert dlg.btn_apply.isEnabled() is False  # clean again


def test_dialog_save_persists_and_closes(tmp_path, qapp):
    store, env, settings = _store(tmp_path, None)
    dlg = _dialog(store)
    accepted = []
    dlg.accepted.connect(lambda: accepted.append(True))

    dlg.chk_voice.setChecked(True)
    dlg._on_save()

    assert accepted == [True]
    assert settings.enable_voice is True
    assert "ENABLE_VOICE=true" in env.read_text(encoding="utf-8")


def test_dialog_cancel_discards_unsaved_changes(tmp_path, qapp):
    store, env, settings = _store(tmp_path, "ENABLE_VOICE=false\nGEMINI_MODEL=gemini-2.5-flash\n")
    dlg = _dialog(store)
    before = env.read_text(encoding="utf-8")

    dlg.chk_voice.setChecked(True)
    dlg.txt_gemini_model.setText("something-else")
    dlg.reject()

    assert settings.enable_voice is False
    assert settings.gemini_model == "gemini-2.5-flash"
    assert env.read_text(encoding="utf-8") == before


def test_apply_does_not_write_on_keystroke(tmp_path, qapp):
    store, env, settings = _store(tmp_path, "ENABLE_VOICE=false\nENABLE_GEMINI=true\n")
    dlg = _dialog(store)
    before = env.read_text(encoding="utf-8")
    initial = (settings.enable_gemini, settings.voice_stt_model_dir)

    dlg.txt_stt_dir.setText("D:/some/model/dir")
    dlg.chk_gemini.setChecked(False)

    # No persistence and no live mutation until Apply/Save.
    assert env.read_text(encoding="utf-8") == before
    assert (settings.enable_gemini, settings.voice_stt_model_dir) == initial
    assert dlg._is_dirty() is True


def test_apply_with_no_changes_is_a_no_op(tmp_path, qapp):
    store, env, _settings = _store(tmp_path, None)
    dlg = _dialog(store)

    dlg._on_apply()

    assert dlg.lbl_status.text() == NO_CHANGES_TEXT
    assert env.exists() is False


# --------------------------------------------------------------------------
# 6. Preservation of unrelated configuration fields
# --------------------------------------------------------------------------

def test_unknown_keys_comments_and_ordering_are_preserved(tmp_path):
    original = (
        "# Vector settings\n"
        "CUSTOM_USER_KEY=please-keep-me\n"
        "\n"
        "ENABLE_VOICE=false\n"
        "# trailing comment\n"
        "ANOTHER_UNKNOWN='quoted value'\n"
    )
    store, env, _settings = _store(tmp_path, original)

    store.apply({"enable_voice": True})

    text = env.read_text(encoding="utf-8")
    assert "# Vector settings" in text
    assert "CUSTOM_USER_KEY=please-keep-me" in text
    assert "# trailing comment" in text
    assert "ANOTHER_UNKNOWN='quoted value'" in text
    assert "ENABLE_VOICE=true" in text
    # The unrelated key was not rewritten.
    assert text.count("CUSTOM_USER_KEY=") == 1
    # Order of the untouched lines is preserved.
    assert text.index("CUSTOM_USER_KEY") < text.index("ENABLE_VOICE")


def test_duplicate_keys_collapse_to_one_updated_value(tmp_path):
    store, env, _settings = _store(tmp_path, "ENABLE_VOICE=false\nENABLE_VOICE=true\n")

    store.apply({"enable_voice": False})

    text = env.read_text(encoding="utf-8")
    assert text.count("ENABLE_VOICE=") == 1
    assert "ENABLE_VOICE=false" in text


def test_write_leaves_no_temp_files(tmp_path):
    store, env, _settings = _store(tmp_path, None)
    store.apply({"enable_voice": True})
    leftovers = [p.name for p in tmp_path.iterdir() if p.name.endswith(".tmp")]
    assert leftovers == []


# --------------------------------------------------------------------------
# 7. Runtime changes vs restart-required changes
# --------------------------------------------------------------------------

def test_restart_required_classification(tmp_path):
    store, _env, _settings = _store(tmp_path, None)

    live = store.apply({"enable_voice": True})
    assert live.restart_required == []

    restart = store.apply({"log_level": "DEBUG"})
    assert restart.restart_required == ["log_level"]

    assert RESTART_REQUIRED_FIELDS == {"log_level", "database_path"}
    assert "enable_voice" not in RESTART_REQUIRED_FIELDS
    assert "require_confirmation_for_dangerous" not in RESTART_REQUIRED_FIELDS


def test_dialog_shows_restart_hint_for_log_level(tmp_path, qapp):
    store, _env, _settings = _store(tmp_path, "LOG_LEVEL=INFO\n")
    dlg = _dialog(store)

    dlg.cmb_log_level.setCurrentIndex(dlg.cmb_log_level.findText("DEBUG"))
    dlg._on_save()

    assert dlg.lbl_status.text() == RESTART_HINT_TEMPLATE.format(
        fields=FIELD_LABELS["log_level"]
    )


def test_live_setting_takes_effect_on_shared_instance(tmp_path):
    # A permission engine created with the *same* settings instance must observe
    # the change immediately -- this is what "takes effect without a restart"
    # means for the shared singleton.
    from app.security.permissions import PermissionEngine

    store, _env, settings = _store(tmp_path, None)
    engine = PermissionEngine(settings=settings)
    assert engine.evaluate(PermissionLevel.DANGEROUS).requires_confirmation is True

    store.apply({"require_confirmation_for_dangerous": False})

    assert engine.evaluate(PermissionLevel.DANGEROUS).requires_confirmation is False


# --------------------------------------------------------------------------
# 8. Voice / STT backend options
# --------------------------------------------------------------------------

@pytest.mark.parametrize("backend", ["online", "offline", "auto"])
def test_stt_backend_options_roundtrip(tmp_path, backend):
    # Start from a *different* backend so the change is never a no-op.
    baseline = "online" if backend != "online" else "offline"
    store, env, settings = _store(tmp_path, f"VOICE_STT_BACKEND={baseline}\n")

    result = store.apply({"voice_stt_backend": backend})

    assert result.ok is True
    assert settings.voice_stt_backend == backend
    assert f"VOICE_STT_BACKEND={backend}" in env.read_text(encoding="utf-8")
    # And the persisted value reloads correctly.
    assert Settings(_env_file=env).voice_stt_backend == backend


def test_stt_model_dir_with_spaces_roundtrips(tmp_path):
    store, env, _settings = _store(tmp_path, None)
    target = r"C:\Program Files\sherpa models\stt"

    result = store.apply({"voice_stt_model_dir": target})

    assert result.ok is True
    raw = env.read_text(encoding="utf-8")
    assert "VOICE_STT_MODEL_DIR=" in raw
    assert Settings(_env_file=env).voice_stt_model_dir == target


# --------------------------------------------------------------------------
# 9. Protection of routing and execution safety invariants
# --------------------------------------------------------------------------

def test_dangerous_toggle_never_bypasses_blocked_or_confirm(tmp_path):
    from app.security.permissions import PermissionEngine

    settings = Settings(_env_file=None, require_confirmation_for_dangerous=False)
    engine = PermissionEngine(settings=settings)

    blocked = engine.evaluate(PermissionLevel.BLOCKED)
    assert blocked.allowed is False  # BLOCKED stays blocked

    confirm = engine.evaluate(PermissionLevel.CONFIRM)
    assert confirm.allowed is True
    assert confirm.requires_confirmation is True  # CONFIRM always prompts

    dangerous = engine.evaluate(PermissionLevel.DANGEROUS)
    assert dangerous.allowed is True
    assert dangerous.requires_confirmation is False  # only DANGEROUS is affected


def test_store_ignores_unknown_or_permission_fields(tmp_path):
    store, env, settings = _store(tmp_path, None)

    result = store.apply(
        {
            "max_tool_calls": 9999,
            "permission_level": "SAFE",
            "allowed_terminal_commands": ["rm"],
        }
    )

    assert result.ok is True
    assert result.changed == []  # nothing recognisable, nothing written
    assert not env.exists()


# --------------------------------------------------------------------------
# 10. UI behavior and headless Qt compatibility
# --------------------------------------------------------------------------

def test_dialog_builds_all_tabs(tmp_path, qapp):
    store, _env, _settings = _store(tmp_path, None)
    dlg = _dialog(store)
    titles = [dlg.tabs.tabText(i) for i in range(dlg.tabs.count())]
    assert titles == ["AI & Routing", "Voice", "Privacy & Permissions", "Appearance", "General"]


def test_dialog_reflects_current_settings(tmp_path, qapp):
    store, _env, _settings = _store(
        tmp_path,
        "ENABLE_VOICE=true\nVOICE_STT_BACKEND=offline\nREQUIRE_CONFIRMATION_FOR_DANGEROUS=false\n",
    )
    dlg = _dialog(store)
    assert dlg.chk_voice.isChecked() is True
    assert dlg.cmb_stt_backend.currentData() == "offline"
    assert dlg.chk_confirm.isChecked() is False


def test_appearance_tab_offers_no_fake_controls(tmp_path, qapp):
    store, _env, _settings = _store(tmp_path, None)
    dlg = _dialog(store)
    appearance = dlg.tabs.widget(3)
    controls = appearance.findChildren(QCheckBox) + appearance.findChildren(QComboBox)
    # Informational only: no cosmetic toggle that would do nothing.
    assert controls == []


def test_api_key_field_is_masked_and_session_only(tmp_path, qapp):
    store, _env, _settings = _store(tmp_path, None)
    dlg = _dialog(store)
    assert dlg.txt_gemini_key.echoMode() == QLineEdit.Password
    assert "gemini_api_key" in SESSION_ONLY_FIELDS


def test_invalid_change_shows_error_and_keeps_dialog_open(tmp_path, qapp):
    store, _env, _settings = _store(tmp_path, None)
    dlg = _dialog(store)
    dlg._collect_changes = lambda: {"voice_stt_backend": "bogus"}
    dlg._initial = {"force": "dirty"}
    accepted = []
    dlg.accepted.connect(lambda: accepted.append(True))

    dlg._on_save()

    assert accepted == []
    assert dlg.lbl_status.property("error") == "true"
    assert "voice_stt_backend" in dlg.lbl_status.text()


def test_background_write_error_shows_generic_message(tmp_path, qapp):
    store, _env, _settings = _store(tmp_path, None)
    dlg = _dialog(store)
    dlg._collect_changes = lambda: {"enable_voice": True}
    dlg._initial = {"force": "dirty"}

    def _boom(_changes):
        raise RuntimeError("disk on fire")

    dlg.store.apply = _boom
    dlg._on_apply()

    assert dlg.lbl_status.property("error") == "true"
    assert dlg.lbl_status.text() == SAVE_ERROR_TEXT


def test_api_key_never_written_to_disk(tmp_path, qapp):
    store, env, settings = _store(tmp_path, "ENABLE_VOICE=false\n")
    dlg = _dialog(store)

    dlg.txt_gemini_key.setText("AIzaSySUPER_SECRET_VALUE_1234567890")
    dlg._on_apply()

    raw = env.read_text(encoding="utf-8")
    assert "SUPER_SECRET_VALUE" not in raw
    assert "GEMINI_API_KEY" not in raw
    # In-memory session value is updated, though.
    assert settings.gemini_api_key == "AIzaSySUPER_SECRET_VALUE_1234567890"


# --------------------------------------------------------------------------
# 11. Regression compatibility with Health / Diagnostics / History / startup
# --------------------------------------------------------------------------

def test_regression_dialog_importable_and_default_store(tmp_path, qapp):
    # No store passed -> uses the live singleton. Constructing must not mutate it.
    from app.config.settings import get_settings

    dlg = SettingsDialog()
    assert dlg.store.settings is get_settings()
    assert dlg.windowTitle() != ""


def test_regression_main_window_exposes_settings_dialog():
    from app.ui.main_window import MainWindow

    assert callable(getattr(MainWindow, "_open_settings", None))


def test_regression_main_window_constructs_with_new_dialog(qapp):
    from app.ui.main_window import MainWindow

    win = MainWindow()
    assert isinstance(win.settings, Settings)
    assert callable(getattr(win, "_open_settings", None))
    win.close()


def test_regression_other_dialogs_import():
    # Importing the settings module must not break the sibling dialogs.
    from app.ui.action_history import ActionHistoryDialog  # noqa: F401
    from app.ui.diagnostics import DiagnosticsDialog  # noqa: F401
    from app.ui.system_health import SystemHealthDialog  # noqa: F401

    assert settings_mod.SettingsDialog is SettingsDialog
