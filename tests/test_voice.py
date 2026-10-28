"""
Unit tests for TTSEngine, STTEngine, VADEngine, VoiceListener, the voice UI
toggle, and local privacy mode.

Everything here is mocked/offline: no microphone is opened, no audio is played,
no network transcription call is made, and no audio content is logged.
"""

import os
import struct
import sys
import types
from pathlib import Path

os.environ["QT_QPA_PLATFORM"] = "offscreen"

import numpy as np
import pytest
from pydantic import ValidationError

from app.config.settings import Settings
from app.core.router import IntentRouter
from app.tools.registry import ToolRegistry
from app.voice.listener import VoiceListener
from app.voice.stt import STTEngine, _SherpaOfflineBackend, _pcm_clipped_fraction
from app.voice.tts import TTSEngine
from app.voice.vad import VADEngine


def test_tts_engine():
    """Verify native Windows SAPI5 TTS engine configuration."""
    tts = TTSEngine(rate=200, volume=0.9)
    assert tts.is_enabled is True
    assert tts.rate == 200

    # Test toggling speech
    tts.set_enabled(False)
    assert tts.is_enabled is False


def test_vad_engine_rms_calculation():
    """Verify Voice Activity Detection RMS energy calculation."""
    vad = VADEngine(energy_threshold=100.0)

    # Empty audio
    assert vad.calculate_rms(b"") == 0.0

    # Silent PCM audio (16-bit zeros)
    silent_pcm = b"\x00\x00" * 100
    assert vad.calculate_rms(silent_pcm) == 0.0
    assert vad.is_speech(silent_pcm) is False


def test_stt_engine_lazy_loading():
    """Verify STT engine lazy loading state."""
    stt = STTEngine(model_size="tiny.en")
    assert stt.is_loaded is False  # Must not load model on __init__


def test_offline_privacy_mode():
    """Verify complete local-only mode when ENABLE_GEMINI=False."""
    settings = Settings(enable_gemini=False)
    reg = ToolRegistry()
    router = IntentRouter(settings=settings, registry=reg)

    res = router.route_and_execute("Write a complex code refactoring plan")
    assert "Gemini unavailable" in res.response_text


# --------------------------------------------------------------------------
# VAD edge-silence semantics
# --------------------------------------------------------------------------

def _speech_chunk(value: int = 1000, count: int = 100) -> bytes:
    """A chunk whose RMS (abs value) exceeds the test threshold."""
    return struct.pack(f"<{count}h", *([value] * count))


def _silence_chunk(count: int = 100) -> bytes:
    return b"\x00\x00" * count


def test_filter_silence_strips_leading():
    vad = VADEngine(energy_threshold=300.0)
    speech = _speech_chunk()
    silence = _silence_chunk()

    assert vad.filter_silence([silence, speech, speech]) == speech + speech


def test_filter_silence_strips_trailing():
    vad = VADEngine(energy_threshold=300.0)
    speech = _speech_chunk()
    silence = _silence_chunk()

    assert vad.filter_silence([speech, silence]) == speech


def test_filter_silence_preserves_interior():
    vad = VADEngine(energy_threshold=300.0)
    speech = _speech_chunk()
    silence = _silence_chunk()

    # An interior pause between words must NOT be clipped.
    assert vad.filter_silence([speech, silence, speech]) == speech + silence + speech


def test_filter_silence_all_silent_is_empty():
    vad = VADEngine(energy_threshold=300.0)
    silence = _silence_chunk()

    assert vad.filter_silence([silence, silence]) == b""
    assert vad.filter_silence([]) == b""


# --------------------------------------------------------------------------
# STTEngine with a mocked SpeechRecognition module (no mic, no network)
# --------------------------------------------------------------------------

class _FakeAudioFile:
    def __init__(self, path):
        self.path = path

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeAudioData:
    """Stands in for speech_recognition.AudioData; holds synthetic PCM only."""

    def __init__(self, raw: bytes = b"\x01\x00" * 160):
        self.raw = raw
        self.conversions = []

    def get_raw_data(self, convert_rate=None, convert_width=None):
        self.conversions.append((convert_rate, convert_width))
        return self.raw


class _FakeRecognizer:
    def __init__(self, phrase="hello vector", fail=None, audio=None, events=None, calibration_fail=None):
        self._phrase = phrase
        self._fail = fail
        self._audio = audio
        self._events = events if events is not None else []
        self._calibration_fail = calibration_fail
        self.google_calls = 0
        self.calibrations = 0
        self.listen_calls = 0

    def record(self, source):
        return _FakeAudioData()

    def adjust_for_ambient_noise(self, source, duration=0.3):
        self.calibrations += 1
        self._events.append("calibrate")
        if self._calibration_fail is not None:
            raise self._calibration_fail
        return None

    def listen(self, source, timeout=None, phrase_time_limit=None):
        self.listen_calls += 1
        self._events.append("listen")
        return self._audio if self._audio is not None else _FakeAudioData()

    def recognize_google(self, audio):
        self.google_calls += 1
        if self._fail is not None:
            raise self._fail
        return self._phrase


class _FakeMic:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _fake_sr_module(recognizer: _FakeRecognizer) -> types.SimpleNamespace:
    class UnknownValueError(Exception):
        pass

    class RequestError(Exception):
        pass

    return types.SimpleNamespace(
        Recognizer=lambda: recognizer,
        AudioFile=_FakeAudioFile,
        Microphone=_FakeMic,
        UnknownValueError=UnknownValueError,
        RequestError=RequestError,
    )


def test_transcribe_file_empty_file_returns_empty(tmp_path):
    empty = tmp_path / "empty.wav"
    empty.write_bytes(b"")
    assert STTEngine().transcribe_file(str(empty)) == ""


def test_transcribe_file_uses_speech_recognition(tmp_path, monkeypatch):
    wav = tmp_path / "clip.wav"
    wav.write_bytes(b"RIFFxxxx")

    monkeypatch.setitem(
        sys.modules,
        "speech_recognition",
        _fake_sr_module(_FakeRecognizer(phrase="  open chrome  ")),
    )

    assert STTEngine().transcribe_file(str(wav)) == "open chrome"


def test_transcribe_file_logs_reason_without_secrets(tmp_path, monkeypatch, caplog):
    wav = tmp_path / "clip.wav"
    wav.write_bytes(b"RIFFxxxx")

    # SpeechRecognition present but broken; faster_whisper absent (not installed).
    monkeypatch.setitem(
        sys.modules,
        "speech_recognition",
        _fake_sr_module(_FakeRecognizer(fail=RuntimeError("no device"))),
    )
    monkeypatch.setitem(sys.modules, "faster_whisper", None)

    with caplog.at_level("WARNING", logger="app.voice.stt"):
        assert STTEngine().transcribe_file(str(wav)) == ""

    logged = " ".join(r.message for r in caplog.records)
    assert "SpeechRecognition failed" in logged
    assert "faster_whisper" in logged


def test_listen_and_transcribe_mic_returns_text(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "speech_recognition",
        _fake_sr_module(_FakeRecognizer(phrase="  turn on the lights  ")),
    )

    assert STTEngine().listen_and_transcribe_mic() == "turn on the lights"


def test_listen_and_transcribe_mic_unintelligible_returns_empty(monkeypatch, caplog):
    fake = _fake_sr_module(_FakeRecognizer())
    rec = _FakeRecognizer(fail=fake.UnknownValueError("nope"))
    fake.Recognizer = lambda: rec
    monkeypatch.setitem(sys.modules, "speech_recognition", fake)

    with caplog.at_level("INFO", logger="app.voice.stt"):
        assert STTEngine().listen_and_transcribe_mic() == ""

    assert any("not intelligible" in r.message for r in caplog.records)


def test_listen_and_transcribe_mic_capture_failure(monkeypatch, caplog):
    class _BrokenMic:
        def __enter__(self):
            raise RuntimeError("no microphone")

        def __exit__(self, *exc):
            return False

    fake = _fake_sr_module(_FakeRecognizer())
    fake.Microphone = _BrokenMic
    monkeypatch.setitem(sys.modules, "speech_recognition", fake)

    with caplog.at_level("WARNING", logger="app.voice.stt"):
        assert STTEngine().listen_and_transcribe_mic() == ""

    assert any("microphone capture failed" in r.message for r in caplog.records)


def test_mic_calibration_completes_before_prompt_and_recording(monkeypatch):
    """
    The user must not be prompted to speak while ambient calibration is still
    consuming microphone audio; recording starts only after the prompt.
    """
    events = []
    rec = _FakeRecognizer(phrase="turn up the volume", events=events)
    monkeypatch.setitem(sys.modules, "speech_recognition", _fake_sr_module(rec))

    result = STTEngine().listen_and_transcribe_mic(on_ready=lambda: events.append("prompt"))

    assert result == "turn up the volume"
    assert events.index("calibrate") < events.index("prompt") < events.index("listen")
    assert rec.calibrations == 1, "ambient calibration must run exactly once per capture"


def test_mic_prompt_and_recording_skipped_when_calibration_fails(monkeypatch, caplog):
    events = []
    rec = _FakeRecognizer(events=events, calibration_fail=RuntimeError("ambient noise"))
    monkeypatch.setitem(sys.modules, "speech_recognition", _fake_sr_module(rec))

    with caplog.at_level("WARNING", logger="app.voice.stt"):
        result = STTEngine().listen_and_transcribe_mic(on_ready=lambda: events.append("prompt"))

    assert result == ""
    assert "prompt" not in events
    assert "listen" not in events
    assert any("microphone capture failed" in r.message for r in caplog.records)


# --------------------------------------------------------------------------
# Offline (sherpa-onnx) backend selection, adapter conversion, and fallback
# --------------------------------------------------------------------------

class _FakeOfflineBackend:
    def __init__(self, text="offline text", error=None):
        self.text = text
        self.error = error
        self.pcm_calls = []
        self.file_calls = []

    def transcribe_pcm(self, pcm, sample_rate=None):
        self.pcm_calls.append((pcm, sample_rate))
        if self.error is not None:
            raise self.error
        return self.text

    def transcribe_file(self, path):
        self.file_calls.append(path)
        if self.error is not None:
            raise self.error
        return self.text


def _inject_offline(engine, fake):
    """Skip real model loading; install a fake adapter as if it were built."""
    engine._offline_backend = fake
    engine._offline_attempted = True
    return fake


def test_backend_defaults_to_online():
    assert STTEngine().backend == "online"


def test_backend_override_is_honored():
    assert STTEngine(backend="offline").backend == "offline"
    assert STTEngine(backend="auto").backend == "auto"


def test_backend_setting_rejects_invalid_value():
    with pytest.raises(ValidationError):
        Settings(voice_stt_backend="bogus")


def test_backend_setting_accepts_valid_values():
    for value in ("online", "offline", "auto"):
        assert Settings(voice_stt_backend=value).voice_stt_backend == value


def test_offline_missing_package_returns_empty_without_online(tmp_path, monkeypatch, caplog):
    monkeypatch.setitem(sys.modules, "sherpa_onnx", None)  # forces ImportError
    rec = _FakeRecognizer(phrase="must not be used")
    monkeypatch.setitem(sys.modules, "speech_recognition", _fake_sr_module(rec))

    engine = STTEngine(backend="offline", model_dir=str(tmp_path))
    with caplog.at_level("WARNING", logger="app.voice.stt"):
        result = engine.listen_and_transcribe_mic()

    assert result == ""
    assert rec.google_calls == 0, "offline mode must never invoke online recognition"
    assert any("sherpa_onnx" in r.message for r in caplog.records)


def test_offline_missing_model_dir_returns_empty(tmp_path, monkeypatch, caplog):
    rec = _FakeRecognizer(phrase="must not be used")
    monkeypatch.setitem(sys.modules, "speech_recognition", _fake_sr_module(rec))

    engine = STTEngine(backend="offline", model_dir=str(tmp_path / "does_not_exist"))
    with caplog.at_level("WARNING", logger="app.voice.stt"):
        assert engine.listen_and_transcribe_mic() == ""

    assert rec.google_calls == 0
    assert any("model directory not found" in r.message for r in caplog.records)


def test_offline_mic_success_uses_offline_only(monkeypatch):
    audio = _FakeAudioData(raw=b"\x10\x27" * 32)
    rec = _FakeRecognizer(audio=audio)
    monkeypatch.setitem(sys.modules, "speech_recognition", _fake_sr_module(rec))
    fake = _FakeOfflineBackend(text="  open chrome  ")

    engine = STTEngine(backend="offline")
    _inject_offline(engine, fake)

    assert engine.listen_and_transcribe_mic() == "open chrome"
    assert rec.google_calls == 0
    assert fake.pcm_calls == [(audio.raw, engine.sample_rate)]
    assert audio.conversions == [(engine.sample_rate, 2)]


def test_pcm_clipped_fraction_counts_full_scale_samples():
    assert _pcm_clipped_fraction(b"") == 0.0

    clean = struct.pack("<4h", 0, 1000, -1000, 20000)
    assert _pcm_clipped_fraction(clean) == 0.0

    hot = struct.pack("<4h", 32767, -32768, 100, -100)
    assert _pcm_clipped_fraction(hot) == 0.5


def test_offline_mic_warns_but_transcribes_when_input_clips(monkeypatch, caplog):
    """A hot capture must be flagged at the source, never silently boosted."""
    hot = struct.pack("<200h", *([32767, -32768] * 100))
    audio = _FakeAudioData(raw=hot)
    rec = _FakeRecognizer(audio=audio)
    monkeypatch.setitem(sys.modules, "speech_recognition", _fake_sr_module(rec))
    fake = _FakeOfflineBackend(text="open chrome")

    engine = STTEngine(backend="offline")
    _inject_offline(engine, fake)

    with caplog.at_level("WARNING", logger="app.voice.stt"):
        assert engine.listen_and_transcribe_mic() == "open chrome"

    assert rec.google_calls == 0
    assert any("clipping" in r.message for r in caplog.records)


def test_offline_mic_no_clipping_warning_for_normal_signal(monkeypatch, caplog):
    audio = _FakeAudioData(raw=struct.pack("<200h", *([3000, -3000] * 100)))
    rec = _FakeRecognizer(audio=audio)
    monkeypatch.setitem(sys.modules, "speech_recognition", _fake_sr_module(rec))
    fake = _FakeOfflineBackend(text="open chrome")

    engine = STTEngine(backend="offline")
    _inject_offline(engine, fake)

    with caplog.at_level("WARNING", logger="app.voice.stt"):
        assert engine.listen_and_transcribe_mic() == "open chrome"

    assert not any("clipping" in r.message for r in caplog.records)


def test_offline_failure_strict_returns_empty_without_online(monkeypatch, caplog):
    rec = _FakeRecognizer(phrase="must not appear")
    monkeypatch.setitem(sys.modules, "speech_recognition", _fake_sr_module(rec))
    fake = _FakeOfflineBackend(error=RuntimeError("boom"))

    engine = STTEngine(backend="offline")
    _inject_offline(engine, fake)

    with caplog.at_level("WARNING", logger="app.voice.stt"):
        assert engine.listen_and_transcribe_mic() == ""

    assert rec.google_calls == 0
    assert any("transcription failed" in r.message for r in caplog.records)


def test_auto_prefers_offline_when_available(monkeypatch):
    rec = _FakeRecognizer(phrase="online result")
    monkeypatch.setitem(sys.modules, "speech_recognition", _fake_sr_module(rec))
    fake = _FakeOfflineBackend(text="offline result")

    engine = STTEngine(backend="auto")
    _inject_offline(engine, fake)

    assert engine.listen_and_transcribe_mic() == "offline result"
    assert rec.google_calls == 0


def test_auto_falls_back_to_online_when_offline_fails(monkeypatch, caplog):
    rec = _FakeRecognizer(phrase="online result")
    monkeypatch.setitem(sys.modules, "speech_recognition", _fake_sr_module(rec))
    fake = _FakeOfflineBackend(error=RuntimeError("boom"))

    engine = STTEngine(backend="auto")
    _inject_offline(engine, fake)

    with caplog.at_level("INFO", logger="app.voice.stt"):
        assert engine.listen_and_transcribe_mic() == "online result"

    assert rec.google_calls == 1
    assert any("falling back" in r.message for r in caplog.records)


def test_transcribe_file_offline_success(tmp_path, monkeypatch):
    wav = tmp_path / "clip.wav"
    wav.write_bytes(b"RIFFxxxx")
    rec = _FakeRecognizer(phrase="online must not run")
    monkeypatch.setitem(sys.modules, "speech_recognition", _fake_sr_module(rec))
    fake = _FakeOfflineBackend(text="offline file text")

    engine = STTEngine(backend="offline")
    _inject_offline(engine, fake)

    assert engine.transcribe_file(str(wav)) == "offline file text"
    assert rec.google_calls == 0
    assert fake.file_calls == [str(wav.resolve())]


def test_transcribe_file_offline_missing_backend_returns_empty(tmp_path, monkeypatch, caplog):
    wav = tmp_path / "clip.wav"
    wav.write_bytes(b"RIFFxxxx")
    rec = _FakeRecognizer(phrase="online must not run")
    monkeypatch.setitem(sys.modules, "speech_recognition", _fake_sr_module(rec))
    monkeypatch.setitem(sys.modules, "sherpa_onnx", None)

    engine = STTEngine(backend="offline", model_dir=str(tmp_path))
    with caplog.at_level("WARNING", logger="app.voice.stt"):
        assert engine.transcribe_file(str(wav)) == ""

    assert rec.google_calls == 0


def _bare_adapter(recognizer_obj):
    """Build an adapter without importing sherpa-onnx (no real model)."""
    backend = object.__new__(_SherpaOfflineBackend)
    backend.sample_rate = 16000
    backend._sherpa = None
    backend.recognizer = recognizer_obj
    return backend


class _CapturingStream:
    def __init__(self, captured):
        self._captured = captured
        self.result = types.SimpleNamespace(text="decoded")

    def accept_waveform(self, sample_rate, samples):
        self._captured["sample_rate"] = sample_rate
        self._captured["samples"] = samples


class _CapturingRecognizer:
    def __init__(self, captured):
        self._captured = captured

    def create_stream(self):
        return _CapturingStream(self._captured)

    def decode_stream(self, stream):
        self._captured["decoded"] = True


def test_offline_adapter_converts_int16_pcm_to_float32():
    captured = {}
    backend = _bare_adapter(_CapturingRecognizer(captured))

    pcm = struct.pack("<4h", -32768, -1, 0, 32767)
    text = backend.transcribe_pcm(pcm, 16000)

    assert text == "decoded"
    assert captured["sample_rate"] == 16000
    assert captured["decoded"] is True
    samples = captured["samples"]
    assert samples.dtype == np.float32
    assert samples[0] == pytest.approx(-1.0)
    assert samples[3] == pytest.approx(32767 / 32768)


def test_offline_adapter_downmixes_stereo_wav(tmp_path):
    import wave

    path = tmp_path / "stereo.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(struct.pack("<8h", 1, 2, 3, 4, 5, 6, 7, 8))

    captured = {}
    backend = _bare_adapter(_CapturingRecognizer(captured))
    text = backend.transcribe_file(str(path))

    assert text == "decoded"
    samples = captured["samples"]
    assert np.allclose(samples, np.array([1, 3, 5, 7], dtype=np.float32) / 32768.0)


def _have_real_offline_model():
    return Path("models/stt").is_dir() and Path("models/stt/test_wavs/0.wav").is_file()


def test_real_offline_model_transcribes_bundled_wavs():
    """
    Integration test: loads the actual sherpa-onnx model from models/stt/ and
    decodes the bundled test WAVs. Never uses the microphone, Google
    recognition, Gemini, or any streaming API.
    """
    pytest.importorskip("sherpa_onnx")
    model_dir = Path("models/stt")
    if not _have_real_offline_model():
        pytest.skip("offline STT model not installed (models/stt)")

    engine = STTEngine(backend="offline", model_dir=str(model_dir))
    text_16k = engine.transcribe_file(str(model_dir / "test_wavs" / "0.wav"))
    assert text_16k, "real offline model produced empty transcription for 16k wav"
    assert "NIGHTFALL" in text_16k.upper()

    text_8k = engine.transcribe_file(str(model_dir / "test_wavs" / "8k.wav"))
    assert text_8k, "real offline model produced empty transcription for 8k wav"
    assert "HESTER" in text_8k.upper()


def test_real_offline_model_strict_offline_never_invokes_online(monkeypatch):
    """
    With the real model loaded and backend='offline', any attempt to reach the
    online recognizer (SpeechRecognition / Google) must fail loudly.
    """
    pytest.importorskip("sherpa_onnx")
    model_dir = Path("models/stt")
    if not _have_real_offline_model():
        pytest.skip("offline STT model not installed (models/stt)")

    class _ForbidOnline:
        def Recognizer(self):
            raise AssertionError("online recognizer invoked in strict offline mode")

        def AudioFile(self, *args, **kwargs):
            raise AssertionError("online AudioFile invoked in strict offline mode")

        def Microphone(self, *args, **kwargs):
            raise AssertionError("online Microphone invoked in strict offline mode")

    monkeypatch.setitem(sys.modules, "speech_recognition", _ForbidOnline())

    engine = STTEngine(backend="offline", model_dir=str(model_dir))
    text = engine.transcribe_file(str(model_dir / "test_wavs" / "0.wav"))
    assert text, "real offline model produced empty transcription"
    assert "NIGHTFALL" in text.upper()


def test_offline_real_existing_dir_without_model_layout_returns_empty(monkeypatch, caplog):
    """
    Real behavior when model_dir exists but has no sherpa-onnx layout
    (e.g. models/ holds only Needle artifacts): offline stays empty, no online.
    """
    pytest.importorskip("sherpa_onnx")
    rec = _FakeRecognizer(phrase="must not be used")
    monkeypatch.setitem(sys.modules, "speech_recognition", _fake_sr_module(rec))
    wav = Path("models/stt/test_wavs/0.wav")
    if not wav.is_file():
        pytest.skip("offline STT test wav not installed")

    engine = STTEngine(backend="offline", model_dir="models")
    with caplog.at_level("WARNING", logger="app.voice.stt"):
        assert engine.transcribe_file(str(wav)) == ""

    assert rec.google_calls == 0
    assert any("failed to initialize" in r.message for r in caplog.records)


# --------------------------------------------------------------------------
# VoiceListener delegation
# --------------------------------------------------------------------------

class _RecordingSTT(STTEngine):
    def __init__(self, text: str):
        super().__init__()
        self._text = text
        self.calls = 0
        self.received_ready = None

    def listen_and_transcribe_mic(self, phrase_time_limit: int = 6, on_ready=None) -> str:
        self.calls += 1
        self.received_ready = on_ready
        return self._text


def test_voice_listener_delegates_to_stt():
    fake = _RecordingSTT("open notepad")
    listener = VoiceListener(stt_engine=fake)

    assert listener.listen_from_mic() == "open notepad"
    assert fake.calls == 1


def test_voice_listener_forwards_ready_callback():
    fake = _RecordingSTT("open notepad")
    listener = VoiceListener(stt_engine=fake)
    events = []

    assert listener.listen_from_mic(on_ready=lambda: events.append("ready")) == "open notepad"
    assert fake.received_ready is not None

    fake.received_ready()
    assert events == ["ready"]


# --------------------------------------------------------------------------
# TTS behavior with pyttsx3 mocked (no audio output)
# --------------------------------------------------------------------------

def _install_fake_pyttsx3(monkeypatch, tts_module):
    calls = []

    class _FakeEngine:
        def setProperty(self, key, value):
            calls.append(("set", key, value))

        def say(self, text):
            calls.append(("say", text))

        def runAndWait(self):
            calls.append(("run",))

    if getattr(tts_module, "pyttsx3", None) is None:
        import pyttsx3 as _py
        tts_module.pyttsx3 = _py
    monkeypatch.setattr(tts_module.pyttsx3, "init", lambda: _FakeEngine())
    return calls


def test_tts_speak_disabled_is_noop(monkeypatch):
    import app.voice.tts as tts_module

    def _boom():
        raise AssertionError("pyttsx3.init must not be called when disabled")

    if getattr(tts_module, "pyttsx3", None) is None:
        import pyttsx3 as _py
        tts_module.pyttsx3 = _py

    monkeypatch.setattr(tts_module.pyttsx3, "init", _boom)

    tts = TTSEngine()
    tts.set_enabled(False)
    tts.speak("should not speak")  # must not raise / must not init


def test_tts_speak_runs_engine_when_enabled(monkeypatch):
    import app.voice.tts as tts_module

    calls = _install_fake_pyttsx3(monkeypatch, tts_module)

    class _SyncThread:
        def __init__(self, target=None, args=(), daemon=False):
            self._target = target
            self._args = args

        def start(self):
            self._target(*self._args)

    monkeypatch.setattr(tts_module.threading, "Thread", _SyncThread)

    tts = TTSEngine(rate=200, volume=0.9)
    tts.speak("hello")

    assert ("say", "hello") in calls
    assert ("run",) in calls


# --------------------------------------------------------------------------
# Voice toggle gating (offscreen; no real capture started)
# --------------------------------------------------------------------------

@pytest.fixture(scope="session")
def qapp():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


def _window_with_voice(qapp, enable_voice: bool):
    from app.ui.main_window import MainWindow

    win = MainWindow()
    # Replace the cached singleton so the flag mutation is test-local.
    win.settings = Settings(enable_gemini=False, enable_voice=enable_voice)

    started = []
    win.thread_pool = types.SimpleNamespace(start=lambda worker: started.append(worker))
    return win, started


def test_voice_toggle_disabled_does_not_start_capture(qapp):
    win, started = _window_with_voice(qapp, enable_voice=False)
    try:
        win._handle_voice_toggle(True)

        assert started == [], "microphone capture started while voice is disabled"
        assert win.command_input.btn_mic.isChecked() is False
    finally:
        win.close()


def test_voice_toggle_enabled_starts_capture(qapp):
    win, started = _window_with_voice(qapp, enable_voice=True)
    try:
        win._handle_voice_toggle(True)

        assert len(started) == 1, "microphone capture worker was not started"
    finally:
        win.close()


def test_voice_toggle_prompts_only_after_calibration_ready(qapp):
    win, started = _window_with_voice(qapp, enable_voice=True)
    try:
        text_before = win.command_input.btn_mic.text()
        win._handle_voice_toggle(True)

        assert len(started) == 1
        assert win.command_input.btn_mic.text() == text_before, (
            "mic button must not prompt the user to speak before calibration completes"
        )

        started[0].signals.ready.emit()
        assert win.command_input.btn_mic.text() == "🔴 Listening..."
    finally:
        win.close()
