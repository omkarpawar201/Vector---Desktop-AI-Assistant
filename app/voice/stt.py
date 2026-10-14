"""
Speech-to-Text (STT) Engine for Vector Desktop AI Assistant.

Three backends are available, selected by the ``voice_stt_backend`` setting:

* ``online``  - the existing SpeechRecognition recognizer (network call).
* ``offline`` - a local sherpa-onnx model only. This never touches the network,
  even when the package or model is missing or transcription fails.
* ``auto``    - prefers the local sherpa-onnx model and falls back to the online
  recognizer when offline is unavailable or fails.

The offline engine is imported and instantiated lazily, so no model is loaded at
application startup. Failures are logged by reason only -- the audio content,
credentials, and sensitive device details are never logged.
"""

import array
import logging
import sys
import wave
from pathlib import Path

logger = logging.getLogger(__name__)

_OFFLINE_BACKENDS = ("offline", "auto")

_CLIPPING_WARN_FRACTION = 0.001


def _pcm_clipped_fraction(pcm_bytes: bytes) -> float:
    """
    Fraction of 16-bit samples pinned at the positive or negative rail.

    The microphone is captured at a fixed Windows input level; if that level is
    too high the device (or a Microphone Boost stage) already clips before the
    driver. This measures that condition so it can be surfaced, without applying
    any software gain or normalization -- clipped samples are unrecoverable.
    """
    count = len(pcm_bytes) // 2
    if count <= 0:
        return 0.0
    samples = array.array("h")
    samples.frombytes(pcm_bytes[: count * 2])
    if sys.byteorder == "big":
        samples.byteswap()
    clipped = sum(1 for s in samples if s >= 32767 or s <= -32768)
    return clipped / count


class _SherpaOfflineBackend:
    """
    Lazy adapter around sherpa-onnx for local (offline) speech recognition.

    The constructor imports sherpa-onnx, so an ImportError propagates to the
    caller when the optional package is not installed. Instantiating this class
    loads the model, which is why STTEngine builds it only on first use.
    """

    def __init__(self, model_dir, sample_rate: int = 16000):
        import sherpa_onnx  # optional dependency; ImportError handled by caller

        self._sherpa = sherpa_onnx
        self.sample_rate = sample_rate
        self.recognizer = self._build(Path(model_dir))

    @staticmethod
    def _pick(files, *needles):
        """
        Return the preferred ``.onnx`` file whose name contains every fragment,
        preferring an int8-quantized variant when both exist.
        """
        candidates = [f for f in files if f.endswith(".onnx") and all(n in f for n in needles)]
        if not candidates:
            return None
        candidates.sort(key=lambda name: (".int8." not in name, name))
        return candidates[0]

    def _build(self, model_dir: Path):
        files = [p.name for p in model_dir.iterdir() if p.is_file()]
        tokens = next((f for f in files if "tokens" in f and f.endswith(".txt")), None)
        if tokens is None:
            raise ValueError("tokens file not found in offline STT model directory")

        encoder = self._pick(files, "encoder")
        decoder = self._pick(files, "decoder")
        joiner = self._pick(files, "joiner")

        # Zipformer/transducer layout (recommended offline model).
        if encoder and decoder and joiner:
            return self._sherpa.OfflineRecognizer.from_transducer(
                encoder=str(model_dir / encoder),
                decoder=str(model_dir / decoder),
                joiner=str(model_dir / joiner),
                tokens=str(model_dir / tokens),
                num_threads=1,
                decoding_method="greedy_search",
                provider="cpu",
            )
        # Whisper layout (encoder + decoder, no joiner).
        if encoder and decoder:
            return self._sherpa.OfflineRecognizer.from_whisper(
                encoder=str(model_dir / encoder),
                decoder=str(model_dir / decoder),
                tokens=str(model_dir / tokens),
                num_threads=1,
                provider="cpu",
            )
        raise ValueError("unrecognized sherpa-onnx model layout (need encoder/decoder[/joiner] + tokens)")

    def transcribe_pcm(self, pcm_bytes: bytes, sample_rate=None) -> str:
        """Decode 16-bit mono PCM bytes; sample_rate defaults to the model rate."""
        import numpy as np

        rate = sample_rate or self.sample_rate
        samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        stream = self.recognizer.create_stream()
        stream.accept_waveform(rate, samples)
        self.recognizer.decode_stream(stream)
        return stream.result.text

    def transcribe_file(self, audio_path: str) -> str:
        """Decode a 16-bit PCM WAV file."""
        import numpy as np

        with wave.open(str(audio_path), "rb") as wav:
            rate = wav.getframerate()
            channels = wav.getnchannels()
            width = wav.getsampwidth()
            frames = wav.readframes(wav.getnframes())

        if width != 2:
            raise ValueError("only 16-bit PCM WAV files are supported for offline STT")
        if channels > 1:
            mono = np.frombuffer(frames, dtype=np.int16).reshape(-1, channels)[:, 0]
            frames = mono.tobytes()
        return self.transcribe_pcm(frames, rate)


class STTEngine:
    """
    Speech-to-Text engine with selectable online/offline backends.

    The backend is resolved from the ``voice_stt_backend`` setting unless an
    explicit ``backend`` is passed (used by callers/tests).
    """

    def __init__(
        self,
        model_size: str = "tiny.en",
        backend: str = None,
        model_dir: str = None,
        sample_rate: int = 16000,
    ):
        self.model_size = model_size
        self.sample_rate = sample_rate
        self._whisper_model = None
        self._offline_backend = None
        self._offline_attempted = False
        self._backend_override = backend
        self._model_dir_override = model_dir

    @property
    def backend(self) -> str:
        """Resolve the active backend, defaulting to 'online' on any error."""
        if self._backend_override is not None:
            return self._backend_override
        try:
            from app.config.settings import get_settings

            return get_settings().voice_stt_backend
        except Exception:
            return "online"

    @property
    def model_dir(self) -> str:
        """Resolve the offline model directory from settings."""
        if self._model_dir_override is not None:
            return self._model_dir_override
        try:
            from app.config.settings import get_settings

            return get_settings().voice_stt_model_dir
        except Exception:
            return "models/stt"

    @property
    def is_loaded(self) -> bool:
        """
        Returns True once a local model (faster-whisper or offline sherpa-onnx)
        has been lazily instantiated, so the UI can show an accurate state
        without forcing an eager model download at startup.
        """
        return self._whisper_model is not None or self._offline_backend is not None

    def _get_offline(self):
        """
        Lazily construct the offline backend, or return None if it is
        unavailable (missing package, missing model directory, or load error).
        The outcome is cached so a failed probe is not retried on every call.
        """
        if self._offline_backend is not None:
            return self._offline_backend
        if self._offline_attempted:
            return None
        self._offline_attempted = True

        model_dir = self.model_dir
        if not model_dir or not Path(model_dir).is_dir():
            logger.warning("STT offline: model directory not found (%s)", model_dir)
            return None

        try:
            self._offline_backend = _SherpaOfflineBackend(model_dir, sample_rate=self.sample_rate)
        except ImportError:
            logger.warning("STT offline: local engine 'sherpa_onnx' is not installed")
        except Exception as e:
            logger.warning("STT offline: failed to initialize engine (%s): %s", type(e).__name__, e)
        return self._offline_backend

    def transcribe_file(self, audio_path: str) -> str:
        """
        Transcribes a WAV audio file using the selected backend. In 'offline'
        mode only the local model is ever consulted; 'auto' may fall back to the
        online recognizer.
        """
        p = Path(audio_path).resolve()
        if not p.exists() or p.stat().st_size == 0:
            return ""

        backend = self.backend
        if backend in _OFFLINE_BACKENDS:
            offline = self._get_offline()
            if offline is not None:
                try:
                    text = offline.transcribe_file(str(p))
                    if text and text.strip():
                        return text.strip()
                    logger.info("STT offline: no text produced for %s", p.name)
                except Exception as e:
                    logger.warning("STT offline: transcription failed (%s): %s", type(e).__name__, e)
                if backend == "offline":
                    return ""
                logger.info("STT: falling back to online recognizer")
            elif backend == "offline":
                logger.warning("STT offline: no offline backend available; returning empty")
                return ""

        return self._transcribe_file_online(p)

    def _transcribe_file_online(self, p: Path) -> str:
        # 1. SpeechRecognition (zero-config on Windows). Note: recognize_google
        #    is a network call, so this is not an offline path.
        try:
            import speech_recognition as sr

            r = sr.Recognizer()
            with sr.AudioFile(str(p)) as source:
                audio_data = r.record(source)
            text = r.recognize_google(audio_data)
            if text and text.strip():
                return text.strip()
            logger.info("STT: recognizer returned no text for %s", p.name)
        except ImportError:
            logger.warning("STT: 'speech_recognition' is not installed")
        except Exception as e:
            logger.warning("STT: SpeechRecognition failed (%s): %s", type(e).__name__, e)

        # 2. Optional local faster-whisper model, only if it is installed.
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            logger.warning("STT: local engine 'faster_whisper' is not installed")
            return ""

        try:
            if self._whisper_model is None:
                self._whisper_model = WhisperModel(self.model_size, device="cpu", compute_type="int8")
            segments, _ = self._whisper_model.transcribe(str(p), beam_size=1)
            return " ".join(s.text for s in segments).strip()
        except Exception as e:
            logger.warning("STT: faster-whisper failed (%s): %s", type(e).__name__, e)
            return ""

    def listen_and_transcribe_mic(self, phrase_time_limit: int = 6, on_ready=None) -> str:
        """
        Captures speech from the default Windows microphone and transcribes it.

        ``on_ready`` is invoked once ambient calibration has finished and just
        before recording starts, so callers can prompt the user to speak only
        when the microphone is actually listening.

        Returns an empty string when nothing intelligible was captured or the
        backend is unavailable. In 'offline' mode the online recognizer is never
        invoked. Failures are logged by reason only -- the audio content and any
        credentials are never logged.
        """
        try:
            import speech_recognition as sr
        except ImportError:
            logger.warning("STT: 'speech_recognition' is not installed; microphone capture unavailable")
            return ""

        recognizer = sr.Recognizer()
        try:
            with sr.Microphone() as source:
                recognizer.adjust_for_ambient_noise(source, duration=0.3)
                if on_ready is not None:
                    on_ready()
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=phrase_time_limit)
        except Exception as e:
            logger.warning("STT: microphone capture failed (%s): %s", type(e).__name__, e)
            return ""

        backend = self.backend
        if backend in _OFFLINE_BACKENDS:
            offline = self._get_offline()
            if offline is not None:
                try:
                    pcm = audio.get_raw_data(convert_rate=self.sample_rate, convert_width=2)
                    clipped = _pcm_clipped_fraction(pcm)
                    if clipped > _CLIPPING_WARN_FRACTION:
                        logger.warning(
                            "STT: microphone input is clipping (%.1f%% of samples at full "
                            "scale); lower the Windows input level or disable Microphone Boost",
                            clipped * 100.0,
                        )
                    text = offline.transcribe_pcm(pcm, self.sample_rate)
                    if text and text.strip():
                        return text.strip()
                    logger.info("STT offline: no speech recognized")
                except Exception as e:
                    logger.warning("STT offline: transcription failed (%s): %s", type(e).__name__, e)
                if backend == "offline":
                    return ""
                logger.info("STT: falling back to online recognizer")
            elif backend == "offline":
                logger.warning("STT offline: no offline backend available; returning empty")
                return ""

        return self._recognize_online(recognizer, audio, sr)

    @staticmethod
    def _recognize_online(recognizer, audio, sr) -> str:
        try:
            return recognizer.recognize_google(audio).strip()
        except sr.UnknownValueError:
            logger.info("STT: speech was not intelligible")
            return ""
        except sr.RequestError as e:
            logger.warning("STT: recognition service unreachable (%s)", type(e).__name__)
            return ""
        except Exception as e:
            logger.warning("STT: transcription failed (%s): %s", type(e).__name__, e)
            return ""
