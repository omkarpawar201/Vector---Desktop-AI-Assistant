"""
Read-only WAV signal analyzer for microphone diagnostic recordings.

Reports format, duration, peak/RMS levels (linear and dBFS), DC offset,
clipping, and frame-level speech/silence statistics. The target file is opened
for reading only and nothing is written; no application modules are imported,
so stt.py, settings, models, datasets, and checkpoints are never touched.

Usage:
    python scripts/analyze_wav.py [wav-path] [--threshold RMS] [--window-ms MS]
"""

import argparse
import math
import struct
import statistics
import wave

DEFAULT_WAV = r"C:\Users\omkar\Documents\Sound Recordings\Recording.wav"
FULL_SCALE = 32768.0


def to_dbfs(amplitude: float) -> float:
    if amplitude <= 0.0:
        return float("-inf")
    return 20.0 * math.log10(amplitude / FULL_SCALE)


def load_mono(path: str):
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sampwidth = wav.getsampwidth()
        rate = wav.getframerate()
        nframes = wav.getnframes()
        comptype = wav.getcomptype()
        raw = wav.readframes(nframes)

    if comptype != "NONE":
        raise ValueError(f"unsupported WAV compression: {comptype}")
    if sampwidth != 2:
        raise ValueError(f"only 16-bit PCM is supported (got {sampwidth * 8}-bit)")

    count = len(raw) // 2
    interleaved = struct.unpack(f"<{count}h", raw[: count * 2])

    if channels > 1:
        samples = tuple(
            round(sum(interleaved[i : i + channels]) / channels)
            for i in range(0, count - channels + 1, channels)
        )
    else:
        samples = interleaved

    return {"channels": channels, "rate": rate, "nframes": nframes, "samples": samples}


def rms_of(samples) -> float:
    if not samples:
        return 0.0
    return math.sqrt(sum(s * s for s in samples) / len(samples))


def analyze(path: str, threshold: float, window_ms: float) -> None:
    info = load_mono(path)
    samples = info["samples"]
    rate = info["rate"]
    total = len(samples)
    duration = total / rate if rate else 0.0

    peak = max((abs(s) for s in samples), default=0)
    peak_index = max(range(total), key=lambda i: abs(samples[i])) if total else 0
    rms = rms_of(samples)
    dc = sum(samples) / total if total else 0.0
    clipped = sum(1 for s in samples if s >= 32767 or s <= -32768)

    window = max(1, int(rate * window_ms / 1000.0))
    windows = [rms_of(samples[i : i + window]) for i in range(0, total, window)]
    active = [w >= threshold for w in windows]

    first = next((i for i, a in enumerate(active) if a), None)
    last = next((len(active) - 1 - i for i, a in enumerate(reversed(active)) if a), None)

    lead_silence = first * window / rate if first is not None else duration
    last_end_sample = min((last + 1) * window, total) if last is not None else 0
    tail_silence = (total - last_end_sample) / rate if last is not None else 0.0
    active_samples = last_end_sample - (first * window if first is not None else 0)
    speech_duration = active_samples / rate if active_samples > 0 else 0.0

    inactive_windows = [w for w in windows if w < threshold]
    noise_floor = statistics.median(inactive_windows) if inactive_windows else 0.0
    loudest = max(windows) if windows else 0.0
    loudest_at = windows.index(loudest) * window / rate if windows else 0.0

    print(f"Read-only analysis of: {path}")
    print(f"Format: {rate} Hz, {info['channels']} channel(s), 16-bit PCM, "
          f"{info['nframes']} frames")
    print(f"Duration: {duration:.3f} s")
    print(f"Peak: {peak} linear, {to_dbfs(peak):.2f} dBFS at {peak_index / rate:.3f} s")
    print(f"RMS: {rms:.1f} linear, {to_dbfs(rms):.2f} dBFS")
    print(f"DC offset: {dc:.2f} linear")
    print(f"Clipped samples (|x| >= full scale): {clipped}")
    print(f"Window: {window_ms:.0f} ms ({len(windows)} windows), "
          f"activity threshold RMS >= {threshold:.0f}")

    if not windows:
        print("No audio samples to analyze.")
        return

    print(f"Noise floor (median of inactive windows): {noise_floor:.1f} RMS")
    if first is None:
        print("Speech onset: none (no window reached the activity threshold)")
        print("Active speech duration: 0.000 s (0% of windows)")
    else:
        print(f"Speech onset: {lead_silence:.3f} s (leading silence {lead_silence:.3f} s)")
        print(f"Speech end: {last_end_sample / rate:.3f} s (trailing silence {tail_silence:.3f} s)")
        print(f"Active speech duration: {speech_duration:.3f} s "
              f"({sum(active) / len(active) * 100:.0f}% of windows)")
        if lead_silence <= window_ms / 1000.0:
            print("Onset note: activity begins in the first window "
                  "(possible onset cut-off / no leading silence captured)")
        if tail_silence <= window_ms / 1000.0:
            print("End note: activity runs to the last window "
                  "(possible trailing cut-off)")
    print(f"Loudest window: RMS {loudest:.1f} at {loudest_at:.3f} s")
    print(f"Threshold hint (1.5 x noise floor): {noise_floor * 1.5:.1f} RMS")
    print("No files were written; analysis is read-only.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only WAV signal statistics.")
    parser.add_argument("wav", nargs="?", default=DEFAULT_WAV, help="path to a 16-bit PCM WAV")
    parser.add_argument("--threshold", type=float, default=300.0,
                        help="frame-level RMS activity threshold (default 300, SR default)")
    parser.add_argument("--window-ms", type=float, default=20.0,
                        help="analysis window in milliseconds (default 20)")
    args = parser.parse_args()

    try:
        analyze(args.wav, args.threshold, args.window_ms)
    except FileNotFoundError:
        print(f"File not found: {args.wav}")
        return 1
    except (ValueError, wave.Error) as exc:
        print(f"Cannot analyze {args.wav}: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
