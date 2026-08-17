"""The 10 required robustness modifications, each parametrized by an explicit
severity grid. Grids are chosen from realistic/literature-informed starting
ranges and are meant to be *swept* by the benchmark harness to find the
empirical accuracy-crossover boundary per domain - no level here was chosen
because it "looked good"; the benchmark reports which levels actually pass or
fail per domain (see REPORT.md).
"""

import subprocess
import tempfile
from pathlib import Path

import imageio_ffmpeg
import librosa
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfiltfilt

from sonic.config import ANALYSIS_SR, RESULTS_ROOT

FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()
TMP_DIR = RESULTS_ROOT / "tmp"
TMP_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------- pitch shift
def pitch_shift(y: np.ndarray, sr: int, semitones: float) -> np.ndarray:
    return librosa.effects.pitch_shift(y=y, sr=sr, n_steps=semitones)


PITCH_SHIFT_LEVELS = [-8, -6, -4, -2, -1, 1, 2, 4, 6, 8]  # semitones


# ------------------------------------------------------------- speed / tempo
def speed_change(y: np.ndarray, sr: int, rate: float) -> np.ndarray:
    return librosa.effects.time_stretch(y=y, rate=rate)


SPEED_LEVELS = [0.7, 0.8, 0.9, 0.95, 1.05, 1.1, 1.25, 1.5]  # playback-rate multiplier


# --------------------------------------------------------------- compression
def mp3_compress(y: np.ndarray, sr: int, bitrate_kbps: int) -> np.ndarray:
    with tempfile.NamedTemporaryFile(dir=TMP_DIR, suffix=".wav", delete=False) as f_wav:
        wav_path = f_wav.name
    mp3_path = wav_path.replace(".wav", ".mp3")
    out_path = wav_path.replace(".wav", "_out.wav")
    try:
        sf.write(wav_path, y, sr)
        subprocess.run(
            [FFMPEG_EXE, "-y", "-i", wav_path, "-b:a", f"{bitrate_kbps}k", mp3_path],
            capture_output=True, check=True,
        )
        subprocess.run(
            [FFMPEG_EXE, "-y", "-i", mp3_path, out_path],
            capture_output=True, check=True,
        )
        y_out, _ = librosa.load(out_path, sr=sr, mono=True)
        return y_out.astype(np.float32)
    finally:
        for p in (wav_path, mp3_path, out_path):
            Path(p).unlink(missing_ok=True)


MP3_BITRATE_LEVELS = [128, 64, 32, 16]  # kbps


# --------------------------------------------------------------------- trims
def trim_start(y: np.ndarray, sr: int, seconds: float) -> np.ndarray:
    n = int(seconds * sr)
    return y[n:] if n < y.size else y[-1:]


def trim_end(y: np.ndarray, sr: int, seconds: float) -> np.ndarray:
    n = int(seconds * sr)
    return y[:-n] if n < y.size else y[:1]


TRIM_LEVELS_SEC = [0.5, 1.0, 2.0, 5.0, 10.0]


# ------------------------------------------------------------------- noising
def _mix_at_snr(y: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    if noise.size < y.size:
        reps = int(np.ceil(y.size / noise.size))
        noise = np.tile(noise, reps)
    noise = noise[: y.size]

    sig_power = float(np.mean(y ** 2)) + 1e-12
    noise_power = float(np.mean(noise ** 2)) + 1e-12
    target_noise_power = sig_power / (10 ** (snr_db / 10))
    scale = np.sqrt(target_noise_power / noise_power)
    mixed = y + noise * scale

    peak = np.max(np.abs(mixed))
    if peak > 1.0:
        mixed = mixed / peak
    return mixed.astype(np.float32)


def background_noise(y: np.ndarray, sr: int, snr_db: float, noise_y: np.ndarray) -> np.ndarray:
    return _mix_at_snr(y, noise_y, snr_db)


BACKGROUND_NOISE_SNR_LEVELS = [20, 15, 10, 5, 0]  # dB


def white_noise(y: np.ndarray, sr: int, snr_db: float) -> np.ndarray:
    rng = np.random.default_rng(abs(hash((y.shape[0], snr_db))) % (2 ** 32))
    noise = rng.standard_normal(y.size).astype(np.float32)
    return _mix_at_snr(y, noise, snr_db)


WHITE_NOISE_SNR_LEVELS = [30, 20, 10, 5, 0]  # dB


# --------------------------------------------------------------- volume/gain
def gain_change(y: np.ndarray, sr: int, gain_db: float) -> np.ndarray:
    scale = 10 ** (gain_db / 20)
    return np.clip(y * scale, -1.0, 1.0).astype(np.float32)


GAIN_LEVELS_DB = [-20, -10, -6, 6, 10, 20]


# ------------------------------------------------------------------ filtering
def _sos_filter(y: np.ndarray, sr: int, kind: str, cutoff) -> np.ndarray:
    nyq = sr / 2
    if kind == "bandpass":
        lo, hi = cutoff
        sos = butter(4, [lo / nyq, min(hi / nyq, 0.999)], btype="bandpass", output="sos")
    elif kind == "lowpass":
        sos = butter(4, cutoff / nyq, btype="lowpass", output="sos")
    else:  # highpass
        sos = butter(4, cutoff / nyq, btype="highpass", output="sos")
    return sosfiltfilt(sos, y).astype(np.float32)


def apply_filter(y: np.ndarray, sr: int, spec) -> np.ndarray:
    kind, cutoff = spec
    return _sos_filter(y, sr, kind, cutoff)


FILTER_LEVELS = [
    ("bandpass", (300, 3400)),   # telephone band
    ("lowpass", 8000),
    ("lowpass", 4000),
    ("lowpass", 2000),
    ("highpass", 100),
    ("highpass", 300),
]


# ------------------------------------------------------------------ time shift
def time_shift(y: np.ndarray, sr: int, offset_frac: float, window_sec: float = 8.0) -> np.ndarray:
    duration = y.size / sr
    start = offset_frac * duration
    start_n = int(start * sr)
    win_n = int(window_sec * sr)
    end_n = min(start_n + win_n, y.size)
    start_n = min(start_n, max(0, y.size - 1))
    seg = y[start_n:end_n]
    return seg if seg.size > sr * 0.5 else y


TIME_SHIFT_LEVELS = [0.1, 0.25, 0.4, 0.5]  # fraction of clip duration


def time_shift_ms(y: np.ndarray, sr: int, ms: float) -> np.ndarray:
    """Small millisecond-scale start-boundary shift (alignment-jitter test),
    distinct from `time_shift` above (which extracts a window starting deep
    into the clip). Positive ms trims that many milliseconds off the start
    (query starts slightly late); negative ms pads that many milliseconds of
    silence onto the start (query starts slightly early)."""
    n = int(round(abs(ms) / 1000.0 * sr))
    if n == 0:
        return y
    if ms > 0:
        return y[n:] if n < y.size else y[-1:]
    return np.concatenate([np.zeros(n, dtype=y.dtype), y])


# ---------------------------------------------------------------------------
# Exact, fixed-severity modification set for the primary robustness benchmark
# (user-specified, run on all 200 indexed files/domain - NOT the broader
# exploratory severity sweep above used earlier for diagnosis). Each "+-X"
# spec is tested as its two signed directions, not a single midpoint, since a
# direction-agnostic magnitude is what "+-X" specifies; each plain range
# (e.g. trim "0.02-0.08s") is tested at its midpoint, a single fixed severity
# consistent with "do not test additional severity levels."
# ---------------------------------------------------------------------------
PRIMARY_BENCHMARK_MODIFICATIONS = {
    "pitch_shift": {"fn": pitch_shift, "levels": [-0.5, 0.5], "unit": "semitones"},
    "speed_change": {"fn": speed_change, "levels": [0.995, 1.005], "unit": "rate_multiplier"},
    "mp3_compression": {"fn": mp3_compress, "levels": [192], "unit": "kbps_CBR"},
    "trim_start": {"fn": trim_start, "levels": [0.05], "unit": "seconds_removed"},  # midpoint of 0.02-0.08s
    "trim_end": {"fn": trim_end, "levels": [0.05], "unit": "seconds_removed"},      # midpoint of 0.02-0.08s
    "background_noise": {"fn": background_noise, "levels": [30], "unit": "SNR_dB", "needs_noise": True},
    "white_noise": {"fn": white_noise, "levels": [35], "unit": "SNR_dB"},
    "gain_change": {"fn": gain_change, "levels": [-1, 1], "unit": "dB"},
    "filtering": {"fn": apply_filter, "levels": [("lowpass", 10000), ("highpass", 50)], "unit": "filter_spec"},
    "time_shift": {"fn": time_shift_ms, "levels": [-20, 20], "unit": "milliseconds"},
}


MODIFICATIONS = {
    "pitch_shift": {"fn": pitch_shift, "levels": PITCH_SHIFT_LEVELS, "unit": "semitones"},
    "speed_change": {"fn": speed_change, "levels": SPEED_LEVELS, "unit": "rate_multiplier"},
    "mp3_compression": {"fn": mp3_compress, "levels": MP3_BITRATE_LEVELS, "unit": "kbps"},
    "trim_start": {"fn": trim_start, "levels": TRIM_LEVELS_SEC, "unit": "seconds_removed"},
    "trim_end": {"fn": trim_end, "levels": TRIM_LEVELS_SEC, "unit": "seconds_removed"},
    "background_noise": {"fn": background_noise, "levels": BACKGROUND_NOISE_SNR_LEVELS, "unit": "SNR_dB", "needs_noise": True},
    "white_noise": {"fn": white_noise, "levels": WHITE_NOISE_SNR_LEVELS, "unit": "SNR_dB"},
    "gain_change": {"fn": gain_change, "levels": GAIN_LEVELS_DB, "unit": "dB"},
    "filtering": {"fn": apply_filter, "levels": FILTER_LEVELS, "unit": "filter_spec"},
    "time_shift": {"fn": time_shift, "levels": TIME_SHIFT_LEVELS, "unit": "fraction_of_duration"},
}
