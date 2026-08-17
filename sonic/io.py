"""Shared audio loading used identically for indexing and querying, so preprocessing
never diverges between the two paths.

Deliberately implemented on soundfile + numpy only, NOT librosa: librosa
transitively requires numba and scikit-learn, and profiling showed that the
first real use of librosa.load/librosa.stft lazily JIT-compiles librosa's
internal numba-accelerated routines, adding a one-time ~100-150MB RSS tax to
the process regardless of dataset size - enough on its own to blow the
<120MB peak-RAM budget. soundfile (libsndfile) decodes wav/mp3 directly with a
tiny footprint, and resampling is done with a plain FFT resample (the same
algorithm scipy.signal.resample uses) so no extra heavy compiled dependency is
pulled into the serving path. librosa is still used offline, in
augment/modifications.py only, for pitch_shift/time_stretch test-data
generation - that is a benchmark-authoring tool, not part of the deployed
retrieval system, so its footprint doesn't count against the serving budget.
"""

import numpy as np
import soundfile as sf

from sonic.config import ANALYSIS_SR


def _next_pow2(n: int) -> int:
    return 1 << max(int(n) - 1, 0).bit_length()


def _resample_fft(y: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """FFT resample (same algorithm as scipy.signal.resample), padded to a
    power-of-two length before transforming. Measured: an unpadded rfft/irfft
    pair on this dataset's exact sample counts landed on prime-length arrays
    (e.g. a 30s/16kHz music file resamples to a 660983-sample target, which is
    prime) - numpy's FFT falls back to an O(n^2)-ish path for such lengths,
    turning a ~15ms resample into ~180ms. Padding to a power of two keeps every
    call on the fast O(n log n) path regardless of the specific file duration.
    """
    if orig_sr == target_sr or y.size == 0:
        return y.astype(np.float32)
    n = y.size
    new_n = max(1, int(round(n * target_sr / orig_sr)))

    n_fft_in = _next_pow2(n)
    y_padded = np.pad(y, (0, n_fft_in - n))
    Y = np.fft.rfft(y_padded)

    n_fft_out = _next_pow2(max(1, int(round(n_fft_in * target_sr / orig_sr))))
    n_freq_new = n_fft_out // 2 + 1
    if n_freq_new <= Y.size:
        Y2 = Y[:n_freq_new]
    else:
        Y2 = np.zeros(n_freq_new, dtype=Y.dtype)
        Y2[: Y.size] = Y

    y2_padded = np.fft.irfft(Y2, n=n_fft_out)
    y2_padded *= n_fft_out / n_fft_in
    return y2_padded[:new_n].astype(np.float32)


def load_audio(path: str, sr: int = ANALYSIS_SR, max_duration: float | None = None,
                offset: float = 0.0) -> np.ndarray:
    """Load an audio file as mono float32 at the shared analysis sample rate,
    peak-normalized to [-1, 1]. Returns an empty array for zero-length input.

    `max_duration`/`offset` (seconds) are converted to a native-sample-rate
    frame range and passed to soundfile so the decoder never reads more of the
    file than needed - this is what keeps query-time latency bounded for the
    long tail of long recordings (see fingerprint/landmarks.py's
    QUERY_MAX_DURATION_SEC). Indexing always uses the full file
    (max_duration=None) so a bounded query can still match any position within
    a long indexed track.
    """
    info = sf.info(path)
    native_sr = info.samplerate
    start_frame = int(offset * native_sr)
    frames = int(max_duration * native_sr) if max_duration is not None else -1

    y, _ = sf.read(path, start=start_frame, frames=frames, dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = y.astype(np.float32)

    y = _resample_fft(y, native_sr, sr)

    peak = np.max(np.abs(y)) if y.size else 0.0
    if peak > 1e-8:
        y = y / peak
    return y


def trim_leading_silence(y: np.ndarray, sr: int, rel_threshold: float = 0.15,
                          frame_sec: float = 0.1) -> int:
    """Returns the sample index where content starts (first frame whose RMS
    exceeds rel_threshold * the array's peak frame RMS), or 0 if the whole
    array is already active or too short to frame. Used to skip a quiet/pause
    opening before taking a bounded query window - see query_window() below."""
    frame_len = max(1, int(frame_sec * sr))
    n_frames = y.size // frame_len
    if n_frames < 2:
        return 0
    frames = y[: n_frames * frame_len].reshape(n_frames, frame_len)
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
    peak_rms = rms.max()
    if peak_rms <= 1e-8:
        return 0
    above = np.nonzero(rms > rel_threshold * peak_rms)[0]
    if above.size == 0:
        return 0
    return int(above[0]) * frame_len


def load_query_window(path: str, target_duration: float, sr: int = ANALYSIS_SR,
                       lookahead_sec: float = 10.0) -> np.ndarray:
    """Loads up to (target_duration + lookahead_sec) seconds from the start of
    a file, skips a leading quiet/pause section if present, and returns a
    window of at most target_duration seconds starting from where content
    begins. This keeps the per-query STFT/fingerprint cost bounded at
    target_duration while still avoiding the accuracy loss a fixed cold t=0
    cutoff causes on recordings (e.g. spontaneous/unscripted speech) whose
    first few seconds are a pause rather than real content - see
    config.QUERY_MAX_DURATION_SEC_BY_DOMAIN for the measured trade-off this
    fixes."""
    y = load_audio(path, sr=sr, max_duration=target_duration + lookahead_sec)
    if y.size == 0:
        return y
    start = trim_leading_silence(y, sr)
    target_samples = int(target_duration * sr)
    return y[start : start + target_samples]
