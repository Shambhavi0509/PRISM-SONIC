"""Primary fingerprint: log-frequency landmark (constellation) hashing with
pitch- and tempo-invariant relative encoding.

Design rationale (see REPORT.md Sec. 7-9 for full literature discussion):

- Classic Shazam-style fingerprints (Wang, 2003) pick spectral peaks on a *linear*
  STFT and hash absolute (f1, f2, dt) triples with an absolute time offset for
  histogram alignment. A pitch shift moves every frequency by a *multiplicative*
  factor, which is not a constant additive offset in linear-frequency space, so
  linear-STFT landmark hashes break under pitch shift. A tempo/speed change scales
  every time gap by a constant factor, which also breaks the fixed-offset histogram
  assumption.
- We instead pick peaks on a LOG-frequency spectrogram (a fast STFT + triangular
  log-spaced filterbank, i.e. a cheap pseudo-CQT that avoids full CQT kernel
  convolution cost). In log-frequency space, a pitch shift becomes an *additive*
  translation of every peak's frequency coordinate.
- We hash PEAK TRIPLETS (anchor p0, p1, p2 with p0 earliest in time), using
  Delta-f1 = f1 - f0 and Delta-f2 = f2 - f0 (both invariant to a global additive
  log-freq translation, i.e. invariant to pitch shift), and a time RATIO
  r = (t2 - t0) / (t1 - t0) (invariant to a global multiplicative time scaling,
  i.e. invariant to tempo/speed change). This is the mechanism used by Panako
  (Six & Leman, 2014) to achieve simultaneous pitch- and time-scale invariance.
- Because the hash itself carries no absolute time/frequency, matching is scored
  by raw hash-collision COUNT between query and candidate (this still identifies
  the correct file even under pitch/tempo change). A secondary, absolute-time
  offset histogram is *also* computed as an auxiliary consistency signal (used for
  confidence calibration and explainability), which is most informative when the
  audio is unmodified or only mildly time-modified.
"""

import itertools
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import maximum_filter

from sonic.config import ANALYSIS_SR

# NOTE: the triplet-building step below is deliberately implemented as pure
# vectorized NumPy rather than a Numba @njit loop. An earlier Numba version was
# ~5x faster per-call, but profiling showed Numba's first JIT compilation pulls
# in its LLVM backend and adds a one-time ~100MB RSS tax to the process - on its
# own enough to blow the <120MB peak-RAM budget regardless of index size. The
# vectorized-NumPy version below has no such fixed cost and is still well within
# the latency budget (see REPORT.md's RAM/latency breakdown for the measured
# trade-off).

# --- STFT / log-frequency filterbank parameters -----------------------------
# HOP_LENGTH=256 (~11.6ms/frame @22050Hz): tried doubling to 512 to roughly
# halve STFT/filterbank/peak-picking cost, but that coarser time resolution
# measurably hurt Environmental duplicate accuracy (197/200 -> 191/200) - many
# environmental sounds are short transients (this dataset's shortest clip is
# 0.32s) where the finer hop matters. Reverted; per-domain query-duration caps
# (see config.QUERY_MAX_DURATION_SEC_BY_DOMAIN) are the latency lever instead.
N_FFT = 1024
HOP_LENGTH = 256
F_MIN = 50.0
F_MAX = 10000.0
NUM_LOG_BINS = 192

# --- Peak-picking parameters --------------------------------------------------
PEAK_FREQ_NEIGHBORHOOD = 9   # bins
PEAK_TIME_NEIGHBORHOOD = 7   # frames
PEAKS_PER_SEC_TARGET = 20
MIN_MAG_PERCENTILE = 60.0    # discard peaks below this percentile of nonzero magnitudes

# --- Triplet / hashing parameters --------------------------------------------
# Kept deliberately tight (short window, few neighbors/triplets per anchor):
# with correct windowed neighbor search (see _build_triplets_vectorized), a wide
# window at PEAKS_PER_SEC_TARGET density produces enough triplets per anchor
# that index size/query latency/RAM all overshoot budget; this configuration
# was chosen by sweeping density down until duplicate accuracy just stays at
# 100% while RAM/latency clear their targets with margin (see REPORT.md).
TARGET_ZONE_MIN_SEC = 0.05
TARGET_ZONE_MAX_SEC = 0.8
MAX_NEIGHBORS_PER_ANCHOR = 5
MAX_TRIPLETS_PER_ANCHOR = 4

DF_BITS = 9          # bits for each of Delta-f1, Delta-f2
DF_CLAMP = 255        # clamp Delta-f (in log-bin units) to [-DF_CLAMP, DF_CLAMP]
R_BITS = 10           # bits for quantized log2(time ratio)
R_MIN, R_MAX = 1.0, 8.0  # clamp range for time ratio r = (t2-t0)/(t1-t0)


@dataclass
class Landmark:
    hash: np.ndarray      # uint32 hash keys, shape (n,)
    anchor_time: np.ndarray  # float32 seconds, shape (n,)


_FILTERBANK_CACHE = {}


def _build_log_filterbank(sr: int, n_fft: int) -> np.ndarray:
    key = (sr, n_fft)
    if key in _FILTERBANK_CACHE:
        return _FILTERBANK_CACHE[key]

    n_linear = n_fft // 2 + 1
    linear_freqs = np.linspace(0, sr / 2, n_linear)

    log_edges = np.geomspace(F_MIN, F_MAX, NUM_LOG_BINS + 2)
    fb = np.zeros((NUM_LOG_BINS, n_linear), dtype=np.float32)
    for i in range(NUM_LOG_BINS):
        f_lo, f_ctr, f_hi = log_edges[i], log_edges[i + 1], log_edges[i + 2]
        left = (linear_freqs - f_lo) / max(f_ctr - f_lo, 1e-9)
        right = (f_hi - linear_freqs) / max(f_hi - f_ctr, 1e-9)
        tri = np.clip(np.minimum(left, right), 0.0, None)
        fb[i] = tri

    _FILTERBANK_CACHE[key] = fb
    return fb


def stft_magnitude(y: np.ndarray, sr: int = ANALYSIS_SR) -> tuple[np.ndarray, np.ndarray]:
    """Returns (magnitude_spectrogram, frame_times) at the shared N_FFT/HOP_LENGTH
    resolution. Computed once per query/index file and reused by BOTH the
    primary landmark log-filterbank and the secondary mel-filterbank embedding
    (see fingerprint/binary_embed.py) - halves per-file STFT cost versus each
    stage computing its own."""
    if y.size < N_FFT:
        y = np.pad(y, (0, N_FFT - y.size))
    stft = compute_stft(y)
    mag = np.abs(stft).astype(np.float32)
    frame_times = np.arange(mag.shape[1], dtype=np.float32) * HOP_LENGTH / sr
    return mag, frame_times


def log_spectrogram_from_magnitude(mag: np.ndarray, sr: int = ANALYSIS_SR) -> np.ndarray:
    fb = _build_log_filterbank(sr, N_FFT)
    return fb @ mag


def log_spectrogram(y: np.ndarray, sr: int = ANALYSIS_SR) -> tuple[np.ndarray, np.ndarray]:
    """Returns (log_spec, frame_times). log_spec shape (NUM_LOG_BINS, n_frames)."""
    mag, frame_times = stft_magnitude(y, sr)
    return log_spectrogram_from_magnitude(mag, sr), frame_times


_HANN_CACHE = {}


def compute_stft(y: np.ndarray, n_fft: int = N_FFT, hop_length: int = HOP_LENGTH) -> np.ndarray:
    """Plain-NumPy STFT (center-padded, periodic-Hann windowed), equivalent to
    librosa.stft(y, n_fft, hop_length, window='hann') but without depending on
    librosa (see io.py's module docstring for why: librosa's numba dependency
    costs ~100MB+ of RSS on first use, which the serving path must avoid)."""
    if n_fft not in _HANN_CACHE:
        _HANN_CACHE[n_fft] = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n_fft) / n_fft)
    win = _HANN_CACHE[n_fft]

    pad = n_fft // 2
    y_padded = np.pad(y, (pad, pad), mode="reflect")
    n_frames = 1 + (y_padded.size - n_fft) // hop_length
    if n_frames <= 0:
        return np.zeros((n_fft // 2 + 1, 0), dtype=np.complex64)

    idx = np.arange(n_fft)[:, None] + hop_length * np.arange(n_frames)[None, :]
    frames = y_padded[idx] * win[:, None]
    return np.fft.rfft(frames, axis=0).astype(np.complex64)


def pick_peaks(log_spec: np.ndarray, frame_times: np.ndarray):
    """Local-maxima peak picking with density control. Returns (freq_bins, times, mags)."""
    if log_spec.size == 0:
        return np.array([], dtype=np.int32), np.array([], dtype=np.float32), np.array([], dtype=np.float32)

    footprint = (PEAK_FREQ_NEIGHBORHOOD, PEAK_TIME_NEIGHBORHOOD)
    local_max = maximum_filter(log_spec, size=footprint, mode="constant")
    is_peak = (log_spec == local_max) & (log_spec > 0)

    mags = log_spec[is_peak]
    if mags.size == 0:
        return np.array([], dtype=np.int32), np.array([], dtype=np.float32), np.array([], dtype=np.float32)

    thresh = np.percentile(mags, MIN_MAG_PERCENTILE)
    is_peak &= log_spec > thresh

    freq_idx, time_idx = np.nonzero(is_peak)
    peak_mags = log_spec[freq_idx, time_idx]
    peak_times = frame_times[time_idx]

    duration = frame_times[-1] - frame_times[0] if frame_times.size > 1 else 1.0
    duration = max(duration, 1.0)
    target_n = int(PEAKS_PER_SEC_TARGET * duration)
    if peak_mags.size > target_n:
        top_idx = np.argpartition(-peak_mags, target_n)[:target_n]
        freq_idx, peak_times, peak_mags = freq_idx[top_idx], peak_times[top_idx], peak_mags[top_idx]

    order = np.argsort(peak_times, kind="stable")
    return freq_idx[order].astype(np.int32), peak_times[order].astype(np.float32), peak_mags[order].astype(np.float32)


_LOG2_R_MAX = float(np.log2(R_MAX))
_R_QUANT_MAX = float((1 << R_BITS) - 1)

# Fixed (o1, o2) offset templates, applied relative to each anchor's window
# START (the first peak at least TARGET_ZONE_MIN_SEC later - see `lo` below),
# not relative to the anchor's own array index. Using the anchor's own index as
# the reference point (an earlier version of this function did that) silently
# under-counts landmarks in dense regions: whenever >6 peaks fall inside the
# target_min gap itself (common, since multiple frequency bins can each be a
# local-maxima "peak" at the same or adjacent time frames), "the next 6 peaks by
# index" never reaches past that gap at all, so nearly every candidate pair
# fails the target_min check and the anchor contributes ~0 triplets. Anchoring
# offsets to the window start instead reproduces the original (pre-vectorization)
# windowed-search semantics exactly, while staying fully vectorizable via
# `np.searchsorted`'s array form.
_PAIR_TEMPLATES = list(itertools.combinations(range(0, MAX_NEIGHBORS_PER_ANCHOR), 2))[:MAX_TRIPLETS_PER_ANCHOR]
_O1 = np.array([o1 for o1, _ in _PAIR_TEMPLATES], dtype=np.int64)
_O2 = np.array([o2 for _, o2 in _PAIR_TEMPLATES], dtype=np.int64)


def _build_triplets_vectorized(freq_idx: np.ndarray, times: np.ndarray):
    n = times.shape[0]
    if n < 3:
        return np.array([], dtype=np.uint32), np.array([], dtype=np.float32)

    base = np.arange(n, dtype=np.int64)
    # First index whose time is >= TARGET_ZONE_MIN_SEC after each anchor -
    # vectorized equivalent of the original per-anchor `while` scan.
    lo = np.searchsorted(times, times + TARGET_ZONE_MIN_SEC, side="left")
    lo = np.maximum(lo, base + 1)[:, None]                  # (n, 1)

    idx1 = lo + _O1[None, :]                                # (n, K)
    idx2 = lo + _O2[None, :]                                # (n, K)
    in_bounds = (idx1 < n) & (idx2 < n)

    idx1_safe = np.clip(idx1, 0, n - 1)
    idx2_safe = np.clip(idx2, 0, n - 1)

    t0 = times[base][:, None]                               # (n, 1) broadcasts
    f0 = freq_idx[base][:, None]
    t1, f1 = times[idx1_safe], freq_idx[idx1_safe]
    t2, f2 = times[idx2_safe], freq_idx[idx2_safe]

    dt1 = t1 - t0
    dt2 = t2 - t0

    valid = (
        in_bounds
        & (dt1 >= TARGET_ZONE_MIN_SEC) & (dt1 <= TARGET_ZONE_MAX_SEC)
        & (dt2 >= TARGET_ZONE_MIN_SEC) & (dt2 <= TARGET_ZONE_MAX_SEC)
        & (dt2 > dt1)
    )

    dt1_safe = np.where(valid, dt1, 1.0)
    r = dt2 / dt1_safe
    valid &= (r >= R_MIN) & (r <= R_MAX)

    df1 = np.clip(f1 - f0, -DF_CLAMP, DF_CLAMP) + DF_CLAMP
    df2 = np.clip(f2 - f0, -DF_CLAMP, DF_CLAMP) + DF_CLAMP

    r_safe = np.clip(r, R_MIN, R_MAX)
    log_r = np.log2(r_safe) / _LOG2_R_MAX
    rq = np.round(log_r * _R_QUANT_MAX).astype(np.int64)

    h = (df1.astype(np.uint32) << np.uint32(DF_BITS + R_BITS)) \
        | (df2.astype(np.uint32) << np.uint32(R_BITS)) \
        | rq.astype(np.uint32)

    flat_valid = valid.reshape(-1)
    flat_hash = h.reshape(-1)[flat_valid]
    flat_time = np.broadcast_to(t0, h.shape).reshape(-1)[flat_valid]

    return flat_hash.astype(np.uint32), flat_time.astype(np.float32)


def build_landmarks_from_magnitude(mag: np.ndarray, frame_times: np.ndarray, sr: int = ANALYSIS_SR) -> Landmark:
    log_spec = log_spectrogram_from_magnitude(mag, sr)
    freq_idx, times, _mags = pick_peaks(log_spec, frame_times)

    if times.size < 3:
        return Landmark(hash=np.array([], dtype=np.uint32), anchor_time=np.array([], dtype=np.float32))

    hashes, anchor_times = _build_triplets_vectorized(freq_idx, times)
    return Landmark(hash=hashes, anchor_time=anchor_times)


def build_landmarks(y: np.ndarray, sr: int = ANALYSIS_SR) -> Landmark:
    log_spec, frame_times = log_spectrogram(y, sr)
    freq_idx, times, _mags = pick_peaks(log_spec, frame_times)

    if times.size < 3:
        return Landmark(hash=np.array([], dtype=np.uint32), anchor_time=np.array([], dtype=np.float32))

    hashes, anchor_times = _build_triplets_vectorized(freq_idx, times)
    return Landmark(hash=hashes, anchor_time=anchor_times)
