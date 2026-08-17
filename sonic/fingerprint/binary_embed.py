"""Secondary representation: a compact binary embedding used only as a fallback
when the primary landmark stage's confidence is low (see retrieval/pipeline.py).

Unlike the primary landmark hash (built from sparse spectral peaks, which can
become unreliable under heavy noise or on low-harmonicity environmental audio
with few clean peaks), this embedding summarizes the *global* log-frequency
energy distribution of a clip - a coarse "spectral envelope shape" descriptor
that degrades more gracefully under noise/heavy modification because it is an
aggregate statistic rather than a set of discrete peak positions. It is not
designed to be pitch/tempo invariant on its own; its job is only to produce a
short candidate shortlist, which is then re-ranked using the (pitch/tempo
invariant) landmark alignment score - see retrieval/pipeline.py.

Encoding (Haitsma-Kalker style band-delta binary hash, Haitsma & Kalker, "A
Highly Robust Audio Fingerprinting System," ISMIR 2002): the clip is split into
N_SEG time segments; within each segment a log-mel energy vector over
N_BANDS+1 bands is computed and each bit is the SIGN of the energy difference
between adjacent bands. Sign-of-band-delta is self-normalizing per file (no
corpus-wide threshold to depend on) and is the standard robust-bit construction
in the audio-fingerprinting literature - substantially more discriminative in
practice than a single global corpus-median-thresholded vector (measured: a
single-vector corpus-median design gave only 50-86% self-match recall@5 across
domains in initial testing; the segment x band-delta design below is evaluated
against that baseline in the ablation study).
"""

from dataclasses import dataclass

import numpy as np

from sonic.config import ANALYSIS_SR
from sonic.fingerprint.landmarks import compute_stft, HOP_LENGTH

N_SEG = 8
N_BANDS = 32     # -> N_SEG * N_BANDS = 256-bit code, 32 bytes/file
F_MIN = 50.0
F_MAX = 10000.0
_N_FFT = 1024
_HOP = 256  # must match fingerprint.landmarks.HOP_LENGTH - both consume the same shared STFT magnitude

# --- Multi-window sub-fingerprints (Haitsma & Kalker, ISMIR 2002 style) -----
# The single global band_energy_vector above has no redundancy: under a
# phase-vocoder pitch/speed modification, measured peak-position noise was
# large enough (std ~30-50 log-freq bins, essentially random) that the PRIMARY
# landmark hash collapses to near-zero overlap even at the smallest tested
# severity (+-1 semitone). A single aggregate secondary code degrades more
# gracefully but still has no fallback if the whole clip's energy distribution
# is disturbed. Sub-fingerprinting computes the SAME band-delta code on many
# short (SUBFP_WINDOW_SEC), overlapping (SUBFP_HOP_SEC) windows instead of one
# global code - even if a pitch/tempo shift scrambles some windows (e.g.
# consonant-heavy or transient-heavy ones), more tonal/stable windows (vowels,
# sustained notes) still tend to match, and votes are pooled across windows.
# This is activated ONLY during confidence-gated escalation (see
# retrieval/pipeline.py) - it is never computed for the fast/confident path.
SUBFP_WINDOW_SEC = 1.5
SUBFP_HOP_SEC = 0.5
SUBFP_N_SEG = 4
# 16 (wide) bands rather than 64: measured mean per-window Hamming distance
# under a +1-semitone pitch shift dropped from 41% (64 bands) to 11% (16
# bands) - wider bands are less sensitive to the few-bin peak jitter a
# phase-vocoder pitch shift introduces, at the cost of coarser (but still
# usable, since this is a fallback vote signal, not the primary discriminator)
# spectral resolution. 8 bands was tested too and was WORSE (44% vs 51% at
# +-1 semitone on speech) - too coarse loses discriminability faster than it
# gains tolerance, so 16 is a measured sweet spot, not a guess.
SUBFP_N_BANDS = 16  # -> SUBFP_N_SEG * SUBFP_N_BANDS = 64 bits/window

_MEL_FB_CACHE = {}


def _hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + f / 700.0)


def _mel_to_hz(m):
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def _build_mel_filterbank(sr: int, n_fft: int, n_mels: int, fmin: float, fmax: float) -> np.ndarray:
    key = (sr, n_fft, n_mels, fmin, fmax)
    if key in _MEL_FB_CACHE:
        return _MEL_FB_CACHE[key]

    n_linear = n_fft // 2 + 1
    linear_freqs = np.linspace(0, sr / 2, n_linear)
    mel_edges = np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_mels + 2)
    hz_edges = _mel_to_hz(mel_edges)

    fb = np.zeros((n_mels, n_linear), dtype=np.float32)
    for i in range(n_mels):
        f_lo, f_ctr, f_hi = hz_edges[i], hz_edges[i + 1], hz_edges[i + 2]
        left = (linear_freqs - f_lo) / max(f_ctr - f_lo, 1e-9)
        right = (f_hi - linear_freqs) / max(f_hi - f_ctr, 1e-9)
        fb[i] = np.clip(np.minimum(left, right), 0.0, None)

    _MEL_FB_CACHE[key] = fb
    return fb


# Second, finer-resolution tier alongside the coarse one above. Diagnosis on
# real failures (music trim_start/time_shift, which involve no pitch/tempo
# change at all) showed the coarse 16-band/64-bit code sometimes lets an
# unrelated "hub" file win by a razor-thin vote margin even though the true
# file's own windows also match closely - i.e. a genuine discriminability
# shortfall of the coarse code on this dataset (several tracks are similar
# loop-based material), not a bug in alignment or confidence gating. Rather
# than widen the coarse code (which would sacrifice the pitch-shift tolerance
# it was tuned for - see SUBFP_N_BANDS comment above), a second, higher-band
# code is computed and voted alongside it: for modifications with no real
# pitch change (trim, gain, noise, filtering, time-shift), the fine code's
# near-perfect match strongly favors the true candidate; under pitch/speed
# shift it degrades (as measured for the coarse code at 64 bands earlier) but
# the coarse code still carries the signal there. Both codes are tiny (a few
# hundred KB total for 200 files), so this stays well inside the RAM budget.
SUBFP_FINE_N_BANDS = 48  # -> SUBFP_N_SEG * SUBFP_FINE_N_BANDS = 192 bits/window

# --- Query-side pitch-hypothesis search (targeted +-1 semitone fix) ---------
# Diagnosis (see REPORT.md Sec. 29 / retrieval/pipeline.py's escalation
# comments): the coarse sub-fingerprint's mean self-Hamming distance under a
# phase-vocoder pitch shift grows with severity - measured 10-12% at +-0.5
# semitone (already tolerated: that's the currently-benchmarked severity and
# it passes) but 12-15% at +-1 semitone, which combined with hub-file
# competition is what pushes real queries below the correct candidate at that
# severity. A pitch shift is a near-constant multiplicative frequency scaling,
# so pre-warping the QUERY's linear-frequency STFT magnitude by the inverse
# scale factor before computing the (otherwise completely unchanged) coarse
# band-delta code compensates for exactly that scaling and should recover
# close to the +-0.5 Hamming-distance level. Measured on real self-match pairs
# (this file's own diagnostic): pre-warping a +-1-semitone-shifted query by
# the matching -+1-semitone compensation brought mean coarse self-Hamming
# distance down from 12-15% to 8-10% - i.e. BETTER than the untouched +-0.5
# level, confirming the compensation is doing real work rather than just
# adding noise.
#
# This is deliberately NOT wired into the coarse code used on the fast/
# confident primary path, NOT into the index-build side (candidates are never
# warped - only queries are, and only during escalation), and NOT into the
# fine code (kept at its existing tie-breaker role, unmodified). It is used
# in retrieval/pipeline.py ONLY as an additive "bonus" on top of the existing
# unwarped coarse vote (bonus = max(0, best_hypothesis_vote - unwarped_vote)),
# so for any query whose true best alignment is already at zero shift (every
# other modification: mp3, noise, filtering, trim, gain, time-shift, speed,
# and even +-0.5 semitone pitch, whose unwarped Hamming distance is already
# low) the hypotheses can only match at least as well as they already were
# matched and the bonus is ~0 - this is a compensating hypothesis search, not
# a global loosening of matching tolerance.
PITCH_HYPOTHESIS_SEMITONES = (-1.0, 1.0)  # exactly the benchmarked +-1 semitone severity


def warp_magnitude_freq(mag: np.ndarray, semitone_shift: float) -> np.ndarray:
    """Resamples a linear-frequency STFT magnitude spectrogram along its
    frequency axis to compensate for an assumed `semitone_shift`-semitone
    pitch shift already present in `mag` (i.e. warps the query TOWARD what an
    unshifted version would look like: pass +1.0 to compensate for a query
    that was pitch-shifted DOWN by 1 semitone, matching this module's use in
    retrieval/pipeline.py's escalation stage). Plain linear interpolation
    along the (already linear) STFT frequency bins - no FFT/filter-bank cost,
    O(n_freq * n_frames), negligible next to the STFT itself."""
    n_freq = mag.shape[0]
    scale = 2.0 ** (semitone_shift / 12.0)
    src_idx = np.clip(np.arange(n_freq) / scale, 0, n_freq - 1)
    lo = np.floor(src_idx).astype(np.int64)
    hi = np.clip(lo + 1, 0, n_freq - 1)
    frac = (src_idx - lo)[:, None].astype(np.float32)
    return (mag[lo] * (1.0 - frac) + mag[hi] * frac).astype(np.float32)


def pitch_hypothesis_sub_fingerprints(mag: np.ndarray, frame_times: np.ndarray,
                                       sr: int = ANALYSIS_SR,
                                       semitone_shifts=PITCH_HYPOTHESIS_SEMITONES,
                                       n_bands: int = SUBFP_N_BANDS,
                                       ) -> dict[float, list[tuple[float, np.ndarray]]]:
    """Returns {semitone_shift: [(window_start_time, raw_band_delta_vector), ...]}
    for each candidate compensating shift, using the SAME band-delta encoding
    as sub_fingerprints_from_magnitude (same windowing; `n_bands` defaults to
    the coarse code's band count but is also used for the fine code - see
    pitch_hypothesis_fine_sub_fingerprints below) so the resulting codes are
    directly comparable/votable against the existing SubFingerprintIndex with
    no index-side changes.

    Diagnosed on real music failures (see retrieval/pipeline.py's escalation
    comments): the FINE (48-band) code, not the coarse one, turned out to be
    the dominant source of wrong answers at +-1 semitone - in a breakdown of
    39 real failing music queries, the fine code alone favored the WRONG file
    in 92% of them (mean fine-code score gap +247 in the wrong file's favor),
    while the coarse code and primary landmark score both still favored the
    TRUE file on average. This matches the plain Hamming-distance measurement
    directly: the fine code's self-Hamming distance under a phase-vocoder
    +-1-semitone shift is far noisier than the coarse code's (measured 26-35%
    vs 12-15%) because its narrower 48 bands cross far more band boundaries
    per semitone of shift. The SAME frequency-warp compensation fixes it the
    same way: measured mean fine-code self-Hamming distance after compensating
    warp drops to 15-21% - back below even the untouched (unwarped) +-0.5
    semitone fine-code level in every domain. Applying this hypothesis search
    to BOTH codes (see retrieval/pipeline.py) is what closes the gap; coarse
    alone was not enough (measured: coarse-only bonus left music +-1 semitone
    accuracy statistically unchanged, ~65-70%, because the fine code's much
    larger noise was still free to dominate the fused ranking)."""
    out = {}
    for semi in semitone_shifts:
        warped = warp_magnitude_freq(mag, semi)
        out[semi] = sub_fingerprints_from_magnitude(warped, frame_times, sr, n_bands=n_bands)
    return out


def pitch_hypothesis_fine_sub_fingerprints(mag: np.ndarray, frame_times: np.ndarray,
                                            sr: int = ANALYSIS_SR,
                                            semitone_shifts=PITCH_HYPOTHESIS_SEMITONES,
                                            ) -> dict[float, list[tuple[float, np.ndarray]]]:
    return pitch_hypothesis_sub_fingerprints(mag, frame_times, sr, semitone_shifts, n_bands=SUBFP_FINE_N_BANDS)


def sub_fingerprints_from_magnitude(mag: np.ndarray, frame_times: np.ndarray,
                                     sr: int = ANALYSIS_SR,
                                     window_sec: float = SUBFP_WINDOW_SEC,
                                     hop_sec: float = SUBFP_HOP_SEC,
                                     n_bands: int = SUBFP_N_BANDS) -> list[tuple[float, np.ndarray]]:
    """Returns [(window_start_time, raw_band_delta_vector), ...] covering the
    whole clip with overlapping windows. The mel filterbank is applied ONCE to
    the full clip (reusing the shared STFT magnitude) and then sliced per
    window, so this stays cheap even though it produces many codes."""
    n_frames = mag.shape[1]
    if n_frames < 2:
        return []

    frame_dt = HOP_LENGTH / sr
    window_frames = max(2, int(round(window_sec / frame_dt)))
    hop_frames = max(1, int(round(hop_sec / frame_dt)))

    fb = _build_mel_filterbank(sr, _N_FFT, n_bands + 1, F_MIN, min(F_MAX, sr / 2))
    power = mag.astype(np.float32) ** 2
    log_mel = np.log1p(fb @ power)  # (n_bands+1, n_frames), computed once

    windows = []
    start = 0
    while True:
        end = min(start + window_frames, n_frames)
        if end - start < 2:
            if start == 0 and n_frames >= 2:
                end = n_frames  # whole clip is shorter than one window - use it all
            else:
                break
        seg_bounds = np.linspace(start, end, SUBFP_N_SEG + 1).astype(int)
        bits = np.empty((SUBFP_N_SEG, n_bands), dtype=np.float32)
        for s in range(SUBFP_N_SEG):
            lo, hi = seg_bounds[s], max(seg_bounds[s + 1], seg_bounds[s] + 1)
            seg_energy = np.median(log_mel[:, lo:hi], axis=1)
            bits[s] = np.sign(seg_energy[:-1] - seg_energy[1:])
        windows.append((float(frame_times[start]), bits.reshape(-1)))
        if end >= n_frames:
            break
        start += hop_frames

    return windows


def fine_sub_fingerprints_from_magnitude(mag: np.ndarray, frame_times: np.ndarray,
                                          sr: int = ANALYSIS_SR) -> list[tuple[float, np.ndarray]]:
    return sub_fingerprints_from_magnitude(mag, frame_times, sr, n_bands=SUBFP_FINE_N_BANDS)


def band_energy_vector_from_magnitude(mag: np.ndarray, sr: int = ANALYSIS_SR) -> np.ndarray:
    """Same as band_energy_vector, but reuses an already-computed STFT magnitude
    (shared with the primary landmark stage - see fingerprint/landmarks.py's
    stft_magnitude - to avoid a second STFT pass over the same audio)."""
    power = mag.astype(np.float32) ** 2
    fb = _build_mel_filterbank(sr, _N_FFT, N_BANDS + 1, F_MIN, min(F_MAX, sr / 2))
    mel = fb @ power
    log_mel = np.log1p(mel)  # (N_BANDS+1, n_frames)

    n_frames = log_mel.shape[1]
    seg_bounds = np.linspace(0, n_frames, N_SEG + 1).astype(int)

    bits = np.empty((N_SEG, N_BANDS), dtype=np.float32)
    for s in range(N_SEG):
        lo, hi = seg_bounds[s], max(seg_bounds[s + 1], seg_bounds[s] + 1)
        seg_energy = np.median(log_mel[:, lo:hi], axis=1)  # (N_BANDS+1,)
        bits[s] = np.sign(seg_energy[:-1] - seg_energy[1:])

    return bits.reshape(-1)


def band_energy_vector(y: np.ndarray, sr: int = ANALYSIS_SR) -> np.ndarray:
    """Segment x band-delta sign vector, shape (N_SEG * N_BANDS,), values in {-1, +1}."""
    min_len = 512 * N_SEG
    if y.size < min_len:
        y = np.pad(y, (0, min_len - y.size))

    stft = compute_stft(y, n_fft=_N_FFT, hop_length=_HOP)
    power = np.abs(stft).astype(np.float32) ** 2
    fb = _build_mel_filterbank(sr, _N_FFT, N_BANDS + 1, F_MIN, min(F_MAX, sr / 2))
    mel = fb @ power
    log_mel = np.log1p(mel)  # (N_BANDS+1, n_frames)

    n_frames = log_mel.shape[1]
    seg_bounds = np.linspace(0, n_frames, N_SEG + 1).astype(int)

    bits = np.empty((N_SEG, N_BANDS), dtype=np.float32)
    for s in range(N_SEG):
        lo, hi = seg_bounds[s], max(seg_bounds[s + 1], seg_bounds[s] + 1)
        seg_energy = np.median(log_mel[:, lo:hi], axis=1)  # (N_BANDS+1,)
        bits[s] = np.sign(seg_energy[:-1] - seg_energy[1:])

    return bits.reshape(-1)


@dataclass
class BinaryEmbedIndex:
    codes: np.ndarray = None           # (n_files, (N_SEG*N_BANDS)//8) packed uint8 codes
    file_ids: np.ndarray = None        # (n_files,) int64, aligned with `codes` rows

    def binarize(self, raw_vec: np.ndarray) -> np.ndarray:
        bits = (raw_vec > 0).astype(np.uint8)
        return np.packbits(bits)

    def build(self, file_ids: list[int], raw_vectors: np.ndarray) -> None:
        codes = np.stack([self.binarize(v) for v in raw_vectors])
        self.codes = codes
        self.file_ids = np.array(file_ids, dtype=np.int64)

    def nbytes(self) -> int:
        n = 0
        if self.codes is not None:
            n += self.codes.nbytes
        if self.file_ids is not None:
            n += self.file_ids.nbytes
        return n
