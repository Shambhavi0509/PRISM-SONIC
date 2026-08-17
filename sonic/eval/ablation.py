"""Ablation study: isolates where SONIC's accuracy/latency actually come from.

  Baseline A - primary landmark hashes, brute-force linear scan (no inverted
              index, no IDF weighting): for each query, directly compare its
              landmark hashes against every indexed file's hash set and score
              by raw overlap count. This is the "textbook Shazam-style landmark
              matching with no optimization" baseline.
  Baseline B - primary landmark hashes + the inverted index / IDF-weighted
              voting (see index/landmark_index.py), but the secondary stage is
              never used.
  Baseline C - B + the secondary binary-embedding stage, but ALWAYS activated
              (no confidence gating) and fused with primary evidence.
  Proposed   - B + confidence-GATED secondary activation (sonic.retrieval.pipeline,
              the actual SONIC system).

A + B isolate the inverted index's contribution (expected: speed, and accuracy
robustness to the "hub file" pathology - see index/landmark_index.py docstring).
B + C + Proposed isolate the secondary stage's contribution and whether gating
it (vs. always running it) changes accuracy/latency.
"""

import time

import numpy as np

from sonic.fingerprint.binary_embed import band_energy_vector_from_magnitude
from sonic.fingerprint.landmarks import build_landmarks_from_magnitude, stft_magnitude
from sonic.retrieval.pipeline import (
    DomainIndex, _is_confident, _pairwise_overlap_score, query_array,
)


def _load_and_fingerprint(idx: DomainIndex, path_or_array, sr=None):
    from sonic.config import ANALYSIS_SR
    from sonic.io import load_audio

    if isinstance(path_or_array, str):
        y = load_audio(path_or_array, max_duration=idx.query_max_duration_sec)
        sr = ANALYSIS_SR
    else:
        y = path_or_array
        sr = sr or ANALYSIS_SR
        max_samples = int(idx.query_max_duration_sec * sr)
        if y.size > max_samples:
            y = y[:max_samples]
    mag, frame_times = stft_magnitude(y, sr)
    q_lm = build_landmarks_from_magnitude(mag, frame_times, sr)
    return q_lm, mag, sr


def baseline_a_bruteforce(idx: DomainIndex, path_or_array, sr=None, top_k: int = 5):
    """Linear scan over every indexed file's (post-prune) hash set, raw overlap count."""
    q_lm, _, _ = _load_and_fingerprint(idx, path_or_array, sr)
    t0 = time.perf_counter()
    scores = []
    for fid in idx.landmark_index.file_id_to_local:
        cand_hashes = idx.landmark_index.get_file_hashes(fid)
        if q_lm.hash.size == 0 or cand_hashes.size == 0:
            scores.append((fid, 0))
            continue
        overlap = np.isin(q_lm.hash, cand_hashes).sum()
        scores.append((fid, int(overlap)))
    scores.sort(key=lambda x: -x[1])
    latency_ms = (time.perf_counter() - t0) * 1000
    top = scores[0] if scores else (None, 0)
    return {
        "predicted_id": top[0] if top[1] > 0 else None,
        "ranked_ids": [s[0] for s in scores[:top_k]],
        "latency_ms": latency_ms,
    }


def baseline_b_primary_only(idx: DomainIndex, path_or_array, sr=None, top_k: int = 5):
    q_lm, _, _ = _load_and_fingerprint(idx, path_or_array, sr)
    t0 = time.perf_counter()
    cands = idx.landmark_index.query(q_lm, top_k=top_k)
    latency_ms = (time.perf_counter() - t0) * 1000
    top = cands[0] if cands else None
    accepted = top is not None and top.weighted_score >= idx.min_accept_score
    return {
        "predicted_id": (top.file_id if accepted else None),
        "ranked_ids": [c.file_id for c in cands],
        "latency_ms": latency_ms,
    }


def baseline_c_always_fuse(idx: DomainIndex, path_or_array, sr=None, top_k: int = 5):
    """Primary + secondary, secondary ALWAYS computed and fused (no confidence gate)."""
    q_lm, mag, sr = _load_and_fingerprint(idx, path_or_array, sr)
    t0 = time.perf_counter()
    cands = idx.landmark_index.query(q_lm, top_k=20)

    embed_vec = band_energy_vector_from_magnitude(mag, sr)
    code = idx.binary_embed.binarize(embed_vec)
    shortlist = idx.binary_hnsw.search(code, top_k=10)

    primary_ids = {c.file_id: c for c in cands}
    combined = {c.file_id: c.offset_confidence * 1000 + c.weighted_score for c in cands}
    for fid, _hamming in shortlist:
        if fid in primary_ids:
            continue
        cand_hashes = idx.landmark_index.get_file_hashes(fid)
        overlap = _pairwise_overlap_score(q_lm, cand_hashes)
        combined[fid] = overlap * q_lm.hash.size * 0.5

    latency_ms = (time.perf_counter() - t0) * 1000
    if not combined:
        return {"predicted_id": None, "ranked_ids": [], "latency_ms": latency_ms}

    ranked = sorted(combined, key=combined.get, reverse=True)[:top_k]
    best = ranked[0]
    accepted = combined[best] >= idx.min_accept_score
    return {
        "predicted_id": (best if accepted else None),
        "ranked_ids": ranked,
        "latency_ms": latency_ms,
    }


def proposed(idx: DomainIndex, path_or_array, sr=None, top_k: int = 5):
    if isinstance(path_or_array, str):
        from sonic.config import ANALYSIS_SR
        from sonic.io import load_audio
        y = load_audio(path_or_array, max_duration=idx.query_max_duration_sec)
        res = query_array(y, idx, top_k=top_k, sr=ANALYSIS_SR)
    else:
        res = query_array(path_or_array, idx, top_k=top_k, sr=sr)
    total_latency = sum(res.timing_ms.values())
    return {"predicted_id": res.file_id, "ranked_ids": res.ranked_ids, "latency_ms": total_latency}


ABLATION_METHODS = {
    "A_bruteforce_no_index": baseline_a_bruteforce,
    "B_primary_index_only": baseline_b_primary_only,
    "C_always_fuse_secondary": baseline_c_always_fuse,
    "Proposed_confidence_gated": proposed,
}
