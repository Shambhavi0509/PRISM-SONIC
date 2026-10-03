"""Confidence-guided adaptive retrieval pipeline - the core novelty of SONIC.

Flow:
  1. Load query audio once (capped to QUERY_MAX_DURATION_SEC), used for both stages.
  2. Primary stage: log-frequency landmark hashing + inverted-index voting
     (pitch/tempo invariant, near-zero cost). Compute a confidence score from
     the top candidate's temporal-offset consistency and its score margin over
     the runner-up.
  3. If confident: return the primary top candidate. This is the fast path and
     is expected to cover the large majority of queries (clean duplicates and
     mildly-edited audio), keeping average latency and RAM low.
  4. If not confident (heavy modification, sparse-landmark environmental audio,
     noise): compute the secondary binary embedding (Haitsma-Kalker style
     band-delta code) from the SAME loaded audio and search the binary-HNSW
     index for a shortlist. Candidates found only via the secondary shortlist
     (i.e. missed by the primary inverted-index lookup entirely) are verified
     with a direct pairwise landmark-hash overlap against the query, and the
     final answer is the candidate with the best available evidence across
     both stages.

Every result carries its evidence (confidence, scores, which stage produced it)
for explainability.
"""

from dataclasses import dataclass, field

import numpy as np

from sonic.config import QUERY_MAX_DURATION_SEC, QUERY_MAX_DURATION_SEC_BY_DOMAIN
from sonic.fingerprint.binary_embed import (
    band_energy_vector_from_magnitude, fine_sub_fingerprints_from_magnitude,
    sub_fingerprints_from_magnitude, pitch_hypothesis_sub_fingerprints,
    pitch_hypothesis_fine_sub_fingerprints, BinaryEmbedIndex,
    SUBFP_N_BANDS, SUBFP_FINE_N_BANDS,
)
from sonic.fingerprint.landmarks import build_landmarks_from_magnitude, stft_magnitude, Landmark
from sonic.index.binary_hnsw_index import BinaryHNSWIndex
from sonic.index.landmark_index import LandmarkIndex
from sonic.index.subfingerprint_index import SubFingerprintIndex
from sonic.io import load_audio

# Confidence-gate defaults. Calibrated empirically per domain in
# scripts/03_run_benchmark.py's calibration sweep (see REPORT.md Sec. "adaptive
# threshold calibration"); these are the pre-calibration starting values.
DEFAULT_OFFSET_CONF_TAU = 0.15
DEFAULT_MARGIN_TAU = 0.20
# Measured bug: under pitch/speed-shifted queries, primary weighted_score
# collapses to noise level (2-17) but the margin between a noisy top1/top2 can
# still easily exceed 20% by chance, so a low min_score_tau let noise pass as
# "confident" and skip the secondary stage entirely (in one 20-query pitch
# test, 11/20 wrongly short-circuited this way; the 9/20 that DID escalate
# were 9/9 correct via the multi-window sub-fingerprint vote). Clean/duplicate
# true matches score in the hundreds-to-thousands, so 50 cleanly separates
# "genuinely confident" from "noise that happens to have some margin."
DEFAULT_MIN_SCORE_TAU = 50.0
DEFAULT_MIN_ACCEPT_SCORE = 0.5  # absolute floor below which we report "no match" rather than a weak guess

# Gate for the +-1 semitone pitch-hypothesis rescue (see its use in
# query_array below) - a SEPARATE, more permissive margin than
# DEFAULT_MARGIN_TAU because this only controls whether a cheap secondary
# re-rank runs, not whether the whole secondary stage runs at all.
PITCH_RESCUE_MARGIN_TAU = 0.35

PRIMARY_POOL_K = 20
SECONDARY_SHORTLIST_K = 10
# Sub-fingerprint vote search breadth. Measured: at the default-scale K=5/10,
# the true match was often missed by the initial FAISS shortlist before voting
# even had a chance to aggregate it (rank@20-with-K=20 found it in 20/20 test
# cases at rank 0-5, vs the pipeline's end-to-end accuracy being far lower at
# K=5/10) - this corpus is small (a few thousand window-codes for 200 files),
# so a wider per-window search is still cheap.
SUBFP_TOP_K_PER_WINDOW = 20
SUBFP_FINAL_TOP_K = 20


@dataclass
class MatchResult:
    file_id: int | None
    confident: bool
    stage_used: str          # "primary" or "primary+secondary"
    primary_confidence: float
    offset_confidence: float
    weighted_score: float
    margin: float
    escalated: bool
    ranked_ids: list = field(default_factory=list)   # top-K file_ids, for recall@K
    secondary_shortlist: list = field(default_factory=list)
    timing_ms: dict = field(default_factory=dict)


@dataclass
class DomainIndex:
    category: str
    landmark_index: LandmarkIndex
    binary_embed: BinaryEmbedIndex
    binary_hnsw: BinaryHNSWIndex
    subfp_index: SubFingerprintIndex
    subfp_fine_index: SubFingerprintIndex
    offset_conf_tau: float = DEFAULT_OFFSET_CONF_TAU
    margin_tau: float = DEFAULT_MARGIN_TAU
    min_score_tau: float = DEFAULT_MIN_SCORE_TAU
    min_accept_score: float = DEFAULT_MIN_ACCEPT_SCORE
    query_max_duration_sec: float = QUERY_MAX_DURATION_SEC

    def nbytes(self) -> int:
        return (self.landmark_index.nbytes() + self.binary_embed.nbytes()
                + self.binary_hnsw.nbytes() + self.subfp_index.nbytes()
                + self.subfp_fine_index.nbytes())



# RAM optimization: build_domain_index() used to accumulate every file's
# Landmark object, raw band-energy vector, and every window's sub-fingerprint
# code in one big Python list each, held for the ENTIRE corpus until the very
# end (measured: ~55MB of pure list/object overhead beyond what the final
# index structures need, at N=2000 files - see the RAM-scaling investigation
# for the measured breakdown). Processing the corpus in fixed-size CHUNKS and
# flattening/compacting each chunk into plain arrays immediately - discarding
# that chunk's raw per-file objects before moving to the next chunk - caps
# the "raw per-file object" memory at chunk size instead of corpus size,
# while producing byte-for-byte identical final index contents (chunking
# changes nothing about what's computed, only when intermediate objects are
# freed). Explicit `del` was verified NOT to shrink already-reached peak RSS
# (CPython/glibc don't return freed heap pages to the OS) - the only way to
# actually lower the peak is to never let it get that high in the first
# place, which is what per-chunk compaction does.
DOMAIN_INDEX_BUILD_CHUNK_SIZE = 64


def build_domain_index(category: str, entries: list[dict]) -> DomainIndex:
    file_ids = [e["file_id"] for e in entries]
    embed_idx = BinaryEmbedIndex()  # only used for its .binarize() helper here

    lidx = LandmarkIndex()
    lidx.file_id_to_local = {fid: i for i, fid in enumerate(file_ids)}
    lidx.local_to_file_id = np.array(file_ids, dtype=np.int64)
    lidx.n_files = len(file_ids)

    hash_chunks, lid_chunks, time_chunks = [], [], []
    raw_vec_chunks = []
    subfp_file_ids, subfp_code_chunks = [], []
    subfp_fine_file_ids, subfp_fine_code_chunks = [], []

    for chunk_start in range(0, len(entries), DOMAIN_INDEX_BUILD_CHUNK_SIZE):
        chunk = entries[chunk_start:chunk_start + DOMAIN_INDEX_BUILD_CHUNK_SIZE]
        chunk_file_ids = [e["file_id"] for e in chunk]
        chunk_landmarks, chunk_raw_vecs = [], []
        chunk_subfp_codes, chunk_subfp_fine_codes = [], []

        for e in chunk:
            y = load_audio(e["path"])
            mag, frame_times = stft_magnitude(y)
            chunk_landmarks.append(build_landmarks_from_magnitude(mag, frame_times))
            chunk_raw_vecs.append(band_energy_vector_from_magnitude(mag))

            for _start_time, raw_vec in sub_fingerprints_from_magnitude(mag, frame_times):
                subfp_file_ids.append(e["file_id"])
                chunk_subfp_codes.append(embed_idx.binarize(raw_vec))

            for _start_time, raw_vec in fine_sub_fingerprints_from_magnitude(mag, frame_times):
                subfp_fine_file_ids.append(e["file_id"])
                chunk_subfp_fine_codes.append(embed_idx.binarize(raw_vec))

        # Compact this chunk into flat/stacked arrays now, then let
        # chunk_landmarks/chunk_raw_vecs/chunk_subfp_codes go out of scope at
        # the top of the next loop iteration - only the compact per-chunk
        # arrays below survive into the accumulator lists.
        c_hash, c_lid, c_time = lidx.flatten_chunk(chunk_file_ids, chunk_landmarks)
        hash_chunks.append(c_hash)
        lid_chunks.append(c_lid)
        time_chunks.append(c_time)
        raw_vec_chunks.append(np.stack(chunk_raw_vecs))
        if chunk_subfp_codes:
            subfp_code_chunks.append(np.stack(chunk_subfp_codes))
        if chunk_subfp_fine_codes:
            subfp_fine_code_chunks.append(np.stack(chunk_subfp_fine_codes))

    lidx.build_from_flat(np.concatenate(hash_chunks), np.concatenate(lid_chunks), np.concatenate(time_chunks))
    del hash_chunks, lid_chunks, time_chunks

    raw_vecs = np.concatenate(raw_vec_chunks)
    del raw_vec_chunks

    bidx = BinaryEmbedIndex()
    bidx.build(file_ids, raw_vecs)

    hidx = BinaryHNSWIndex()
    hidx.build(file_ids, bidx.codes)

    sidx = SubFingerprintIndex()
    if subfp_code_chunks:
        sidx.build(subfp_file_ids, np.concatenate(subfp_code_chunks))

    sidx_fine = SubFingerprintIndex()
    if subfp_fine_code_chunks:
        sidx_fine.build(subfp_fine_file_ids, np.concatenate(subfp_fine_code_chunks))

    return DomainIndex(
        category=category,
        landmark_index=lidx,
        binary_embed=bidx,
        binary_hnsw=hidx,
        subfp_index=sidx,
        subfp_fine_index=sidx_fine,
        query_max_duration_sec=QUERY_MAX_DURATION_SEC_BY_DOMAIN.get(category, QUERY_MAX_DURATION_SEC),
    )


def _pairwise_overlap_score(query_lm: Landmark, cand_hashes: np.ndarray) -> float:
    if query_lm.hash.size == 0 or cand_hashes.size == 0:
        return 0.0
    cand_set = np.unique(cand_hashes)
    hits = np.isin(query_lm.hash, cand_set)
    return float(hits.sum()) / query_lm.hash.size


def _is_confident(top, second, idx: DomainIndex) -> tuple[bool, float]:
    if top is None:
        return False, 0.0
    margin = 1.0
    if second is not None and top.weighted_score > 0:
        margin = (top.weighted_score - second.weighted_score) / top.weighted_score
    # Both the offset-consistency path and the margin path used to accept a
    # candidate on RATIO evidence alone (e.g. offset_confidence=0.8 from just
    # 4-5 total hash matches). With so few matches the ratio is statistically
    # meaningless - measured: 23/100 pitch-shifted speech queries wrongly
    # short-circuited to the (wrong) primary answer this way, all with
    # weighted_score in the 2-15 range. Both paths now also require the
    # absolute min_score_tau floor, so a handful of coincidental matches can
    # no longer look "confident" regardless of what ratio they happen to form.
    confident = top.weighted_score >= idx.min_score_tau and (
        top.offset_confidence >= idx.offset_conf_tau or margin >= idx.margin_tau
    )
    return confident, margin


def query(y_full_path: str, idx: DomainIndex, top_k: int = 5) -> MatchResult:
    """Query from a file on disk (used for duplicate-audio benchmarking)."""
    import time

    t0 = time.perf_counter()
    y = load_audio(y_full_path, max_duration=idx.query_max_duration_sec)
    load_ms = (time.perf_counter() - t0) * 1000
    return query_array(y, idx, top_k=top_k, load_ms=load_ms)


def query_array(y: np.ndarray, idx: DomainIndex, top_k: int = 5, load_ms: float = 0.0,
                 sr: int = None) -> MatchResult:
    """Query from an already-loaded/modified in-memory audio array (used for
    edited-audio benchmarking, where the modification is generated in memory).
    The domain's query_max_duration_sec cap is applied here too, so edited-query
    latency stays comparable to file-path queries regardless of how long the
    modified array is.
    """
    import time
    from sonic.config import ANALYSIS_SR

    sr = sr or ANALYSIS_SR
    max_samples = int(idx.query_max_duration_sec * sr)
    if y.size > max_samples:
        y = y[:max_samples]

    timing = {"load_ms": load_ms}

    t0 = time.perf_counter()
    mag, frame_times = stft_magnitude(y, sr)
    q_lm = build_landmarks_from_magnitude(mag, frame_times, sr)
    timing["primary_fingerprint_ms"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    cands = idx.landmark_index.query(q_lm, top_k=PRIMARY_POOL_K)
    timing["primary_search_ms"] = (time.perf_counter() - t0) * 1000

    top = cands[0] if cands else None
    second = cands[1] if len(cands) > 1 else None
    confident, margin = _is_confident(top, second, idx)

    if confident:
        accepted = top is not None and top.weighted_score >= idx.min_accept_score
        return MatchResult(
            file_id=(top.file_id if accepted else None),
            confident=True,
            stage_used="primary",
            primary_confidence=top.query_confidence if top else 0.0,
            offset_confidence=top.offset_confidence if top else 0.0,
            weighted_score=top.weighted_score if top else 0.0,
            margin=margin,
            escalated=False,
            ranked_ids=[c.file_id for c in cands[:top_k]],
            timing_ms=timing,
        )

    # --- Escalate to secondary stage(s) ---
    t0 = time.perf_counter()
    embed_vec = band_energy_vector_from_magnitude(mag, sr)
    code = idx.binary_embed.binarize(embed_vec)
    shortlist = idx.binary_hnsw.search(code, top_k=SECONDARY_SHORTLIST_K)

    # Multi-window sub-fingerprint majority vote (Haitsma-Kalker style - see
    # fingerprint/binary_embed.py and index/subfingerprint_index.py). This is
    # what actually recovers pitch/tempo-shifted queries: the single global
    # secondary code above and the primary landmark hash both collapse under
    # phase-vocoder pitch/speed modification, but overlapping short windows
    # retain redundancy - some windows usually still match even when the
    # aggregate doesn't.
    subfp_windows = sub_fingerprints_from_magnitude(mag, frame_times, sr)
    subfp_votes = []
    if subfp_windows:
        query_codes = np.stack([idx.binary_embed.binarize(v) for _t, v in subfp_windows])
        subfp_votes = idx.subfp_index.vote(
            query_codes, top_k_per_window=SUBFP_TOP_K_PER_WINDOW, final_top_k=SUBFP_FINAL_TOP_K,
        )

    # Fine-resolution sub-fingerprint vote (see fingerprint/binary_embed.py's
    # SUBFP_FINE_N_BANDS comment): more discriminative than the coarse code
    # above, at the cost of being less pitch/tempo tolerant. Diagnosed on real
    # music trim_start/time_shift failures where the coarse code let an
    # unrelated file win by a razor-thin margin even though the true file also
    # matched closely - the fine code breaks that kind of near-tie decisively
    # for modifications that don't actually change pitch/tempo.
    fine_windows = fine_sub_fingerprints_from_magnitude(mag, frame_times, sr)
    subfp_fine_votes = []
    if fine_windows:
        fine_codes = np.stack([idx.binary_embed.binarize(v) for _t, v in fine_windows])
        subfp_fine_votes = idx.subfp_fine_index.vote(
            fine_codes, top_k_per_window=SUBFP_TOP_K_PER_WINDOW, final_top_k=SUBFP_FINAL_TOP_K,
        )

    timing["secondary_ms"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    primary_ids = {c.file_id: c for c in cands}
    # Fuse primary landmark evidence with BOTH secondary evidence sources,
    # ADDITIVELY for every candidate that appears in any pool - not "primary
    # score if available, else re-derive from primary hashes again". An
    # earlier version fell back to a primary-hash overlap score for
    # secondary-only candidates, which is exactly the signal that's near-zero
    # for the queries this stage exists to rescue (heavy pitch/tempo shift):
    # landmark hashes barely survive a phase-vocoder pitch/tempo change (see
    # REPORT.md's pitch/tempo robustness analysis), so that fallback almost
    # never found the true candidate.
    combined_scores = {c.file_id: c.weighted_score for c in cands}

    n_bits = idx.binary_hnsw.n_bits
    for fid, hamming in shortlist:
        combined_scores[fid] = combined_scores.get(fid, 0.0) + float(n_bits - hamming)

    # Sub-fingerprint votes are summed over multiple windows, so they can
    # reach a much larger scale than a single-window Hamming score (up to
    # n_bits * n_query_windows) precisely when many windows agree - that
    # larger scale is intentional: strong multi-window agreement should be
    # able to override weak/absent primary and single-code evidence.
    for fid, vote_score in subfp_votes:
        combined_scores[fid] = combined_scores.get(fid, 0.0) + vote_score

    # Fine-code votes are scaled down to the coarse code's per-bit scale
    # (n_bits ratio) before fusing, rather than added raw. The fine code has
    # 3x the bits of the coarse one, so unscaled it would dominate the fused
    # score whenever it has ANY signal - including the noisy signal it
    # produces under pitch/speed shift (measured earlier: a 64-band code's
    # Hamming distance nearly doubled under a mere +1 semitone). Scaling keeps
    # its role as a tie-breaker on top of the coarse/pitch-tolerant code
    # rather than letting it override that code's judgment under pitch/tempo
    # modification, where it is the less trustworthy of the two.
    fine_scale = SUBFP_N_BANDS / SUBFP_FINE_N_BANDS
    for fid, vote_score in subfp_fine_votes:
        combined_scores[fid] = combined_scores.get(fid, 0.0) + vote_score * fine_scale

    # --- Targeted +-1 semitone pitch-hypothesis rescue (see
    # fingerprint/binary_embed.py's pitch_hypothesis_sub_fingerprints
    # docstring for the measured motivation/validation). Gated to only run
    # when the UNWARPED fusion above is already ambiguous (top candidate's
    # lead over the runner-up is thin) - i.e. this is a tie-breaker rescue,
    # not something that runs on every escalated query. Two reasons, both
    # measured:
    #   1. Latency: most escalated queries (background_noise, mp3, filtering,
    #      etc. that still needed to escalate) are already resolved
    #      correctly and decisively by the unwarped fusion above: recomputing
    #      two extra coarse + two extra fine sub-fingerprint codes and voting
    #      each against the full index for every one of them was measured to
    #      roughly double average escalated-query latency for no accuracy
    #      benefit on those conditions.
    #   2. Precision: gating on an existing decisive lead is what keeps this
    #      change a true no-op on already-correct calls. Without the gate, a
    #      small number of ALREADY-CORRECT +-0.5-semitone (and, rarely, other-
    #      modification) queries flipped to wrong: a competing candidate could
    #      occasionally score higher under an (incorrect, since the true shift
    #      wasn't really +-1 semitone) warp hypothesis than the true
    #      candidate's own already-strong, already-decisive unwarped lead,
    #      because the bonus rule only compares each candidate's hypothesis
    #      score against ITS OWN base score, not against the current leader.
    #      Skipping the whole rescue whenever the unwarped call is already
    #      decisive removes that failure mode entirely, since a decisive
    #      unwarped leader is never even put up for reconsideration.
    # PITCH_RESCUE_MARGIN_TAU chosen as roughly half of DEFAULT_MARGIN_TAU
    # (the primary stage's own "confident" margin) - deliberately more
    # permissive than the primary gate (this is a cheap secondary-stage
    # re-rank, not a full stage skip) while still excluding the clearly-
    # decisive majority of escalated calls from the extra computation.
    top_fid_prebonus = max(combined_scores, key=combined_scores.get) if combined_scores else None
    prebonus_top = combined_scores.get(top_fid_prebonus, 0.0) if top_fid_prebonus is not None else 0.0
    prebonus_second = max(
        (v for fid, v in combined_scores.items() if fid != top_fid_prebonus), default=0.0,
    )
    prebonus_margin = ((prebonus_top - prebonus_second) / prebonus_top) if prebonus_top > 0 else 0.0
    run_pitch_rescue = subfp_windows and prebonus_margin < PITCH_RESCUE_MARGIN_TAU

    pitch_hyp_votes_by_shift = {}
    pitch_hyp_fine_votes_by_shift = {}
    if run_pitch_rescue:
        hyp_windows_by_shift = pitch_hypothesis_sub_fingerprints(mag, frame_times, sr)
        for semi, hyp_windows in hyp_windows_by_shift.items():
            if not hyp_windows:
                continue
            hyp_codes = np.stack([idx.binary_embed.binarize(v) for _t, v in hyp_windows])
            pitch_hyp_votes_by_shift[semi] = dict(idx.subfp_index.vote(
                hyp_codes, top_k_per_window=SUBFP_TOP_K_PER_WINDOW, final_top_k=SUBFP_FINAL_TOP_K,
            ))
        if fine_windows:
            hyp_fine_windows_by_shift = pitch_hypothesis_fine_sub_fingerprints(mag, frame_times, sr)
            for semi, hyp_windows in hyp_fine_windows_by_shift.items():
                if not hyp_windows:
                    continue
                hyp_codes = np.stack([idx.binary_embed.binarize(v) for _t, v in hyp_windows])
                pitch_hyp_fine_votes_by_shift[semi] = dict(idx.subfp_fine_index.vote(
                    hyp_codes, top_k_per_window=SUBFP_TOP_K_PER_WINDOW, final_top_k=SUBFP_FINAL_TOP_K,
                ))

    # Pitch-hypothesis bonus: for each candidate, only the EXCESS of the best
    # compensated-shift vote over the already-fused unwarped coarse vote is
    # added - never the raw hypothesis vote itself. This makes the bonus
    # strictly additive evidence for candidates whose match genuinely improves
    # under a +-1 semitone compensation (the true match, when the query really
    # was shifted by about that much) while leaving every other candidate's
    # score (and therefore every already-passing, non-pitch-shifted condition)
    # unchanged whenever the hypotheses don't outperform the unwarped code -
    # measured to be the case for +-0.5 semitone and all other modification
    # types (see fingerprint/binary_embed.py's PITCH_HYPOTHESIS_SEMITONES
    # docstring).
    # Restricted to candidates that ALREADY appear in the unwarped coarse
    # shortlist (base_coarse_votes) - i.e. this can only strengthen a
    # candidate the unwarped evidence already considers plausible, never
    # introduce a brand-new candidate purely off hypothesis-vote noise.
    # Measured: without this restriction, a handful of candidates that show
    # up ONLY under a warped hypothesis (never in the unwarped vote at all -
    # a sign the "match" is coincidental to that specific compensation, not a
    # real one) picked up spurious bonuses and net accuracy went DOWN
    # slightly on music +-1 semitone despite helping several individual
    # cases; restricting to already-plausible candidates removed that
    # regression.
    if pitch_hyp_votes_by_shift:
        base_coarse_votes = dict(subfp_votes)
        for fid in base_coarse_votes:
            base = base_coarse_votes[fid]
            best_hyp = max((votes.get(fid, 0.0) for votes in pitch_hyp_votes_by_shift.values()), default=0.0)
            bonus = max(0.0, best_hyp - base)
            if bonus > 0.0:
                combined_scores[fid] = combined_scores.get(fid, 0.0) + bonus

    # Same bonus treatment for the FINE code, scaled down by the same
    # fine_scale used for its base vote above (so a fine-code bonus can never
    # outweigh what an equivalently-strong coarse-code bonus would). This is
    # the component that turned out to matter most in practice: diagnosed on
    # real music +-1 semitone failures, the fine code (not the coarse one) was
    # the dominant source of wrong answers (favored the wrong file in 92% of
    # 39 diagnosed failures - see pitch_hypothesis_sub_fingerprints'
    # docstring). Same strict-improvement-only, already-plausible-candidate
    # restriction as the coarse bonus above, for the same reason (avoids
    # rewarding a candidate that only "matches" a specific warped hypothesis
    # by coincidence).
    if pitch_hyp_fine_votes_by_shift:
        base_fine_votes = dict(subfp_fine_votes)
        for fid in base_fine_votes:
            base = base_fine_votes[fid]
            best_hyp = max((votes.get(fid, 0.0) for votes in pitch_hyp_fine_votes_by_shift.values()), default=0.0)
            bonus = max(0.0, best_hyp - base)
            if bonus > 0.0:
                combined_scores[fid] = combined_scores.get(fid, 0.0) + bonus * fine_scale

    timing["rerank_ms"] = (time.perf_counter() - t0) * 1000

    if not combined_scores:
        return MatchResult(
            file_id=None, confident=False, stage_used="primary+secondary",
            primary_confidence=0.0, offset_confidence=0.0, weighted_score=0.0,
            margin=margin, escalated=True,
            secondary_shortlist=shortlist, timing_ms=timing,
        )

    ranked = sorted(combined_scores, key=combined_scores.get, reverse=True)[:top_k]
    best_fid = ranked[0]
    best_c = primary_ids.get(best_fid)
    accepted = combined_scores[best_fid] >= idx.min_accept_score

    return MatchResult(
        file_id=(best_fid if accepted else None),
        confident=False,
        stage_used="primary+secondary",
        primary_confidence=(best_c.query_confidence if best_c else 0.0),
        offset_confidence=(best_c.offset_confidence if best_c else 0.0),
        weighted_score=(best_c.weighted_score if best_c else combined_scores[best_fid]),
        margin=margin,
        escalated=True,
        ranked_ids=ranked,
        secondary_shortlist=shortlist,
        timing_ms=timing,
    )
