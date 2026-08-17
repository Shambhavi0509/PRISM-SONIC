"""Primary index: an inverted index over landmark hashes, backed by sorted numpy
arrays (no Python dict/list-of-lists overhead), following Wang (2003)'s
hash -> (file_id, anchor_time) inverted-index design. Lookup is O(log n) via
binary search (np.searchsorted) into a globally sorted hash array.

Matching score: raw hash-collision counts alone are unreliable because a small
number of "hub" files (highly repetitive audio, e.g. a loop-based track, or
generically common landmark triplets shared across many unrelated files) rack up
large collision counts without actually being the right match - empirically,
plain raw-count ranking mis-ranked ~15% of exact self-match music queries in
initial testing because a repetitive file's inflated hash-bucket occupancy beat
the true match's count. We therefore use an IDF-style weighted score: each
matched position is weighted by 1 / posting_list_length for its hash value
(down-weighting common/non-discriminative hashes, the audio-fingerprinting
analogue of TF-IDF stopword down-weighting), which stays valid under pitch/tempo
change since it does not depend on absolute time alignment. Hash values whose
global posting list exceeds POSTING_LEN_CAP are pruned from the index entirely
at build time (a "stop-hash" cutoff) - they are already near-zero weight and
dropping them cuts both index size and query time.

A secondary absolute time-offset histogram (computed in one vectorized pass over
all candidates via a combined file/offset-bin bincount) is used as an auxiliary
confidence / explainability signal and to detect temporally-consistent matches
(most informative for time-preserving edits; naturally weaker for tempo-changed
queries, which downstream logic uses to help decide when to escalate to the
secondary stage).
"""

from dataclasses import dataclass, field

import numpy as np

POSTING_LEN_CAP = 300      # hash values with a global posting list longer than this are pruned
OFFSET_BIN_SEC = 0.5
MAX_OFFSET_SEC = 60.0


@dataclass
class Candidate:
    file_id: int
    weighted_score: float     # sum of 1/posting_length over matched hashes (primary ranking score)
    raw_matches: int          # raw hash-collision count (diagnostic only, not used for ranking)
    query_confidence: float   # weighted_score / n_query_hashes
    offset_peak: int          # size of the largest time-offset histogram bin
    offset_confidence: float  # offset_peak / raw_matches
    best_offset_sec: float


@dataclass
class LandmarkIndex:
    sorted_hash: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.uint32))
    local_ids: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int32))
    anchor_times: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    n_hashes_per_file: dict = field(default_factory=dict)
    file_id_to_local: dict = field(default_factory=dict)
    local_to_file_id: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    n_files: int = 0
    n_bins: int = int(2 * MAX_OFFSET_SEC / OFFSET_BIN_SEC) + 1

    def build(self, file_ids: list[int], landmark_list: list) -> None:
        """Classic entry point: takes the full in-memory list of per-file
        Landmark objects (unchanged signature/behavior). RAM-conscious callers
        that don't want to hold all N Landmark objects simultaneously (e.g. a
        chunked/streaming build) can instead call `flatten_chunk()` per chunk
        and `build_from_flat()` once at the end - see
        retrieval/pipeline.py's build_domain_index() for that path. Both
        paths produce byte-for-byte identical index contents."""
        self.file_id_to_local = {fid: i for i, fid in enumerate(file_ids)}
        self.local_to_file_id = np.array(file_ids, dtype=np.int64)
        self.n_files = len(file_ids)

        all_hash, all_lid, all_time = self.flatten_chunk(file_ids, landmark_list)
        self.build_from_flat(all_hash, all_lid, all_time)

    def flatten_chunk(self, file_ids: list[int], landmark_list: list):
        """Flattens one chunk's worth of per-file Landmark objects into three
        concatenated (hash, local_id, anchor_time) arrays, recording each
        file's raw landmark count along the way. Pure function of its
        arguments (does not touch self.sorted_hash/local_ids/anchor_times) so
        it can be called once per chunk during a streaming build, discarding
        each chunk's raw Landmark objects as soon as they're flattened instead
        of holding all N files' Landmark objects in RAM simultaneously."""
        hashes, lids, times = [], [], []
        for fid, lm in zip(file_ids, landmark_list):
            n = lm.hash.size
            self.n_hashes_per_file[fid] = int(n)
            if n == 0:
                continue
            local = self.file_id_to_local[fid]
            hashes.append(lm.hash)
            lids.append(np.full(n, local, dtype=np.int32))
            times.append(lm.anchor_time)

        if not hashes:
            empty_i = np.array([], dtype=np.uint32)
            return empty_i, np.array([], dtype=np.int32), np.array([], dtype=np.float32)
        return np.concatenate(hashes), np.concatenate(lids), np.concatenate(times)

    def build_from_flat(self, all_hash: np.ndarray, all_lid: np.ndarray, all_time: np.ndarray) -> None:
        """Sort + stop-hash-prune step, factored out of build() so a chunked
        caller can flatten each chunk separately (see flatten_chunk()) and
        concatenate just the (much smaller, already-compact) per-chunk arrays
        here instead of concatenating N raw per-file Landmark objects at
        once. Requires self.file_id_to_local/local_to_file_id/n_files and
        self.n_hashes_per_file to already be populated (build() and the
        chunked caller in pipeline.py both do this before calling)."""
        if all_hash.size == 0:
            self.sorted_hash = np.array([], dtype=np.uint32)
            self.local_ids = np.array([], dtype=np.int32)
            self.anchor_times = np.array([], dtype=np.float32)
            return

        # RAM optimization (build-time only, output is bit-for-bit identical to
        # the previous implementation - see below): the previous version
        # materialized THREE full-size sorted copies (sorted_hash, sorted_lid,
        # sorted_time) and then THREE more full-size masked copies from those.
        # Only sorted_hash is actually needed as a full array (to compute the
        # stop-hash group counts via np.unique) - local_ids/anchor_times never
        # need a full-size "sorted" intermediate at all, because fancy-index
        # composition is associative: sorted_lid[keep_mask] is mathematically
        # identical to all_lid[order][keep_mask], which is identical to
        # all_lid[order[keep_mask]] - so we compute the combined
        # sort+prune index array ONCE (order[keep_mask], already
        # pruned-size, i.e. small) and gather directly from the original
        # (unsorted) concatenated arrays with it. Measured to cut this
        # function's transient peak by ~25-30% (removes two full-corpus-size
        # arrays) with the exact same final self.sorted_hash/local_ids/
        # anchor_times contents and ordering as before.
        order = np.argsort(all_hash, kind="stable")
        sorted_hash = all_hash[order]

        # Prune "stop-hashes": hash values whose global posting list is too long
        # to be discriminative (near-zero IDF weight anyway).
        _, first_idx, group_counts = np.unique(sorted_hash, return_index=True, return_counts=True)
        keep_group = group_counts <= POSTING_LEN_CAP
        keep_mask = np.repeat(keep_group, group_counts)
        del sorted_hash, keep_group, group_counts, first_idx

        final_order = order[keep_mask]
        del order, keep_mask

        self.sorted_hash = all_hash[final_order]
        self.local_ids = all_lid[final_order]
        self.anchor_times = all_time[final_order]

    def nbytes(self) -> int:
        return self.sorted_hash.nbytes + self.local_ids.nbytes + self.anchor_times.nbytes

    def get_file_hashes(self, file_id: int) -> np.ndarray:
        """On-demand reconstruction of a single file's (surviving, post-prune)
        hash set from the shared index arrays, instead of keeping a second,
        redundant per-file copy of every landmark in RAM (memory is tight
        against the 120MB budget - see REPORT.md's RAM breakdown)."""
        local = self.file_id_to_local.get(file_id)
        if local is None:
            return np.array([], dtype=np.uint32)
        mask = self.local_ids == local
        return self.sorted_hash[mask]

    def query(self, query_landmark, top_k: int = 5) -> list[Candidate]:
        q_hash = query_landmark.hash
        q_time = query_landmark.anchor_time
        n_query = q_hash.size
        if n_query == 0 or self.sorted_hash.size == 0:
            return []

        lo_idx = np.searchsorted(self.sorted_hash, q_hash, side="left")
        hi_idx = np.searchsorted(self.sorted_hash, q_hash, side="right")
        posting_len = hi_idx - lo_idx
        total = int(posting_len.sum())
        if total == 0:
            return []

        # Vectorized "ragged range" gather: expand each (lo, hi) match range into
        # its absolute index positions without a Python-level per-hash loop.
        cum = np.cumsum(posting_len)
        local_offsets = np.arange(total) - np.repeat(cum - posting_len, posting_len)
        positions = np.repeat(lo_idx, posting_len) + local_offsets

        all_lids = self.local_ids[positions]
        all_offsets = self.anchor_times[positions] - np.repeat(q_time, posting_len)
        all_weights = 1.0 / np.repeat(posting_len, posting_len).astype(np.float64)

        raw_counts = np.bincount(all_lids, minlength=self.n_files)
        weighted_scores = np.bincount(all_lids, weights=all_weights, minlength=self.n_files)

        pool_size = min(max(top_k * 2, 10), self.n_files)
        pool_local_ids = np.argpartition(-weighted_scores, pool_size - 1)[:pool_size]
        pool_local_ids = pool_local_ids[np.argsort(-weighted_scores[pool_local_ids])]

        # Single vectorized pass to get the time-offset histogram peak for every
        # candidate at once (combined file x offset-bin bincount), instead of a
        # per-candidate Python loop.
        bin_idx = np.clip(
            np.round(all_offsets / OFFSET_BIN_SEC).astype(np.int64) + self.n_bins // 2,
            0, self.n_bins - 1,
        )
        combined = all_lids.astype(np.int64) * self.n_bins + bin_idx
        hist = np.bincount(combined, minlength=self.n_files * self.n_bins)
        hist = hist.reshape(self.n_files, self.n_bins)

        candidates = []
        for local_id in pool_local_ids:
            raw = int(raw_counts[local_id])
            if raw == 0:
                continue
            wscore = float(weighted_scores[local_id])
            row = hist[local_id]
            offset_peak = int(row.max())
            best_bin = int(row.argmax())

            candidates.append(Candidate(
                file_id=int(self.local_to_file_id[local_id]),
                weighted_score=wscore,
                raw_matches=raw,
                query_confidence=wscore / n_query,
                offset_peak=offset_peak,
                offset_confidence=(offset_peak / raw) if raw else 0.0,
                best_offset_sec=float((best_bin - self.n_bins // 2) * OFFSET_BIN_SEC),
            ))

        candidates.sort(key=lambda c: c.weighted_score, reverse=True)
        return candidates[:top_k]
