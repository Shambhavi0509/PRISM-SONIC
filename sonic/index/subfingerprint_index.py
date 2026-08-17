"""Multi-window sub-fingerprint index (Haitsma & Kalker, ISMIR 2002 style
majority-vote matching). Unlike BinaryHNSWIndex (one global code per file),
this stores MANY short overlapping window codes per indexed file and matches
a query's own window codes against all of them, voting per candidate file.

This is the fix for the redundancy gap diagnosed in pitch/tempo-shifted
queries: a single aggregate code has no fallback if scrambled by a
phase-vocoder pitch/speed modification, but with many independent overlapping
windows, some (e.g. more tonal/stable segments) usually still match even when
others don't. See fingerprint/binary_embed.py's module docstring for the
measured motivation and retrieval/pipeline.py for when this runs (only during
confidence-gated escalation, never on the fast/confident path).
"""

from dataclasses import dataclass, field

import faiss
import numpy as np

HNSW_M = 32


@dataclass
class SubFingerprintIndex:
    index: object = None
    window_file_ids: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    n_bits: int = 0
    idf_weight: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))

    def build(self, window_file_ids: list[int], codes: np.ndarray) -> None:
        if len(window_file_ids) == 0:
            return
        self.n_bits = codes.shape[1] * 8
        self.index = faiss.IndexBinaryHNSW(self.n_bits, HNSW_M)
        self.index.add(codes)
        self.window_file_ids = np.array(window_file_ids, dtype=np.int64)
        self._compute_idf_weights(codes)

    def _compute_idf_weights(self, codes: np.ndarray, threshold_frac: float = 0.15, k: int = 30) -> None:
        """Diagnosed on real failures: a few 'hub' window codes (generic
        content shared across many DIFFERENT files, e.g. similar drum/loop
        patterns in this music dataset) win votes for the wrong file even when
        the true file's own windows also match closely - the same hub
        pathology fixed for the primary landmark index via IDF weighting (see
        index/landmark_index.py), applied here the same way: down-weight
        window codes whose near-neighborhood spans many other files, one-time
        cost at index build, not part of query-time latency."""
        n = codes.shape[0]
        k = min(k, n)
        distances, labels = self.index.search(codes, k)
        threshold = max(1, int(round(self.n_bits * threshold_frac)))

        commonness = np.zeros(n, dtype=np.int32)
        for i in range(n):
            my_fid = self.window_file_ids[i]
            other_file_mask = (labels[i] >= 0) & (labels[i] != i) & (self.window_file_ids[labels[i].clip(min=0)] != my_fid)
            close_mask = distances[i] <= threshold
            commonness[i] = int(np.sum(other_file_mask & close_mask))

        self.idf_weight = 1.0 / (1.0 + commonness.astype(np.float64))

    def vote(self, query_codes: np.ndarray, top_k_per_window: int = 5, final_top_k: int = 5) -> list[tuple[int, float]]:
        """query_codes: (n_windows, n_bytes) packed codes, one per query window.
        Returns [(file_id, vote_score), ...] sorted descending. Each query
        window contributes its BEST match per candidate file (so a file with
        many windows near one query window isn't over-counted), summed across
        all query windows - this is the majority-vote step. Each match is
        weighted by the matched index entry's IDF weight (see
        _compute_idf_weights), down-weighting generic/common window codes."""
        if self.index is None or query_codes.shape[0] == 0 or len(self.window_file_ids) == 0:
            return []

        k = min(top_k_per_window, len(self.window_file_ids))
        distances, labels = self.index.search(query_codes, k)

        scores: dict[int, float] = {}
        for w in range(labels.shape[0]):
            best_per_file: dict[int, float] = {}
            for lbl, dist in zip(labels[w], distances[w]):
                if lbl < 0:
                    continue
                fid = int(self.window_file_ids[lbl])
                weight = self.idf_weight[lbl] if self.idf_weight.size else 1.0
                sim = float(self.n_bits - dist) * weight
                if fid not in best_per_file or sim > best_per_file[fid]:
                    best_per_file[fid] = sim
            for fid, sim in best_per_file.items():
                scores[fid] = scores.get(fid, 0.0) + sim

        ranked = sorted(scores.items(), key=lambda kv: -kv[1])[:final_top_k]
        return ranked

    def nbytes(self) -> int:
        if self.index is None:
            return 0
        code_bytes = (self.n_bits // 8) * len(self.window_file_ids)
        graph_overhead = len(self.window_file_ids) * HNSW_M * 2 * 4
        return code_bytes + graph_overhead
