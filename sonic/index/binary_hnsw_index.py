"""FAISS IndexBinaryHNSW wrapper for the secondary (fallback-only) embedding
stage. See fingerprint/binary_embed.py for why this exists and when it's used;
see retrieval/pipeline.py for the confidence-gated activation logic.
"""

from dataclasses import dataclass, field

import faiss
import numpy as np

HNSW_M = 32  # HNSW graph out-degree (faiss default-scale choice for small corpora)


@dataclass
class BinaryHNSWIndex:
    index: object = None
    file_ids: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    n_bits: int = 0

    def build(self, file_ids: list[int], codes: np.ndarray) -> None:
        self.n_bits = codes.shape[1] * 8
        self.index = faiss.IndexBinaryHNSW(self.n_bits, HNSW_M)
        self.index.add(codes)
        self.file_ids = np.array(file_ids, dtype=np.int64)

    def search(self, query_code: np.ndarray, top_k: int = 5):
        """Returns list of (file_id, hamming_distance) sorted by distance ascending."""
        q = query_code.reshape(1, -1)
        distances, labels = self.index.search(q, min(top_k, len(self.file_ids)))
        results = []
        for lbl, dist in zip(labels[0], distances[0]):
            if lbl < 0:
                continue
            results.append((int(self.file_ids[lbl]), int(dist)))
        return results

    def nbytes(self) -> int:
        # faiss doesn't expose exact graph memory directly; approximate with the
        # stored code data + a per-vector HNSW graph overhead estimate.
        code_bytes = (self.n_bits // 8) * len(self.file_ids)
        graph_overhead = len(self.file_ids) * HNSW_M * 2 * 4  # rough: M links x 2 (up/down) x 4 bytes
        return code_bytes + graph_overhead
