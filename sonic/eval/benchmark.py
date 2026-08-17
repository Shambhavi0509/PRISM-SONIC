"""Benchmark harness. Deliberately imports only sonic.io / sonic.fingerprint /
sonic.index / sonic.retrieval (never sonic.augment, never librosa) so that the
process this runs in reflects the actual deployed retrieval system's memory
footprint - modified query audio is pre-generated offline by
scripts/02_generate_modified_queries.py and simply read as WAV here.
"""

import json
import os
import time

import psutil
import soundfile as sf

from sonic.config import RESULTS_ROOT
from sonic.retrieval.pipeline import build_domain_index, query, query_array

_PROC = psutil.Process(os.getpid())


def rss_mb() -> float:
    return _PROC.memory_info().rss / 1024 / 1024


class RSSTracker:
    """Samples RSS at each `.sample()` call; reports the observed peak."""

    def __init__(self):
        self.peak = rss_mb()
        self.samples = [self.peak]

    def sample(self):
        v = rss_mb()
        self.samples.append(v)
        if v > self.peak:
            self.peak = v
        return v


def run_duplicate_benchmark(category: str, idx, entries: list[dict], tracker: RSSTracker) -> list[dict]:
    records = []
    for i, e in enumerate(entries):
        t0 = time.perf_counter()
        res = query(e["path"], idx)
        latency_ms = (time.perf_counter() - t0) * 1000

        records.append({
            "category": category,
            "condition": "duplicate",
            "level": "0",
            "true_id": e["file_id"],
            "predicted_id": res.file_id,
            "correct": res.file_id == e["file_id"],
            "ranked_ids": res.ranked_ids,
            "latency_ms": latency_ms,
            "escalated": res.escalated,
            "confidence": res.primary_confidence,
            "offset_confidence": res.offset_confidence,
        })
        if i % 20 == 0:
            tracker.sample()
    tracker.sample()
    return records


def run_negative_benchmark(category: str, idx, negative_entries: list[dict], tracker: RSSTracker) -> list[dict]:
    records = []
    for e in negative_entries:
        t0 = time.perf_counter()
        res = query(e["path"], idx)
        latency_ms = (time.perf_counter() - t0) * 1000
        records.append({
            "category": category,
            "condition": "negative",
            "true_id": None,
            "predicted_id": res.file_id,
            "latency_ms": latency_ms,
            "escalated": res.escalated,
        })
    tracker.sample()
    return records


def run_edited_benchmark(category: str, idx, modified_records: list[dict], tracker: RSSTracker) -> list[dict]:
    """modified_records: entries from results/modified_queries_manifest.json for
    this category (already filtered), each with modified_path/true_file_id/
    modification/level."""
    out = []
    for i, rec in enumerate(modified_records):
        if rec.get("modified_path") is None:
            out.append({
                "category": category, "condition": rec["modification"], "level": rec["level"],
                "true_id": rec["true_file_id"], "predicted_id": None, "correct": False,
                "ranked_ids": [], "latency_ms": 0.0, "escalated": False,
                "generation_failed": True,
            })
            continue

        y, sr = sf.read(rec["modified_path"], dtype="float32", always_2d=False)
        t0 = time.perf_counter()
        res = query_array(y, idx, sr=sr)
        latency_ms = (time.perf_counter() - t0) * 1000

        out.append({
            "category": category,
            "condition": rec["modification"],
            "level": rec["level"],
            "true_id": rec["true_file_id"],
            "predicted_id": res.file_id,
            "correct": res.file_id == rec["true_file_id"],
            "ranked_ids": res.ranked_ids,
            "latency_ms": latency_ms,
            "escalated": res.escalated,
            "confidence": res.primary_confidence,
            "offset_confidence": res.offset_confidence,
        })
        if i % 50 == 0:
            tracker.sample()
    tracker.sample()
    return out


def save_records(records: list[dict], name: str) -> None:
    path = RESULTS_ROOT / f"{name}.json"
    with open(path, "w") as f:
        json.dump(records, f)
    print(f"wrote {len(records)} records to {path}")
