"""Metric computation over raw per-query benchmark records. Kept dependency-free
(no numpy needed) so it can run inside the same clean, librosa-free benchmark
process used for the RAM/latency measurements.
"""

import math


def _percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return float("nan")
    k = (len(sorted_vals) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def latency_stats(latencies_ms: list[float]) -> dict:
    if not latencies_ms:
        return {"mean_ms": None, "p50_ms": None, "p95_ms": None, "p99_ms": None,
                "max_ms": None, "throughput_qps": None}
    s = sorted(latencies_ms)
    mean = sum(s) / len(s)
    return {
        "mean_ms": mean,
        "p50_ms": _percentile(s, 0.50),
        "p95_ms": _percentile(s, 0.95),
        "p99_ms": _percentile(s, 0.99),
        "max_ms": s[-1],
        "throughput_qps": (1000.0 / mean) if mean > 0 else None,
    }


def accuracy_stats(records: list[dict]) -> dict:
    """records: dicts with at least {"correct": bool, "predicted_id", "true_id",
    "ranked_ids": list}."""
    n = len(records)
    if n == 0:
        return {"n": 0, "accuracy": None, "recall_at_1": None, "recall_at_5": None, "precision": None}

    correct = sum(1 for r in records if r["correct"])
    recall5 = sum(1 for r in records if r["true_id"] in r.get("ranked_ids", []))
    accepted = sum(1 for r in records if r.get("predicted_id") is not None)

    return {
        "n": n,
        "accuracy": correct / n,
        "recall_at_1": correct / n,
        "recall_at_5": recall5 / n,
        "precision": (correct / accepted) if accepted else None,
        "n_accepted": accepted,
        "n_rejected_no_match": n - accepted,
    }


def false_positive_rate(negative_records: list[dict]) -> dict:
    """negative_records: queries whose true file is NOT in the index; a
    "false positive" is any non-None prediction (the system should have said
    no-match)."""
    n = len(negative_records)
    if n == 0:
        return {"n": 0, "fpr": None}
    fp = sum(1 for r in negative_records if r.get("predicted_id") is not None)
    return {"n": n, "fpr": fp / n, "n_false_positives": fp}


def summarize_condition(records: list[dict]) -> dict:
    """Combines accuracy + latency stats for one (domain, modification, level) condition."""
    acc = accuracy_stats(records)
    lat = latency_stats([r["latency_ms"] for r in records])
    n_escalated = sum(1 for r in records if r.get("escalated"))
    return {**acc, **lat, "n_escalated_to_secondary": n_escalated}
