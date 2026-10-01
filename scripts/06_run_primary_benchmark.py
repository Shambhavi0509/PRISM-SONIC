"""Runs the user-specified primary robustness benchmark: duplicate accuracy
(full 200/domain) + edited accuracy for exactly the 10 modifications at their
specified fixed severities (see sonic/augment/modifications.py's
PRIMARY_BENCHMARK_MODIFICATIONS), also on the full 200/domain.

Deliberately imports only sonic.io/fingerprint/index/retrieval (never
sonic.augment, never librosa) - see sonic/eval/benchmark.py's docstring for
why: this keeps the peak-RAM measurement honest (reflects the deployed
retrieval system, not the librosa-based test-data generator).

Outputs results/primary_benchmark_results.json with, per domain: duplicate
accuracy, edited accuracy (overall + per modification), recall@1, recall@5,
average/P50/P95/P99 query time, and peak RAM (shared across all domains since
it's one process). Also prints the exact query count per modification.
"""

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import psutil
import soundfile as sf

from sonic.config import RESULTS_ROOT
from sonic.eval.metrics import accuracy_stats, latency_stats
from sonic.retrieval.pipeline import build_domain_index, query, query_array

PROC = psutil.Process()


def rss_mb():
    return PROC.memory_info().rss / 1024 / 1024


def main():
    print(f"RSS at start: {rss_mb():.1f} MB", flush=True)

    with open(RESULTS_ROOT / "manifest.json") as f:
        manifest = json.load(f)
    with open(RESULTS_ROOT / "primary_benchmark_manifest.json") as f:
        mod_records_all = json.load(f)["records"]

    peak_rss = rss_mb()
    results = {}
    all_edited_by_mod = defaultdict(list)  # mod_name -> records across all domains, for exact-count reporting

    for category in ["speech", "music", "environment"]:
        entries = manifest["categories"][category]  # full 200

        t0 = time.perf_counter()
        idx = build_domain_index(category, entries)
        build_time = time.perf_counter() - t0
        peak_rss = max(peak_rss, rss_mb())
        print(f"[{category}] index built in {build_time:.1f}s, RSS={rss_mb():.1f}MB, "
              f"index_bytes={idx.nbytes()/1024/1024:.2f}MB", flush=True)

        # --- Duplicate accuracy (all 200) ---
        dup_records = []
        for e in entries:
            t0 = time.perf_counter()
            res = query(e["path"], idx)
            lat = (time.perf_counter() - t0) * 1000
            dup_records.append({
                "correct": res.file_id == e["file_id"], "predicted_id": res.file_id,
                "true_id": e["file_id"], "ranked_ids": res.ranked_ids, "latency_ms": lat,
            })
        peak_rss = max(peak_rss, rss_mb())
        dup_stats = accuracy_stats(dup_records)
        dup_lat = latency_stats([r["latency_ms"] for r in dup_records])
        print(f"[{category}] duplicate accuracy: {dup_stats['accuracy']:.3f} "
              f"recall@5={dup_stats['recall_at_5']:.3f} mean_lat={dup_lat['mean_ms']:.1f}ms", flush=True)

        # --- Edited accuracy, per modification (all 200 each) ---
        cat_mod_records = [r for r in mod_records_all if r["category"] == category]
        by_mod = defaultdict(list)
        for r in cat_mod_records:
            by_mod[r["modification"]].append(r)

        edited_all = []
        per_mod_results = {}
        for mod_name, recs in sorted(by_mod.items()):
            mod_query_records = []
            for r in recs:
                all_edited_by_mod[mod_name].append(r)
                if r.get("modified_path") is None:
                    mod_query_records.append({
                        "correct": False, "predicted_id": None, "true_id": r["true_file_id"],
                        "ranked_ids": [], "latency_ms": 0.0,
                    })
                    continue
                y, sr = sf.read(r["modified_path"], dtype="float32", always_2d=False)
                t0 = time.perf_counter()
                res = query_array(y, idx, sr=sr)
                lat = (time.perf_counter() - t0) * 1000
                mod_query_records.append({
                    "correct": res.file_id == r["true_file_id"], "predicted_id": res.file_id,
                    "true_id": r["true_file_id"], "ranked_ids": res.ranked_ids, "latency_ms": lat,
                    "level": r["level"], "unit": r["unit"],
                })
            peak_rss = max(peak_rss, rss_mb())
            stats = accuracy_stats(mod_query_records)
            lat_stats = latency_stats([r["latency_ms"] for r in mod_query_records])
            levels_tested = sorted(set(r["level"] for r in recs))
            per_mod_results[mod_name] = {
                "levels_tested": levels_tested, "unit": recs[0]["unit"],
                "n_queries": stats["n"], "accuracy": stats["accuracy"],
                "recall_at_1": stats["recall_at_1"], "recall_at_5": stats["recall_at_5"],
                **lat_stats,
            }
            print(f"[{category}] {mod_name} (levels={levels_tested}): "
                  f"acc={stats['accuracy']:.3f} n={stats['n']} mean_lat={lat_stats['mean_ms']:.1f}ms", flush=True)
            edited_all.extend(mod_query_records)

        edited_overall_stats = accuracy_stats(edited_all)
        edited_overall_lat = latency_stats([r["latency_ms"] for r in edited_all])

        results[category] = {
            "duplicate": {**dup_stats, **dup_lat},
            "edited_overall": {**edited_overall_stats, **edited_overall_lat},
            "edited_by_modification": per_mod_results,
            "index_build_time_sec": build_time,
            "index_bytes": idx.nbytes(),
        }

    results["peak_rss_mb"] = peak_rss
    results["query_counts_per_modification"] = {
        mod: len(recs) for mod, recs in all_edited_by_mod.items()
    }

    with open(RESULTS_ROOT / "primary_benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nPeak RSS: {peak_rss:.1f} MB", flush=True)
    print("Query counts per modification (pooled across 3 domains):", flush=True)
    for mod, n in results["query_counts_per_modification"].items():
        print(f"  {mod}: {n}", flush=True)
    print(f"Results written to {RESULTS_ROOT / 'primary_benchmark_results.json'}", flush=True)

    print_summary_table(results)


def print_summary_table(results: dict) -> None:
    """Purely a display step: derives everything below from the already-computed
    `results` dict (accuracy_stats/latency_stats outputs) - no benchmark logic
    here, just formatting the final numbers into a clean summary table."""
    display_names = [("speech", "Speech"), ("music", "Music"), ("environment", "Environmental")]

    rows = []
    pooled_latency_sum = 0.0
    pooled_n = 0
    for category, label in display_names:
        dup = results[category]["duplicate"]
        edited = results[category]["edited_overall"]

        dup_correct = round(dup["accuracy"] * dup["n"])
        edited_correct = round(edited["accuracy"] * edited["n"])

        dup_str = f"{dup['accuracy'] * 100:.2f}% ({dup_correct}/{dup['n']})"
        edited_str = f"{edited['accuracy'] * 100:.2f}% ({edited_correct}/{edited['n']})"
        rows.append((label, dup_str, edited_str))

        pooled_latency_sum += dup["mean_ms"] * dup["n"] + edited["mean_ms"] * edited["n"]
        pooled_n += dup["n"] + edited["n"]

    pooled_avg_ms = pooled_latency_sum / pooled_n if pooled_n else float("nan")

    col1_w = max(16, max(len(r[0]) for r in rows) + 2)
    col2_w = max(24, max(len(r[1]) for r in rows) + 2)

    width = 65
    print()
    print("=" * width)
    print("BENCHMARK RESULTS".center(width))
    print("=" * width)
    print()
    print(f"{'Dataset':<{col1_w}}{'Duplicate Accuracy':<{col2_w}}{'Edited Accuracy'}")
    print("-" * width)
    for label, dup_str, edited_str in rows:
        print(f"{label:<{col1_w}}{dup_str:<{col2_w}}{edited_str}")
    print("-" * width)
    print()
    print(f"Peak RAM: {results['peak_rss_mb']:.1f} MB")
    print(f"Pooled average query time: {pooled_avg_ms:.2f} ms")
    print("=" * width)


if __name__ == "__main__":
    main()
