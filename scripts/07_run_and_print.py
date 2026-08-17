"""Runs the primary benchmark (duplicate + edited accuracy, all 10 modifications,
all domains) and prints a clean summary table directly to the terminal, in
addition to (not instead of) writing the full results JSON for programmatic use.

Same underlying pipeline as scripts/06_run_primary_benchmark.py.
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

    for category in ["speech", "music", "environment"]:
        entries = manifest["categories"][category]

        t0 = time.perf_counter()
        idx = build_domain_index(category, entries)
        build_time = time.perf_counter() - t0
        peak_rss = max(peak_rss, rss_mb())
        print(f"[{category}] index built in {build_time:.1f}s over {len(entries)} files, "
              f"RSS={rss_mb():.1f}MB", flush=True)

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
        print(f"[{category}] duplicate accuracy: {dup_stats['accuracy']:.4f} "
              f"({int(round(dup_stats['accuracy']*dup_stats['n']))}/{dup_stats['n']})", flush=True)

        cat_mod_records = [r for r in mod_records_all if r["category"] == category]
        by_mod = defaultdict(list)
        for r in cat_mod_records:
            by_mod[r["modification"]].append(r)

        edited_all = []
        per_mod_results = {}
        for mod_name, recs in sorted(by_mod.items()):
            mod_query_records = []
            for r in recs:
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
                  f"acc={stats['accuracy']:.4f} n={stats['n']} mean_lat={lat_stats['mean_ms']:.1f}ms", flush=True)
            edited_all.extend(mod_query_records)

        edited_overall_stats = accuracy_stats(edited_all)
        edited_overall_lat = latency_stats([r["latency_ms"] for r in edited_all])
        print(f"[{category}] EDITED overall accuracy: {edited_overall_stats['accuracy']:.4f} "
              f"({int(round(edited_overall_stats['accuracy']*edited_overall_stats['n']))}/{edited_overall_stats['n']})", flush=True)

        results[category] = {
            "duplicate": {**dup_stats, **dup_lat},
            "edited_overall": {**edited_overall_stats, **edited_overall_lat},
            "edited_by_modification": per_mod_results,
            "index_build_time_sec": build_time,
            "index_bytes": idx.nbytes(),
        }

    results["peak_rss_mb"] = peak_rss

    with open(RESULTS_ROOT / "primary_benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # ---------------- clean terminal summary table ----------------
    def fmt_acc(stats):
        n = stats["n"]
        correct = int(round(stats["accuracy"] * n))
        return f"{stats['accuracy']*100:6.2f}% ({correct}/{n})"

    print("", flush=True)
    print("=" * 64, flush=True)
    print("                   BENCHMARK RESULTS".center(64), flush=True)
    print("=" * 64, flush=True)
    print("", flush=True)
    print(f"{'Dataset':<16}{'Duplicate Accuracy':<26}{'Edited Accuracy':<22}", flush=True)
    print("-" * 64, flush=True)
    for cat, label in [("speech", "Speech"), ("music", "Music"), ("environment", "Environmental")]:
        d = results[cat]
        dup_str = fmt_acc(d["duplicate"])
        edit_str = fmt_acc(d["edited_overall"])
        print(f"{label:<16}{dup_str:<26}{edit_str:<22}", flush=True)
    print("-" * 64, flush=True)
    print("", flush=True)
    print(f"Peak RAM: {peak_rss:.1f} MB", flush=True)
    all_dup_n = sum(results[c]["duplicate"]["n"] for c in ["speech", "music", "environment"])
    all_dup_lat = sum(results[c]["duplicate"]["n"] * results[c]["duplicate"]["mean_ms"] for c in ["speech", "music", "environment"])
    all_ed_n = sum(results[c]["edited_overall"]["n"] for c in ["speech", "music", "environment"])
    all_ed_lat = sum(results[c]["edited_overall"]["n"] * results[c]["edited_overall"]["mean_ms"] for c in ["speech", "music", "environment"])
    pooled_mean = (all_dup_lat + all_ed_lat) / (all_dup_n + all_ed_n)
    print(f"Pooled average query time (all {all_dup_n + all_ed_n} queries): {pooled_mean:.1f} ms", flush=True)
    print("=" * 64, flush=True)
    print(f"\nFull results also written to {RESULTS_ROOT / 'primary_benchmark_results.json'}", flush=True)


if __name__ == "__main__":
    main()
