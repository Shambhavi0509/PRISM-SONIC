"""Runs the primary benchmark for exactly ONE domain in its own process, so
Peak RSS reflects that domain alone (not the cumulative 3-domain process used
by 06_run_primary_benchmark.py). Invoked once per domain (see
scripts/run_per_domain_peak_ram.ps1 or the loop in the calling shell) so each
run starts from a clean interpreter with no other domain's index in memory.

RSS is sampled at every checkpoint (after index build, after every 20 queries
within duplicate/edited loops) to catch the true peak, not just stage
boundaries.

Usage:
    python scripts/11_run_per_domain_peak_ram.py --category speech
    python scripts/11_run_per_domain_peak_ram.py --category music
    python scripts/11_run_per_domain_peak_ram.py --category environment

Writes results/peak_ram_<category>.json
"""

import argparse
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", required=True, choices=["speech", "music", "environment"])
    args = parser.parse_args()
    category = args.category

    peak_rss = rss_mb()
    rss_at_start = peak_rss
    print(f"[{category}] RSS at start: {rss_at_start:.1f} MB", flush=True)

    with open(RESULTS_ROOT / "manifest.json") as f:
        manifest = json.load(f)
    with open(RESULTS_ROOT / "primary_benchmark_manifest.json") as f:
        mod_records_all = json.load(f)["records"]

    entries = manifest["categories"][category]

    t0 = time.perf_counter()
    idx = build_domain_index(category, entries)
    build_time = time.perf_counter() - t0
    peak_rss = max(peak_rss, rss_mb())
    print(f"[{category}] index built in {build_time:.1f}s, RSS={rss_mb():.1f}MB, "
          f"index_bytes={idx.nbytes()/1024/1024:.2f}MB", flush=True)

    # --- Duplicate accuracy (all N in this domain) ---
    dup_records = []
    for i, e in enumerate(entries):
        t0 = time.perf_counter()
        res = query(e["path"], idx)
        lat = (time.perf_counter() - t0) * 1000
        dup_records.append({
            "correct": res.file_id == e["file_id"], "predicted_id": res.file_id,
            "true_id": e["file_id"], "ranked_ids": res.ranked_ids, "latency_ms": lat,
        })
        if (i + 1) % 20 == 0:
            peak_rss = max(peak_rss, rss_mb())
    peak_rss = max(peak_rss, rss_mb())
    dup_stats = accuracy_stats(dup_records)
    dup_lat = latency_stats([r["latency_ms"] for r in dup_records])
    print(f"[{category}] duplicate accuracy: {dup_stats['accuracy']:.3f} "
          f"recall@5={dup_stats['recall_at_5']:.3f} mean_lat={dup_lat['mean_ms']:.1f}ms "
          f"RSS={rss_mb():.1f}MB", flush=True)

    # --- Edited accuracy, per modification (all N each) ---
    cat_mod_records = [r for r in mod_records_all if r["category"] == category]
    by_mod = defaultdict(list)
    for r in cat_mod_records:
        by_mod[r["modification"]].append(r)

    edited_all = []
    per_mod_results = {}
    for mod_name, recs in sorted(by_mod.items()):
        mod_query_records = []
        for i, r in enumerate(recs):
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
            if (i + 1) % 20 == 0:
                peak_rss = max(peak_rss, rss_mb())
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
              f"acc={stats['accuracy']:.3f} n={stats['n']} mean_lat={lat_stats['mean_ms']:.1f}ms "
              f"RSS={rss_mb():.1f}MB", flush=True)
        edited_all.extend(mod_query_records)

    edited_overall_stats = accuracy_stats(edited_all)
    edited_overall_lat = latency_stats([r["latency_ms"] for r in edited_all])
    peak_rss = max(peak_rss, rss_mb())

    result = {
        "category": category,
        "rss_at_start_mb": round(rss_at_start, 2),
        "rss_after_index_build_mb": None,  # filled below for clarity in output file
        "peak_rss_mb": round(peak_rss, 2),
        "rss_growth_over_baseline_mb": round(peak_rss - rss_at_start, 2),
        "index_build_time_sec": round(build_time, 2),
        "index_bytes": idx.nbytes(),
        "duplicate": {**dup_stats, **dup_lat},
        "edited_overall": {**edited_overall_stats, **edited_overall_lat},
        "edited_by_modification": per_mod_results,
    }

    out_path = RESULTS_ROOT / f"peak_ram_{category}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n[{category}] Peak RSS (isolated process): {peak_rss:.1f} MB "
          f"(baseline {rss_at_start:.1f} MB, growth {peak_rss - rss_at_start:.1f} MB)", flush=True)
    print(f"[{category}] Results written to {out_path}", flush=True)


if __name__ == "__main__":
    main()
