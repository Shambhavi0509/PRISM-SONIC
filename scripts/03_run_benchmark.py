"""Main benchmark run. Deliberately never imports sonic.augment or librosa (see
sonic/eval/benchmark.py's module docstring) - run scripts/02_generate_modified_queries.py
first to produce results/modified_queries_manifest.json.

Produces:
  results/duplicate_records.json   - one record per duplicate (unmodified) query
  results/edited_records.json      - one record per modified query
  results/negative_records.json    - one record per "not in index" query
  results/index_stats.json         - per-domain index size / build time / RSS
  results/summary.json             - aggregated metrics tables (per REPORT.md)
"""

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sonic.config import RESULTS_ROOT
from sonic.eval.benchmark import (
    RSSTracker, rss_mb,
    run_duplicate_benchmark, run_edited_benchmark, run_negative_benchmark,
    save_records,
)
from sonic.eval.metrics import accuracy_stats, false_positive_rate, latency_stats, summarize_condition
from sonic.retrieval.pipeline import build_domain_index

CATEGORIES = ["speech", "music", "environment"]


def main():
    print(f"RSS at process start: {rss_mb():.1f} MB", flush=True)

    with open(RESULTS_ROOT / "manifest.json") as f:
        manifest = json.load(f)

    modified_manifest_path = RESULTS_ROOT / "modified_queries_manifest.json"
    modified_records_all = []
    if modified_manifest_path.exists():
        with open(modified_manifest_path) as f:
            modified_records_all = json.load(f)["records"]
    else:
        print("WARNING: results/modified_queries_manifest.json not found - run "
              "scripts/02_generate_modified_queries.py first. Edited-audio benchmark will be skipped.")

    tracker = RSSTracker()
    index_stats = {}
    duplicate_records, edited_records, negative_records = [], [], []

    for category in CATEGORIES:
        entries = manifest["categories"][category]

        t0 = time.perf_counter()
        idx = build_domain_index(category, entries)
        build_time = time.perf_counter() - t0
        tracker.sample()

        index_stats[category] = {
            "n_files": len(entries),
            "build_time_sec": build_time,
            "index_bytes": idx.nbytes(),
            "landmark_index_bytes": idx.landmark_index.nbytes(),
            "binary_embed_bytes": idx.binary_embed.nbytes(),
            "binary_hnsw_bytes": idx.binary_hnsw.nbytes(),
        }
        print(f"[{category}] index built: {build_time:.1f}s, {idx.nbytes()/1024/1024:.2f}MB, "
              f"RSS={rss_mb():.1f}MB", flush=True)

        dup_recs = run_duplicate_benchmark(category, idx, entries, tracker)
        duplicate_records.extend(dup_recs)
        dup_acc = accuracy_stats(dup_recs)
        print(f"[{category}] duplicate accuracy: {dup_acc['accuracy']:.3f} "
              f"({dup_acc['n'] - int(round((1-dup_acc['accuracy'])*dup_acc['n']))}/{dup_acc['n']})", flush=True)

        cat_mod_records = [r for r in modified_records_all if r["category"] == category]
        if cat_mod_records:
            ed_recs = run_edited_benchmark(category, idx, cat_mod_records, tracker)
            edited_records.extend(ed_recs)
            ed_acc = accuracy_stats(ed_recs)
            print(f"[{category}] edited accuracy (all modifications pooled): {ed_acc['accuracy']:.3f}", flush=True)

        neg_entries = manifest.get("held_out_negative", {}).get(category, [])
        if neg_entries:
            neg_recs = run_negative_benchmark(category, idx, neg_entries, tracker)
            negative_records.extend(neg_recs)
            fpr = false_positive_rate(neg_recs)
            print(f"[{category}] false positive rate: {fpr['fpr']}", flush=True)

    save_records(duplicate_records, "duplicate_records")
    save_records(edited_records, "edited_records")
    save_records(negative_records, "negative_records")

    with open(RESULTS_ROOT / "index_stats.json", "w") as f:
        json.dump(index_stats, f, indent=2)

    # ---- Aggregate summary ----
    summary = {"peak_rss_mb": tracker.peak, "index_stats": index_stats, "per_domain": {}}

    for category in CATEGORIES:
        dup = [r for r in duplicate_records if r["category"] == category]
        ed = [r for r in edited_records if r["category"] == category]
        neg = [r for r in negative_records if r["category"] == category]

        by_condition = defaultdict(list)
        for r in ed:
            by_condition[(r["condition"], r["level"])].append(r)

        summary["per_domain"][category] = {
            "duplicate": summarize_condition(dup),
            "edited_overall": summarize_condition(ed) if ed else None,
            "edited_by_condition": {
                f"{cond}@{level}": summarize_condition(recs)
                for (cond, level), recs in by_condition.items()
            },
            "false_positive_rate": false_positive_rate(neg),
        }

    all_latencies = [r["latency_ms"] for r in duplicate_records + edited_records]
    summary["overall_latency"] = latency_stats(all_latencies)

    with open(RESULTS_ROOT / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nPeak RSS observed: {tracker.peak:.1f} MB", flush=True)
    print(f"Summary written to {RESULTS_ROOT / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
