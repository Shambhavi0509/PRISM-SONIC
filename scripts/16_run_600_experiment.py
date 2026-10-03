"""600-clip augmentation experiment, end to end (200 Speech / 200 Music / 200 Environmental).

Orchestration only - no fingerprinting, matching, or augmentation logic lives here:

  1. scripts/10_augment_experiment.py   select 200/domain (seeded) + generate augmented queries
  2. (this script)                      turn the generated files into a benchmark query manifest
  3. scripts/11_run_per_domain_peak_ram.py  existing SONIC benchmark (build_domain_index + query),
                                        one isolated process per dataset -> per-dataset Peak RAM
  4. (this script)                      verify outputs, write detail CSVs, print the concise summary

Detail goes to files (results/*.csv, results/experiment_600/*.log, verification.txt);
the terminal only gets the final high-level metrics.

    python scripts/16_run_600_experiment.py
    python scripts/16_run_600_experiment.py --skip-augmentation   # reuse existing generated files
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import soundfile as sf

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
EXP = RESULTS / "experiment_600_corrected"
BENCH = EXP / "benchmark"
PY = sys.executable

SELECTED_CSV = EXP / "selected_600_manifest.csv"
METADATA_CSV = EXP / "augmentation_metadata.csv"
MANIFEST_JSON = EXP / "manifest_600.json"
QUERIES_JSON = EXP / "benchmark_queries.json"
RESULTS_CSV = EXP / "benchmark_results.csv"
SUMMARY_CSV = EXP / "benchmark_summary.csv"
VERIFY_TXT = EXP / "verification.txt"

DOMAINS = [("speech", "Speech"), ("music", "Music"), ("environment", "Environmental")]
LABEL_TO_CATEGORY = {label: cat for cat, label in DOMAINS}
EXPECTED_PER_DOMAIN = 200


def run_step(cmd, log_path):
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    with open(log_path, "w", encoding="utf-8") as lf:
        rc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=REPO, env=env).returncode
    if rc != 0:
        raise SystemExit(f"Step failed (exit {rc}); see {log_path}")


def build_queries_manifest():
    """Benchmark query records (existing 06/11 record format) for every GENERATED
    file (status Applied/Adjusted). Skipped/Failed augmentations produced no file,
    so they are not queries and are not counted for or against accuracy."""
    with open(MANIFEST_JSON) as f:
        manifest = json.load(f)
    id_by_key = {(cat, e["filename"]): e["file_id"] for cat, ents in manifest["categories"].items() for e in ents}

    records, n_invalid = [], 0
    with open(METADATA_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["status"] not in ("Applied", "Adjusted") or not row["generated_path"]:
                continue
            gp = Path(row["generated_path"])
            try:
                ok = gp.exists() and sf.info(str(gp)).frames > 0
            except Exception:
                ok = False
            if not ok:
                n_invalid += 1
                continue
            cat = LABEL_TO_CATEGORY[row["dataset"]]
            records.append({
                "category": cat, "modification": row["augmentation"], "level": row["requested_range"],
                "unit": "randomized-per-clip", "true_file_id": id_by_key[(cat, row["original_filename"])],
                "orig_filename": row["original_filename"], "modified_path": str(gp),
                "param_used": row["actual_param_used"],
            })
    with open(QUERIES_JSON, "w") as f:
        json.dump({"records": records}, f)
    return records, n_invalid


def verify(t_start, records, n_invalid):
    lines, ok_all = [], True

    def check(name, ok, detail=""):
        nonlocal ok_all
        ok_all &= bool(ok)
        lines.append(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))

    with open(MANIFEST_JSON) as f:
        manifest = json.load(f)
    counts = {cat: len(manifest["categories"][cat]) for cat, _ in DOMAINS}
    total = sum(counts.values())
    check("Selected clips per dataset", all(c == EXPECTED_PER_DOMAIN for c in counts.values()),
          ", ".join(f"{lab}={counts[cat]}" for cat, lab in DOMAINS))
    check("Total selected originals == 600", total == 600, str(total))

    originals = [Path(e["path"]) for ents in manifest["categories"].values() for e in ents]
    check("All originals exist", all(p.exists() for p in originals))
    check("Originals not modified during run (mtime before run start)",
          all(p.stat().st_mtime < t_start for p in originals))
    dataset_root = (REPO / "dataset").resolve()
    check("No generated file written inside dataset/",
          not any(dataset_root in Path(r["modified_path"]).resolve().parents for r in records))

    n_bad = 0
    for r in records:
        try:
            if sf.info(r["modified_path"]).frames <= 0:
                n_bad += 1
        except Exception:
            n_bad += 1
    check("Generated files all readable and non-empty", n_bad == 0 and n_invalid == 0,
          f"{len(records)} files checked, {n_bad + n_invalid} bad")

    with open(METADATA_CSV, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    status = {}
    for r in rows:
        status[r["status"]] = status.get(r["status"], 0) + 1
    check("Metadata has one row per (clip x augmentation)", len(rows) == total * 8, f"{len(rows)} rows")
    eight = {"Pitch Shift", "Speed Change", "Compression", "Noise", "Gain", "High-pass", "Trim", "Time Shift"}
    check("Only the eight requested augmentation types appear", {r["augmentation"] for r in rows} == eight)
    lines.append("Status counts: " + ", ".join(f"{k}={v}" for k, v in sorted(status.items())))
    n_adj_skip = [r for r in rows if r["status"] in ("Adjusted", "Skipped") and not r["reason"].strip()]
    check("Every Adjusted/Skipped row has a recorded reason", not n_adj_skip)
    VERIFY_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return ok_all


def pct(x):
    return "Not recorded" if x is None else f"{100 * x:.1f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-augmentation", action="store_true")
    ap.add_argument("--resume", action="store_true",
                    help="skip any dataset whose benchmark result json already exists")
    args = ap.parse_args()

    t_start = time.time()
    EXP.mkdir(parents=True, exist_ok=True)
    BENCH.mkdir(parents=True, exist_ok=True)

    if not args.skip_augmentation:
        run_step([PY, "scripts/10_augment_experiment.py", "--n-per-dataset", str(EXPECTED_PER_DOMAIN),
                  "--seed", "42", "--no-primary-benchmark-section"], EXP / "augmentation_run.log")

    records, n_invalid = build_queries_manifest()

    for cat, _label in DOMAINS:
        if args.resume and (BENCH / f"benchmark_{cat}.json").exists():
            continue
        run_step([PY, "scripts/11_run_per_domain_peak_ram.py", "--category", cat,
                  "--manifest", str(MANIFEST_JSON), "--queries", str(QUERIES_JSON),
                  "--out", str(BENCH / f"benchmark_{cat}.json"),
                  "--details-csv", str(BENCH / f"details_{cat}.csv")], BENCH / f"benchmark_{cat}.log")

    # detailed per-query results + per-dataset summary (all values read from the benchmark outputs)
    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as out:
        w = None
        for cat, label in DOMAINS:
            with open(BENCH / f"details_{cat}.csv", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    row["dataset"] = label
                    if w is None:
                        w = csv.DictWriter(out, fieldnames=list(row.keys()))
                        w.writeheader()
                    w.writerow(row)

    summary = []
    for cat, label in DOMAINS:
        res = json.load(open(BENCH / f"benchmark_{cat}.json"))
        dup, ed = res["duplicate"], res["edited_overall"]
        pooled_n = dup["n"] + ed["n"]
        summary.append({
            "dataset": label, "n_tracks": dup["n"],
            "n_duplicate_queries": dup["n"], "duplicate_accuracy": dup["accuracy"],
            "n_edited_queries": ed["n"], "edited_accuracy": ed["accuracy"],
            "avg_query_time_ms_edited": ed["mean_ms"], "avg_query_time_ms_duplicate": dup["mean_ms"],
            "avg_query_time_ms_pooled": (dup["mean_ms"] * dup["n"] + ed["mean_ms"] * ed["n"]) / pooled_n,
            "peak_ram_mb": res["peak_rss_mb"], "baseline_ram_mb": res["rss_at_start_mb"],
        })
    with open(SUMMARY_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    verified = verify(t_start, records, n_invalid)

    # ---- concise terminal output ----
    total_tracks = sum(s["n_tracks"] for s in summary)
    total_edited = sum(s["n_edited_queries"] for s in summary)
    print(f"The benchmark across the entire dataset has completed successfully. Here are the high-level "
          f"metrics evaluated over all ~{total_tracks:,} tracks and ~{total_edited:,} modified queries:\n")
    for s in summary:
        print(f"### {s['dataset']} Dataset ({s['n_tracks']} tracks)\n")
        print(f"- **Duplicate Accuracy:** {pct(s['duplicate_accuracy'])}")
        print(f"- **Edited Accuracy:** {pct(s['edited_accuracy'])}")
        print(f"- **Avg Query Time:** {s['avg_query_time_ms_edited']:.2f} ms")
        print(f"- **Peak RAM Usage:** {s['peak_ram_mb']:.1f} MB\n")
    if not verified:
        print(f"WARNING: one or more verification checks failed - see {VERIFY_TXT}")


if __name__ == "__main__":
    main()
