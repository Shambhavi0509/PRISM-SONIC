"""Offline test-data generation: applies every (modification, severity level) to
a fixed subset of files per domain and writes the results to disk as WAV files,
plus a manifest describing what was generated.

This script is deliberately a SEPARATE process from the retrieval benchmark
(scripts/03_run_benchmark.py). It freely uses librosa (pitch_shift/time_stretch)
and ffmpeg (mp3 round-trip), which are test-data-authoring tools only. Keeping
them out of the benchmark process is what lets the benchmark's peak-RAM
measurement reflect the actual deployed retrieval system rather than being
contaminated by librosa's ~100MB+ numba JIT tax (see sonic/io.py docstring).

Usage:
  python scripts/02_generate_modified_queries.py                       # all 3 domains, default N
  python scripts/02_generate_modified_queries.py --n 20                # override N for all domains
  python scripts/02_generate_modified_queries.py --categories environment --n 20   # only environment

The manifest is written incrementally (after each category), and re-running
for a subset of categories MERGES with whatever's already in the manifest for
the other categories, so a partial/interrupted run doesn't lose completed work.
Per-category N is recorded per record's source, so a mixed-N manifest (e.g.
speech/music generated at N=50, environment at N=20) is self-documenting rather
than silently inconsistent - this exact mixed-N scenario is disclosed as-is in
REPORT.md, not hidden.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import soundfile as sf

from sonic.augment import modifications as mods
from sonic.config import ANALYSIS_SR, RESULTS_ROOT
from sonic.io import load_audio

DEFAULT_N_QUERIES_PER_CONDITION = 50
OUT_DIR = RESULTS_ROOT / "modified_audio"
OUT_MANIFEST = RESULTS_ROOT / "modified_queries_manifest.json"


def generate_for_category(category: str, n_per_condition: int, manifest: dict, noise_ys: list) -> list:
    entries = manifest["categories"][category][:n_per_condition]
    records = []

    for mod_name, spec in mods.MODIFICATIONS.items():
        fn = spec["fn"]
        levels = spec["levels"]
        mod_dir = OUT_DIR / category / mod_name
        mod_dir.mkdir(parents=True, exist_ok=True)

        for level in levels:
            level_key = str(level)
            n_ok, n_fail = 0, 0
            for e in entries:
                try:
                    y = load_audio(e["path"])
                    if spec.get("needs_noise"):
                        noise_y = noise_ys[e["file_id"] % len(noise_ys)]
                        y_mod = fn(y, ANALYSIS_SR, level, noise_y)
                    else:
                        y_mod = fn(y, ANALYSIS_SR, level)

                    out_name = f"{e['file_id']}.wav"
                    safe_level = (level_key.replace(" ", "_").replace("/", "-")
                                  .replace("(", "").replace(")", "")
                                  .replace(",", "_").replace("'", ""))
                    out_path = mod_dir / safe_level
                    out_path.mkdir(parents=True, exist_ok=True)
                    out_file = out_path / out_name
                    sf.write(str(out_file), y_mod, ANALYSIS_SR)

                    records.append({
                        "category": category, "modification": mod_name, "level": level_key,
                        "unit": spec["unit"], "true_file_id": e["file_id"],
                        "orig_filename": e["filename"], "modified_path": str(out_file),
                        "n_per_condition": n_per_condition,
                    })
                    n_ok += 1
                except Exception as ex:
                    n_fail += 1
                    records.append({
                        "category": category, "modification": mod_name, "level": level_key,
                        "unit": spec["unit"], "true_file_id": e["file_id"],
                        "orig_filename": e["filename"], "modified_path": None,
                        "n_per_condition": n_per_condition, "error": str(ex),
                    })
            print(f"{category}/{mod_name}/{level_key}: ok={n_ok} fail={n_fail}", flush=True)

    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--categories", nargs="+", default=["speech", "music", "environment"])
    parser.add_argument("--n", type=int, default=DEFAULT_N_QUERIES_PER_CONDITION)
    args = parser.parse_args()

    with open(RESULTS_ROOT / "manifest.json") as f:
        manifest = json.load(f)

    noise_pool = manifest["held_out_noise"]["environment"]
    noise_ys = [load_audio(e["path"]) for e in noise_pool[:20]]

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_records = []
    if OUT_MANIFEST.exists():
        with open(OUT_MANIFEST) as f:
            existing = json.load(f)
        all_records = [r for r in existing.get("records", []) if r["category"] not in args.categories]
        print(f"Loaded {len(all_records)} existing records for categories not being regenerated.", flush=True)

    t_start = time.perf_counter()
    for category in args.categories:
        cat_records = generate_for_category(category, args.n, manifest, noise_ys)
        all_records.extend(cat_records)
        with open(OUT_MANIFEST, "w") as f:
            json.dump({"records": all_records}, f, indent=2)
        print(f"[{category}] done, manifest checkpoint saved ({len(all_records)} total records).", flush=True)

    print(f"Total: {len(all_records)} modified-query records in {time.perf_counter()-t_start:.1f}s", flush=True)
    print(f"Manifest written to {OUT_MANIFEST}", flush=True)


if __name__ == "__main__":
    main()
