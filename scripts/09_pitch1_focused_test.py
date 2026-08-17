"""Focused robustness test: +-1 semitone pitch shift ONLY, 100 files/domain,
against the current unmodified SONIC pipeline (no algorithm/parameter changes
made for this test). Generates its own +-1 semitone queries (separate from any
other benchmark data), builds a fresh 100-file/domain index using the existing
build_domain_index() unmodified, and reports duplicate + pitch-shifted accuracy,
latency, and peak RAM directly to the terminal.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import psutil
import soundfile as sf

from sonic.augment.modifications import pitch_shift
from sonic.config import ANALYSIS_SR, RESULTS_ROOT
from sonic.io import load_audio
from sonic.retrieval.pipeline import build_domain_index, query, query_array

PROC = psutil.Process()
N_FILES = 100
LEVELS = [-1, 1]  # semitones - ONLY this severity, both directions

OUT_DIR = RESULTS_ROOT / "pitch1_focused_audio"


def rss_mb():
    return PROC.memory_info().rss / 1024 / 1024


def main():
    peak_rss = rss_mb()
    print(f"RSS at start: {peak_rss:.1f} MB", flush=True)

    with open(RESULTS_ROOT / "manifest.json") as f:
        full_manifest = json.load(f)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {}

    for category in ["speech", "music", "environment"]:
        entries = full_manifest["categories"][category][:N_FILES]  # first 100, deterministic
        assert len(entries) == N_FILES, f"{category}: only {len(entries)} files available"

        # --- generate +-1 semitone modified queries for these 100 files ---
        mod_dir = OUT_DIR / category
        mod_dir.mkdir(parents=True, exist_ok=True)
        pitch_records = []
        t0 = time.perf_counter()
        for e in entries:
            y = load_audio(e["path"])
            for level in LEVELS:
                y_mod = pitch_shift(y, ANALYSIS_SR, level)
                out_file = mod_dir / f"{e['file_id']}_{level}.wav"
                sf.write(str(out_file), y_mod, ANALYSIS_SR)
                pitch_records.append({"true_file_id": e["file_id"], "level": level, "path": str(out_file)})
        gen_time = time.perf_counter() - t0
        print(f"[{category}] generated {len(pitch_records)} pitch-shifted queries "
              f"({N_FILES} files x {len(LEVELS)} directions) in {gen_time:.1f}s", flush=True)
        peak_rss = max(peak_rss, rss_mb())

        # --- build index (unmodified pipeline) ---
        t0 = time.perf_counter()
        idx = build_domain_index(category, entries)
        build_time = time.perf_counter() - t0
        peak_rss = max(peak_rss, rss_mb())
        print(f"[{category}] index built in {build_time:.1f}s over {N_FILES} files, RSS={rss_mb():.1f}MB", flush=True)

        # --- duplicate queries ---
        dup_correct = 0
        dup_times = []
        for e in entries:
            t0 = time.perf_counter()
            res = query(e["path"], idx)
            dup_times.append((time.perf_counter() - t0) * 1000)
            dup_correct += int(res.file_id == e["file_id"])
        peak_rss = max(peak_rss, rss_mb())

        # --- pitch-shifted queries (+-1 semitone) ---
        pitch_correct = 0
        pitch_times = []
        for r in pitch_records:
            y, sr = sf.read(r["path"], dtype="float32", always_2d=False)
            t0 = time.perf_counter()
            res = query_array(y, idx, sr=sr)
            pitch_times.append((time.perf_counter() - t0) * 1000)
            pitch_correct += int(res.file_id == r["true_file_id"])
        peak_rss = max(peak_rss, rss_mb())

        results[category] = {
            "dup_correct": dup_correct, "dup_n": N_FILES,
            "dup_mean_ms": sum(dup_times) / len(dup_times),
            "pitch_correct": pitch_correct, "pitch_n": len(pitch_records),
            "pitch_mean_ms": sum(pitch_times) / len(pitch_times),
        }
        print(f"[{category}] duplicate: {dup_correct}/{N_FILES}   "
              f"pitch+-1: {pitch_correct}/{len(pitch_records)}", flush=True)

    # ---------------- final table ----------------
    print("", flush=True)
    print("=" * 100, flush=True)
    print("PITCH SHIFT +-1 SEMITONE - FOCUSED ROBUSTNESS TEST (100 files/domain, current unmodified pipeline)".center(100), flush=True)
    print("=" * 100, flush=True)
    header = f"{'Dataset':<14}{'Pitch Shift':<14}{'Accuracy':<12}{'Correct':<10}{'Failed':<9}{'Avg Query Time':<17}{'Peak RAM':<10}"
    print(header, flush=True)
    print("-" * 100, flush=True)
    for cat, label in [("speech", "Speech"), ("music", "Music"), ("environment", "Environmental")]:
        d = results[cat]
        acc = d["pitch_correct"] / d["pitch_n"]
        failed = d["pitch_n"] - d["pitch_correct"]
        row = (f"{label:<14}{'+/-1 semitone':<14}{acc*100:>6.2f}%     "
               f"{d['pitch_correct']:<10}{failed:<9}{d['pitch_mean_ms']:>7.1f} ms      {peak_rss:>6.1f} MB")
        print(row, flush=True)
    print("-" * 100, flush=True)
    print("", flush=True)
    print("Duplicate accuracy (unmodified, same 100-file index, for reference):", flush=True)
    for cat, label in [("speech", "Speech"), ("music", "Music"), ("environment", "Environmental")]:
        d = results[cat]
        print(f"  {label:<14} {d['dup_correct']}/{d['dup_n']} = {d['dup_correct']/d['dup_n']*100:.2f}%   "
              f"avg_lat={d['dup_mean_ms']:.1f} ms", flush=True)
    print("", flush=True)
    print(f"Peak RAM (single process, all 3 domain indexes + all queries): {peak_rss:.1f} MB", flush=True)
    print("=" * 100, flush=True)


if __name__ == "__main__":
    main()
