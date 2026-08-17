"""Re-measures latency and peak RAM for the +-1 semitone focused test in a
CLEAN process (imports only sonic.io/fingerprint/index/retrieval - no
sonic.augment, no librosa), reusing the already-generated pitch-shifted WAV
files from scripts/09_pitch1_focused_test.py. This is what makes the RAM
number representative of the actual deployed serving system, consistent with
how every other benchmark in this project was measured (see
sonic/eval/benchmark.py's docstring)."""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import psutil
import soundfile as sf

from sonic.config import RESULTS_ROOT
from sonic.retrieval.pipeline import build_domain_index, query, query_array

PROC = psutil.Process()
N_FILES = 100
LEVELS = [-1, 1]
AUDIO_DIR = RESULTS_ROOT / "pitch1_focused_audio"


def rss_mb():
    return PROC.memory_info().rss / 1024 / 1024


def main():
    peak_rss = rss_mb()
    print(f"RSS at start (clean, no librosa): {peak_rss:.1f} MB", flush=True)

    with open(RESULTS_ROOT / "manifest.json") as f:
        full_manifest = json.load(f)

    results = {}
    for category in ["speech", "music", "environment"]:
        entries = full_manifest["categories"][category][:N_FILES]

        t0 = time.perf_counter()
        idx = build_domain_index(category, entries)
        build_time = time.perf_counter() - t0
        peak_rss = max(peak_rss, rss_mb())
        print(f"[{category}] index built in {build_time:.1f}s, RSS={rss_mb():.1f}MB", flush=True)

        dup_correct = 0
        dup_times = []
        for e in entries:
            t0 = time.perf_counter()
            res = query(e["path"], idx)
            dup_times.append((time.perf_counter() - t0) * 1000)
            dup_correct += int(res.file_id == e["file_id"])
        peak_rss = max(peak_rss, rss_mb())

        pitch_correct = 0
        pitch_times = []
        n_pitch = 0
        for e in entries:
            for level in LEVELS:
                path = AUDIO_DIR / category / f"{e['file_id']}_{level}.wav"
                y, sr = sf.read(str(path), dtype="float32", always_2d=False)
                t0 = time.perf_counter()
                res = query_array(y, idx, sr=sr)
                pitch_times.append((time.perf_counter() - t0) * 1000)
                pitch_correct += int(res.file_id == e["file_id"])
                n_pitch += 1
        peak_rss = max(peak_rss, rss_mb())

        results[category] = {
            "dup_correct": dup_correct, "dup_n": N_FILES, "dup_mean_ms": sum(dup_times) / len(dup_times),
            "pitch_correct": pitch_correct, "pitch_n": n_pitch, "pitch_mean_ms": sum(pitch_times) / len(pitch_times),
        }
        print(f"[{category}] duplicate: {dup_correct}/{N_FILES}   pitch+-1: {pitch_correct}/{n_pitch}", flush=True)

    print("", flush=True)
    print("=" * 100, flush=True)
    print("PITCH SHIFT +-1 SEMITONE - CLEAN RAM RE-MEASUREMENT (serving-path only, no librosa)".center(100), flush=True)
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
    print(f"\nPeak RAM (clean serving-path process, all 3 domain indexes + all queries): {peak_rss:.1f} MB", flush=True)
    print("=" * 100, flush=True)


if __name__ == "__main__":
    main()
