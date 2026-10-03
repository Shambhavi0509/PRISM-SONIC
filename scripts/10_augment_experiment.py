"""
Augmentation experiment (Phase 1): apply 8 randomized augmentations to N clips
per dataset (default 200/dataset, 600 total) with safe short-clip handling.

Rules enforced (per spec):
  - Never force a modification that would produce empty/corrupted/unusably
    short audio.
  - If the requested parameter range is not valid for a clip, fall back to
    the maximum valid value within (or, when the whole range is infeasible,
    below) the allowed range; if no valid modification exists at all, skip
    that augmentation for that clip.
  - Every skip/adjustment is logged with a reason in the metadata CSV.
  - Fixed random seed -> reproducible clip selection and reproducible
    per-clip parameter draws.
  - Modular: re-run with --n-per-dataset <bigger number> (up to the full
    dataset size) to scale the experiment up later.

Usage:
    python scripts/10_augment_experiment.py
    python scripts/10_augment_experiment.py --n-per-dataset 200 --seed 42
"""

import argparse
import csv
import random
import sys
import time
import traceback
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from sonic.augment.modifications import (  # noqa: E402
    apply_filter,
    mp3_compress,
    pitch_shift,
)

DATASET_ROOT = REPO_ROOT / "dataset"
OUT_ROOT = REPO_ROOT / "results" / "experiment_600_corrected"
SELECTED_MANIFEST_CSV = OUT_ROOT / "selected_600_manifest.csv"
AUGMENTATION_METADATA_CSV = OUT_ROOT / "augmentation_metadata.csv"
AUGMENTATION_COUNTS_CSV = OUT_ROOT / "augmentation_counts.csv"
SELECTED_MANIFEST_JSON = OUT_ROOT / "manifest_600.json"

# Dataset1/2/3 <-> underlying category folder. Order fixed for reproducibility.
DATASETS = [
    ("Dataset1", "speech"),
    ("Dataset2", "music"),
    ("Dataset3", "environment"),
]

SEED = 42

# ----------------------------------------------------------------- ranges
PITCH_RANGE_ST = (-1.0, 4.0)          # semitones
SPEED_RANGE_FRAC = (0.05, 0.20)       # fractional change magnitude (either direction)
COMPRESSION_RANGE_KBPS = (64, 128)
NOISE_SNR_RANGE_DB = (10.0, 20.0)
GAIN_RANGE_DB = (3.0, 6.0)
HPF_RANGE_HZ = (150.0, 200.0)
TRIM_RANGE_SEC = (3.0, 10.0)
TIME_SHIFT_RANGE_SEC = (5.0, 20.0)

MIN_USABLE_SEC = 0.5   # never emit audio shorter than this
MIN_PROCESS_SEC = 0.05  # below this, skip every augmentation for the clip
# Trim and Time Shift remove/relocate content, so the shortened clip (trim) or each of the two
# segments (time shift) must stay usable by the retrieval system: at least one full secondary-
# fingerprint window (= sonic.fingerprint.binary_embed.SUBFP_WINDOW_SEC, 1.5 s). Derived from the
# system's own window length, not tuned to accuracy.
MIN_REMAINING_SEC = 1.5

AUGMENTATIONS = [
    "pitch_shift",
    "speed_change",
    "compression",
    "noise",
    "gain",
    "highpass",
    "trim",
    "time_shift",
]

SUFFIX = {
    "pitch_shift": "pitch",
    "speed_change": "speed",
    "compression": "compression",
    "noise": "noise",
    "gain": "gain",
    "highpass": "highpass",
    "trim": "trim",
    "time_shift": "timeshift",
}

REQUESTED_RANGE_STR = {
    "pitch_shift": "-1 to +4 semitones",
    "speed_change": "5% to 20% change",
    "compression": "64-128 kbps",
    "noise": "10-20 dB SNR",
    "gain": "+3 to +6 dB",
    "highpass": "150-200 Hz",
    "trim": "3-10 sec",
    "time_shift": "5-20 sec",
}


class Record(dict):
    pass


def select_clips(seed, n_per_dataset):
    """Randomly selects n readable files per domain directly from
    dataset/<domain>/ (no dependency on any shared manifest), using a
    per-domain RNG derived from `seed` so the selection is reproducible.
    A domain with fewer than n readable files (Music has exactly 200) simply
    contributes all of them - files are never duplicated to reach n."""
    selected = {}
    for _dname, category in DATASETS:
        d = DATASET_ROOT / category
        names = sorted(p.name for p in d.iterdir() if p.is_file())
        random.Random(f"{seed}|{category}").shuffle(names)
        entries = []
        for name in names:
            if len(entries) >= n_per_dataset:
                break
            path = d / name
            try:
                info = sf.info(str(path))
                duration = info.frames / info.samplerate if info.samplerate else 0.0
            except Exception:
                continue  # unreadable file: skip, take the next one in the shuffled order
            if duration <= 0:
                continue
            entries.append({
                "category": category, "filename": name, "path": str(path),
                "orig_samplerate": info.samplerate, "orig_channels": info.channels,
                "duration_sec": round(duration, 4),
            })
        entries.sort(key=lambda e: e["filename"])
        selected[category] = entries

    file_id = 0
    for _dname, category in DATASETS:
        for e in selected[category]:
            e["file_id"] = file_id
            file_id += 1
    return selected


def write_selected_manifest(selected, csv_path, json_path):
    """results/selected_600_manifest.csv (human-readable record of exactly which
    originals were used) + a SONIC-format manifest json consumed by the benchmark."""
    import json

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "original_filename", "original_path", "duration_sec",
                    "sample_rate", "channels", "file_id"])
        for _dname, category in DATASETS:
            for e in selected[category]:
                w.writerow([DATASET_DISPLAY_NAME[category], e["filename"], e["path"],
                            e["duration_sec"], e["orig_samplerate"], e["orig_channels"], e["file_id"]])
    with open(json_path, "w") as f:
        json.dump({"categories": selected}, f, indent=2)


def load_audio(path):
    y, sr = librosa.load(path, sr=None, mono=True)
    return y.astype(np.float32), sr


def safe_write(path, y, sr):
    if y is None or y.size == 0:
        raise ValueError("empty audio buffer")
    if not np.all(np.isfinite(y)):
        raise ValueError("non-finite samples")
    sf.write(str(path), y, sr, subtype="FLOAT")  # float: no clipping/quantization from the container


# ------------------------------------------------------------ augmentations
def do_pitch_shift(rng, y, sr, orig_dur):
    if orig_dur < MIN_PROCESS_SEC:
        return None, None, "skipped", "clip too short to process (<0.05s)"
    semitones = round(rng.uniform(*PITCH_RANGE_ST), 3)
    out = pitch_shift(y, sr, semitones)
    return out, f"{semitones:+.2f} semitones", "applied", ""


def do_speed_change(rng, y, sr, orig_dur):
    if orig_dur < MIN_PROCESS_SEC:
        return None, None, "skipped", "clip too short to process (<0.05s)"
    magnitude = rng.uniform(*SPEED_RANGE_FRAC)
    direction = rng.choice([-1, 1])
    rate = 1.0 + direction * magnitude
    try:
        out = librosa.effects.time_stretch(y=y, rate=rate)
    except Exception as e:  # pragma: no cover - defensive
        return None, None, "skipped", f"time_stretch failed: {e}"
    if out.size < MIN_USABLE_SEC * sr:
        return None, None, "skipped", "resulting speed-changed clip would be unusably short"
    pct = direction * magnitude * 100
    return out, f"{pct:+.1f}% (rate={rate:.3f})", "applied", ""


def do_compression(rng, y, sr, orig_dur):
    if orig_dur < MIN_PROCESS_SEC:
        return None, None, "skipped", "clip too short to process (<0.05s)"
    kbps = rng.choice(range(int(COMPRESSION_RANGE_KBPS[0]), int(COMPRESSION_RANGE_KBPS[1]) + 1, 8))
    try:
        out = mp3_compress(y, sr, kbps)
    except Exception as e:
        return None, None, "skipped", f"mp3 encode/decode failed: {e}"
    if out.size < MIN_USABLE_SEC * sr:
        return None, None, "skipped", "compressed output unusably short"
    return out, f"{kbps} kbps", "applied", ""


def do_noise(rng, y, sr, orig_dur):
    if orig_dur < MIN_PROCESS_SEC:
        return None, None, "skipped", "clip too short to process (<0.05s)"
    snr = round(rng.uniform(*NOISE_SNR_RANGE_DB), 2)
    noise = rng_np_standard_normal(rng, y.size).astype(np.float32)
    sig_power = float(np.mean(y ** 2)) + 1e-12
    noise_power = float(np.mean(noise ** 2)) + 1e-12
    target_noise_power = sig_power / (10 ** (snr / 10))
    scale = np.sqrt(target_noise_power / noise_power)
    mixed = y + noise * scale
    peak = np.max(np.abs(mixed))
    if peak > 1.0:
        mixed = mixed / peak
    return mixed.astype(np.float32), f"{snr:.1f} dB SNR", "applied", ""


def rng_np_standard_normal(rng, n):
    seed = rng.randint(0, 2**31 - 1)
    local = np.random.default_rng(seed)
    return local.standard_normal(n)


def do_gain(rng, y, sr, orig_dur):
    if orig_dur < MIN_PROCESS_SEC:
        return None, None, "skipped", "clip too short to process (<0.05s)"
    gain_db = round(rng.uniform(*GAIN_RANGE_DB), 2)
    out = (y * 10 ** (gain_db / 20)).astype(np.float32)  # pure gain; gain_change() would hard-clip at +-1
    return out, f"{gain_db:+.2f} dB", "applied", ""


def do_highpass(rng, y, sr, orig_dur):
    min_samples_for_filtfilt = 32  # sosfiltfilt needs > padlen; order-4 filter needs a modest margin
    if y.size < min_samples_for_filtfilt:
        return None, None, "skipped", "clip too short for stable filtering (<32 samples)"
    cutoff = round(rng.uniform(*HPF_RANGE_HZ), 1)
    try:
        out = apply_filter(y, sr, ("highpass", cutoff))
    except Exception as e:
        return None, None, "skipped", f"filter failed: {e}"
    return out, f"highpass {cutoff:.0f} Hz", "applied", ""


def do_trim(rng, y, sr, orig_dur):
    target = round(rng.uniform(*TRIM_RANGE_SEC), 2)
    max_valid = orig_dur - MIN_REMAINING_SEC  # most we can remove and still leave a usable clip
    if max_valid < TRIM_RANGE_SEC[0]:
        return None, None, "skipped", (
            f"clip too short for any trim inside {TRIM_RANGE_SEC[0]:g}-{TRIM_RANGE_SEC[1]:g}s "
            f"(duration={orig_dur:.2f}s; needs >= {TRIM_RANGE_SEC[0] + MIN_REMAINING_SEC:g}s so >= {MIN_REMAINING_SEC:g}s remains)")
    if target <= max_valid:
        trim_sec, status, reason = target, "applied", ""
    else:
        trim_sec = min(TRIM_RANGE_SEC[1], max_valid)  # largest valid value inside the allowed range
        status = "adjusted"
        reason = (f"requested trim {target:.2f}s leaves < {MIN_REMAINING_SEC:g}s of a {orig_dur:.2f}s clip; "
                  f"used the largest valid value in range, {trim_sec:.2f}s")
    n = int(round(trim_sec * sr))
    out = y[:-n] if 0 < n < y.size else None
    if out is None or out.size < MIN_REMAINING_SEC * sr * 0.99:  # safety re-check
        return None, None, "skipped", "trim would leave unusably short audio"
    return out, f"{trim_sec:.2f} sec removed from end", status, reason


def do_time_shift(rng, y, sr, orig_dur):
    target = round(rng.uniform(*TIME_SHIFT_RANGE_SEC), 2)
    max_valid = orig_dur - MIN_REMAINING_SEC  # circular shift: both segments must stay >= MIN_REMAINING_SEC
    if max_valid < TIME_SHIFT_RANGE_SEC[0]:
        return None, None, "skipped", (
            f"clip too short for any shift inside {TIME_SHIFT_RANGE_SEC[0]:g}-{TIME_SHIFT_RANGE_SEC[1]:g}s "
            f"(duration={orig_dur:.2f}s; needs >= {TIME_SHIFT_RANGE_SEC[0] + MIN_REMAINING_SEC:g}s)")
    if target <= max_valid:
        shift_sec, status, reason = target, "applied", ""
    else:
        shift_sec = min(TIME_SHIFT_RANGE_SEC[1], max_valid)  # largest valid value inside the allowed range
        status = "adjusted"
        reason = (f"requested shift {target:.2f}s too large for a {orig_dur:.2f}s clip; "
                  f"used the largest valid value in range, {shift_sec:.2f}s")
    n = int(round(shift_sec * sr)) % y.size
    if n == 0:
        return None, None, "skipped", "computed shift amount rounds to 0 samples"
    return np.roll(y, n), f"{shift_sec:.2f} sec (circular shift)", status, reason


AUG_FUNCS = {
    "pitch_shift": do_pitch_shift,
    "speed_change": do_speed_change,
    "compression": do_compression,
    "noise": do_noise,
    "gain": do_gain,
    "highpass": do_highpass,
    "trim": do_trim,
    "time_shift": do_time_shift,
}


def process_dataset(dataset_name, category, entries, out_dir, seed):
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for idx, entry in enumerate(entries):
        filename = entry["filename"]
        src_path = DATASET_ROOT / category / filename
        stem = Path(filename).stem

        clip_rng = random.Random(f"{seed}|{dataset_name}|{filename}")

        try:
            y, sr = load_audio(src_path)
        except Exception as e:
            for aug in AUGMENTATIONS:
                records.append(dict(
                    dataset=dataset_name, category=category, original_filename=filename,
                    original_duration_sec="", augmentation=aug,
                    requested_range=REQUESTED_RANGE_STR[aug], param_used="",
                    final_duration_sec="", status="failed",
                    reason=f"could not load source audio: {e}", output_filename="",
                ))
            continue

        orig_dur = y.size / sr

        for aug in AUGMENTATIONS:
            fn = AUG_FUNCS[aug]
            out_filename = f"{dataset_name}_{stem}_{SUFFIX[aug]}.wav"
            out_path = out_dir / out_filename
            try:
                out_y, param_used, status, reason = fn(clip_rng, y, sr, orig_dur)
            except Exception as e:
                out_y, param_used, status, reason = None, None, "failed", f"unexpected error: {e}\n{traceback.format_exc(limit=1)}"

            final_dur = ""
            output_filename_recorded = ""
            if status in ("applied", "adjusted") and out_y is not None:
                try:
                    safe_write(out_path, out_y, sr)
                    final_dur = round(out_y.size / sr, 4)
                    output_filename_recorded = out_filename
                except Exception as e:
                    status = "failed"
                    reason = f"write failed: {e}"

            records.append(dict(
                dataset=dataset_name, category=category, original_filename=filename,
                original_duration_sec=round(orig_dur, 4), augmentation=aug,
                requested_range=REQUESTED_RANGE_STR[aug], param_used=param_used or "",
                final_duration_sec=final_dur, status=status, reason=reason,
                output_filename=output_filename_recorded,
            ))

        if (idx + 1) % 25 == 0:
            print(f"  [{dataset_name}/{category}] {idx + 1}/{len(entries)} clips done", flush=True)

    return records


def write_metadata_csv(all_records, path):
    """One row per (clip, augmentation) attempt, including skipped/failed ones."""
    fieldnames = [
        "dataset", "original_filename", "generated_filename", "generated_path",
        "original_duration_sec", "augmentation", "requested_range",
        "actual_param_used", "final_duration_sec", "status", "reason",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in all_records:
            generated = r["output_filename"]
            gen_path = ""
            if generated:
                gen_path = str(OUT_ROOT / f"{r['dataset']}_{r['category']}" / generated)
            w.writerow({
                "dataset": DATASET_DISPLAY_NAME[r["category"]],
                "original_filename": r["original_filename"],
                "generated_filename": generated,
                "generated_path": gen_path,
                "original_duration_sec": r["original_duration_sec"],
                "augmentation": DISPLAY_NAME[r["augmentation"]],
                "requested_range": r["requested_range"],
                "actual_param_used": r["param_used"],
                "final_duration_sec": r["final_duration_sec"],
                "status": r["status"].capitalize(),
                "reason": r["reason"],
            })


import re

_PARAM_PATTERNS = {
    "pitch_shift": re.compile(r"([+-]?\d+\.?\d*)\s*semitones"),
    "speed_change": re.compile(r"([+-]?\d+\.?\d*)%"),
    "compression": re.compile(r"(\d+)\s*kbps"),
    "noise": re.compile(r"([+-]?\d+\.?\d*)\s*dB"),
    "gain": re.compile(r"([+-]?\d+\.?\d*)\s*dB"),
    "highpass": re.compile(r"highpass\s+(\d+\.?\d*)\s*Hz"),
    "trim": re.compile(r"([\d.]+)\s*sec removed"),
    "time_shift": re.compile(r"([\d.]+)\s*sec"),
}


def extract_numeric_param(aug, param_used_str):
    """Best-effort numeric extraction from a record's human-readable param_used
    string, for reporting the actual min/max parameter values drawn during the
    run (display-only; does not affect any augmentation logic)."""
    if not param_used_str:
        return None
    pattern = _PARAM_PATTERNS.get(aug)
    if not pattern:
        return None
    m = pattern.search(param_used_str)
    return float(m.group(1)) if m else None


def build_summary_tables(all_records, n_per_dataset):
    from collections import defaultdict

    per_clip_status = defaultdict(lambda: defaultdict(set))  # dataset -> status -> set(filenames)
    dataset_table = {}
    for dname, _cat in DATASETS:
        dataset_table[dname] = dict(original_clips=n_per_dataset, successfully_processed=0,
                                     adjusted=0, skipped=0, failed=0)

    aug_table = defaultdict(lambda: dict(applied=0, adjusted=0, skipped=0, failed=0))
    # (dataset, augmentation) -> list of extracted numeric param values actually used
    param_values = defaultdict(list)

    clip_had = defaultdict(lambda: defaultdict(str))  # (dataset, filename) -> aggregate

    for r in all_records:
        key = (r["dataset"], r["augmentation"])
        st = r["status"]
        if st in aug_table[key]:
            aug_table[key][st] += 1

        if st in ("applied", "adjusted"):
            v = extract_numeric_param(r["augmentation"], r["param_used"])
            if v is not None:
                param_values[key].append(v)

        clip_key = (r["dataset"], r["original_filename"])
        cur = clip_had[clip_key]
        cur[st] = cur.get(st, 0) + 1

    # per-clip aggregate: a clip counts as "successfully processed" if it had
    # >=1 applied/adjusted augmentation and no failures; "adjusted" if any
    # augmentation needed adjustment; "failed" if any augmentation errored
    # unexpectedly; clips are not double counted across these buckets.
    clip_status = {}
    for (dname, fname), counts in clip_had.items():
        if counts.get("failed", 0) > 0:
            clip_status[(dname, fname)] = "failed"
        elif counts.get("adjusted", 0) > 0:
            clip_status[(dname, fname)] = "adjusted"
        elif counts.get("applied", 0) > 0:
            clip_status[(dname, fname)] = "processed"
        else:
            clip_status[(dname, fname)] = "skipped"

    for (dname, _fname), status in clip_status.items():
        if status == "processed":
            dataset_table[dname]["successfully_processed"] += 1
        elif status == "adjusted":
            dataset_table[dname]["adjusted"] += 1
            dataset_table[dname]["successfully_processed"] += 1
        elif status == "failed":
            dataset_table[dname]["failed"] += 1
        else:
            dataset_table[dname]["skipped"] += 1

    return dataset_table, aug_table, param_values


def print_and_save_summary(dataset_table, aug_table, param_values, n_per_dataset, out_dir):
    lines = []

    lines.append("\n=== SUMMARY TABLE (per dataset) ===")
    header = f"{'Dataset':<12}{'Original':>10}{'Processed':>12}{'Adjusted':>10}{'Skipped':>10}{'Failed':>8}"
    lines.append(header)
    tot = dict(original_clips=0, successfully_processed=0, adjusted=0, skipped=0, failed=0)
    for dname, _cat in DATASETS:
        row = dataset_table[dname]
        lines.append(f"{dname:<12}{row['original_clips']:>10}{row['successfully_processed']:>12}"
                      f"{row['adjusted']:>10}{row['skipped']:>10}{row['failed']:>8}")
        for k in tot:
            tot[k] += row[k]
    lines.append(f"{'TOTAL':<12}{tot['original_clips']:>10}{tot['successfully_processed']:>12}"
                  f"{tot['adjusted']:>10}{tot['skipped']:>10}{tot['failed']:>8}")

    lines.append("\n=== AUGMENTATION-LEVEL TABLE ===")
    header2 = f"{'Dataset':<12}{'Augmentation':<14}{'Requested Range':<22}{'Applied':>9}{'Adjusted':>10}{'Skipped':>9}{'Failed':>8}"
    lines.append(header2)
    aug_totals = {aug: dict(applied=0, adjusted=0, skipped=0, failed=0) for aug in AUGMENTATIONS}
    for dname, _cat in DATASETS:
        for aug in AUGMENTATIONS:
            row = aug_table[(dname, aug)]
            lines.append(f"{dname:<12}{aug:<14}{REQUESTED_RANGE_STR[aug]:<22}"
                          f"{row['applied']:>9}{row['adjusted']:>10}{row['skipped']:>9}{row['failed']:>8}")
            for k in aug_totals[aug]:
                aug_totals[aug][k] += row[k]
    lines.append("--- TOTAL across datasets, per augmentation ---")
    for aug in AUGMENTATIONS:
        row = aug_totals[aug]
        lines.append(f"{'TOTAL':<12}{aug:<14}{REQUESTED_RANGE_STR[aug]:<22}"
                      f"{row['applied']:>9}{row['adjusted']:>10}{row['skipped']:>9}{row['failed']:>8}")

    lines.append("\n=== ACTUAL PARAMETER RANGE USED (per augmentation, per dataset) ===")
    header3 = f"{'Dataset':<12}{'Augmentation':<14}{'Configured Range':<22}{'Actual Min':>12}{'Actual Max':>12}"
    lines.append(header3)
    for dname, _cat in DATASETS:
        for aug in AUGMENTATIONS:
            vals = param_values.get((dname, aug), [])
            vmin = f"{min(vals):.2f}" if vals else "N/A"
            vmax = f"{max(vals):.2f}" if vals else "N/A"
            lines.append(f"{dname:<12}{aug:<14}{REQUESTED_RANGE_STR[aug]:<22}{vmin:>12}{vmax:>12}")
    lines.append("--- TOTAL (across all 3 datasets) ---")
    for aug in AUGMENTATIONS:
        vals = []
        for dname, _cat in DATASETS:
            vals.extend(param_values.get((dname, aug), []))
        vmin = f"{min(vals):.2f}" if vals else "N/A"
        vmax = f"{max(vals):.2f}" if vals else "N/A"
        lines.append(f"{'TOTAL':<12}{aug:<14}{REQUESTED_RANGE_STR[aug]:<22}{vmin:>12}{vmax:>12}")
    lines.append(
        "\nNote: Trim and Time Shift actual ranges include values below the configured\n"
        "floor (e.g. Trim < 3s, Time Shift < 5s) - these come from short-clip\n"
        "adjustments, not out-of-spec draws. Speed Change actual min/max are signed\n"
        "percentages (both directions were drawn); the configured range describes\n"
        "the magnitude (5-20%), applied in either direction."
    )

    text = "\n".join(lines)
    print(text)
    with open(out_dir / "summary_tables.txt", "w", encoding="utf-8") as f:
        f.write(text + "\n")


def print_benchmark_results_from_json():
    """Reads results/primary_benchmark_results.json (produced separately by
    scripts/06_run_primary_benchmark.py - never run from here) and prints the
    same BENCHMARK RESULTS table that script prints, so both sections can
    appear together at the end of this script's output. Purely a display
    step: no benchmark logic is executed or duplicated here beyond
    re-deriving correct/total counts from the stored accuracy + n."""
    import json

    results_path = OUT_ROOT.parent / "primary_benchmark_results.json"
    width = 75

    print("=" * width)
    print("BENCHMARK RESULTS".center(width))
    print("=" * width)
    print()

    if not results_path.exists():
        print("Primary benchmark results not found")
        print()
        return

    try:
        with open(results_path) as f:
            results = json.load(f)
    except Exception:
        print("Primary benchmark results not found")
        print()
        return

    display_names = [("speech", "Speech"), ("music", "Music"), ("environment", "Environmental")]
    rows = []
    for category, label in display_names:
        if category not in results:
            continue
        dup = results[category]["duplicate"]
        edited = results[category]["edited_overall"]
        dup_correct = round(dup["accuracy"] * dup["n"])
        edited_correct = round(edited["accuracy"] * edited["n"])
        dup_str = f"{dup['accuracy'] * 100:.2f}% ({dup_correct}/{dup['n']})"
        edited_str = f"{edited['accuracy'] * 100:.2f}% ({edited_correct}/{edited['n']})"
        query_time_str = f"{edited['mean_ms']:.2f} ms"
        rows.append((label, dup_str, edited_str, query_time_str))

    if not rows:
        print("Primary benchmark results not found")
        print()
        return

    col1_w = max(16, max(len(r[0]) for r in rows) + 2)
    col2_w = max(24, max(len(r[1]) for r in rows) + 2)
    col3_w = max(21, max(len(r[2]) for r in rows) + 2)

    print(f"{'Dataset':<{col1_w}}{'Duplicate Accuracy':<{col2_w}}{'Edited Accuracy':<{col3_w}}{'Avg Query Time'}")
    print("-" * width)
    for label, dup_str, edited_str, query_time_str in rows:
        print(f"{label:<{col1_w}}{dup_str:<{col2_w}}{edited_str:<{col3_w}}{query_time_str}")
    print("-" * width)
    print()

    peak_rss = results.get("peak_rss_mb")
    if peak_rss is not None:
        print(f"Peak RAM Usage: {peak_rss:.1f} MB")
    else:
        print("Peak RAM Usage: Not recorded")
    print("-" * 65)
    print()


DISPLAY_NAME = {
    "pitch_shift": "Pitch Shift",
    "speed_change": "Speed Change",
    "compression": "Compression",
    "noise": "Noise",
    "gain": "Gain",
    "highpass": "High-pass",
    "trim": "Trim",
    "time_shift": "Time Shift",
}

DATASET_DISPLAY_NAME = {
    "speech": "Speech",
    "music": "Music",
    "environment": "Environmental",
}


def print_augmentation_results_table(aug_table, out_dir):
    """Final clean flat table: Dataset | Augmentation | Requested Range | Applied
    (Applied = status == "applied" only; see summary_tables.txt for the full
    applied/adjusted/skipped breakdown and actual parameter ranges used)."""
    rows = []
    for dname, category in DATASETS:
        label = DATASET_DISPLAY_NAME[category]
        for aug in AUGMENTATIONS:
            row = aug_table[(dname, aug)]
            rows.append((label, DISPLAY_NAME[aug], REQUESTED_RANGE_STR[aug], row["applied"]))

    col1_w = max(len(r[0]) for r in rows) + 2
    col2_w = max(len(r[1]) for r in rows) + 2
    col3_w = max(len(r[2]) for r in rows) + 2
    width = 65

    lines = []
    lines.append("=" * width)
    lines.append("AUGMENTATION RESULTS".center(width))
    lines.append("=" * width)
    lines.append("")
    lines.append(f"{'Dataset':<{col1_w}}{'Augmentation':<{col2_w}}{'Requested Range':<{col3_w}}{'Applied'}")
    lines.append("-" * width)
    for label, aug_label, req_range, applied in rows:
        lines.append(f"{label:<{col1_w}}{aug_label:<{col2_w}}{req_range:<{col3_w}}{applied}")
    lines.append("-" * width)

    text = "\n".join(lines)
    print(text)
    with open(out_dir / "augmentation_results_table.txt", "w", encoding="utf-8") as f:
        f.write(text + "\n")


# Requested ranges, checked against the parameters actually used (|speed|: both directions are drawn)
RANGE_CHECKS = {
    "pitch_shift": (-1.0, 4.0, False), "speed_change": (5.0, 20.0, True), "compression": (64.0, 128.0, False),
    "noise": (10.0, 20.0, False), "gain": (3.0, 6.0, False), "highpass": (150.0, 200.0, False),
    "trim": (3.0, 10.0, False), "time_shift": (5.0, 20.0, False),
}


def write_counts_and_check_ranges(all_records, csv_path):
    """augmentation_counts.csv (dataset x augmentation x status) + verification that only the eight
    requested augmentations exist and every used parameter lies inside its requested range.
    Returns the list of violations (empty = OK)."""
    from collections import Counter

    counts = Counter((r["category"], r["augmentation"], r["status"]) for r in all_records)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "augmentation", "applied", "adjusted", "skipped", "failed", "generated_files"])
        for _d, cat in DATASETS:
            for aug in AUGMENTATIONS:
                a, j, k, x = (counts[(cat, aug, st)] for st in ("applied", "adjusted", "skipped", "failed"))
                w.writerow([DATASET_DISPLAY_NAME[cat], DISPLAY_NAME[aug], a, j, k, x, a + j])

    violations = []
    unexpected = {r["augmentation"] for r in all_records} - set(AUGMENTATIONS)
    if unexpected:
        violations.append(f"unexpected augmentation types: {sorted(unexpected)}")
    tol = 0.011  # parameters are stored rounded to 2 decimals
    for r in all_records:
        if r["status"] not in ("applied", "adjusted"):
            continue
        v = extract_numeric_param(r["augmentation"], r["param_used"])
        lo, hi, use_abs = RANGE_CHECKS[r["augmentation"]]
        if v is None:
            violations.append(f"{r['dataset']} {r['original_filename']} {r['augmentation']}: unparseable '{r['param_used']}'")
            continue
        if use_abs:
            v = abs(v)
        if not (lo - tol <= v <= hi + tol):
            violations.append(f"{r['dataset']} {r['original_filename']} {r['augmentation']}: {v} outside [{lo}, {hi}]")
    return violations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-per-dataset", type=int, default=200)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--no-primary-benchmark-section", action="store_true",
                        help="skip the end-of-run BENCHMARK RESULTS section (which reads "
                             "results/primary_benchmark_results.json - a different experiment)")
    args = parser.parse_args()

    t_start = time.time()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    print(f"Selecting {args.n_per_dataset} clips per dataset (seed={args.seed})...")
    selected = select_clips(args.seed, args.n_per_dataset)
    for _dname, category in DATASETS:
        print(f"  {category}: selected {len(selected[category])} clips")
    write_selected_manifest(selected, SELECTED_MANIFEST_CSV, SELECTED_MANIFEST_JSON)
    print(f"Wrote selection manifest: {SELECTED_MANIFEST_CSV}")

    all_records = []
    for dname, category in DATASETS:
        out_dir = OUT_ROOT / f"{dname}_{category}"
        print(f"\nProcessing {dname} ({category}) -> {out_dir}")
        recs = process_dataset(dname, category, selected[category], out_dir, args.seed)
        all_records.extend(recs)

    metadata_path = AUGMENTATION_METADATA_CSV
    write_metadata_csv(all_records, metadata_path)
    print(f"\nWrote metadata CSV: {metadata_path} ({len(all_records)} rows)")
    violations = write_counts_and_check_ranges(all_records, AUGMENTATION_COUNTS_CSV)
    if violations:
        print(f"\nRANGE CHECK FAILED ({len(violations)} violations); first 10:")
        for v in violations[:10]:
            print("  ", v)
        sys.exit(1)
    print("Range check passed: only the eight requested augmentations, all parameters inside their requested ranges.")

    dataset_table, aug_table, param_values = build_summary_tables(all_records, args.n_per_dataset)
    print_and_save_summary(dataset_table, aug_table, param_values, args.n_per_dataset, OUT_ROOT)

    print()
    if not args.no_primary_benchmark_section:
        print_benchmark_results_from_json()
    print_augmentation_results_table(aug_table, OUT_ROOT)

    n_files_generated = sum(1 for r in all_records if r["status"] in ("applied", "adjusted"))
    elapsed = time.time() - t_start
    print(f"\nTotal augmented audio files generated: {n_files_generated}")
    print(f"Total processing time: {elapsed:.1f} sec ({elapsed/60:.1f} min)")

    with open(OUT_ROOT / "run_info.txt", "w") as f:
        f.write(f"n_per_dataset={args.n_per_dataset}\nseed={args.seed}\n")
        f.write(f"total_records={len(all_records)}\n")
        f.write(f"files_generated={n_files_generated}\n")
        f.write(f"elapsed_sec={elapsed:.1f}\n")


if __name__ == "__main__":
    main()
