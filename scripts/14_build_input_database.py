"""Builds 'Input Database/' - 10 test-query audio files simulating what a
real user might upload, generated FROM the 600-file Testing Output Database
(so ground truth is known) plus 2 genuinely-absent files for the non-match
case. Ground truth is written to Input Database/ground_truth.csv - used only
by the automated test harness (scripts/15_audio_matching_system.py --test),
never shown to the interactive "real user" flow.

Reuses the EXISTING modification functions from sonic/augment/modifications.py
unchanged - no new modification method is introduced. Does not touch the
SONIC retrieval architecture in any way.

Mixture (10 total, across all three domains, demonstrating all three cases
the system must handle):
  - 3 exact duplicates  (Speech, Music, Environment)
  - 5 edited versions    (2 Speech, 1 Music, 2 Environment)
  - 2 non-match          (Speech, Environment - Music has no spare files
                          outside the 600-file DB, since it only has 200
                          files total, so a music non-match is impossible
                          without reusing an indexed file)
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import librosa
import numpy as np
import soundfile as sf

from sonic.augment.modifications import (  # noqa: E402 - existing, unmodified functions
    apply_filter, background_noise, gain_change, mp3_compress, pitch_shift,
    trim_end,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = REPO_ROOT / "dataset"
TESTING_DB = REPO_ROOT / "Testing Output Database"
OUT_DIR = REPO_ROOT / "Input Database"

DOMAIN_DIRS = {"Speech": "speech", "Music": "music", "Environment": "environment"}


def load(path):
    y, sr = librosa.load(str(path), sr=None, mono=True)
    return y.astype(np.float32), sr


def testing_db_files(domain):
    return sorted(p.name for p in (TESTING_DB / domain).iterdir() if p.is_file())


def full_dataset_files(category):
    return sorted(p.name for p in (DATASET_ROOT / category).iterdir() if p.is_file())


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records = []

    # ---- 3 exact duplicates -------------------------------------------------
    dup_plan = [("Speech", 0), ("Music", 0), ("Environment", 0)]
    for i, (domain, idx) in enumerate(dup_plan, start=1):
        files = testing_db_files(domain)
        src_name = files[idx]
        src_path = TESTING_DB / domain / src_name
        y, sr = load(src_path)
        out_name = f"duplicate_{domain.lower()}_{i:02d}.wav"
        sf.write(str(OUT_DIR / out_name), y, sr)
        records.append(dict(
            input_filename=out_name, source_filename=src_name, domain=domain,
            input_type="duplicate", modification="none",
        ))
        print(f"[duplicate] {out_name}  <-  {domain}/{src_name}")

    # ---- 5 edited versions ---------------------------------------------------
    edit_plan = [
        ("Speech", 49, "pitch_shift"),
        ("Speech", 99, "mp3_compression"),
        ("Music", 49, "speed_change"),
        ("Environment", 49, "background_noise"),
        ("Environment", 99, "trim_end"),
    ]
    env_files_full = full_dataset_files("environment")
    testing_env_names = set(testing_db_files("Environment"))
    noise_bed_name = next(n for n in env_files_full if n not in testing_env_names)
    noise_y, noise_sr = load(DATASET_ROOT / "environment" / noise_bed_name)

    for i, (domain, idx, mod) in enumerate(edit_plan, start=1):
        files = testing_db_files(domain)
        src_name = files[idx]
        src_path = TESTING_DB / domain / src_name
        y, sr = load(src_path)

        if mod == "pitch_shift":
            y_out = pitch_shift(y, sr, 2.0)
            param = "+2.0 semitones"
        elif mod == "mp3_compression":
            y_out = mp3_compress(y, sr, 96)
            param = "96 kbps"
        elif mod == "speed_change":
            y_out = librosa.effects.time_stretch(y=y, rate=1.10)
            param = "rate=1.10 (+10%)"
        elif mod == "background_noise":
            noise_local = noise_y if noise_sr == sr else librosa.resample(noise_y, orig_sr=noise_sr, target_sr=sr)
            y_out = background_noise(y, sr, 15.0, noise_local)
            param = "15 dB SNR (real env. noise bed, outside the 600-file DB)"
        elif mod == "trim_end":
            y_out = trim_end(y, sr, 2.0)
            param = "2.0 sec removed from end"
        else:
            raise ValueError(mod)

        out_name = f"edited_{domain.lower()}_{i:02d}.wav"
        sf.write(str(OUT_DIR / out_name), y_out, sr)
        records.append(dict(
            input_filename=out_name, source_filename=src_name, domain=domain,
            input_type="edited", modification=f"{mod} ({param})",
        ))
        print(f"[edited]    {out_name}  <-  {domain}/{src_name}  [{mod}: {param}]")

    # ---- 2 non-match (genuinely absent from the 600-file DB) -----------------
    nonmatch_plan = [("Speech", "speech"), ("Environment", "environment")]
    for i, (domain, category) in enumerate(nonmatch_plan, start=1):
        testing_names = set(testing_db_files(domain))
        full_names = full_dataset_files(category)
        candidate = next(n for n in full_names if n not in testing_names)
        src_path = DATASET_ROOT / category / candidate
        y, sr = load(src_path)
        out_name = f"nonmatch_{domain.lower()}_{i:02d}.wav"
        sf.write(str(OUT_DIR / out_name), y, sr)
        records.append(dict(
            input_filename=out_name, source_filename="(none - not in Testing Output Database)",
            domain=domain, input_type="non-match", modification="none",
        ))
        print(f"[non-match] {out_name}  <-  {category}/{candidate} (outside the 600-file DB)")

    # ---- ground truth CSV (internal only, not shown during normal use) ------
    gt_path = OUT_DIR / "ground_truth.csv"
    with open(gt_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["input_filename", "source_filename", "domain", "input_type", "modification"])
        w.writeheader()
        for r in records:
            w.writerow(r)

    print(f"\nTOTAL input files generated: {len(records)}")
    print(f"Ground truth written to: {gt_path}")
    assert len(records) == 10, f"expected exactly 10 inputs, got {len(records)}"
    print("OK - exactly 10 input files confirmed.")


if __name__ == "__main__":
    main()
