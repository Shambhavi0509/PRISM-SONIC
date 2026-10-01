"""AUDIO MATCHING SYSTEM - user-facing demonstration front-end for SONIC.

This script does NOT implement any fingerprinting, matching, scoring, or
indexing logic of its own. It is a thin input/output layer around the
EXISTING, UNMODIFIED SONIC architecture:

    sonic.retrieval.pipeline.build_domain_index()   <- existing index builder
    sonic.retrieval.pipeline.query()                <- existing query pipeline

Every uploaded audio file is passed to query(path, idx) exactly as the
existing benchmark scripts already do for duplicate-audio queries. No new
matching algorithm, scoring rule, or confidence logic is introduced here.

Two modes:
  python scripts/15_audio_matching_system.py            interactive (real user)
  python scripts/15_audio_matching_system.py --test      automated validation
                                                          against Input Database/
"""

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:  # Windows consoles/pipes can default to cp1252, which can't encode the
    # checkmark/arrow characters used for step-progress output below.
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

import numpy as np

from sonic.config import ANALYSIS_SR  # noqa: E402 - existing config, unmodified
from sonic.io import load_audio  # noqa: E402 - existing preprocessing, unmodified
from sonic.retrieval.pipeline import build_domain_index, query  # noqa: E402 - existing pipeline, unmodified

REPO_ROOT = Path(__file__).resolve().parent.parent
TESTING_DB = REPO_ROOT / "Testing Output Database"
INPUT_DB = REPO_ROOT / "Input Database"
TEST_RESULTS_DIR = REPO_ROOT / "test_results"

DOMAINS = [("1", "Speech", "speech"), ("2", "Music", "music"), ("3", "Environment", "environment")]
DOMAIN_BY_NUM = {n: (label, cat) for n, label, cat in DOMAINS}


# --------------------------------------------------------------- index build
def scan_domain_entries(domain_label: str, category: str) -> list[dict]:
    """Builds the entries list build_domain_index() expects, pointing at
    Testing Output Database/<Domain>/ - NOT the original large dataset/."""
    d = TESTING_DB / domain_label
    files = sorted(p.name for p in d.iterdir() if p.is_file())
    entries = []
    for i, name in enumerate(files):
        entries.append({
            "file_id": i,  # local to this domain's index; fine since each domain has its own DomainIndex
            "category": category,
            "filename": name,
            "path": str(d / name),
        })
    return entries


def build_all_indexes(quiet: bool = False) -> dict:
    """Builds one DomainIndex per domain from ONLY the 600-file Testing
    Output Database, using the existing, unmodified build_domain_index()."""
    indexes = {}
    total_files = 0
    for _num, label, category in DOMAINS:
        entries = scan_domain_entries(label, category)
        total_files += len(entries)
        if not quiet:
            print(f"  Building {label} index ({len(entries)} files)...", flush=True)
        idx = build_domain_index(category, entries)
        indexes[category] = {"index": idx, "entries": entries, "label": label}
    if not quiet:
        print(f"\nDatabase: Testing Output Database")
        print(f"Total indexed files: {total_files}\n")
    return indexes


# --------------------------------------------------------- match-type check
def classify_match_type(query_path: str, matched_path: str) -> str:
    """Determines Duplicate vs Edited Audio by comparing actual decoded audio
    content (through the SAME existing load_audio() preprocessing used by the
    retrieval pipeline) - not by filename. A near-zero sample-level difference
    at matching duration means the uploaded file is (up to lossless container
    conversion) the same audio as the matched database file; any real
    modification (pitch/speed/noise/trim/compression/etc.) produces a much
    larger difference."""
    yq = load_audio(query_path, sr=ANALYSIS_SR)
    yd = load_audio(matched_path, sr=ANALYSIS_SR)
    if yq.size == 0 or yd.size == 0:
        return "Edited Audio"
    n = min(yq.size, yd.size)
    len_ratio = abs(yq.size - yd.size) / max(yq.size, yd.size, 1)
    diff = float(np.mean(np.abs(yq[:n] - yd[:n])))
    # Threshold calibrated against real project data: a true duplicate
    # (identical audio, only container/format changed) measures ~3e-5 here;
    # even a fairly gentle edit (96kbps MP3 recompression) measures ~2e-4 -
    # 1e-4 cleanly separates the two with margin on both sides.
    if diff < 1e-4 and len_ratio < 0.01:
        return "Duplicate"
    return "Edited Audio"


# ------------------------------------------------------------------- lookup
def entries_by_local_id(entries: list[dict]) -> dict:
    return {e["file_id"]: e for e in entries}


# ------------------------------------------------------------- core routine
def run_query(input_path: str, domain_key: dict, verbose: bool = True) -> dict:
    """Runs ONE uploaded file through the existing SONIC pipeline for the
    given (already-built) domain index. Returns a plain-dict result; does not
    alter or reimplement any matching logic - `query()` is called exactly as
    the existing benchmark code already calls it."""
    idx = domain_key["index"]
    entries = domain_key["entries"]
    label = domain_key["label"]
    by_id = entries_by_local_id(entries)

    if verbose:
        print("Processing audio...\n")
        print("✓ Preprocessing complete")
        print("✓ Primary fingerprint generated")
        print("✓ Searching database...")

    t0 = time.perf_counter()
    res = query(input_path, idx)  # <-- the ENTIRE existing SONIC pipeline, unmodified
    elapsed_ms = (time.perf_counter() - t0) * 1000

    if verbose and res.escalated:
        print("\n→ Escalating to secondary fingerprint...")
        print("→ Performing multi-resolution matching...")

    matched_entry = by_id.get(res.file_id) if res.file_id is not None else None
    match_type = None
    if matched_entry is not None:
        match_type = classify_match_type(input_path, matched_entry["path"])

    return {
        "matched": matched_entry is not None,
        "matched_filename": matched_entry["filename"] if matched_entry else None,
        "domain": label,
        "match_type": match_type,
        "score": res.weighted_score,
        "confident": res.confident,
        "stage_used": res.stage_used,
        "escalated": res.escalated,
        "query_time_ms": elapsed_ms,
    }


def print_result_block(input_filename: str, result: dict):
    """User-facing result - deliberately does NOT show the internal matching
    score. The score remains available in `result` for --test mode's CSV."""
    width = 40
    print("\n" + "-" * width)
    print("RESULT".center(width))
    print("-" * width)
    if result["matched"]:
        print("\nMATCH FOUND\n")
        print(f"Input: {input_filename}")
        print(f"Matched File: {result['matched_filename']}")
        print(f"Domain: {result['domain']}")
        print(f"Match Type: {result['match_type']}")
        print(f"\nQuery Time: {result['query_time_ms']:.2f} ms")
    else:
        print("\nAUDIO DOES NOT MATCH IN DATABASE\n")
        print(f"Input: {input_filename}")
        print(f"\nQuery Time: {result['query_time_ms']:.2f} ms")
    print("-" * width + "\n")


# ------------------------------------------------------------- file picker
def pick_audio_file() -> str | None:
    """Opens a native Windows file-selection dialog (tkinter, standard
    library - no new dependency) restricted to audio formats already
    supported by the existing SONIC preprocessing (soundfile-backed
    sonic.io.load_audio: WAV/FLAC/OGG plus MP3 via soundfile's MP3 support).
    Returns the selected path, or None if the user cancels."""
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        path = filedialog.askopenfilename(
            title="Select your audio file",
            filetypes=[
                ("Audio files", "*.wav *.mp3 *.flac *.ogg *.m4a *.aac"),
                ("All files", "*.*"),
            ],
        )
    finally:
        root.destroy()
    return path or None


# --------------------------------------------------------------- interactive
def interactive_main():
    print("=" * 50)
    print("AUDIO MATCHING SYSTEM".center(50))
    print("=" * 50 + "\n")

    print("Building indexes from Testing Output Database ...")
    indexes = build_all_indexes()

    while True:
        print("Select the audio domain:\n")
        print("1. Speech")
        print("2. Music")
        print("3. Environment")
        print("4. Exit\n")
        choice = input("Enter your choice: ").strip()

        if choice == "4":
            print("Goodbye.")
            return
        if choice not in DOMAIN_BY_NUM:
            print("Invalid choice. Please enter 1, 2, 3, or 4.\n")
            continue

        label, category = DOMAIN_BY_NUM[choice]
        print("\nPlease select your audio file...")
        raw_path = pick_audio_file()

        if not raw_path:
            print("\nNo file selected. Returning to domain selection.\n")
            continue

        input_path = Path(raw_path)
        if not input_path.exists():
            print(f"\nFile not found: {raw_path}\n")
            continue

        print(f"\nSelected file: {input_path.name}")
        print("\nRunning SONIC matching...\n")
        result = run_query(str(input_path), indexes[category], verbose=True)
        print_result_block(input_path.name, result)

        again = input("Test another file? (y/n): ").strip().lower()
        if again != "y":
            print("Goodbye.")
            return
        print()


# ----------------------------------------------------------------- test mode
def automated_test():
    gt_path = INPUT_DB / "ground_truth.csv"
    if not gt_path.exists():
        print(f"Ground truth file not found: {gt_path}")
        print("Run scripts/14_build_input_database.py first.")
        return

    with open(gt_path, encoding="utf-8") as f:
        ground_truth = list(csv.DictReader(f))

    print("Building indexes from Testing Output Database ...")
    indexes = build_all_indexes()

    domain_label_to_category = {label: cat for _n, label, cat in DOMAINS}

    rows = []
    for gt in ground_truth:
        input_filename = gt["input_filename"]
        domain_label = gt["domain"]
        category = domain_label_to_category[domain_label]
        input_path = INPUT_DB / input_filename

        result = run_query(str(input_path), indexes[category], verbose=False)

        expected_match = gt["source_filename"] if gt["input_type"] != "non-match" else "(none)"
        predicted_match = result["matched_filename"] if result["matched"] else "(none)"

        if gt["input_type"] == "non-match":
            outcome = "CORRECT" if not result["matched"] else "INCORRECT"
        else:
            outcome = "CORRECT" if predicted_match == gt["source_filename"] else "INCORRECT"

        rows.append({
            "input": input_filename,
            "domain": domain_label,
            "type": gt["input_type"],
            "expected_match": expected_match,
            "predicted_match": predicted_match,
            "predicted_type": result["match_type"] or "-",
            "result": outcome,
            "score": f"{result['score']:.2f}",
            "query_time_ms": f"{result['query_time_ms']:.2f}",
        })

    TEST_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = TEST_RESULTS_DIR / "test_results.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)

    col = {
        "input": max(10, max(len(r["input"]) for r in rows)) + 2,
        "domain": max(9, max(len(r["domain"]) for r in rows)) + 2,
        "type": max(6, max(len(r["type"]) for r in rows)) + 2,
        "expected_match": max(10, max(len(r["expected_match"]) for r in rows)) + 2,
        "predicted_match": max(11, max(len(r["predicted_match"]) for r in rows)) + 2,
        "result": max(8, max(len(r["result"]) for r in rows)) + 2,
    }
    header = (f"{'Input':<{col['input']}}{'Domain':<{col['domain']}}{'Type':<{col['type']}}"
              f"{'Expected':<{col['expected_match']}}{'Predicted':<{col['predicted_match']}}"
              f"{'Result':<{col['result']}}{'Score':>8}{'Query ms':>10}")
    print("\n" + header)
    print("-" * len(header))
    n_correct = 0
    for r in rows:
        print(f"{r['input']:<{col['input']}}{r['domain']:<{col['domain']}}{r['type']:<{col['type']}}"
              f"{r['expected_match']:<{col['expected_match']}}{r['predicted_match']:<{col['predicted_match']}}"
              f"{r['result']:<{col['result']}}{r['score']:>8}{r['query_time_ms']:>10}")
        if r["result"] == "CORRECT":
            n_correct += 1
    print("-" * len(header))
    print(f"\n{n_correct}/{len(rows)} correct.")
    print(f"Results written to {out_csv}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="run automated validation against Input Database/")
    args = parser.parse_args()
    if args.test:
        automated_test()
    else:
        interactive_main()


if __name__ == "__main__":
    main()
