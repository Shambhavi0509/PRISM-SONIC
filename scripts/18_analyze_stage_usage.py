"""Read-only analysis: how the existing two-stage SONIC pipeline (primary landmark stage ->
confidence gate -> secondary multi-resolution stage) behaves on a finished experiment's queries.

Nothing in sonic/ is modified or bypassed. For every query this calls the pipeline exactly as the
benchmark does (index = build_domain_index, query = query_array) and, separately and read-only,
recomputes the PRIMARY stage's top-1 candidate and the gate decision with the pipeline's own
functions, so that we can tell:
  - which queries the gate let through (confident) vs sent to the secondary stage (escalated),
  - whether the final answer was right,
  - whether the primary top-1 alone was already right (secondary only confirmed) or wrong
    (secondary RECOVERED it), and
  - that the gate's decision equals its own rule on every query (implementation-bug check).

    python scripts/18_analyze_stage_usage.py [--run-dir results/experiment_600]
"""

import argparse
import csv
import importlib.util
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from sonic.config import ANALYSIS_SR  # noqa: E402
from sonic.fingerprint.landmarks import build_landmarks_from_magnitude, stft_magnitude  # noqa: E402
from sonic.io import load_audio  # noqa: E402
from sonic.retrieval.pipeline import PRIMARY_POOL_K, _is_confident, build_domain_index, query_array  # noqa: E402

DOMAINS = [("speech", "Speech"), ("music", "Music"), ("environment", "Environmental")]
GROUPS = [  # (label, selector on (augmentation, signed param)) - harsh first, then easy, then reference
    ("Pitch > 2 st", lambda a, p: a == "Pitch Shift" and p is not None and p > 2),
    ("Noise 10-20 dB", lambda a, p: a == "Noise"),
    ("Trim", lambda a, p: a == "Trim"),
    ("Time Shift", lambda a, p: a == "Time Shift"),
    ("Pitch <= 2 st", lambda a, p: a == "Pitch Shift" and p is not None and p <= 2),
    ("Speed", lambda a, p: a == "Speed Change"),
    ("Compression", lambda a, p: a == "Compression"),
    ("Gain", lambda a, p: a == "Gain"),
    ("High-pass", lambda a, p: a == "High-pass"),
    ("Duplicate (ref.)", lambda a, p: a == "DUPLICATE"),
]


def load_exp10():
    spec = importlib.util.spec_from_file_location("exp10", REPO / "scripts" / "10_augment_experiment.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def primary_view(y, idx):
    """Primary stage only (same first steps as query_array): top-1 candidate id and gate decision."""
    ys = y[: int(idx.query_max_duration_sec * ANALYSIS_SR)]
    mag, frame_times = stft_magnitude(ys, ANALYSIS_SR)
    q_lm = build_landmarks_from_magnitude(mag, frame_times, ANALYSIS_SR)
    cands = idx.landmark_index.query(q_lm, top_k=PRIMARY_POOL_K)
    top = cands[0] if cands else None
    second = cands[1] if len(cands) > 1 else None
    confident, _margin = _is_confident(top, second, idx)
    return (top.file_id if top is not None else None), confident


def classify(r):
    if not r["escalated"]:
        return "primary_correct" if r["final_correct"] else "primary_confident_wrong"
    if r["final_correct"]:
        return "escalated_confirmed" if r["primary_top_correct"] else "escalated_recovered"
    return "escalated_lost" if r["primary_top_correct"] else "escalated_failed"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=str(REPO / "results" / "experiment_600"))
    args = ap.parse_args()
    run = Path(args.run_dir)

    exp10 = load_exp10()
    key_of = {v: k for k, v in exp10.DISPLAY_NAME.items()}
    manifest = json.load(open(run / "manifest_600.json"))
    queries = json.load(open(run / "benchmark_queries.json"))["records"]

    rows, t0 = [], time.time()
    for cat, label in DOMAINS:
        entries = manifest["categories"][cat]
        print(f"[{label}] building index over {len(entries)} files ...", flush=True)
        idx = build_domain_index(cat, entries)

        work = [("DUPLICATE", None, e["file_id"], e["path"], True) for e in entries]
        for r in queries:
            if r["category"] == cat:
                p = exp10.extract_numeric_param(key_of[r["modification"]], r["param_used"])
                work.append((r["modification"], p, r["true_file_id"], r["modified_path"], False))

        for n, (aug, param, true_id, path, is_dup) in enumerate(work):
            y = load_audio(path, max_duration=idx.query_max_duration_sec) if is_dup else load_audio(path)
            res = query_array(y, idx)  # the unmodified pipeline, exactly as the benchmark calls it
            top_id, confident = primary_view(y, idx)
            rows.append({
                "dataset": label, "augmentation": aug, "param": param,
                "final_correct": res.file_id == true_id, "escalated": bool(res.escalated),
                "primary_top_correct": top_id == true_id, "gate_confident_recomputed": confident,
            })
            if (n + 1) % 400 == 0:
                print(f"   [{label}] {n + 1}/{len(work)} queries  ({time.time() - t0:.0f}s elapsed)", flush=True)

    for r in rows:
        r["outcome"] = classify(r)

    # ---- per-query detail file
    with open(run / "stage_usage_details.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    out_lines = []

    def out(s=""):
        print(s)
        out_lines.append(s)

    # ---- gate implementation check: escalated must equal "not confident" on every query
    mism = [r for r in rows if r["escalated"] == r["gate_confident_recomputed"]]
    out(f"\nGate consistency check: escalated == (not confident under the pipeline's own rule) on "
        f"{len(rows) - len(mism)}/{len(rows)} queries; mismatches = {len(mism)}")

    def table(title, subset_rows):
        out(f"\n{title}")
        hdr = (f"{'Group':<18}{'Queries':>8}{'Primary':>9}{'Escalated':>11}{'(esc %)':>9}{'Sec.recovered':>15}"
              f"{'Still failed':>14}{'Wrong w/o esc.':>16}")
        out(hdr)
        out("-" * len(hdr))
        for label, sel in GROUPS:
            g = [r for r in subset_rows if sel(r["augmentation"], r["param"])]
            if not g:
                continue
            c = defaultdict(int)
            for r in g:
                c[r["outcome"]] += 1
            esc = sum(v for k, v in c.items() if k.startswith("escalated"))
            rec = c["escalated_recovered"] + c["escalated_confirmed"]
            fail = c["escalated_failed"] + c["escalated_lost"]
            out(f"{label:<18}{len(g):>8}{c['primary_correct']:>9}{esc:>11}{100 * esc / len(g):>8.1f}%{rec:>15}"
                f"{fail:>14}{c['primary_confident_wrong']:>16}")

    out("\nColumns: Primary = answered by the primary stage (gate confident) and correct | Escalated = sent to the secondary stage")
    out("Sec.recovered = escalated AND final answer correct | Still failed = escalated AND final answer wrong")
    out("Wrong w/o esc. = gate said 'confident' but the answer was wrong (never reached the secondary stage)")
    table("=== ALL DATASETS POOLED ===", rows)
    for _c, label in DOMAINS:
        table(f"=== {label} ===", [r for r in rows if r["dataset"] == label])

    # ---- detail of what the secondary stage did, harsh modifications pooled
    out("\n=== What the secondary stage did with the queries it received ===")
    out(f"{'Group':<18}{'Escalated':>10}{'primary top-1 already right':>29}{'-> final right':>16}{'-> final wrong':>16}"
        f"{'primary top-1 wrong':>21}{'-> RECOVERED':>14}{'-> still wrong':>16}")
    for label, sel in GROUPS:
        g = [r for r in rows if sel(r["augmentation"], r["param"]) and r["escalated"]]
        if not g:
            continue
        c = defaultdict(int)
        for r in g:
            c[r["outcome"]] += 1
        pr = c["escalated_confirmed"] + c["escalated_lost"]
        pw = c["escalated_recovered"] + c["escalated_failed"]
        out(f"{label:<18}{len(g):>10}{pr:>29}{c['escalated_confirmed']:>16}{c['escalated_lost']:>16}"
            f"{pw:>21}{c['escalated_recovered']:>14}{c['escalated_failed']:>16}")

    # ---- overall summary over edited queries
    ed = [r for r in rows if r["augmentation"] != "DUPLICATE"]
    c = defaultdict(int)
    for r in ed:
        c[r["outcome"]] += 1
    out("\n=== Overall summary (all edited queries) ===")
    out(f"Edited queries:                                   {len(ed)}")
    out(f"Primary-only correct:                             {c['primary_correct']}")
    out(f"Escalated to secondary:                           {sum(v for k, v in c.items() if k.startswith('escalated'))}")
    out(f"Secondary-stage recoveries (primary wrong->right): {c['escalated_recovered']}")
    out(f"Secondary confirmed an already-right primary top-1: {c['escalated_confirmed']}")
    out(f"Secondary-stage failures (final wrong):           {c['escalated_failed'] + c['escalated_lost']}"
        f"   (of which primary top-1 had been right: {c['escalated_lost']})")
    out(f"Wrong answers returned without ever escalating:   {c['primary_confident_wrong']}")
    (run / "stage_usage_summary.txt").write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    print(f"\nTotal analysis time {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
