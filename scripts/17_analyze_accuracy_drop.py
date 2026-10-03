"""Explains an experiment's edited accuracy from its own result files, and compares it with
the earlier Phase-1 benchmark (results/benchmark06_v2.log: 200/200/200 files, fixed mild
severities). Read-only: touches no SONIC code and re-runs nothing.

    python scripts/17_analyze_accuracy_drop.py [--run-dir results/experiment_600]

Sections: (A) queries/status per augmentation, (B) actual parameters vs the old severities,
(C) old vs new accuracy per augmentation + exact decomposition of the drop,
(D) accuracy by severity bin, (E) Applied vs Adjusted.
"""

import argparse
import csv
import importlib.util
import re
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PHASE1_LOG = REPO / "results" / "benchmark06_v2.log"
DOMAINS = [("speech", "Speech"), ("music", "Music"), ("environment", "Environmental")]
AUGS = ["Pitch Shift", "Speed Change", "Compression", "Noise", "Gain", "High-pass", "Trim", "Time Shift"]

# Phase-1 modification names -> the eight experiment augmentations (several pool into one)
OLD_TO_NEW = {
    "pitch_shift": "Pitch Shift", "speed_change": "Speed Change", "mp3_compression": "Compression",
    "background_noise": "Noise", "white_noise": "Noise", "gain_change": "Gain",
    "filtering": "High-pass", "trim_start": "Trim", "trim_end": "Trim", "time_shift": "Time Shift",
}
OLD_SEVERITY = {
    "Pitch Shift": "+/-0.5 semitones", "Speed Change": "x0.995 / x1.005 (+/-0.5%)", "Compression": "192 kbps",
    "Noise": "30 dB (real noise) / 35 dB (white)", "Gain": "+/-1 dB",
    "High-pass": "highpass 50 Hz + lowpass 10 kHz (pooled)", "Trim": "0.05 s", "Time Shift": "+/-20 ms",
}


def load_exp10():
    spec = importlib.util.spec_from_file_location("exp10", REPO / "scripts" / "10_augment_experiment.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def parse_phase1():
    """{label: {new_aug: [correct, n]}} from the Phase-1 log (acc * n recovers the correct count)."""
    out = {lab: defaultdict(lambda: [0.0, 0]) for _c, lab in DOMAINS}
    cat_to_label = dict(DOMAINS)
    pat = re.compile(r"^\[(\w+)\] (\w+) \(levels=.*\): acc=([\d.]+) n=(\d+)")
    for line in open(PHASE1_LOG, encoding="utf-8"):
        m = pat.match(line)
        if m and m.group(2) in OLD_TO_NEW:
            lab, aug = cat_to_label[m.group(1)], OLD_TO_NEW[m.group(2)]
            out[lab][aug][0] += round(float(m.group(3)) * int(m.group(4)))  # acc is printed to 3 decimals; acc*n is an integer count
            out[lab][aug][1] += int(m.group(4))
    return out


def pct(c, n):
    return f"{100 * c / n:5.1f}%" if n else "   n/a"


def binned(rows, key, edges, labels, title, out):
    out(f"  {title}")
    for i, lab in enumerate(labels):
        sel = [r for r in rows if r[key] is not None and edges[i] <= r[key] < edges[i + 1]]
        out(f"     {lab:<22}{sum(r['correct'] for r in sel):4d}/{len(sel):<4d} {pct(sum(r['correct'] for r in sel), len(sel))}")


def load_rows(run, exp10, key_of):
    """One row per generated edited query: dataset, augmentation, correct, status, durations, numeric param."""
    meta = list(csv.DictReader(open(run / "augmentation_metadata.csv", encoding="utf-8")))
    meta_by_file = {(m["dataset"], m["generated_filename"]): m for m in meta if m["generated_filename"]}
    results = [r for r in csv.DictReader(open(run / "benchmark_results.csv", encoding="utf-8")) if r["query_type"] == "edited"]
    rows = []
    for r in results:
        m = meta_by_file[(r["dataset"], r["query_file"])]
        rows.append({
            "dataset": r["dataset"], "aug": r["augmentation"], "correct": int(r["correct"]),
            "status": m["status"], "orig_dur": float(m["original_duration_sec"]), "final_dur": float(m["final_duration_sec"]),
            "param": exp10.extract_numeric_param(key_of[r["augmentation"]], m["actual_param_used"]),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=str(REPO / "results" / "experiment_600"))
    ap.add_argument("--previous-run-dir", default=None, help="optional: also compare against this earlier run (section F)")
    args = ap.parse_args()
    run = Path(args.run_dir)
    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    exp10 = load_exp10()
    key_of = {v: k for k, v in exp10.DISPLAY_NAME.items()}

    meta = list(csv.DictReader(open(run / "augmentation_metadata.csv", encoding="utf-8")))
    rows = load_rows(run, exp10, key_of)

    out(f"=== Analysis of {run} ===\n")
    out("(A) Generated edited queries and statuses per dataset / augmentation")
    out(f"    {'':14}" + "".join(f"{a:>14}" for a in AUGS))
    for _c, lab in DOMAINS:
        for st in ("Applied", "Adjusted", "Skipped", "Failed"):
            cnt = [sum(1 for m in meta if m["dataset"] == lab and m["augmentation"] == a and m["status"] == st) for a in AUGS]
            out(f"    {lab[:9]:<9}{st:<10}" + "".join(f"{c:>14}" for c in cnt))
    tot = {st: sum(1 for m in meta if m["status"] == st) for st in ("Applied", "Adjusted", "Skipped", "Failed")}
    out(f"    totals: {tot}   generated queries = {tot['Applied'] + tot['Adjusted']}\n")

    out("(B) Actual parameters used (min / mean / max over generated files) vs the Phase-1 benchmark severity")
    for a in AUGS:
        vals = [r["param"] for r in rows if r["aug"] == a and r["param"] is not None]
        if vals:
            out(f"    {a:<13} actual {min(vals):8.2f} / {sum(vals) / len(vals):8.2f} / {max(vals):8.2f}     Phase-1: {OLD_SEVERITY[a]}")
    out()

    old = parse_phase1()
    out("(C) Edited accuracy per augmentation: Phase-1 (old, mild severities) vs this run, and the exact decomposition")
    out("    drop_a = share_of_queries_a x (old_acc_a - new_acc_a), holding the new query mix fixed; sum(drop_a) is")
    out("    the gap between 'Phase-1 accuracies on this mix' and this run's overall edited accuracy.")
    for _c, lab in DOMAINS:
        sub = [r for r in rows if r["dataset"] == lab]
        N = len(sub)
        new_overall = sum(r["correct"] for r in sub) / N
        old_cc = sum(v[0] for v in old[lab].values())
        old_n = sum(v[1] for v in old[lab].values())
        out(f"\n    {lab}: this run edited accuracy {100 * new_overall:.1f}% over {N} queries; Phase-1 actual {100 * old_cc / old_n:.1f}% over {old_n}")
        out(f"      {'augmentation':<13}{'queries':>8}{'share':>7}{'old acc':>9}{'new acc':>9}{'drop (pp)':>10}{'% of drop':>10}")
        drops = {}
        for a in AUGS:
            s = [r for r in sub if r["aug"] == a]
            if not s:
                continue
            oa = old[lab][a][0] / old[lab][a][1]
            na = sum(r["correct"] for r in s) / len(s)
            drops[a] = (len(s), len(s) / N, oa, na, (len(s) / N) * (oa - na) * 100)
        total_drop = sum(v[4] for v in drops.values())
        for a, (n, w, oa, na, d) in drops.items():
            out(f"      {a:<13}{n:>8}{100 * w:>6.1f}%{100 * oa:>8.1f}%{100 * na:>8.1f}%{d:>10.2f}{100 * d / total_drop if total_drop else 0:>9.1f}%")
        out(f"      {'TOTAL':<13}{N:>8}{'':>7}{'':>9}{100 * new_overall:>8.1f}%{total_drop:>10.2f}")

    out("\n(D) Accuracy by actual severity (this run)")
    for _c, lab in DOMAINS:
        sub = [r for r in rows if r["dataset"] == lab]
        out(f"\n  --- {lab} ---")
        p = [dict(r, param=abs(r["param"])) for r in sub if r["aug"] == "Pitch Shift"]
        binned(p, "param", [0, 0.5, 1.0, 2.0, 3.0, 4.01], ["|shift| 0-0.5 st (Phase-1 level)", "0.5-1 st", "1-2 st", "2-3 st", "3-4 st"], "Pitch Shift by |semitones|", out)
        s = [dict(r, param=abs(r["param"])) for r in sub if r["aug"] == "Speed Change"]
        binned(s, "param", [5, 10, 15, 20.01], ["5-10 %", "10-15 %", "15-20 %"], "Speed Change by |%| change", out)
        n = [r for r in sub if r["aug"] == "Noise"]
        binned(n, "param", [10, 12.5, 15, 17.5, 20.01], ["SNR 10-12.5 dB", "12.5-15", "15-17.5", "17.5-20"], "Noise by SNR", out)
        t = [dict(r, param=r["final_dur"]) for r in sub if r["aug"] == "Trim"]
        binned(t, "param", [0, 1, 2, 4, 8, 1e9], ["remaining < 1 s", "1-2 s", "2-4 s", "4-8 s", ">= 8 s"], "Trim by REMAINING duration of the query", out)
        t2 = [r for r in sub if r["aug"] == "Trim"]
        binned(t2, "param", [0, 3, 5, 7.5, 10.01], ["trim < 3 s (outside range)", "trim 3-5 s", "5-7.5 s", "7.5-10 s"], "Trim by seconds removed", out)
        ts = [r for r in sub if r["aug"] == "Time Shift"]
        binned(ts, "param", [0, 5, 10, 15, 20.01], ["shift < 5 s (outside range)", "shift 5-10 s", "10-15 s", "15-20 s"], "Time Shift by seconds shifted", out)
        for a, edges, labs, ttl in [
            ("Compression", [64, 80, 96, 112, 128.01], ["64-80", "80-96", "96-112", "112-128"], "Compression by kbps"),
        ]:
            binned([r for r in sub if r["aug"] == a], "param", edges, labs, ttl, out)

    out("\n(E) Applied vs Adjusted accuracy (Adjusted = parameter changed because the clip was too short)")
    for _c, lab in DOMAINS:
        for a in AUGS:
            for st in ("Applied", "Adjusted"):
                s = [r for r in rows if r["dataset"] == lab and r["aug"] == a and r["status"] == st]
                if s:
                    out(f"    {lab[:9]:<9}{a:<13}{st:<9}{sum(r['correct'] for r in s):4d}/{len(s):<4d} {pct(sum(r['correct'] for r in s), len(s))}")

    if args.previous_run_dir:
        prev_rows = load_rows(Path(args.previous_run_dir), exp10, key_of)
        out("\n(F) Comparison: earlier benchmark (Phase-1, mild severities) vs previous 600-clip run vs this run")
        out("\n    Edited accuracy per dataset")
        out(f"    {'Dataset':<15}{'Earlier (Phase-1)':>19}{'Previous run':>14}{'This run':>10}{'This - Earlier':>16}{'This - Previous':>17}")
        for _c, lab in DOMAINS:
            oc = sum(v[0] for v in old[lab].values())
            on = sum(v[1] for v in old[lab].values())
            pr = [r for r in prev_rows if r["dataset"] == lab]
            cu = [r for r in rows if r["dataset"] == lab]
            oa, pa, ca = oc / on, sum(r["correct"] for r in pr) / len(pr), sum(r["correct"] for r in cu) / len(cu)
            out(f"    {lab:<15}{100 * oa:>18.1f}%{100 * pa:>13.1f}%{100 * ca:>9.1f}%{100 * (ca - oa):>+15.1f} pp{100 * (ca - pa):>+16.1f} pp")
        out("\n    Edited accuracy per dataset / augmentation  (queries in brackets)")
        out(f"    {'Dataset':<15}{'Augmentation':<14}{'Earlier':>9}{'Previous run':>16}{'This run':>16}{'This - Previous':>17}")
        for _c, lab in DOMAINS:
            for a in AUGS:
                pr = [r for r in prev_rows if r["dataset"] == lab and r["aug"] == a]
                cu = [r for r in rows if r["dataset"] == lab and r["aug"] == a]
                oa = old[lab][a][0] / old[lab][a][1]
                pa = sum(r["correct"] for r in pr) / len(pr) if pr else float("nan")
                ca = sum(r["correct"] for r in cu) / len(cu) if cu else float("nan")
                out(f"    {lab:<15}{a:<14}{100 * oa:>8.1f}%{100 * pa:>9.1f}% ({len(pr):>4}){100 * ca:>9.1f}% ({len(cu):>4}){100 * (ca - pa):>+14.1f} pp")

    (run / "accuracy_analysis.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
