"""Prints the extended results tables (RAM usage, query counts, per-domain
latency) from the already-completed benchmark results JSON, in addition to the
core accuracy table scripts/07 already printed. Read-only - does not re-run
the benchmark or touch the saved data."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sonic.config import RESULTS_ROOT

with open(RESULTS_ROOT / "primary_benchmark_results.json") as f:
    r = json.load(f)

DOMAINS = [("speech", "Speech"), ("music", "Music"), ("environment", "Environmental")]


def line(ch="=", n=72):
    print(ch * n)


def row(cols, widths):
    print("".join(str(c).ljust(w) for c, w in zip(cols, widths)))


# ---------------- Peak RAM ----------------
print()
line()
print("RAM USAGE".center(72))
line()
print(f"Peak RSS (all 3 domain indexes held in memory simultaneously): {r['peak_rss_mb']:.1f} MB")
print(f"Budget: < 120 MB")
line("-")
w = [18, 16, 18, 20]
row(["Domain", "Index Build (s)", "Index Size (MB)", ""], w)
line("-")
for key, label in DOMAINS:
    d = r[key]
    row([label, f"{d['index_build_time_sec']:.1f}", f"{d['index_bytes']/1024/1024:.2f}", ""], w)
line("-")
print()

# ---------------- Query counts ----------------
line()
print("NUMBER OF QUERIES".center(72))
line()
w = [18, 14, 14, 14]
row(["Domain", "Duplicate", "Edited", "Total"], w)
line("-")
grand_dup, grand_ed = 0, 0
for key, label in DOMAINS:
    d = r[key]
    n_dup = d["duplicate"]["n"]
    n_ed = d["edited_overall"]["n"]
    grand_dup += n_dup
    grand_ed += n_ed
    row([label, str(n_dup), str(n_ed), str(n_dup + n_ed)], w)
line("-")
row(["TOTAL", str(grand_dup), str(grand_ed), str(grand_dup + grand_ed)], w)
line("-")
print()

# ---------------- Latency per domain ----------------
line()
print("QUERY LATENCY BY DOMAIN (milliseconds)".center(72))
line()
w = [16, 12, 12, 12, 12, 12]
row(["Domain", "Mean", "P50", "P95", "P99", "Max"], w)
line("-")
for key, label in DOMAINS:
    d = r[key]["edited_overall"]
    row([label, f"{d['mean_ms']:.1f}", f"{d['p50_ms']:.1f}", f"{d['p95_ms']:.1f}", f"{d['p99_ms']:.1f}", f"{d['max_ms']:.1f}"], w)
line("-")

# pooled
all_n = sum(r[k]["duplicate"]["n"] + r[k]["edited_overall"]["n"] for k, _ in DOMAINS)
all_lat = sum(
    r[k]["duplicate"]["n"] * r[k]["duplicate"]["mean_ms"] + r[k]["edited_overall"]["n"] * r[k]["edited_overall"]["mean_ms"]
    for k, _ in DOMAINS
)
print()
print(f"Pooled average query time (duplicate + edited, all {all_n} queries): {all_lat/all_n:.1f} ms")
line()
