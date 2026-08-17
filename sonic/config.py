"""Shared paths and constants for the SONIC Phase-1 (200-file) pipeline."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = REPO_ROOT / "dataset"
RESULTS_ROOT = REPO_ROOT / "results"

CATEGORIES = ["speech", "music", "environment"]

# Per-domain indexed-file counts. Not a single global N: dataset/music contains
# exactly 200 files total, so it cannot supply more than that regardless of
# what speech/environment use (both have thousands of files available).
N_PER_CATEGORY_BY_DOMAIN = {
    "speech": 333,
    "music": 200,
    "environment": 333,
}
N_PER_CATEGORY = 200  # fallback default for any unlisted domain

N_HELD_OUT_NOISE = 100  # extra environment clips reserved as a real-noise bed, disjoint from the indexed set
N_HELD_OUT_NEGATIVE = 30  # extra clips per domain reserved as "not in the index" queries, for false-positive-rate testing

ANALYSIS_SR = 22050  # common sample rate every file is resampled to before fingerprinting

# Query-time analysis window cap: indexing always processes the FULL file, but a
# query only decodes/fingerprints up to this many seconds. Long-tail recordings
# (speech clips up to ~71s in this dataset) would otherwise dominate P95/P99
# latency; capping the query window is standard practice in landmark-hash audio
# fingerprinting (a short excerpt is enough to get a confident match) and keeps
# worst-case query latency bounded regardless of source file length.
#
# Tuned PER DOMAIN, not globally, because the right trade-off differs by domain:
# - speech: dataset is "spontaneous" (unscripted) speech, where several files'
#   opening seconds are a pause/hesitation before content starts. A small cap
#   (8-10s) measurably hurt duplicate accuracy (~190/200) because those queries
#   never see real content. 20s originally recovered it (200/200) but cost too
#   much latency once the multi-window sub-fingerprint escalation stage (see
#   fingerprint/binary_embed.py) was added; re-tested at 12s after that stage
#   existed and 200/200 duplicate accuracy held with much lower latency (the
#   escalation stage's own redundancy, not a large raw query window, is what
#   was actually carrying accuracy by then). (A leading-silence-skip heuristic
#   was also tried instead of a larger cap, but it introduced new mismatches on
#   files that didn't need it and made overall accuracy worse.)
# - music: every file is a fixed ~30s clip with dense content throughout, so
#   there is no "silent opening" failure mode - self-match is 100% at 8s.
# - environment: highly variable duration (0.3-30s) but content is typically
#   dense from t=0; re-tested down from 12s to 8s after the escalation stage
#   was strengthened and accuracy held (100% duplicate, >=95.5% pitch/speed).
QUERY_MAX_DURATION_SEC_BY_DOMAIN = {
    "speech": 12.0,
    "music": 8.0,
    "environment": 8.0,
}
QUERY_MAX_DURATION_SEC = 8.0  # fallback default for any unlisted domain

MANIFEST_PATH = RESULTS_ROOT / "manifest.json"
