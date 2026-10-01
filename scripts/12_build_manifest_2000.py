"""Builds the 2000-track manifest for the scaled-up primary benchmark:
music=200 (its full available count), speech=900, environment=900
(900+900+200=2000). Speech and Environmental are split evenly since neither
dataset is capacity-constrained (8396 and 10231 files available respectively)
and no other distribution was specified.

Mirrors build_manifest_333.py's approach exactly, just with different counts.
Writes results/manifest.json (the caller is responsible for backing up the
previous manifest first, since this overwrites it).
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sonic.config import MANIFEST_PATH, N_HELD_OUT_NEGATIVE, N_HELD_OUT_NOISE
from sonic.eval.manifest import _file_entry, _list_sorted

COUNTS = {"speech": 900, "music": 200, "environment": 900}

manifest = {"categories": {}, "held_out_noise": {}, "held_out_negative": {}}
global_id = 0

for category, n in COUNTS.items():
    dir_path, files = _list_sorted(category)
    if len(files) < n:
        raise ValueError(f"{category}: only {len(files)} files found, need {n}")
    entries = []
    for name in files[:n]:
        entries.append(_file_entry(category, dir_path, name, global_id))
        global_id += 1
    manifest["categories"][category] = entries

env_dir, env_files = _list_sorted("environment")
noise_names = env_files[COUNTS["environment"]:COUNTS["environment"] + N_HELD_OUT_NOISE]
noise_entries = []
for name in noise_names:
    noise_entries.append(_file_entry("environment_noise_bed", env_dir, name, global_id))
    global_id += 1
manifest["held_out_noise"]["environment"] = noise_entries

for category, offset in [("speech", COUNTS["speech"]), ("environment", COUNTS["environment"] + N_HELD_OUT_NOISE)]:
    dir_path, files = _list_sorted(category)
    neg_entries = []
    for name in files[offset:offset + N_HELD_OUT_NEGATIVE]:
        neg_entries.append(_file_entry(f"{category}_negative", dir_path, name, global_id))
        global_id += 1
    manifest["held_out_negative"][category] = neg_entries

manifest["meta"] = {
    "n_per_category": COUNTS,
    "n_held_out_noise": N_HELD_OUT_NOISE,
    "n_held_out_negative": N_HELD_OUT_NEGATIVE,
    "total_indexed_files": sum(len(v) for v in manifest["categories"].values()),
}

MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(MANIFEST_PATH, "w") as f:
    json.dump(manifest, f, indent=2)

for cat, entries in manifest["categories"].items():
    print(f"{cat}: {len(entries)} files")
print(f"TOTAL indexed: {manifest['meta']['total_indexed_files']}")
print(f"manifest written to {MANIFEST_PATH}")
