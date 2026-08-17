import sys, json
sys.path.insert(0, ".")
from sonic.config import DATASET_ROOT, MANIFEST_PATH, N_HELD_OUT_NOISE, N_HELD_OUT_NEGATIVE
from sonic.eval.manifest import _list_sorted, _file_entry

COUNTS = {"speech": 333, "music": 200, "environment": 333}

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

manifest["meta"] = {"n_per_category": COUNTS, "n_held_out_noise": N_HELD_OUT_NOISE, "n_held_out_negative": N_HELD_OUT_NEGATIVE}

MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(MANIFEST_PATH, "w") as f:
    json.dump(manifest, f, indent=2)

for cat, entries in manifest["categories"].items():
    print(f"{cat}: {len(entries)} files")
print(f"manifest written to {MANIFEST_PATH}")