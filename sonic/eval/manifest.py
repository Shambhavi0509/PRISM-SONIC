"""Builds the deterministic Phase-1 manifest: first N files (by sorted filename,
count set PER DOMAIN via N_PER_CATEGORY_BY_DOMAIN) per category, plus a disjoint
held-out Environmental pool used as a real-world noise bed for the Background
Noise modification (never used as an index/query file).
"""

import json

import soundfile as sf

from sonic.config import (
    CATEGORIES,
    DATASET_ROOT,
    MANIFEST_PATH,
    N_HELD_OUT_NEGATIVE,
    N_HELD_OUT_NOISE,
    N_PER_CATEGORY,
    N_PER_CATEGORY_BY_DOMAIN,
)


def _list_sorted(category: str):
    d = DATASET_ROOT / category
    files = sorted(p.name for p in d.iterdir() if p.is_file())
    return d, files


def _file_entry(category: str, dir_path, name: str, file_id: int) -> dict:
    path = dir_path / name
    info = sf.info(str(path))
    duration = info.frames / info.samplerate if info.samplerate else 0.0
    return {
        "file_id": file_id,
        "category": category,
        "filename": name,
        "path": str(path),
        "orig_samplerate": info.samplerate,
        "orig_channels": info.channels,
        "duration_sec": round(duration, 4),
    }


def build_manifest() -> dict:
    manifest = {"categories": {}, "held_out_noise": {}, "held_out_negative": {}}
    global_id = 0

    for category in CATEGORIES:
        dir_path, files = _list_sorted(category)
        if len(files) < N_PER_CATEGORY:
            raise ValueError(
                f"{category}: only {len(files)} files found, need {N_PER_CATEGORY}"
            )
        selected = files[:N_PER_CATEGORY]
        entries = []
        for name in selected:
            entries.append(_file_entry(category, dir_path, name, global_id))
            global_id += 1
        manifest["categories"][category] = entries

    # Held-out real-noise pool: environment files immediately after the 200 indexed
    # ones, disjoint from every category's index/query set.
    env_dir, env_files = _list_sorted("environment")
    held_out_names = env_files[N_PER_CATEGORY : N_PER_CATEGORY + N_HELD_OUT_NOISE]
    if len(held_out_names) < N_HELD_OUT_NOISE:
        raise ValueError(
            f"environment: only {len(held_out_names)} held-out noise files available, "
            f"need {N_HELD_OUT_NOISE}"
        )
    noise_entries = []
    for name in held_out_names:
        entries = _file_entry("environment_noise_bed", env_dir, name, global_id)
        noise_entries.append(entries)
        global_id += 1
    manifest["held_out_noise"]["environment"] = noise_entries

    # Held-out "not in the index" pool per domain, used for false-positive-rate
    # testing (query files that have no correct answer in the index - the system
    # should reject/report no-match rather than confidently pick a wrong file).
    # music has exactly N_PER_CATEGORY files total in this dataset, so no spare
    # files exist for it; this is reported as a documented limitation, not faked.
    for category, reserved_offset in [("speech", N_PER_CATEGORY), ("environment", N_PER_CATEGORY + N_HELD_OUT_NOISE)]:
        dir_path, files = _list_sorted(category)
        neg_names = files[reserved_offset : reserved_offset + N_HELD_OUT_NEGATIVE]
        neg_entries = []
        for name in neg_names:
            neg_entries.append(_file_entry(f"{category}_negative", dir_path, name, global_id))
            global_id += 1
        manifest["held_out_negative"][category] = neg_entries

    manifest["meta"] = {
        "n_per_category": N_PER_CATEGORY,
        "n_held_out_noise": N_HELD_OUT_NOISE,
        "n_held_out_negative": N_HELD_OUT_NEGATIVE,
        "total_indexed_files": sum(len(v) for v in manifest["categories"].values()),
        "music_has_no_negative_pool": "dataset/music has exactly 200 files total; no files remain outside the index for FPR testing",
    }
    return manifest


def save_manifest(manifest: dict) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)


def load_manifest() -> dict:
    with open(MANIFEST_PATH) as f:
        return json.load(f)


if __name__ == "__main__":
    m = build_manifest()
    save_manifest(m)
    for cat, entries in m["categories"].items():
        durs = [e["duration_sec"] for e in entries]
        print(f"{cat}: {len(entries)} files, duration min={min(durs):.2f}s "
              f"max={max(durs):.2f}s mean={sum(durs)/len(durs):.2f}s")
    noise = m["held_out_noise"]["environment"]
    print(f"held-out noise bed: {len(noise)} files")
    print(f"manifest written to {MANIFEST_PATH}")
