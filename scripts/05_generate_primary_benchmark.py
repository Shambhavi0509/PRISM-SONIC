"""Generates modified queries for the EXACT, fixed-severity primary robustness
benchmark the user specified (see sonic/augment/modifications.py's
PRIMARY_BENCHMARK_MODIFICATIONS): 10 modification types, each at one fixed
mild severity (or its +-X two signed directions), run on ALL 200 indexed files
per domain - not the broader exploratory severity sweep used earlier for
diagnosis (results/modified_audio/, results/modified_queries_manifest.json),
which is left untouched.

Output: results/primary_benchmark_audio/<category>/<mod>/<level>/<file_id>.wav
        results/primary_benchmark_manifest.json
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import soundfile as sf

from sonic.augment.modifications import PRIMARY_BENCHMARK_MODIFICATIONS as MODS
from sonic.config import ANALYSIS_SR, RESULTS_ROOT
from sonic.io import load_audio

OUT_DIR = RESULTS_ROOT / "primary_benchmark_audio"
OUT_MANIFEST = RESULTS_ROOT / "primary_benchmark_manifest.json"


def safe_level_name(level) -> str:
    s = str(level)
    return (s.replace(" ", "_").replace("/", "-").replace("(", "")
            .replace(")", "").replace(",", "_").replace("'", ""))


def main():
    with open(RESULTS_ROOT / "manifest.json") as f:
        manifest = json.load(f)

    noise_pool = manifest["held_out_noise"]["environment"]
    noise_ys = [load_audio(e["path"]) for e in noise_pool[:20]]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    t_start = time.perf_counter()

    for category in ["speech", "music", "environment"]:
        entries = manifest["categories"][category]  # ALL 200 files

        for mod_name, spec in MODS.items():
            fn = spec["fn"]
            mod_dir = OUT_DIR / category / mod_name
            mod_dir.mkdir(parents=True, exist_ok=True)

            for level in spec["levels"]:
                safe = safe_level_name(level)
                out_level_dir = mod_dir / safe
                out_level_dir.mkdir(parents=True, exist_ok=True)

                n_ok, n_fail = 0, 0
                for e in entries:
                    try:
                        y = load_audio(e["path"])
                        if spec.get("needs_noise"):
                            noise_y = noise_ys[e["file_id"] % len(noise_ys)]
                            y_mod = fn(y, ANALYSIS_SR, level, noise_y)
                        else:
                            y_mod = fn(y, ANALYSIS_SR, level)

                        out_file = out_level_dir / f"{e['file_id']}.wav"
                        sf.write(str(out_file), y_mod, ANALYSIS_SR)

                        records.append({
                            "category": category, "modification": mod_name, "level": str(level),
                            "unit": spec["unit"], "true_file_id": e["file_id"],
                            "orig_filename": e["filename"], "modified_path": str(out_file),
                        })
                        n_ok += 1
                    except Exception as ex:
                        n_fail += 1
                        records.append({
                            "category": category, "modification": mod_name, "level": str(level),
                            "unit": spec["unit"], "true_file_id": e["file_id"],
                            "orig_filename": e["filename"], "modified_path": None, "error": str(ex),
                        })
                print(f"{category}/{mod_name}/{level}: ok={n_ok} fail={n_fail}", flush=True)

        with open(OUT_MANIFEST, "w") as f:
            json.dump({"records": records}, f, indent=2)
        print(f"[{category}] done, checkpoint saved ({len(records)} total records).", flush=True)

    print(f"Total: {len(records)} records in {time.perf_counter()-t_start:.1f}s", flush=True)
    print(f"Manifest written to {OUT_MANIFEST}", flush=True)


if __name__ == "__main__":
    main()
