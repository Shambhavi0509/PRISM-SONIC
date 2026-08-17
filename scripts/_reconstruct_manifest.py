"""One-off: rebuild modified_queries_manifest.json records for categories whose
WAV files already exist on disk but whose manifest was never written (process
was interrupted before scripts/02's final/checkpoint save existed)."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sonic.augment import modifications as mods
from sonic.config import RESULTS_ROOT

OUT_DIR = RESULTS_ROOT / "modified_audio"


def safe_level_name(level) -> str:
    level_key = str(level)
    return (level_key.replace(" ", "_").replace("/", "-")
            .replace("(", "").replace(")", "")
            .replace(",", "_").replace("'", ""))


def main():
    with open(RESULTS_ROOT / "manifest.json") as f:
        manifest = json.load(f)

    records = []
    for category in ["speech", "music"]:
        by_id = {e["file_id"]: e for e in manifest["categories"][category]}
        for mod_name, spec in mods.MODIFICATIONS.items():
            for level in spec["levels"]:
                safe = safe_level_name(level)
                d = OUT_DIR / category / mod_name / safe
                if not d.exists():
                    continue
                for wav_path in sorted(d.glob("*.wav")):
                    fid = int(wav_path.stem)
                    entry = by_id.get(fid)
                    if entry is None:
                        continue
                    records.append({
                        "category": category, "modification": mod_name, "level": str(level),
                        "unit": spec["unit"], "true_file_id": fid,
                        "orig_filename": entry["filename"], "modified_path": str(wav_path),
                        "n_per_condition": 50,
                    })
        print(f"{category}: reconstructed {sum(1 for r in records if r['category']==category)} records")

    with open(RESULTS_ROOT / "modified_queries_manifest.json", "w") as f:
        json.dump({"records": records}, f, indent=2)
    print(f"Total {len(records)} records written.")


if __name__ == "__main__":
    main()
