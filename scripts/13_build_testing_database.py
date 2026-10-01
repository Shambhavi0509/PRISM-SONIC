"""Builds 'Testing Output Database/' - a 600-file (200/200/200) copy of real
original files from the supplied dataset, used as the searchable database for
the user-facing demo (scripts/15_audio_matching_system.py).

This script does NOT touch the SONIC architecture (sonic/*) at all - it only
selects and copies files. Selection is deterministic (sorted filename, first
200 per domain), matching the same convention already used for the project's
Phase-1 manifest (see sonic/eval/manifest.py / build_manifest_333.py), so the
same 200 files are chosen every time this is re-run.

Originals under dataset/ are never modified or deleted - only copied.
"""

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = REPO_ROOT / "dataset"
OUT_ROOT = REPO_ROOT / "Testing Output Database"

N_PER_DOMAIN = 200
DOMAINS = [("Speech", "speech"), ("Music", "music"), ("Environment", "environment")]


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    summary = {}

    for display_name, category in DOMAINS:
        src_dir = DATASET_ROOT / category
        dst_dir = OUT_ROOT / display_name
        dst_dir.mkdir(parents=True, exist_ok=True)

        files = sorted(p.name for p in src_dir.iterdir() if p.is_file())
        if len(files) < N_PER_DOMAIN:
            raise ValueError(f"{category}: only {len(files)} files available, need {N_PER_DOMAIN}")
        selected = files[:N_PER_DOMAIN]

        copied = 0
        for name in selected:
            src = src_dir / name
            dst = dst_dir / name
            if not dst.exists():
                shutil.copy2(src, dst)
            copied += 1
        summary[display_name] = copied
        print(f"{display_name}: {copied} files copied to {dst_dir}")

    total = sum(summary.values())
    print(f"\nTOTAL files in Testing Output Database: {total}")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    assert total == 600, f"expected exactly 600 files, got {total}"
    print("\nOK - exactly 600 files confirmed.")


if __name__ == "__main__":
    main()
