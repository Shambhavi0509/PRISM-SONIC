import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sonic.eval.manifest import build_manifest, save_manifest

if __name__ == "__main__":
    m = build_manifest()
    save_manifest(m)
    for cat, entries in m["categories"].items():
        durs = [e["duration_sec"] for e in entries]
        print(f"{cat}: {len(entries)} files, duration min={min(durs):.2f}s "
              f"max={max(durs):.2f}s mean={sum(durs)/len(durs):.2f}s")
    noise = m["held_out_noise"]["environment"]
    print(f"held-out noise bed: {len(noise)} files")
