"""Ablation study: runs Baselines A/B/C and the Proposed system (see
sonic/eval/ablation.py) over a fixed sample of duplicate + edited queries per
domain, so the report can show exactly where accuracy/latency gains come from.

Sample size is capped (not the full 200/58-condition sweep) since this runs 4
methods x N queries x 3 domains - kept deliberately small to stay a tractable
extra pass on top of the main benchmark, not a second multi-hour run.
"""

import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import soundfile as sf

from sonic.config import RESULTS_ROOT
from sonic.eval.ablation import ABLATION_METHODS
from sonic.retrieval.pipeline import build_domain_index

N_DUPLICATE_SAMPLE = 40
N_EDITED_SAMPLE = 40
RNG_SEED = 42


def main():
    with open(RESULTS_ROOT / "manifest.json") as f:
        manifest = json.load(f)

    modified_manifest_path = RESULTS_ROOT / "modified_queries_manifest.json"
    modified_records_all = []
    if modified_manifest_path.exists():
        with open(modified_manifest_path) as f:
            modified_records_all = json.load(f)["records"]

    rng = random.Random(RNG_SEED)
    results = {}

    for category in ["speech", "music", "environment"]:
        entries = manifest["categories"][category]
        idx = build_domain_index(category, entries)

        dup_sample = rng.sample(entries, min(N_DUPLICATE_SAMPLE, len(entries)))

        cat_mod_records = [r for r in modified_records_all
                            if r["category"] == category and r.get("modified_path")]
        edited_sample = rng.sample(cat_mod_records, min(N_EDITED_SAMPLE, len(cat_mod_records))) \
            if cat_mod_records else []

        for method_name, method_fn in ABLATION_METHODS.items():
            n_correct, n = 0, 0
            latencies = []

            for e in dup_sample:
                out = method_fn(idx, e["path"])
                latencies.append(out["latency_ms"])
                n_correct += int(out["predicted_id"] == e["file_id"])
                n += 1

            for rec in edited_sample:
                y, sr = sf.read(rec["modified_path"], dtype="float32", always_2d=False)
                out = method_fn(idx, y, sr=sr)
                latencies.append(out["latency_ms"])
                n_correct += int(out["predicted_id"] == rec["true_file_id"])
                n += 1

            latencies.sort()
            results.setdefault(category, {})[method_name] = {
                "n": n,
                "accuracy": n_correct / n if n else None,
                "mean_latency_ms": sum(latencies) / len(latencies) if latencies else None,
                "p95_latency_ms": latencies[int(len(latencies) * 0.95)] if latencies else None,
            }
            print(f"[{category}] {method_name}: acc={results[category][method_name]['accuracy']:.3f} "
                  f"mean_lat={results[category][method_name]['mean_latency_ms']:.1f}ms", flush=True)

    with open(RESULTS_ROOT / "ablation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Ablation results written to {RESULTS_ROOT / 'ablation_results.json'}")


if __name__ == "__main__":
    main()
