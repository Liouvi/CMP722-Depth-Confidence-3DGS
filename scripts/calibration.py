"""Run the confidence-calibration analysis for every scene of a benchmark.

For each scene with a reference model, calls cwdepth.calib.sparsification on
the sparse training views and writes results/calibration/<benchmark>/<scene>.json,
then a summary table results/calibration/<benchmark>.md.

    python scripts/calibration.py configs/llff3.yaml --reference llff-ref
    python scripts/calibration.py configs/statues12.yaml --reference statues-ref
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
KEYS = ("ause", "aurg", "spearman_conf_vs_error", "mean_abs_rel_error", "valid_fraction")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config", type=Path)
    parser.add_argument("--reference", required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    bench = cfg["benchmark"]
    out_dir = REPO_ROOT / "results" / "calibration" / bench
    results = []
    for scene in cfg["scenes"]:
        ref_run = REPO_ROOT / "outputs" / args.reference / "V0" / scene / "seed0"
        out = out_dir / f"{scene}.json"
        if not (ref_run / "metrics.json").exists():
            print(f"{scene}: no reference model, skipped")
            continue
        if not out.exists():
            cmd = [sys.executable, "-m", "cwdepth.calib.sparsification",
                   "--scene_dir", str(REPO_ROOT / cfg["data_root"] / scene), "--reference_run", str(ref_run),
                   "--prior_dir", str(REPO_ROOT / cfg["prior_root"] / scene), "--out", str(out),
                   "--n_views", str(cfg["train_args"]["n_views"]),
                   "--resolution", str(cfg["train_args"]["resolution"])]
            result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"{scene}: FAILED\n{result.stdout[-2000:]}\n{result.stderr[-2000:]}")
                continue
        results.append(json.loads(out.read_text()))
        m = results[-1]["mean"]
        print(f"{scene}: AUSE {m['ause']:.3f} AURG {m['aurg']:+.3f} Spearman {m['spearman_conf_vs_error']:+.3f}",
              flush=True)
    if not results:
        return 1

    lines = [f"# Confidence calibration: {bench} (reference: {args.reference})", "",
             "Per training view, the Depth Anything 3 depth is aligned to the reference depth by least-squares "
             "scale and shift; error is absolute relative error. AUSE: area between the confidence and oracle "
             "sparsification curves (0 is perfect). AURG: area between random removal and the confidence curve "
             "(> 0 means confidence ranks errors better than chance). Spearman: rank correlation of confidence "
             "with error (negative is the desired sign).", "",
             "| Scene | AUSE | AURG | Spearman | AbsRel of prior | Valid px |", "|---|---|---|---|---|---|"]
    for r in results:
        m = r["mean"]
        lines.append(f"| {r['scene']} | " + " | ".join(f"{m[k]:.3f}" for k in KEYS) + " |")
    means = {k: np.mean([r["mean"][k] for r in results]) for k in KEYS}
    lines.append("| **mean** | " + " | ".join(f"**{means[k]:.3f}**" for k in KEYS) + " |")
    positive = sum(r["mean"]["aurg"] > 0 for r in results)
    lines += ["", f"AURG > 0 in {positive}/{len(results)} scenes."]
    (out_dir.parent / f"{bench}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
