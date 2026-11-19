"""Collect finished runs into results/<benchmark>/ and build the tables.

Reads only metrics.json and run.json written by cwdepth.train. Writes:
  runs.csv       one row per run (scene, variant, seed, metrics, cost)
  summary.md     per-variant mean +- std over seeds, per-scene PSNR, paired tests

    python scripts/aggregate.py llff3 --baseline V0 --proposed V2 --plain V1
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]
METRICS = ("psnr", "ssim", "lpips")
HIGHER_IS_BETTER = {"psnr": True, "ssim": True, "lpips": False}


def collect(root: Path) -> list[dict]:
    rows = []
    for metrics_path in sorted(root.glob("*/*/seed*/metrics.json")):
        run_dir = metrics_path.parent
        metrics = json.loads(metrics_path.read_text())
        run = json.loads((run_dir / "run.json").read_text())
        rows.append({
            "variant": run_dir.parts[-3], "scene": run_dir.parts[-2], "seed": run["seed"],
            **{m: metrics["mean"][m] for m in METRICS},
            "num_gaussians": run["num_gaussians"], "train_minutes": run["train_seconds"] / 60,
            "peak_vram_mb": run["peak_vram_mb"], "depth_weight": run["depth"]["weight"],
            "depth_mode": run["depth"]["mode"], "git_commit": run["git_commit"],
        })
    return rows


def by_variant(rows):
    table = defaultdict(dict)
    for r in rows:
        table[r["variant"]][(r["scene"], r["seed"])] = r
    return table


def seed_means(runs: dict, metric: str, scenes) -> np.ndarray:
    """Scene-averaged metric for each seed that has every scene."""
    seeds = sorted({seed for _, seed in runs})
    out = []
    for seed in seeds:
        if all((scene, seed) in runs for scene in scenes):
            out.append(np.mean([runs[(scene, seed)][metric] for scene in scenes]))
    return np.array(out)


def paired(a_runs: dict, b_runs: dict, metric: str) -> dict:
    """Paired comparison of variant a against b on their common (scene, seed) runs."""
    keys = sorted(set(a_runs) & set(b_runs))
    diff = np.array([a_runs[k][metric] - b_runs[k][metric] for k in keys])
    scenes = sorted({scene for scene, _ in keys})
    scene_diff = np.array([np.mean([d for d, k in zip(diff, keys) if k[0] == scene]) for scene in scenes])
    out = {"n_runs": len(keys), "n_scenes": len(scenes), "mean_diff": float(diff.mean()) if len(keys) else float("nan")}
    out["p_runs"] = float(stats.wilcoxon(diff).pvalue) if len(keys) >= 6 and np.any(diff != 0) else float("nan")
    out["p_scenes"] = (float(stats.wilcoxon(scene_diff).pvalue)
                       if len(scenes) >= 6 and np.any(scene_diff != 0) else float("nan"))
    wins = int(np.sum(scene_diff > 0)) if HIGHER_IS_BETTER[metric] else int(np.sum(scene_diff < 0))
    out["scenes_improved"] = wins
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("benchmark")
    parser.add_argument("--baseline", default="V0")
    parser.add_argument("--plain", default="V1")
    parser.add_argument("--proposed", default="V2")
    args = parser.parse_args()

    rows = collect(REPO_ROOT / "outputs" / args.benchmark)
    if not rows:
        print(f"no finished runs under outputs/{args.benchmark}")
        return 1
    out_dir = REPO_ROOT / "results" / args.benchmark
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "runs.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    table = by_variant(rows)
    scenes = sorted({r["scene"] for r in rows})
    lines = [f"# {args.benchmark}", "",
             f"{len(rows)} runs, {len(scenes)} scenes. Mean and standard deviation are over seeds of the "
             "scene-averaged metric; a seed counts only if all scenes finished.", "",
             "| Variant | Seeds | PSNR | SSIM | LPIPS | Gaussians | Train min |", "|---|---|---|---|---|---|---|"]
    for variant in sorted(table):
        runs = table[variant]
        cells = []
        n_seeds = 0
        for metric in METRICS:
            values = seed_means(runs, metric, scenes)
            n_seeds = len(values)
            if n_seeds == 0:
                cells.append("incomplete")
            elif n_seeds == 1:
                cells.append(f"{values[0]:.3f}")
            else:
                cells.append(f"{values.mean():.3f} ± {values.std(ddof=1):.3f}")
        gaussians = np.mean([r["num_gaussians"] for r in runs.values()])
        minutes = np.mean([r["train_minutes"] for r in runs.values()])
        lines.append(f"| {variant} | {n_seeds} | {cells[0]} | {cells[1]} | {cells[2]} | {gaussians:,.0f} | {minutes:.1f} |")

    lines += ["", "## PSNR per scene (mean over seeds)", "",
              "| Variant | " + " | ".join(scenes) + " |", "|---|" + "---|" * len(scenes)]
    for variant in sorted(table):
        cells = []
        for scene in scenes:
            values = [r["psnr"] for (s, _), r in table[variant].items() if s == scene]
            cells.append(f"{np.mean(values):.2f}" if values else "–")
        lines.append(f"| {variant} | " + " | ".join(cells) + " |")

    lines += ["", "## Paired comparisons", "",
              "Difference is first minus second. `p (runs)` is a Wilcoxon signed-rank test over scene × seed "
              "pairs; `p (scenes)` uses one seed-averaged difference per scene and is the conservative one, "
              "because seeds of the same scene are not independent.", "",
              "| Comparison | Metric | Mean diff | p (runs) | p (scenes) | Scenes improved |", "|---|---|---|---|---|---|"]
    # The last three are the controls: does the placement of the weights matter
    # (V2 against shuffled V4), and do hard masking or random weights differ from plain V1?
    comparisons = [(args.proposed, args.baseline), (args.proposed, args.plain), (args.plain, args.baseline),
                   (args.proposed, "V4"), ("V3", args.plain), ("V4", args.plain)]
    for a, b in comparisons:
        if a not in table or b not in table:
            continue
        for metric in METRICS:
            p = paired(table[a], table[b], metric)
            lines.append(f"| {a} − {b} | {metric.upper()} | {p['mean_diff']:+.4f} | {p['p_runs']:.4f} | "
                         f"{p['p_scenes']:.4f} | {p['scenes_improved']}/{p['n_scenes']} |")

    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
