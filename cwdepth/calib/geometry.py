"""Depth accuracy of trained sparse-view models at the held-out views.

Each finished run of a benchmark is rendered at the test cameras and its depth
is compared with the depth of that scene's dense reference model (the
baseline trained on every non-test view). Both models live in the same COLMAP
frame, so no scale or shift is fitted: a model whose surfaces sit in the wrong
place is penalised.

Depth is the alpha-normalised composite depth. Pixels count where the
reference is opaque (accumulated opacity > MIN_ALPHA) and the evaluated model
covers the pixel at all (> MIN_MODEL_ALPHA).

    python -m cwdepth.calib.geometry configs/llff3.yaml --reference llff-ref
    python -m cwdepth.calib.geometry configs/statues12.yaml --reference statues-ref

Writes results/<benchmark>/geometry.csv (one row per run) and geometry.md.
"""

from __future__ import annotations

import argparse
import csv
import sys
from argparse import Namespace
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[2]
MIN_ALPHA = 0.99
MIN_MODEL_ALPHA = 0.5
ITERATION = 10000


def depth_errors(depth: torch.Tensor, reference: torch.Tensor) -> dict:
    """AbsRel and threshold accuracy of ``depth`` against ``reference`` (1-D, positive)."""
    if depth.shape != reference.shape or depth.dim() != 1 or depth.numel() == 0:
        raise ValueError("depth and reference must be non-empty 1-D tensors of equal length")
    ref = reference.clamp_min(1e-6)
    est = depth.clamp_min(1e-6)
    ratio = torch.maximum(est / ref, ref / est)
    return {"absrel": float(((est - ref).abs() / ref).mean()),
            "delta1": float((ratio < 1.25).float().mean()),
            "delta05": float((ratio < 1.05).float().mean())}


def normalised_depth(out: dict) -> tuple[torch.Tensor, torch.Tensor]:
    alpha = out["alpha"]
    return out["depth"] / alpha.clamp_min(1e-6), alpha


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config", type=Path)
    parser.add_argument("--reference", required=True, help="benchmark name of the dense reference runs")
    args = parser.parse_args()

    sys.path.insert(0, str(REPO_ROOT / "third_party" / "DropGaussian"))
    from scene import GaussianModel, Scene
    from cwdepth.render import render_with_depth

    cfg = yaml.safe_load(args.config.read_text())
    bench = cfg["benchmark"]
    runs_by_scene = defaultdict(list)
    for metrics in sorted((REPO_ROOT / "outputs" / bench).glob("*/*/seed*/metrics.json")):
        run_dir = metrics.parent
        runs_by_scene[run_dir.parts[-2]].append(run_dir)
    pipe = Namespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
    background = torch.zeros(3, device="cuda")
    rows = []
    for scene_name in sorted(runs_by_scene):
        ref_run = REPO_ROOT / "outputs" / args.reference / "V0" / scene_name / "seed0"
        if not (ref_run / "metrics.json").exists():
            print(f"{scene_name}: no reference model yet, skipped")
            continue
        scene_args = Namespace(
            source_path=str(REPO_ROOT / cfg["data_root"] / scene_name), model_path=str(ref_run), images="images",
            resolution=cfg["train_args"]["resolution"], white_background=False, data_device="cuda", eval=True,
            n_views=cfg["train_args"]["n_views"], rand_pcd=False, sh_degree=3)
        ref_gaussians = GaussianModel(3)
        scene = Scene(scene_args, ref_gaussians, load_iteration=ITERATION, shuffle=False)
        cameras = sorted(scene.getTestCameras(), key=lambda c: c.image_name)
        with torch.no_grad():
            references = []
            for cam in cameras:
                depth, alpha = normalised_depth(render_with_depth(cam, ref_gaussians, pipe, background))
                references.append((depth, alpha > MIN_ALPHA))
            for run_dir in runs_by_scene[scene_name]:
                gaussians = GaussianModel(3)
                gaussians.load_ply(str(run_dir / "point_cloud" / f"iteration_{ITERATION}" / "point_cloud.ply"))
                per_view = []
                for cam, (ref_depth, ref_valid) in zip(cameras, references):
                    depth, alpha = normalised_depth(render_with_depth(cam, gaussians, pipe, background))
                    valid = ref_valid & (alpha > MIN_MODEL_ALPHA)
                    errors = depth_errors(depth[valid], ref_depth[valid])
                    errors["coverage"] = float(valid.sum() / ref_valid.sum().clamp_min(1))
                    per_view.append(errors)
                row = {"variant": run_dir.parts[-3], "scene": scene_name, "seed": int(run_dir.name[4:])}
                row.update({k: float(np.mean([v[k] for v in per_view])) for k in per_view[0]})
                rows.append(row)
                print(f"{row['variant']} {scene_name} seed{row['seed']}: AbsRel {row['absrel']:.4f} "
                      f"d1 {row['delta1']:.3f}", flush=True)
        del scene, ref_gaussians
        torch.cuda.empty_cache()
    if not rows:
        print("nothing evaluated")
        return 1

    out_dir = REPO_ROOT / "results" / bench
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "geometry.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "geometry.md").write_text(summarise(bench, args.reference, rows))
    print((out_dir / "geometry.md").read_text())
    return 0


def summarise(bench: str, reference: str, rows: list[dict]) -> str:
    table = defaultdict(dict)
    for r in rows:
        table[r["variant"]][(r["scene"], r["seed"])] = r
    lines = [f"# {bench}: depth at held-out views against {reference}", "",
             "Mean over all finished runs of each variant (scenes x seeds). Lower AbsRel is better; "
             "d1 and d05 are the fractions of pixels within 25% and 5% of the reference depth.", "",
             "| Variant | Runs | AbsRel | d1 | d05 | Coverage |", "|---|---|---|---|---|---|"]
    for variant in sorted(table):
        values = list(table[variant].values())
        lines.append(f"| {variant} | {len(values)} | " + " | ".join(
            f"{np.mean([v[k] for v in values]):.4f}" for k in ("absrel", "delta1", "delta05", "coverage")) + " |")
    lines += ["", "Paired over common runs; p is a Wilcoxon test over per-scene mean differences.", "",
              "| Comparison | Mean AbsRel diff | p (scenes) | Scenes improved |", "|---|---|---|---|"]
    for a, b in (("V1", "V0"), ("V2", "V0"), ("V2", "V1"), ("V3", "V1"), ("V4", "V1"), ("V2", "V4")):
        if a not in table or b not in table:
            continue
        keys = sorted(set(table[a]) & set(table[b]))
        scenes = sorted({s for s, _ in keys})
        diff = {s: np.mean([table[a][k]["absrel"] - table[b][k]["absrel"] for k in keys if k[0] == s]) for s in scenes}
        d = np.array(list(diff.values()))
        p = float(stats.wilcoxon(d).pvalue) if len(d) >= 6 and np.any(d != 0) else float("nan")
        lines.append(f"| {a} − {b} | {np.mean(d):+.4f} | {p:.4f} | {int(np.sum(d < 0))}/{len(d)} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    sys.exit(main())
