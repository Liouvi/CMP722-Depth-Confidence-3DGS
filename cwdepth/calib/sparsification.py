"""Does the depth model's confidence track its depth error?

For each sparse training view of a scene, the cached depth prior is compared
with a reference depth rendered from a model trained on all non-test views.
The reference is a reconstruction, not ground truth; it is well constrained
where many views overlap and less so near image borders and occlusions.

Outputs per scene (JSON): AUSE, AURG, Spearman correlation, and the
sparsification curves (confidence ordering, oracle ordering).

    python -m cwdepth.calib.sparsification --scene_dir data/llff/fern \
        --reference_run outputs/llff-ref/V0/fern/seed0 \
        --prior_dir data/priors/da3-large/llff/fern --out results/calibration/da3-large/fern.json

Definitions. Pixels are removed in order of increasing confidence and the mean
error of the remaining pixels is recorded (the sparsification curve). The
oracle removes pixels in order of decreasing error. With both curves divided
by the error of the full image:
  AUSE = area between the confidence curve and the oracle curve (0 is perfect)
  AURG = area between the random baseline (constant 1) and the confidence
         curve (positive means confidence is better than random)
"""

from __future__ import annotations

import argparse
import json
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[2]
NUM_STEPS = 100      # points on the sparsification curve
MAX_REMOVED = 0.99   # the curve stops when 1% of the pixels remain
MIN_ALPHA = 0.99     # reference depth is used only where the render is opaque


def align_scale_shift(prior: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Least-squares scale and shift that map ``prior`` onto ``reference`` (1-D tensors)."""
    design = torch.stack([prior, torch.ones_like(prior)], dim=1)
    solution = torch.linalg.lstsq(design, reference[:, None]).solution[:, 0]
    return prior * solution[0] + solution[1]


def sparsification_curve(error: torch.Tensor, order_key: torch.Tensor) -> np.ndarray:
    """Mean error of the pixels that remain as those with the lowest ``order_key`` are removed."""
    if error.shape != order_key.shape or error.dim() != 1:
        raise ValueError("error and order_key must be 1-D tensors of equal length")
    n = error.numel()
    sorted_error = error[torch.argsort(order_key, stable=True)].double()
    # suffix_mean[k] = mean error after removing the k lowest-key pixels
    suffix_sum = torch.flip(torch.cumsum(torch.flip(sorted_error, [0]), 0), [0])
    removed = torch.linspace(0.0, MAX_REMOVED, NUM_STEPS, dtype=torch.float64)
    start = torch.clamp((removed * n).long(), max=n - 1).to(error.device)
    return (suffix_sum[start] / (n - start).double()).cpu().numpy()


def calibration_metrics(error: torch.Tensor, conf: torch.Tensor) -> dict:
    """AUSE, AURG and rank correlation for one image (1-D tensors over valid pixels)."""
    full_error = float(error.double().mean())
    if full_error <= 0.0:
        raise ValueError("mean error is zero; nothing to calibrate against")
    by_conf = sparsification_curve(error, conf) / full_error
    oracle = sparsification_curve(error, -error) / full_error
    x = np.linspace(0.0, MAX_REMOVED, NUM_STEPS)
    spearman = stats.spearmanr(conf.cpu().numpy(), error.cpu().numpy()).statistic
    return {
        "ause": float(np.trapezoid(by_conf - oracle, x)),
        "aurg": float(np.trapezoid(1.0 - by_conf, x)),
        "spearman_conf_vs_error": float(spearman),
        "mean_abs_rel_error": full_error,
        "curve_confidence": by_conf.tolist(),
        "curve_oracle": oracle.tolist(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scene_dir", type=Path, required=True)
    parser.add_argument("--reference_run", type=Path, required=True, help="run trained with --n_views 0")
    parser.add_argument("--prior_dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n_views", type=int, default=3)
    parser.add_argument("--resolution", type=int, default=8)
    parser.add_argument("--iteration", type=int, default=10000)
    args = parser.parse_args()

    sys.path.insert(0, str(REPO_ROOT / "third_party" / "DropGaussian"))
    from scene import GaussianModel, Scene

    from cwdepth.priors.io import load_prior
    from cwdepth.render import render_with_depth

    # Cameras of the sparse setting, Gaussians of the dense reference model.
    scene_args = Namespace(
        source_path=str(args.scene_dir.resolve()), model_path=str(args.reference_run.resolve()), images="images",
        resolution=args.resolution, white_background=False, data_device="cuda", eval=True, n_views=args.n_views,
        rand_pcd=False, sh_degree=3)
    pipe = Namespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
    gaussians = GaussianModel(scene_args.sh_degree)
    scene = Scene(scene_args, gaussians, load_iteration=args.iteration, shuffle=False)
    background = torch.zeros(3, device="cuda")

    per_view = []
    with torch.no_grad():
        for cam in sorted(scene.getTrainCameras(), key=lambda c: c.image_name):
            out = render_with_depth(cam, gaussians, pipe, background)
            valid = out["alpha"] > MIN_ALPHA
            reference = (out["depth"] / out["alpha"].clamp_min(1e-6))[valid]
            prior, conf = load_prior(args.prior_dir, cam.image_name, cam.image_height, cam.image_width)
            aligned = align_scale_shift(prior[valid], reference)
            error = (aligned - reference).abs() / reference.clamp_min(1e-6)
            metrics = calibration_metrics(error, conf[valid])
            metrics.update({"image": cam.image_name, "valid_fraction": float(valid.float().mean())})
            per_view.append(metrics)

    summary = {key: float(np.mean([m[key] for m in per_view]))
               for key in ("ause", "aurg", "spearman_conf_vs_error", "mean_abs_rel_error", "valid_fraction")}
    result = {"scene": args.scene_dir.name, "prior_dir": str(args.prior_dir), "n_views": args.n_views,
              "reference_run": str(args.reference_run), "mean": summary, "per_view": per_view}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps({"scene": result["scene"], **summary}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
