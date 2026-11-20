"""Training entry point.

Mirrors the optimisation loop of the DropGaussian baseline
(third_party/DropGaussian/train.py) and adds three things: a settable seed, an
optional depth-prior loss, and machine-readable outputs. With
``--depth_weight 0`` the loop calls the baseline's own renderer and loss, so it
is the baseline.

Example:
    python -m cwdepth.train -s data/llff/fern -m outputs/llff3/V2/fern/seed0 \
        --eval -r 8 --n_views 3 --seed 0 \
        --depth_weight 0.05 --depth_mode rank --prior_dir data/priors/da3-large/llff/fern
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import time
from argparse import ArgumentParser, Namespace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_ROOT = REPO_ROOT / "third_party" / "DropGaussian"
sys.path.insert(0, str(BASELINE_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torchvision  # noqa: E402

from arguments import ModelParams, OptimizationParams, PipelineParams  # noqa: E402
from gaussian_renderer import render as baseline_render  # noqa: E402
from lpipsPyTorch import lpips  # noqa: E402
from scene import GaussianModel, Scene  # noqa: E402
from utils.general_utils import safe_state  # noqa: E402
from utils.image_utils import psnr  # noqa: E402
from utils.loss_utils import l1_loss, ssim  # noqa: E402

from cwdepth.losses.depth import WEIGHT_MODES, make_weights, weighted_pearson_loss  # noqa: E402
from cwdepth.priors.io import load_prior  # noqa: E402
from cwdepth.render import render_with_depth  # noqa: E402

LOG_EVERY = 500


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def git_commit() -> str:
    try:
        out = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10, check=True)
        dirty = subprocess.run(["git", "-C", str(REPO_ROOT), "status", "--porcelain"],
                               capture_output=True, text=True, timeout=10, check=True)
        return out.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")
    except (subprocess.SubprocessError, OSError):
        return "unknown"


def load_priors(cameras, args):
    """Depth prior and loss weights for each training camera, keyed by image name."""
    priors = {}
    for index, cam in enumerate(sorted(cameras, key=lambda c: c.image_name)):
        depth, conf = load_prior(args.prior_dir, cam.image_name, cam.image_height, cam.image_width)
        weights = make_weights(conf, args.depth_mode, keep_fraction=args.depth_keep_fraction,
                               seed=args.seed * 1000 + index)
        priors[cam.image_name] = (depth, weights)
    return priors


@torch.no_grad()
def evaluate(scene, gaussians, pipe, background, out_dir: Path, iteration: int) -> dict:
    """Held-out metrics with the baseline's own functions (PSNR, SSIM, LPIPS-VGG)."""
    render_dir = out_dir / "test" / f"ours_{iteration}" / "renders"
    gt_dir = out_dir / "test" / f"ours_{iteration}" / "gt"
    render_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)
    per_image = []
    cameras = sorted(scene.getTestCameras(), key=lambda c: c.image_name)
    for idx, cam in enumerate(cameras):
        gt = cam.original_image[0:3].cuda()
        image = baseline_render(cam, gaussians, pipe, background)["render"]
        torchvision.utils.save_image(image, render_dir / f"{idx:05d}.png")
        torchvision.utils.save_image(gt, gt_dir / f"{idx:05d}.png")
        per_image.append({
            "image": cam.image_name,
            "psnr": psnr(image[None], gt[None]).mean().item(),
            "ssim": ssim(image[None], gt[None]).item(),
            "lpips": lpips(image[None], gt[None], net_type="vgg").item(),
        })
    if not per_image:
        raise RuntimeError("no test cameras: run with --eval")
    mean = {k: float(np.mean([m[k] for m in per_image])) for k in ("psnr", "ssim", "lpips")}
    return {"iteration": iteration, "lpips_net": "vgg", "num_test_images": len(per_image),
            "mean": mean, "per_image": per_image}


def train(args, dataset, opt, pipe) -> None:
    out_dir = Path(dataset.model_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "cfg_args").write_text(str(Namespace(**vars(args))))

    use_depth = args.depth_weight > 0.0
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    background = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0],
                              dtype=torch.float32, device="cuda")
    priors = load_priors(scene.getTrainCameras(), args) if use_depth else {}

    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    log = []
    viewpoint_stack = None
    for iteration in range(1, opt.iterations + 1):
        gaussians.update_learning_rate(iteration)
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        cam = viewpoint_stack.pop(random.randint(0, len(viewpoint_stack) - 1))
        gt_image = cam.original_image.cuda()
        bg = torch.rand((3), device="cuda") if opt.random_background else background

        depth_active = use_depth and iteration >= args.depth_from_iter
        if depth_active:
            pkg = render_with_depth(cam, gaussians, pipe, bg, is_train=not args.no_drop, iteration=iteration)
        else:
            pkg = baseline_render(cam, gaussians, pipe, bg, is_train=not args.no_drop, iteration=iteration)
        image, viewspace_points, visibility, radii = (
            pkg["render"], pkg["viewspace_points"], pkg["visibility_filter"], pkg["radii"])

        photo_loss = l1_loss(image, gt_image) + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss = photo_loss
        depth_loss_value = float("nan")
        if depth_active:
            prior_depth, weights = priors[cam.image_name]
            depth_loss = weighted_pearson_loss(pkg["depth"], prior_depth, weights)
            depth_loss_value = depth_loss.item()
            loss = loss + args.depth_weight * depth_loss
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at iteration {iteration}")
        loss.backward()

        with torch.no_grad():
            if iteration % LOG_EVERY == 0 or iteration == 1:
                log.append({"iteration": iteration, "photo_loss": photo_loss.item(),
                            "depth_loss": depth_loss_value, "num_gaussians": gaussians.get_xyz.shape[0]})
                print(f"[{iteration:>6}] photo {photo_loss.item():.4f}  depth {depth_loss_value:.4f}  "
                      f"gaussians {gaussians.get_xyz.shape[0]}", flush=True)

            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility] = torch.max(gaussians.max_radii2D[visibility], radii[visibility])
                gaussians.add_densification_stats(viewspace_points, visibility)
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, None)
                if iteration % opt.opacity_reset_interval == 0 or (
                        dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

    train_seconds = time.time() - start
    peak_vram_mb = torch.cuda.max_memory_allocated() / 2**20
    scene.save(opt.iterations)

    if scene.getTestCameras():
        metrics = evaluate(scene, gaussians, pipe, background, out_dir, opt.iterations)
        (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    else:  # trained on every view (no --eval): nothing is held out
        metrics = None
    run = {
        "scene": Path(dataset.source_path).name,
        "seed": args.seed,
        "n_views": dataset.n_views,
        "resolution": dataset.resolution,
        "iterations": opt.iterations,
        "drop": not args.no_drop,
        "depth": {"weight": args.depth_weight, "mode": args.depth_mode, "from_iter": args.depth_from_iter,
                  "keep_fraction": args.depth_keep_fraction, "prior_dir": args.prior_dir},
        "num_gaussians": int(gaussians.get_xyz.shape[0]),
        "train_seconds": train_seconds,
        "peak_vram_mb": peak_vram_mb,
        "git_commit": git_commit(),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "log": log,
    }
    (out_dir / "run.json").write_text(json.dumps(run, indent=2))
    cost = f"({train_seconds / 60:.1f} min, peak {peak_vram_mb:.0f} MB, {run['num_gaussians']} Gaussians)"
    if metrics is None:
        print(f"done: no held-out views {cost}")
    else:
        m = metrics["mean"]
        print(f"done: PSNR {m['psnr']:.3f}  SSIM {m['ssim']:.4f}  LPIPS {m['lpips']:.4f}  {cost}")


def main(argv=None) -> None:
    parser = ArgumentParser(description="Sparse-view 3DGS with an optional confidence-weighted depth loss")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--depth_weight", type=float, default=0.0, help="0 disables the depth loss (baseline)")
    parser.add_argument("--depth_mode", choices=WEIGHT_MODES, default="uniform")
    parser.add_argument("--depth_from_iter", type=int, default=1)
    parser.add_argument("--depth_keep_fraction", type=float, default=0.5, help="only for --depth_mode binary")
    parser.add_argument("--prior_dir", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no_drop", action="store_true",
                        help="disable DropGaussian's random dropping (for dense-view models, whose rate would "
                             "otherwise grow past 0.2 beyond 10k iterations)")
    args = parser.parse_args(argv)

    if not args.model_path:
        parser.error("-m/--model_path is required")
    if args.depth_weight < 0.0:
        parser.error("--depth_weight must be >= 0")
    if args.depth_weight > 0.0 and not args.prior_dir:
        parser.error("--prior_dir is required when --depth_weight > 0")
    if args.depth_from_iter < 1:
        parser.error("--depth_from_iter must be >= 1")
    if args.compute_cov3D_python or args.convert_SHs_python:
        parser.error("the python covariance / SH paths are not supported by the depth pass")
    if any((Path(args.model_path) / f).exists() for f in ("metrics.json", "run.json")) and not args.overwrite:
        parser.error(f"{args.model_path} already holds a finished run; pass --overwrite to redo it")

    # The baseline's safe_state seeds with 0 and wraps stdout; reseed afterwards.
    safe_state(False)
    seed_everything(args.seed)
    os.chdir(REPO_ROOT)
    train(args, lp.extract(args), op.extract(args), pp.extract(args))


if __name__ == "__main__":
    main()
