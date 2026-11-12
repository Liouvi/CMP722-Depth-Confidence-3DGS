"""Integration tests for the depth pass, run against a trained baseline scene.

Skipped unless a GPU, the built rasteriser and a finished baseline run exist.
"""

import sys
from argparse import Namespace
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = REPO_ROOT / "outputs" / "llff3" / "V0" / "fern" / "seed0"
ITERATION = 10000

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not (RUN_DIR / "point_cloud" / f"iteration_{ITERATION}" / "point_cloud.ply").exists(),
    reason="needs a GPU and a finished baseline run of fern",
)


@pytest.fixture(scope="module")
def scene_bundle():
    sys.path.insert(0, str(REPO_ROOT / "third_party" / "DropGaussian"))
    pytest.importorskip("diff_gaussian_rasterization")
    from gaussian_renderer import render as baseline_render
    from scene import GaussianModel, Scene

    from cwdepth.render import camera_depths, render_with_depth

    args = Namespace(
        source_path=str(REPO_ROOT / "data" / "llff" / "fern"), model_path=str(RUN_DIR), images="images",
        resolution=8, white_background=False, data_device="cuda", eval=True, n_views=3, rand_pcd=False,
        sh_degree=3)
    pipe = Namespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
    gaussians = GaussianModel(args.sh_degree)
    scene = Scene(args, gaussians, load_iteration=ITERATION, shuffle=False)
    background = torch.zeros(3, device="cuda")
    return {
        "gaussians": gaussians, "pipe": pipe, "background": background,
        "train": scene.getTrainCameras(), "test": scene.getTestCameras(),
        "baseline_render": baseline_render, "render_with_depth": render_with_depth,
        "camera_depths": camera_depths,
    }


def test_colour_matches_baseline_at_test_time(scene_bundle):
    b = scene_bundle
    with torch.no_grad():
        for cam in b["test"][:2]:
            ours = b["render_with_depth"](cam, b["gaussians"], b["pipe"], b["background"])
            ref = b["baseline_render"](cam, b["gaussians"], b["pipe"], b["background"])
            assert torch.equal(ours["render"], ref["render"])
            assert torch.equal(ours["radii"], ref["radii"])


def test_colour_matches_baseline_in_training_mode_with_same_seed(scene_bundle):
    """Same random stream, same dropped Gaussians, same image."""
    b = scene_bundle
    cam = b["train"][0]
    with torch.no_grad():
        torch.manual_seed(123)
        ours = b["render_with_depth"](cam, b["gaussians"], b["pipe"], b["background"], is_train=True, iteration=5000)
        torch.manual_seed(123)
        ref = b["baseline_render"](cam, b["gaussians"], b["pipe"], b["background"], is_train=True, iteration=5000)
        plain = b["baseline_render"](cam, b["gaussians"], b["pipe"], b["background"])
    assert torch.equal(ours["render"], ref["render"])
    assert not torch.equal(ref["render"], plain["render"]), "dropout had no effect; the comparison proves nothing"


def test_depth_and_alpha_are_plausible(scene_bundle):
    b = scene_bundle
    cam = b["test"][0]
    with torch.no_grad():
        out = b["render_with_depth"](cam, b["gaussians"], b["pipe"], b["background"])
        z = b["camera_depths"](b["gaussians"].get_xyz, cam.world_view_transform)
    depth, alpha = out["depth"], out["alpha"]
    assert depth.shape == alpha.shape == (cam.image_height, cam.image_width)
    assert torch.isfinite(depth).all() and torch.isfinite(alpha).all()
    assert alpha.min() >= 0.0 and alpha.max() <= 1.0 + 1e-4
    assert alpha.mean() > 0.9, "a trained forward-facing scene should be almost fully opaque"
    # Composited depth is a convex combination of the depths of visible Gaussians.
    visible = z[out["radii"] > 0]
    solid = alpha > 0.99
    normalised = depth[solid] / alpha[solid]
    assert normalised.min() >= visible.min() - 1e-3
    assert normalised.max() <= visible.max() + 1e-3
    assert normalised.std() > 0.0


def test_depth_gradient_reaches_gaussians_and_screen_points(scene_bundle):
    b = scene_bundle
    cam = b["train"][0]
    gaussians = b["gaussians"]
    out = b["render_with_depth"](cam, gaussians, b["pipe"], b["background"], is_train=True, iteration=5000)
    out["depth"].mean().backward()
    assert gaussians._xyz.grad is not None and gaussians._xyz.grad.abs().sum() > 0
    assert gaussians._opacity.grad is not None and gaussians._opacity.grad.abs().sum() > 0
    assert out["viewspace_points"].grad is not None and out["viewspace_points"].grad.abs().sum() > 0
    gaussians._xyz.grad = None
    gaussians._opacity.grad = None
