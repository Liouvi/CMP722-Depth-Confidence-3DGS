"""Rendering with an extra depth pass.

The baseline rasteriser returns colour only. Depth is obtained with a second
call to the same rasteriser in which each Gaussian's "colour" is its depth in
the camera frame, so no CUDA code is changed. The result is the alpha-composited
depth  sum_i T_i * alpha_i * z_i  (not divided by accumulated opacity), which is
the convention of FSGS-style depth losses. A third channel carries accumulated
opacity for analysis.

Both passes share one DropGaussian mask, so colour and depth describe the same
set of Gaussians, and one screen-space tensor, so the depth loss contributes to
the densification statistics exactly as it would in a single-pass rasteriser.
"""

from __future__ import annotations

import math

import torch
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

DROP_GAMMA = 0.2          # DropGaussian paper default
DROP_TOTAL_ITERS = 10000  # denominator hard-coded in the baseline renderer


def camera_depths(xyz: torch.Tensor, world_view_transform: torch.Tensor) -> torch.Tensor:
    """Depth of each point along the camera's viewing axis.

    ``world_view_transform`` is stored transposed (row-vector convention), as
    in the baseline's ``Camera`` class.
    """
    ones = torch.ones_like(xyz[:, :1])
    return (torch.cat([xyz, ones], dim=1) @ world_view_transform)[:, 2]


def _rasterizer(cam, pc, pipe, bg, scaling_modifier):
    settings = GaussianRasterizationSettings(
        image_height=int(cam.image_height),
        image_width=int(cam.image_width),
        tanfovx=math.tan(cam.FoVx * 0.5),
        tanfovy=math.tan(cam.FoVy * 0.5),
        bg=bg,
        scale_modifier=scaling_modifier,
        viewmatrix=cam.world_view_transform,
        projmatrix=cam.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=cam.camera_center,
        prefiltered=False,
        debug=pipe.debug,
    )
    return GaussianRasterizer(raster_settings=settings)


def render_with_depth(cam, pc, pipe, bg_color, is_train=False, iteration=None, scaling_modifier=1.0):
    """Colour pass identical to the baseline's ``render``, plus depth and opacity.

    Returns the baseline's dictionary with two extra keys: ``depth`` (H, W) and
    ``alpha`` (H, W).
    """
    if is_train and iteration is None:
        raise ValueError("iteration is required when is_train=True")

    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    screenspace_points.retain_grad()

    means3D = pc.get_xyz
    opacity = pc.get_opacity
    scales = pc.get_scaling
    rotations = pc.get_rotation

    if is_train:
        # Same operation as the baseline: random drop with rescaling of the kept Gaussians.
        keep = torch.ones(opacity.shape[0], dtype=torch.float32, device="cuda")
        drop_rate = DROP_GAMMA * (iteration / DROP_TOTAL_ITERS)
        keep = torch.nn.functional.dropout(keep, p=drop_rate, training=True)
        opacity = opacity * keep[:, None]

    rasterizer = _rasterizer(cam, pc, pipe, bg_color, scaling_modifier)
    image, radii = rasterizer(
        means3D=means3D, means2D=screenspace_points, shs=pc.get_features, colors_precomp=None,
        opacities=opacity, scales=scales, rotations=rotations, cov3D_precomp=None)

    z = camera_depths(means3D, cam.world_view_transform)
    geom = torch.stack([z, torch.ones_like(z), torch.zeros_like(z)], dim=1)
    geom_rasterizer = _rasterizer(cam, pc, pipe, torch.zeros_like(bg_color), scaling_modifier)
    geom_image, _ = geom_rasterizer(
        means3D=means3D, means2D=screenspace_points, shs=None, colors_precomp=geom,
        opacities=opacity, scales=scales, rotations=rotations, cov3D_precomp=None)

    return {
        "render": image.clamp(0, 1),
        "viewspace_points": screenspace_points,
        "visibility_filter": (radii > 0).nonzero(),
        "radii": radii,
        "depth": geom_image[0],
        "alpha": geom_image[1],
    }
