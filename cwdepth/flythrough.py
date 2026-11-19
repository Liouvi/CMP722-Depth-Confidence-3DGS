"""Render a fly-through along the capture path, several trained models side by side.

The path passes through every captured camera of the scene (training and
held-out views), in image-name order, smoothed and resampled. For the statue
walk-arounds this is a full orbit, so most frames show the statue from angles
that were not among the training views.

    python -m cwdepth.flythrough --scene_dir data/statues/statue05 \
        --run "Baseline (V0)=outputs/statues12/V0/statue05/seed0" \
        --run "Depth (V1)=outputs/statues12/V1/statue05/seed0" \
        --out demo/statue05.mp4 --resolution 2 --depth

Each panel is labelled; with ``--depth`` a second row shows each model's
rendered depth (near = bright).
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from argparse import Namespace
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation, Slerp

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "third_party" / "DropGaussian"))

from scene import GaussianModel  # noqa: E402
from scene.cameras import PseudoCamera  # noqa: E402
from scene.dataset_readers import readColmapSceneInfo  # noqa: E402

from cwdepth.render import render_with_depth  # noqa: E402

FFMPEG = Path.home() / "miniforge3/envs/colmap/bin/ffmpeg"
ITERATION = 10000


def camera_path(cam_infos, n_frames: int, smooth: float, loop: bool):
    """Smoothed, resampled (R, T) along the captured cameras, in the baseline's convention."""
    cams = sorted(cam_infos, key=lambda c: c.image_name)
    w2c = Rotation.from_matrix(np.stack([c.R.T for c in cams]))
    centers = np.stack([-c.R @ c.T for c in cams])  # c.R is the transposed world-to-camera rotation
    mode = "wrap" if loop else "nearest"
    centers = gaussian_filter1d(centers, smooth, axis=0, mode=mode)
    quats = w2c.as_quat()
    quats *= np.sign(np.sum(quats * quats[:1], axis=1, keepdims=True))  # one hemisphere before averaging
    quats = gaussian_filter1d(quats, smooth, axis=0, mode=mode)
    rotations = Rotation.from_quat(quats / np.linalg.norm(quats, axis=1, keepdims=True))
    keys = np.arange(len(cams), dtype=float)
    times = np.linspace(0, len(cams) - 1, n_frames)
    slerp = Slerp(keys, rotations)(times)
    path_centers = np.stack([np.interp(times, keys, centers[:, i]) for i in range(3)], axis=1)
    out = []
    for rot, center in zip(slerp, path_centers):
        r_w2c = rot.as_matrix()
        out.append((r_w2c.T, -r_w2c @ center))
    return out, cams[0]


def colourise_depth(depth: np.ndarray, lo: float, hi: float) -> np.ndarray:
    inv = 1.0 / np.clip(depth, lo, hi)
    norm = (inv - 1.0 / hi) / (1.0 / lo - 1.0 / hi)
    return cv2.applyColorMap((np.clip(norm, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)


def label(img: np.ndarray, text: str) -> np.ndarray:
    img = img.copy()
    scale = max(0.4, img.shape[1] / 700)
    thick = max(1, round(scale * 2))
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    cv2.rectangle(img, (0, 0), (tw + 12, th + 14), (0, 0, 0), -1)
    cv2.putText(img, text, (6, th + 6), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thick, cv2.LINE_AA)
    return img


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scene_dir", type=Path, required=True)
    parser.add_argument("--run", action="append", required=True, metavar="LABEL=RUN_DIR")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=2)
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--smooth", type=float, default=2.0, help="Gaussian smoothing of the path, in cameras")
    parser.add_argument("--no_loop", action="store_true", help="path is open (forward-facing scenes)")
    parser.add_argument("--depth", action="store_true")
    parser.add_argument("--stills", type=int, nargs="*", default=[], help="also save these frame indices as PNG")
    args = parser.parse_args()

    runs = []
    for item in args.run:
        name, _, path = item.partition("=")
        ply = Path(path) / "point_cloud" / f"iteration_{ITERATION}" / "point_cloud.ply"
        if not ply.is_file():
            parser.error(f"no trained model at {ply}")
        gaussians = GaussianModel(3)
        gaussians.load_ply(str(ply))
        runs.append((name, gaussians))

    info = readColmapSceneInfo(str(args.scene_dir), "images", eval=False, n_views=0)
    path, ref = camera_path(info.train_cameras, args.frames, args.smooth, loop=not args.no_loop)
    width, height = round(ref.width / args.resolution), round(ref.height / args.resolution)
    width, height = width - width % 2, height - height % 2
    pipe = Namespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
    background = torch.zeros(3, device="cuda")
    lo, hi = np.percentile(np.asarray(ref.bounds), [0, 100]) if ref.bounds is not None else (0.5, 20.0)
    lo, hi = max(lo * 0.8, 1e-3), hi * 1.2

    tmp = Path(tempfile.mkdtemp(prefix="flythrough_", dir=args.out.parent if args.out.parent.exists() else None))
    try:
        with torch.no_grad():
            for index, (R, T) in enumerate(path):
                cam = PseudoCamera(R=R, T=T, FoVx=ref.FovX, FoVy=ref.FovY, width=width, height=height)
                colour_row, depth_row = [], []
                for name, gaussians in runs:
                    out = render_with_depth(cam, gaussians, pipe, background)
                    rgb = (out["render"].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)[:, :, ::-1]
                    colour_row.append(label(rgb, name))
                    if args.depth:
                        d = (out["depth"] / out["alpha"].clamp_min(1e-6)).cpu().numpy()
                        depth_row.append(colourise_depth(d, lo, hi))
                frame = np.hstack(colour_row)
                if args.depth:
                    frame = np.vstack([frame, np.hstack(depth_row)])
                cv2.imwrite(str(tmp / f"{index:05d}.png"), frame)
                if index in args.stills:
                    still = args.out.with_name(f"{args.out.stem}_frame{index:04d}.png")
                    cv2.imwrite(str(still), frame)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([str(FFMPEG), "-v", "error", "-y", "-framerate", str(args.fps), "-i", str(tmp / "%05d.png"),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
                        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", str(args.out)], check=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"wrote {args.out} ({len(path)} frames, {len(runs)} models)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
