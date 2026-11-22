"""Export the all-view "art" models as upright, centred .splat files to share.

3DGS models live in COLMAP's frame, where "up" is roughly -y, so most viewers
(SuperSplat, PlayCanvas, three.js; all y-up) show them upside down and far
from the origin. Each model is rotated so the mean camera up becomes +y and
translated so the point the cameras look at (least-squares intersection of
their viewing rays: the statue) is the origin. Positions and Gaussian
orientations are rotated; scales are unchanged.

The .splat layout keeps degree-0 colour only (no view-dependent colour), at
32 bytes per Gaussian instead of about 248 in the .ply.

    python -m cwdepth.export_splat --out demo/splats [statue01 ...]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cwdepth.viewer import OUTPUTS, pack_splat, ply_gaussians

ART = OUTPUTS / "statues-art"


def quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product a * b of (w, x, y, z) quaternions; a or b may be a single quaternion."""
    aw, ax, ay, az = np.moveaxis(np.broadcast_to(a, np.broadcast_shapes(a.shape, b.shape)), -1, 0)
    bw, bx, by, bz = np.moveaxis(np.broadcast_to(b, np.broadcast_shapes(a.shape, b.shape)), -1, 0)
    return np.stack([aw * bw - ax * bx - ay * by - az * bz,
                     aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw], axis=-1)


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def rotation_to_y(up: np.ndarray) -> np.ndarray:
    """Quaternion of the smallest rotation taking unit vector ``up`` to +y."""
    up = up / np.linalg.norm(up)
    target = np.array([0.0, 1.0, 0.0])
    axis = np.cross(up, target)
    sin, cos = np.linalg.norm(axis), float(up @ target)
    if sin < 1e-9:  # already +y, or exactly -y: turn half way round the x axis
        return np.array([1.0, 0.0, 0.0, 0.0]) if cos > 0 else np.array([0.0, 1.0, 0.0, 0.0])
    angle = np.arctan2(sin, cos)
    return np.concatenate([[np.cos(angle / 2)], np.sin(angle / 2) * axis / sin])


def scene_frame(cameras: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Rotation quaternion (mean camera up -> +y) and look-at centre, from 3DGS cameras.json entries."""
    rot = np.array([c["rotation"] for c in cameras])       # camera-to-world; columns are the camera axes
    pos = np.array([c["position"] for c in cameras])
    forward, up = rot[:, :, 2], -rot[:, :, 1]
    # Point nearest all viewing rays: sum (I - f f^T) p = sum (I - f f^T) c.
    proj = np.eye(3)[None] - forward[:, :, None] * forward[:, None, :]
    centre = np.linalg.solve(proj.sum(axis=0), np.einsum("nij,nj->i", proj, pos))
    return rotation_to_y(up.mean(axis=0)), centre


def export(statue: str, out_dir: Path) -> Path:
    run = ART / statue
    ply = run / "point_cloud" / "iteration_30000" / "point_cloud.ply"
    q, centre = scene_frame(json.loads((run / "cameras.json").read_text()))
    g = ply_gaussians(ply)
    g["xyz"] = (g["xyz"] - centre) @ quat_to_matrix(q).T
    g["rot"] = quat_multiply(q, g["rot"])
    path = out_dir / f"{statue}.splat"
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(pack_splat(g))
    tmp.replace(path)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("statues", nargs="*", help="default: every finished art model")
    parser.add_argument("--out", type=Path, default=Path("demo/splats"))
    args = parser.parse_args()
    statues = args.statues or sorted(p.name for p in ART.glob("statue*") if (p / "run.json").exists())
    args.out.mkdir(parents=True, exist_ok=True)
    for s in statues:
        if not (ART / s / "run.json").exists():
            print(f"{s}: not finished, skipped", flush=True)
            continue
        path = export(s, args.out)
        print(f"{s}: {path} ({path.stat().st_size / 1e6:.0f} MB)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
