"""Turn the Hacettepe Statues phone videos into scenes the baseline loader reads.

Each video is one walk around one statue (1080x1920 portrait, about 40 s),
shot on a Samsung Galaxy S26 Ultra at FHD 60 fps, 1x zoom, without HDR, flash
or motion compensation. With stabilisation off and the zoom fixed, one camera
model per clip is a sound assumption.
For each video this script:

1. samples ``--frames`` frames: the clip is cut into that many equal time
   windows and the sharpest frame (variance of the Laplacian) of each window
   is kept, resized to ``--long_side`` pixels on the long side;
2. runs COLMAP on all sampled frames (one shared OPENCV camera, exhaustive
   matching, incremental mapping) and undistorts the result to PINHOLE;
3. builds the initial point cloud for the sparse-view runs from the training
   views only: the full model is cut down to the ``--n_views`` training views
   (``cwdepth.data.views.split_views``, the baseline's rule), points are
   re-triangulated from those views alone, then fused by COLMAP PatchMatch
   stereo. Held-out views contribute poses for evaluation but no geometry;
4. writes ``poses_bounds.npy`` (near/far per image; the loader requires the file).

Output, per scene ``data/statues/<scene>/``:
    images/  sparse/0/  poses_bounds.npy  <n>_views/dense/fused.ply  0_views/dense/
    prep.json   (source video, chosen frame indices, registration, timings)
    colmap/     (database, raw and subset models, MVS workspace; can be deleted)

Run in the acv-da3 environment (OpenCV and pycolmap) from the repository root:

    python scripts/prepare_statues.py --videos "data/Hacettepe Statues" --out data/statues

Finished scenes (those with prep.json) are skipped.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pycolmap

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from cwdepth.data.views import split_views  # noqa: E402

COLMAP = Path.home() / "miniforge3/envs/colmap/bin/colmap"


def sharpness(frame: np.ndarray) -> float:
    gray = cv2.cvtColor(cv2.resize(frame, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA),
                        cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def sample_frames(video: Path, out_dir: Path, n_frames: int, long_side: int) -> list[dict]:
    """Keep the sharpest frame of each of ``n_frames`` equal time windows."""
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < n_frames:
        raise RuntimeError(f"{video.name}: {total} frames, fewer than the {n_frames} requested")
    edges = np.linspace(0, total, n_frames + 1).astype(int)
    best: list[tuple[float, int, np.ndarray | None]] = [(-1.0, -1, None)] * n_frames
    window = 0
    for index in range(total):
        ok, frame = cap.read()
        if not ok:
            break
        while window < n_frames - 1 and index >= edges[window + 1]:
            window += 1
        if index % 2:  # every other frame is enough to find a sharp one
            continue
        score = sharpness(frame)
        if score > best[window][0]:
            best[window] = (score, index, frame)
    cap.release()
    missing = [w for w, b in enumerate(best) if b[2] is None]
    if missing:
        raise RuntimeError(f"{video.name}: no frame decoded in windows {missing}")
    out_dir.mkdir(parents=True, exist_ok=True)
    chosen = []
    for w, (score, index, frame) in enumerate(best):
        h, wd = frame.shape[:2]
        if h < wd:
            raise RuntimeError(f"{video.name}: frame is landscape after rotation; orientation tag not applied")
        scale = long_side / max(h, wd)
        small = cv2.resize(frame, (round(wd * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(out_dir / f"f{w:03d}.jpg"), small, [cv2.IMWRITE_JPEG_QUALITY, 95])
        chosen.append({"name": f"f{w:03d}", "frame_index": index, "sharpness": score})
    return chosen


def colmap(log: Path, *args: str) -> None:
    with open(log, "a") as f:
        f.write(f"\n$ colmap {' '.join(args)}\n")
        f.flush()
        result = subprocess.run([str(COLMAP), *args, "--log_target", "stdout"], stdout=f, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise RuntimeError(f"colmap {args[0]} failed (exit {result.returncode}); see {log}")


def largest_model(sparse_dir: Path) -> Path:
    models = [p for p in sparse_dir.iterdir() if (p / "images.bin").exists()]
    if not models:
        raise RuntimeError(f"COLMAP produced no model in {sparse_dir}")
    return max(models, key=lambda p: pycolmap.Reconstruction(str(p)).num_reg_images())


def write_bounds(model_dir: Path, out_path: Path) -> None:
    """LLFF-style poses_bounds.npy: rows in image-name order, last two columns near/far."""
    rec = pycolmap.Reconstruction(str(model_dir))
    rows = []
    for image in sorted(rec.images.values(), key=lambda im: im.name):
        world_to_cam = image.cam_from_world()
        depths = [(world_to_cam * rec.points3D[p.point3D_id].xyz)[2]
                  for p in image.points2D if p.has_point3D()]
        depths = np.asarray(depths)
        depths = depths[depths > 0]
        if depths.size < 10:
            raise RuntimeError(f"{image.name}: only {depths.size} points in front of the camera")
        row = np.zeros(17, dtype=np.float64)
        row[15:] = np.percentile(depths, 0.1), np.percentile(depths, 99.9)
        rows.append(row)
    np.save(out_path, np.stack(rows))


def reconstruct(scene_dir: Path, n_views: int) -> dict:
    work = scene_dir / "colmap"
    work.mkdir(exist_ok=True)
    log = work / "colmap.log"
    db = work / "database.db"
    frames = scene_dir / "input"

    colmap(log, "feature_extractor", "--database_path", str(db), "--image_path", str(frames),
           "--ImageReader.single_camera", "1", "--ImageReader.camera_model", "OPENCV")
    colmap(log, "exhaustive_matcher", "--database_path", str(db))
    raw = work / "sparse"
    raw.mkdir(exist_ok=True)
    colmap(log, "mapper", "--database_path", str(db), "--image_path", str(frames), "--output_path", str(raw))
    full = largest_model(raw)
    rec = pycolmap.Reconstruction(str(full))
    registered = sorted(Path(im.name).stem for im in rec.images.values())

    # Full model, undistorted: this is what the loader reads.
    und = work / "undistorted"
    colmap(log, "image_undistorter", "--image_path", str(frames), "--input_path", str(full),
           "--output_path", str(und), "--output_type", "COLMAP")
    for name in ("images", "sparse"):
        if (scene_dir / name).exists():
            shutil.rmtree(scene_dir / name)
    shutil.copytree(und / "images", scene_dir / "images")
    (scene_dir / "sparse" / "0").mkdir(parents=True)
    for f in (und / "sparse").glob("*.bin"):
        shutil.copy(f, scene_dir / "sparse" / "0" / f.name)
    write_bounds(scene_dir / "sparse" / "0", scene_dir / "poses_bounds.npy")
    (scene_dir / "0_views" / "dense").mkdir(parents=True, exist_ok=True)

    # Training views only: re-triangulate, then dense fusion.
    train, test = split_views(registered, n_views)
    drop = work / "drop.txt"
    drop.write_text("".join(f"{n}.jpg\n" for n in registered if n not in train))
    sub = work / f"sub{n_views}"
    tri = work / f"sub{n_views}_tri"
    mvs = work / f"mvs{n_views}"
    for d in (sub, tri, mvs):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir()
    colmap(log, "image_deleter", "--input_path", str(full), "--output_path", str(sub),
           "--image_names_path", str(drop))
    colmap(log, "point_triangulator", "--database_path", str(db), "--image_path", str(frames),
           "--input_path", str(sub), "--output_path", str(tri))
    colmap(log, "image_undistorter", "--image_path", str(frames), "--input_path", str(tri),
           "--output_path", str(mvs), "--output_type", "COLMAP")
    colmap(log, "patch_match_stereo", "--workspace_path", str(mvs))
    fused = scene_dir / f"{n_views}_views" / "dense" / "fused.ply"
    fused.parent.mkdir(parents=True, exist_ok=True)
    colmap(log, "stereo_fusion", "--workspace_path", str(mvs), "--output_path", str(fused))

    tri_rec = pycolmap.Reconstruction(str(tri))
    with open(fused, "rb") as f:
        header = f.read(512).decode("latin-1")
    n_fused = int(header.split("element vertex ")[1].split()[0])
    return {
        "registered": len(registered), "models": len(list(raw.iterdir())),
        "mean_reproj_error": rec.compute_mean_reprojection_error(),
        "sparse_points_all_views": rec.num_points3D(),
        "train_views": train, "num_test_views": len(test),
        "sparse_points_train_views": tri_rec.num_points3D(), "fused_points_train_views": n_fused,
        "image_size": [rec.cameras[next(iter(rec.cameras))].width, rec.cameras[next(iter(rec.cameras))].height],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--videos", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=96)
    parser.add_argument("--long_side", type=int, default=1280)
    parser.add_argument("--n_views", type=int, default=12)
    parser.add_argument("--only", nargs="*", help="scene names to process (default: all)")
    args = parser.parse_args()

    videos = sorted(args.videos.glob("*.mp4"))
    if not videos:
        parser.error(f"no .mp4 files in {args.videos}")
    args.out.mkdir(parents=True, exist_ok=True)
    # Scene names follow capture order; the mapping is written once and kept.
    mapping = {f"statue{i:02d}": v.name for i, v in enumerate(videos, start=1)}
    (args.out / "scenes.json").write_text(json.dumps(mapping, indent=2))
    failed = []
    for scene, video_name in mapping.items():
        if args.only and scene not in args.only:
            continue
        scene_dir = args.out / scene
        if (scene_dir / "prep.json").exists():
            print(f"{scene}: done, skipping")
            continue
        start = time.time()
        try:
            chosen = sample_frames(args.videos / video_name, scene_dir / "input", args.frames, args.long_side)
            info = reconstruct(scene_dir, args.n_views)
        except RuntimeError as err:
            failed.append(scene)
            print(f"{scene}: FAILED: {err}", flush=True)
            continue
        info.update({"scene": scene, "video": video_name, "frames": chosen, "seconds": time.time() - start})
        (scene_dir / "prep.json").write_text(json.dumps(info, indent=2))
        print(f"{scene}: {info['registered']}/{args.frames} registered in {info['models']} model(s), "
              f"reproj {info['mean_reproj_error']:.2f} px, {info['sparse_points_train_views']} sparse / "
              f"{info['fused_points_train_views']} fused points from {args.n_views} views "
              f"({info['seconds'] / 60:.1f} min)", flush=True)
    if failed:
        print(f"failed: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
