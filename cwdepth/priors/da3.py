"""Compute and cache Depth Anything 3 depth and confidence for the training views.

Runs in the ``acv-da3`` environment (the depth model's dependencies are kept
apart from the training environment). Output: one ``.npz`` per training image
under ``--out_dir``, read by ``cwdepth.priors.io.load_prior``.

    python -m cwdepth.priors.da3 --scene_dir data/llff/fern --n_views 3 \
        --out_dir data/priors/da3-large/llff/fern

Modes:
  joint      all training views in one forward pass (multi-view consistent)
  per_image  each view alone (monocular use of the same model)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from cwdepth.data.views import list_images, split_views
from cwdepth.priors.io import save_prior


def run_model(model, image_paths: list[str], process_res: int):
    prediction = model.inference(image_paths, process_res=process_res, process_res_method="upper_bound_resize")
    depth = np.asarray(prediction.depth, dtype=np.float32)
    conf = getattr(prediction, "conf", None)
    if conf is None:
        raise RuntimeError("this model returns no confidence map; choose a DA3 any-view model")
    conf = np.asarray(conf, dtype=np.float32)
    if depth.shape != conf.shape or depth.shape[0] != len(image_paths):
        raise RuntimeError(f"unexpected output shapes: depth {depth.shape}, conf {conf.shape}")
    return depth, conf


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scene_dir", type=Path, required=True)
    parser.add_argument("--images", default="images")
    parser.add_argument("--n_views", type=int, default=3, help="<= 0 uses every non-test image")
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--model", default="depth-anything/DA3-LARGE-1.1")
    parser.add_argument("--mode", choices=("joint", "per_image"), default="joint")
    parser.add_argument("--process_res", type=int, default=504)
    args = parser.parse_args()

    from depth_anything_3.api import DepthAnything3  # imported late: heavy, and only in the acv-da3 env

    paths = list_images(args.scene_dir / args.images)
    by_stem = {p.stem: p for p in paths}
    train, _ = split_views(list(by_stem), args.n_views)
    image_paths = [str(by_stem[name]) for name in train]
    print(f"{args.scene_dir.name}: {len(train)} training views: {train}")

    device = torch.device("cuda")
    model = DepthAnything3.from_pretrained(args.model).to(device=device).eval()
    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    with torch.no_grad():
        if args.mode == "joint":
            depth, conf = run_model(model, image_paths, args.process_res)
        else:
            outs = [run_model(model, [p], args.process_res) for p in image_paths]
            depth = np.concatenate([o[0] for o in outs])
            conf = np.concatenate([o[1] for o in outs])
    seconds = time.time() - start
    peak_mb = torch.cuda.max_memory_allocated() / 2**20

    for name, d, c in zip(train, depth, conf):
        save_prior(args.out_dir, name, d, c, model=args.model)
    meta = {
        "scene": args.scene_dir.name, "model": args.model, "mode": args.mode, "process_res": args.process_res,
        "views": train, "shape": list(depth.shape[1:]), "seconds": seconds, "peak_vram_mb": peak_mb,
        "depth_range": [float(depth.min()), float(depth.max())],
        "conf_range": [float(conf.min()), float(conf.max())],
        "conf_percentiles_5_50_95": [float(v) for v in np.percentile(conf, [5, 50, 95])],
    }
    (args.out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
