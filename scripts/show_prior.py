"""Save a panel of image, prior depth and confidence for a scene's training views.

    python scripts/show_prior.py --scene_dir data/llff/fern --prior_dir data/priors/da3-base/llff/fern --out fern.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cwdepth.data.views import list_images, split_views  # noqa: E402
from cwdepth.priors.io import prior_path  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scene_dir", type=Path, required=True)
    parser.add_argument("--prior_dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n_views", type=int, default=3)
    args = parser.parse_args()

    by_stem = {p.stem: p for p in list_images(args.scene_dir / "images")}
    train, _ = split_views(list(by_stem), args.n_views)
    fig, axes = plt.subplots(len(train), 4, figsize=(16, 3.2 * len(train)), squeeze=False)
    for row, name in enumerate(train):
        with np.load(prior_path(args.prior_dir, name)) as data:
            depth, conf = data["depth"], data["conf"]
        image = Image.open(by_stem[name]).convert("RGB").resize((depth.shape[1], depth.shape[0]))
        rank = conf.reshape(-1).argsort().argsort().reshape(conf.shape) / (conf.size - 1)
        panels = [(image, None, name), (depth, "turbo", "prior depth"),
                  (conf, "viridis", f"confidence [{conf.min():.1f}, {conf.max():.1f}]"),
                  (rank, "viridis", "confidence rank (loss weight)")]
        for ax, (data_, cmap, title) in zip(axes[row], panels):
            ax.imshow(data_, cmap=cmap)
            ax.set_title(title, fontsize=10)
            ax.axis("off")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=70)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
