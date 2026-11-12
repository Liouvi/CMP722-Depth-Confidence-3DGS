"""Train/test view selection, reproduced from the baseline's COLMAP scene reader.

The depth-prior script runs in its own environment and cannot import the
baseline, so the rule is restated here. ``tests/test_views.py`` checks it
against the baseline's loader.

Rule (third_party/DropGaussian/scene/dataset_readers.py, readColmapSceneInfo):
images sorted by name; every ``llffhold``-th image is a test view; from the
rest, ``n_views`` are taken at evenly spaced positions, rounded with Python's
built-in ``round``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

IMAGE_SUFFIXES = (".jpg", ".JPG", ".png")


def list_images(image_dir: str | Path) -> list[Path]:
    paths = sorted(p for p in Path(image_dir).iterdir() if p.name.endswith(IMAGE_SUFFIXES))
    if not paths:
        raise FileNotFoundError(f"no images found in {image_dir}")
    return sorted(paths, key=lambda p: p.stem)


def split_views(names: list[str], n_views: int, llffhold: int = 8) -> tuple[list[str], list[str]]:
    """Return (train, test) names. ``n_views <= 0`` keeps every non-test image."""
    if llffhold < 1:
        raise ValueError("llffhold must be >= 1")
    ordered = sorted(names)
    train = [n for i, n in enumerate(ordered) if i % llffhold != 0]
    test = [n for i, n in enumerate(ordered) if i % llffhold == 0]
    if n_views > 0:
        if n_views > len(train):
            raise ValueError(f"asked for {n_views} training views, only {len(train)} available")
        keep = {round(i) for i in np.linspace(0, len(train) - 1, n_views)}
        train = [n for i, n in enumerate(train) if i in keep]
        if len(train) != n_views:
            raise ValueError(f"view selection produced {len(train)} views instead of {n_views}")
    return train, test
