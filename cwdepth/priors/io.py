"""Reading and writing cached depth priors.

One ``.npz`` per training image, named after the image stem, holding
``depth`` and ``conf`` as float32 arrays of the same shape, plus the name of
the model that produced them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def prior_path(prior_dir: str | Path, image_name: str) -> Path:
    return Path(prior_dir) / f"{Path(image_name).stem}.npz"


def save_prior(prior_dir: str | Path, image_name: str, depth: np.ndarray, conf: np.ndarray, model: str) -> Path:
    depth = np.asarray(depth, dtype=np.float32)
    conf = np.asarray(conf, dtype=np.float32)
    if depth.ndim != 2 or depth.shape != conf.shape:
        raise ValueError(f"depth {depth.shape} and conf {conf.shape} must be equal 2-D shapes")
    if not (np.isfinite(depth).all() and np.isfinite(conf).all()):
        raise ValueError(f"non-finite values in prior for {image_name}")
    path = prior_path(prior_dir, image_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, depth=depth, conf=conf, model=np.array(model))
    return path


def load_prior(prior_dir: str | Path, image_name: str, height: int, width: int, device="cuda"):
    """Return (depth, conf) as (H, W) float32 tensors resized to the training resolution."""
    path = prior_path(prior_dir, image_name)
    if not path.is_file():
        raise FileNotFoundError(f"no depth prior for image {image_name!r}: expected {path}")
    with np.load(path) as data:
        depth = torch.from_numpy(data["depth"].astype(np.float32))
        conf = torch.from_numpy(data["conf"].astype(np.float32))
    if depth.shape != conf.shape or depth.dim() != 2:
        raise ValueError(f"malformed prior {path}: depth {tuple(depth.shape)}, conf {tuple(conf.shape)}")
    if not (torch.isfinite(depth).all() and torch.isfinite(conf).all()):
        raise ValueError(f"non-finite values in prior {path}")

    def resize(x):
        if x.shape == (height, width):
            return x
        return F.interpolate(x[None, None], size=(height, width), mode="bilinear", align_corners=False)[0, 0]

    return resize(depth).to(device), resize(conf).to(device)
