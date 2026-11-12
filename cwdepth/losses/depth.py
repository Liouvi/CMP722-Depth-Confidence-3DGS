"""Depth-prior losses for sparse-view Gaussian Splatting.

All functions take single-image tensors of shape (H, W). The rendered depth
carries gradients; the prior depth and the weights are treated as constants.

The weighted Pearson loss follows the form used by Pi-GS (arXiv 2602.03327):
weighted means, weighted covariance, weighted standard deviations, and
``loss = 1 - correlation``. With uniform weights it equals the plain Pearson
depth loss used by FSGS-style methods, which is checked by a unit test.
"""

from __future__ import annotations

import torch

WEIGHT_MODES = ("uniform", "rank", "minmax", "binary", "shuffled")


def _check_map(name: str, x: torch.Tensor) -> None:
    if x.dim() != 2:
        raise ValueError(f"{name} must have shape (H, W), got {tuple(x.shape)}")
    if not torch.isfinite(x).all():
        raise ValueError(f"{name} contains non-finite values")


def rank_normalise(conf: torch.Tensor) -> torch.Tensor:
    """Map confidence to [0, 1] by its rank within the image.

    Monotone and independent of the confidence scale, so the same setting
    works across depth models whose confidence ranges differ. Tied values
    share the mean of their ranks, so equal confidence gets equal weight.
    """
    _check_map("conf", conf)
    flat = conf.reshape(-1)
    n = flat.numel()
    if n < 2:
        return torch.ones_like(conf)
    values, inverse, counts = torch.unique(flat, return_inverse=True, return_counts=True)
    if values.numel() == 1:
        return torch.ones_like(conf)
    last = torch.cumsum(counts, dim=0).to(conf.dtype) - 1.0
    first = last - counts.to(conf.dtype) + 1.0
    mean_rank = 0.5 * (first + last)
    return (mean_rank[inverse] / float(n - 1)).reshape(conf.shape)


def minmax_normalise(conf: torch.Tensor) -> torch.Tensor:
    """Map confidence linearly to [0, 1] using the image minimum and maximum."""
    _check_map("conf", conf)
    lo = conf.min()
    span = conf.max() - lo
    if float(span) <= 0.0:
        return torch.ones_like(conf)
    return (conf - lo) / span


def binary_mask(conf: torch.Tensor, keep_fraction: float) -> torch.Tensor:
    """Weight 1 for the ``keep_fraction`` most confident pixels, 0 elsewhere."""
    _check_map("conf", conf)
    if not 0.0 < keep_fraction <= 1.0:
        raise ValueError(f"keep_fraction must be in (0, 1], got {keep_fraction}")
    return (rank_normalise(conf) >= (1.0 - keep_fraction)).to(conf.dtype)


def shuffle_weights(weights: torch.Tensor, seed: int) -> torch.Tensor:
    """Permute the weights spatially, keeping their histogram.

    Control condition: the same amount of down-weighting, placed at random.
    The permutation depends only on ``seed`` and the image size.
    """
    _check_map("weights", weights)
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    perm = torch.randperm(weights.numel(), generator=gen).to(weights.device)
    return weights.reshape(-1)[perm].reshape(weights.shape)


def make_weights(
    conf: torch.Tensor,
    mode: str,
    keep_fraction: float = 0.5,
    seed: int = 0,
) -> torch.Tensor:
    """Build the per-pixel loss weights for one training image.

    mode: ``uniform`` (plain loss), ``rank`` or ``minmax`` (confidence
    weighting), ``binary`` (hard mask on the most confident pixels),
    ``shuffled`` (rank weights placed at random).
    """
    if mode == "uniform":
        _check_map("conf", conf)
        return torch.ones_like(conf)
    if mode == "rank":
        return rank_normalise(conf)
    if mode == "minmax":
        return minmax_normalise(conf)
    if mode == "binary":
        return binary_mask(conf, keep_fraction)
    if mode == "shuffled":
        return shuffle_weights(rank_normalise(conf), seed)
    raise ValueError(f"unknown weight mode {mode!r}; expected one of {WEIGHT_MODES}")


def weighted_pearson_loss(
    rendered: torch.Tensor,
    prior: torch.Tensor,
    weights: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Return ``1 - r``, where r is the weighted Pearson correlation.

    Invariant to a positive scale and any shift of either depth map, so the
    prior needs no alignment to the scene scale. If either map has no weighted
    variance (constant depth, or all weights zero) the correlation is
    undefined; the function then returns a zero that is still attached to the
    graph, so the training step proceeds without a depth gradient.
    """
    _check_map("rendered", rendered)
    _check_map("prior", prior)
    if rendered.shape != prior.shape:
        raise ValueError(f"shape mismatch: rendered {tuple(rendered.shape)}, prior {tuple(prior.shape)}")
    if weights is None:
        weights = torch.ones_like(prior)
    else:
        _check_map("weights", weights)
        if weights.shape != prior.shape:
            raise ValueError(f"shape mismatch: weights {tuple(weights.shape)}, prior {tuple(prior.shape)}")
        if (weights < 0).any():
            raise ValueError("weights must be non-negative")

    w = weights.detach().reshape(-1)
    x = rendered.reshape(-1)
    y = prior.detach().reshape(-1)

    w_sum = w.sum()
    if float(w_sum) <= eps:
        return rendered.sum() * 0.0

    x_c = x - (w * x).sum() / w_sum
    y_c = y - (w * y).sum() / w_sum
    var_x = (w * x_c * x_c).sum()
    var_y = (w * y_c * y_c).sum()
    if float(var_x) <= eps or float(var_y) <= eps:
        return rendered.sum() * 0.0

    corr = (w * x_c * y_c).sum() / torch.sqrt(var_x * var_y)
    return 1.0 - corr
