import pytest
import torch

from cwdepth.losses.depth import (
    binary_mask,
    make_weights,
    minmax_normalise,
    rank_normalise,
    shuffle_weights,
    weighted_pearson_loss,
)

H, W = 12, 17


def _maps(seed=0, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    prior = torch.rand(H, W, generator=g, dtype=dtype) * 5 + 1
    rendered = (prior * 0.7 + torch.rand(H, W, generator=g, dtype=dtype)).requires_grad_(True)
    conf = torch.rand(H, W, generator=g, dtype=dtype) * 10 + 1
    return rendered, prior, conf


def test_uniform_weights_equal_plain_pearson():
    rendered, prior, _ = _maps()
    ref = 1.0 - torch.corrcoef(torch.stack([rendered.detach().reshape(-1), prior.reshape(-1)]))[0, 1]
    assert weighted_pearson_loss(rendered, prior).item() == pytest.approx(ref.item(), abs=1e-10)
    ones = torch.ones_like(prior)
    assert weighted_pearson_loss(rendered, prior, ones).item() == pytest.approx(ref.item(), abs=1e-10)
    # A constant weight of any size is still uniform.
    assert weighted_pearson_loss(rendered, prior, ones * 3.7).item() == pytest.approx(ref.item(), abs=1e-10)


def test_invariant_to_scale_and_shift_of_prior_and_render():
    rendered, prior, conf = _maps()
    w = rank_normalise(conf)
    base = weighted_pearson_loss(rendered, prior, w).item()
    assert weighted_pearson_loss(rendered, prior * 42.0 - 3.0, w).item() == pytest.approx(base, abs=1e-9)
    assert weighted_pearson_loss(rendered * 0.01 + 8.0, prior, w).item() == pytest.approx(base, abs=1e-9)


def test_perfect_and_inverted_depth():
    _, prior, conf = _maps()
    w = rank_normalise(conf)
    assert weighted_pearson_loss(prior * 2 + 1, prior, w).item() == pytest.approx(0.0, abs=1e-9)
    assert weighted_pearson_loss(-prior, prior, w).item() == pytest.approx(2.0, abs=1e-9)


def test_zero_weight_pixels_are_ignored():
    rendered, prior, conf = _maps()
    w = rank_normalise(conf)
    w[:4] = 0.0
    loss = weighted_pearson_loss(rendered, prior, w)
    (grad,) = torch.autograd.grad(loss, rendered)
    assert torch.count_nonzero(grad[:4]) == 0
    assert torch.count_nonzero(grad[4:]) > 0
    # Corrupting the prior where the weight is zero must not change the loss.
    corrupted = prior.clone()
    corrupted[:4] = 1e6
    assert weighted_pearson_loss(rendered, corrupted, w).item() == pytest.approx(loss.item(), abs=1e-9)


def test_degenerate_inputs_give_finite_zero_with_graph():
    rendered, prior, _ = _maps()
    for loss in (
        weighted_pearson_loss(rendered, torch.full_like(prior, 2.0)),
        weighted_pearson_loss(rendered * 0.0 + 1.0, prior),
        weighted_pearson_loss(rendered, prior, torch.zeros_like(prior)),
    ):
        assert torch.isfinite(loss) and loss.item() == 0.0
        loss.backward()
    assert torch.isfinite(rendered.grad).all()


def test_gradcheck():
    rendered, prior, conf = _maps()
    w = rank_normalise(conf)
    assert torch.autograd.gradcheck(lambda r: weighted_pearson_loss(r, prior, w), (rendered,), atol=1e-6)


def test_input_validation():
    rendered, prior, conf = _maps()
    with pytest.raises(ValueError):
        weighted_pearson_loss(rendered[None], prior)
    with pytest.raises(ValueError):
        weighted_pearson_loss(rendered, prior[:, :-1])
    with pytest.raises(ValueError):
        weighted_pearson_loss(rendered, prior, -torch.ones_like(prior))
    bad = prior.clone()
    bad[0, 0] = float("nan")
    with pytest.raises(ValueError):
        weighted_pearson_loss(rendered, bad)
    with pytest.raises(ValueError):
        make_weights(conf, "nonsense")
    with pytest.raises(ValueError):
        binary_mask(conf, 0.0)


def test_rank_normalise_range_order_and_scale_freedom():
    _, _, conf = _maps()
    w = rank_normalise(conf)
    assert w.min().item() == 0.0 and w.max().item() == 1.0
    order = torch.argsort(conf.reshape(-1))
    assert (torch.diff(w.reshape(-1)[order]) >= 0).all()
    # Any monotone rescaling of the confidence gives the same weights.
    assert torch.equal(rank_normalise(torch.log(conf) * 9 + 4), w)


def test_rank_normalise_ties_and_constant():
    conf = torch.tensor([[1.0, 1.0, 1.0], [5.0, 5.0, 9.0]], dtype=torch.float64)
    w = rank_normalise(conf)
    assert w[0, 0] == w[0, 1] == w[0, 2]
    assert w[1, 0] == w[1, 1]
    assert w[0, 0] < w[1, 0] < w[1, 2]
    assert torch.equal(rank_normalise(torch.full((3, 4), 2.5)), torch.ones(3, 4))


def test_minmax_range_and_constant():
    _, _, conf = _maps()
    w = minmax_normalise(conf)
    assert w.min().item() == 0.0 and w.max().item() == pytest.approx(1.0)
    assert torch.equal(minmax_normalise(torch.full((3, 4), 2.5)), torch.ones(3, 4))


def test_binary_mask_keeps_requested_fraction():
    _, _, conf = _maps()
    for q in (0.25, 0.5, 0.9, 1.0):
        m = binary_mask(conf, q)
        assert set(m.unique().tolist()) <= {0.0, 1.0}
        assert m.mean().item() == pytest.approx(q, abs=1.5 / conf.numel())
    assert conf[binary_mask(conf, 0.5) == 1].min() >= conf[binary_mask(conf, 0.5) == 0].max()


def test_shuffle_keeps_histogram_and_is_deterministic():
    _, _, conf = _maps()
    w = rank_normalise(conf)
    s1, s2, s3 = shuffle_weights(w, 7), shuffle_weights(w, 7), shuffle_weights(w, 8)
    assert torch.equal(s1, s2)
    assert not torch.equal(s1, s3)
    assert not torch.equal(s1, w)
    assert torch.equal(torch.sort(s1.reshape(-1)).values, torch.sort(w.reshape(-1)).values)


def test_make_weights_modes():
    _, _, conf = _maps()
    assert torch.equal(make_weights(conf, "uniform"), torch.ones_like(conf))
    assert torch.equal(make_weights(conf, "rank"), rank_normalise(conf))
    assert torch.equal(make_weights(conf, "minmax"), minmax_normalise(conf))
    assert torch.equal(make_weights(conf, "binary", keep_fraction=0.3), binary_mask(conf, 0.3))
    assert torch.equal(make_weights(conf, "shuffled", seed=3), shuffle_weights(rank_normalise(conf), 3))
