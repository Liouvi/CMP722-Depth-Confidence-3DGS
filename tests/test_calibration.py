import numpy as np
import pytest
import torch

from cwdepth.calib.sparsification import align_scale_shift, calibration_metrics, sparsification_curve


def _error(n=20000, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(n, generator=g, dtype=torch.float64) ** 2 + 1e-3


def test_align_recovers_scale_and_shift():
    g = torch.Generator().manual_seed(1)
    reference = torch.rand(5000, generator=g, dtype=torch.float64) * 4 + 1
    prior = (reference - 0.7) / 3.2
    assert torch.allclose(align_scale_shift(prior, reference), reference, atol=1e-8)


def test_curve_starts_at_full_mean_and_oracle_decreases():
    error = _error()
    oracle = sparsification_curve(error, -error)
    assert oracle[0] == pytest.approx(error.mean().item())
    assert np.all(np.diff(oracle) <= 1e-12)


def test_perfect_confidence_has_zero_ause_and_positive_aurg():
    error = _error()
    m = calibration_metrics(error, -error)   # confidence is the exact negative of the error
    assert m["ause"] == pytest.approx(0.0, abs=1e-9)
    assert m["aurg"] > 0.3
    assert m["spearman_conf_vs_error"] == pytest.approx(-1.0)


def test_random_confidence_has_aurg_near_zero():
    error = _error()
    conf = torch.rand(error.numel(), generator=torch.Generator().manual_seed(9), dtype=torch.float64)
    m = calibration_metrics(error, conf)
    assert abs(m["aurg"]) < 0.02
    assert abs(m["spearman_conf_vs_error"]) < 0.03
    assert m["ause"] > 0.3


def test_inverted_confidence_is_worse_than_random():
    error = _error()
    m = calibration_metrics(error, error)    # most confident where the error is largest
    assert m["aurg"] < -0.3


def test_metrics_are_invariant_to_monotone_rescaling_of_confidence():
    error = _error()
    conf = -error + 0.05 * torch.rand(error.numel(), generator=torch.Generator().manual_seed(3), dtype=torch.float64)
    a = calibration_metrics(error, conf)
    b = calibration_metrics(error, torch.exp(conf * 5.0))
    assert a["ause"] == pytest.approx(b["ause"], abs=1e-12)
    assert a["aurg"] == pytest.approx(b["aurg"], abs=1e-12)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_curve_runs_on_gpu_tensors():
    error = _error().cuda()
    m = calibration_metrics(error, -error)
    assert m["ause"] == pytest.approx(0.0, abs=1e-9)


def test_depth_errors_are_zero_for_identical_depth_and_penalise_offsets():
    from cwdepth.calib.geometry import depth_errors
    ref = torch.linspace(1.0, 5.0, 1000, dtype=torch.float64)
    same = depth_errors(ref, ref)
    assert same["absrel"] == pytest.approx(0.0) and same["delta1"] == 1.0
    off = depth_errors(ref * 1.1, ref)
    assert off["absrel"] == pytest.approx(0.1) and off["delta1"] == 1.0 and off["delta05"] == 0.0


def test_depth_control_separates_confidence_that_only_encodes_distance():
    gen = torch.Generator().manual_seed(0)
    depth = torch.rand(20000, generator=gen) + 0.5
    error = depth * 0.1 + 0.01 * torch.rand(20000, generator=gen)   # error grows with distance only
    m = calibration_metrics(error, -depth, depth)                      # confidence = nearness
    assert m["aurg"] > 0.1
    assert abs(m["aurg_depth"] - m["aurg"]) < 1e-9                    # nothing beyond distance
    assert m["spearman_conf_vs_depth"] < -0.99
    informative = calibration_metrics(error, -error, depth)            # confidence knows the residual too
    assert informative["aurg"] > m["aurg_depth"]
    assert informative["partial_spearman"] < -0.5
