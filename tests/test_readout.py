import numpy as np
import pytest

from nvsim.readout import mean_counts_per_shot, sample_counts

CFG = {"r_hz": 6.0e7, "contrast": 0.25, "t_read_s": 0.4e-6, "f_pump": 1.0}


def test_mean_counts_bright_and_dark_states():
    lam0 = mean_counts_per_shot(1.0, CFG)          # ms=0: R * t_read
    lam1 = mean_counts_per_shot(0.0, CFG)          # ms=±1: R(1-C) * t_read
    assert lam0 == pytest.approx(6.0e7 * 0.4e-6)   # 24 photons/shot
    assert lam1 == pytest.approx(lam0 * (1 - 0.25))


def test_counts_are_poisson_fano_factor_one():
    rng = np.random.default_rng(0)
    counts = np.array([sample_counts(0.5, CFG, 1, rng)[0] for _ in range(20000)])
    fano = counts.var() / counts.mean()
    assert fano == pytest.approx(1.0, abs=0.05)
    assert counts.dtype.kind == "i"


def test_n_shots_sum_scales_mean_and_snr():
    rng = np.random.default_rng(1)
    n = 400
    reps = np.array([sample_counts(1.0, CFG, n, rng)[0] for _ in range(3000)])
    lam = mean_counts_per_shot(1.0, CFG)
    assert reps.mean() == pytest.approx(n * lam, rel=0.01)
    # SNR = mean/std = sqrt(n*lam) for Poisson
    assert reps.mean() / reps.std() == pytest.approx(np.sqrt(n * lam), rel=0.05)


def test_imperfect_pump_compresses_contrast():
    cfg = dict(CFG, f_pump=0.8)
    lam0 = mean_counts_per_shot(1.0, cfg)
    lam1 = mean_counts_per_shot(0.0, cfg)
    ideal0 = mean_counts_per_shot(1.0, CFG)
    # unpumped 1/3-per-state background: both levels move toward 1 - (2/3)C
    assert lam0 < ideal0
    assert (lam0 - lam1) == pytest.approx(0.8 * 0.25 * 6.0e7 * 0.4e-6, rel=1e-9)


def test_vectorized_over_p0():
    rng = np.random.default_rng(2)
    p0 = np.linspace(0, 1, 7)
    out = sample_counts(p0, CFG, 100, rng)
    assert out.shape == (7,)
    means = mean_counts_per_shot(p0, CFG)
    assert means.shape == (7,)
    assert np.all(np.diff(means) > 0)  # brighter with more ms=0 population
