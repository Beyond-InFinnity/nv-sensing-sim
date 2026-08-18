import numpy as np
import pytest

from nvsim.estimators.adaptive import DeltaPosterior
from nvsim.estimators.model import expected_counts

READOUT = {"r_hz": 6e7, "contrast": 0.25, "t_read_s": 0.4e-6, "f_pump": 0.95}


def _posterior():
    return DeltaPosterior(readout_cfg=READOUT)


def test_prior_is_flat_and_normalized():
    p = _posterior()
    assert p.p.sum() == pytest.approx(1.0, rel=1e-12)
    assert p.p.std() / p.p.mean() < 1e-9
    assert p.sigma() > 0.9e6  # flat over 3.8 MHz span -> sigma ~ span/sqrt(12)


def test_sequential_updates_concentrate_on_truth():
    rng = np.random.default_rng(0)
    truth = 2.3e6
    p = _posterior()
    for tau in (0.3e-6, 0.7e-6, 1.1e-6, 1.5e-6) * 5:
        lam = expected_counts([tau], truth, 1.5e-6, READOUT, 100)[0]
        p.update(rng.poisson(lam), tau, 100)
    assert abs(p.mean() - truth) < 3 * p.sigma()
    assert p.sigma() < 30e3


def test_update_is_bayes_consistent_with_batch_likelihood():
    """One update with counts k must equal renormalized prior * Poisson lik."""
    prior = _posterior()
    tau, n, k = 0.8e-6, 50, 600
    p2 = _posterior()
    p2.update(k, tau, n)
    lams = np.array([expected_counts([tau], d, 1.5e-6, READOUT, n)[0]
                     for d in prior.grid])
    logl = k * np.log(lams) - lams
    manual = prior.p * np.exp(logl - logl.max())
    manual /= manual.sum()
    np.testing.assert_allclose(p2.p, manual, atol=1e-12)


from nvsim.estimators.adaptive import choose_tau, make_schedule  # noqa: E402

TAU_GRID = np.geomspace(0.05e-6, 4.5e-6, 60)


def test_choose_tau_returns_grid_value():
    p = _posterior()
    rng = np.random.default_rng(1)
    lam = expected_counts([0.4e-6], 2.0e6, 1.5e-6, READOUT, 100)[0]
    p.update(rng.poisson(lam), 0.4e-6, 100)
    tau = choose_tau(p, TAU_GRID, 100)
    assert TAU_GRID[0] <= tau <= TAU_GRID[-1]


def test_narrow_posterior_prefers_longer_tau():
    wide = _posterior()
    narrow = _posterior()
    rng = np.random.default_rng(2)
    truth = 2.0e6
    for tau in (0.2e-6, 0.5e-6, 0.9e-6, 1.3e-6) * 6:
        lam = expected_counts([tau], truth, 1.5e-6, READOUT, 200)[0]
        narrow.update(rng.poisson(lam), tau, 200)
    t_wide = choose_tau(wide, TAU_GRID, 100)
    t_narrow = choose_tau(narrow, TAU_GRID, 100)
    assert t_narrow > 1.5 * t_wide


def test_fixed_schedules_cycle_and_are_posterior_blind():
    cfg = {"tau_min_s": 0.1e-6, "tau_max_s": 3.2e-6, "n_sweep_points": 8}
    lin = make_schedule("linear_sweep", cfg)
    taus = [lin(i, None) for i in range(16)]
    assert taus[:8] == taus[8:]
    assert taus[0] == pytest.approx(0.1e-6) and taus[7] == pytest.approx(3.2e-6)
    exp = make_schedule("exp_ladder", cfg)
    ladder = [exp(i, None) for i in range(6)]
    assert ladder[1] / ladder[0] == pytest.approx(2.0)
