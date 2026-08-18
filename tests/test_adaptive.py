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
