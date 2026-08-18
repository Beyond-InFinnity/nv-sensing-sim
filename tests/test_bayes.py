import numpy as np
import pytest

from nvsim.estimators.bayes import fit_bayes
from nvsim.estimators.crb import crb_sigma_delta
from nvsim.estimators.model import expected_counts

READOUT = {"r_hz": 6e7, "contrast": 0.25, "t_read_s": 0.4e-6, "f_pump": 0.95}
TAUS = np.linspace(0, 5e-6, 150)


def test_posterior_mean_near_truth_high_snr():
    rng = np.random.default_rng(3)
    counts = rng.poisson(expected_counts(TAUS, 2.1e6, 1.5e-6, READOUT, 20000))
    est = fit_bayes(counts, TAUS, READOUT, 20000)
    assert est["delta_hz"] == pytest.approx(2.1e6, rel=2e-3)


def test_posterior_std_tracks_crb_at_high_snr():
    rng = np.random.default_rng(4)
    truth = (2.0e6, 1.5e-6)
    n_shots = 5000
    sigmas = []
    for _ in range(20):
        counts = rng.poisson(expected_counts(TAUS, *truth, READOUT, n_shots))
        sigmas.append(fit_bayes(counts, TAUS, READOUT, n_shots)["delta_sigma_hz"])
    crb = crb_sigma_delta(TAUS, truth, READOUT, n_shots)
    assert np.mean(sigmas) == pytest.approx(crb, rel=0.25)


def test_posterior_coverage_two_sigma():
    rng = np.random.default_rng(5)
    truth = (1.7e6, 2.0e-6)
    hits = 0
    n_rec = 40
    for _ in range(n_rec):
        counts = rng.poisson(expected_counts(TAUS, *truth, READOUT, 500))
        est = fit_bayes(counts, TAUS, READOUT, 500)
        hits += abs(est["delta_hz"] - truth[0]) < 2 * est["delta_sigma_hz"]
    assert hits / n_rec >= 0.85  # ~95% nominal, small-sample slack
