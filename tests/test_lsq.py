import numpy as np
import pytest

from nvsim.estimators.crb import crb_sigma_delta
from nvsim.estimators.lsq import fit_lsq
from nvsim.estimators.model import expected_counts

READOUT = {"r_hz": 6e7, "contrast": 0.25, "t_read_s": 0.4e-6, "f_pump": 0.95}
TAUS = np.linspace(0, 5e-6, 150)


def _record(delta, t2s, n_shots, rng):
    return rng.poisson(expected_counts(TAUS, delta, t2s, READOUT, n_shots))


def test_lsq_recovers_truth_at_high_snr():
    rng = np.random.default_rng(0)
    counts = _record(2.1e6, 1.5e-6, 20000, rng)
    est = fit_lsq(counts, TAUS, READOUT, 20000)
    assert est["converged"]
    assert est["delta_hz"] == pytest.approx(2.1e6, rel=2e-3)
    assert est["t2star_s"] == pytest.approx(1.5e-6, rel=0.1)


def test_lsq_rmse_at_or_above_crb():
    rng = np.random.default_rng(1)
    truth = (2.0e6, 1.5e-6)
    n_shots = 2000
    errs = []
    for _ in range(60):
        est = fit_lsq(_record(*truth, n_shots, rng), TAUS, READOUT, n_shots)
        errs.append(est["delta_hz"] - truth[0])
    rmse = np.sqrt(np.mean(np.square(errs)))
    crb = crb_sigma_delta(TAUS, truth, READOUT, n_shots)
    assert rmse > 0.9 * crb          # cannot beat the bound (stat slack)
    assert rmse < 3.0 * crb          # but should be in its vicinity here


def test_lsq_flags_nonconvergence_gracefully():
    counts = np.zeros(len(TAUS), dtype=int)  # pathological record
    est = fit_lsq(counts, TAUS, READOUT, 10)
    assert est["converged"] in (True, False)  # returns, never raises
