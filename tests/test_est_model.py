import numpy as np
import pytest

from nvsim.estimators.model import expected_counts, ramsey_p0
from nvsim.pulsed import ramsey

READOUT = {"r_hz": 6e7, "contrast": 0.25, "t_read_s": 0.4e-6, "f_pump": 0.95}


def test_analytic_matches_mesolve_generator():
    taus = np.linspace(0, 5e-6, 40)
    for delta, t2s in ((2e6, 1.5e-6), (0.7e6, 3e-6), (-1.2e6, 2e-6)):
        np.testing.assert_allclose(
            ramsey_p0(taus, delta, t2s),
            ramsey(taus, detuning_hz=delta, t2star_s=t2s), atol=1e-8)


def test_analytic_matches_mesolve_no_decay():
    taus = np.linspace(0, 3e-6, 30)
    np.testing.assert_allclose(
        ramsey_p0(taus, 1.5e6, None), ramsey(taus, detuning_hz=1.5e6), atol=1e-8)


def test_expected_counts_positive_and_scaled():
    taus = np.linspace(0, 5e-6, 150)
    lam = expected_counts(taus, 2e6, 1.5e-6, READOUT, 2000)
    assert lam.shape == (150,)
    assert np.all(lam > 0)
    np.testing.assert_allclose(
        expected_counts(taus, 2e6, 1.5e-6, READOUT, 4000), 2 * lam, rtol=1e-12)


def test_experiment_ramsey_unchanged_by_fast_path():
    # regression: the API's ideal curve equals the analytic model
    from nvsim.experiment import run_experiment
    cfg = {"name": "t", "protocol": "ramsey", "seed": 11,
           "sweep": {"min": 0.0, "max": 4e-6, "n_points": 40},
           "n_sweeps": 2, "n_shots": 200,
           "truth": {"detuning_hz": 2e6, "t2star_s": 3e-6},
           "timing": {"t_init_s": 2e-6, "t_read_s": 0.4e-6, "t_dead_s": 1e-6},
           "readout": READOUT, "drift": {}}
    ds = run_experiment(cfg)
    taus = ds["sweep_values"]
    np.testing.assert_allclose(ds["truth"]["p0_ideal"],
                               ramsey_p0(taus, 2e6, 3e-6), atol=1e-8)
