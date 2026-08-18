import numpy as np
import pytest

from nvsim.estimators.crb import crb_sigma_delta, fisher_matrix, sigma_b_tesla

READOUT = {"r_hz": 6e7, "contrast": 0.25, "t_read_s": 0.4e-6, "f_pump": 0.95}
TAUS = np.linspace(0, 5e-6, 150)
THETA = (2e6, 1.5e-6)


def test_fisher_symmetric_positive_definite():
    fim = fisher_matrix(TAUS, THETA, READOUT, 2000)
    assert fim.shape == (2, 2)
    np.testing.assert_allclose(fim, fim.T, rtol=1e-9)
    assert np.all(np.linalg.eigvalsh(fim) > 0)


def test_crb_scales_as_one_over_sqrt_n():
    s1 = crb_sigma_delta(TAUS, THETA, READOUT, 200)
    s2 = crb_sigma_delta(TAUS, THETA, READOUT, 20000)
    assert s1 / s2 == pytest.approx(10.0, rel=1e-3)


def test_crb_against_analytic_single_parameter():
    """Hand-computed Poisson Fisher for delta with negligible decay:
    I_dd = sum_i (dlam_i/dd)^2 / lam_i."""
    taus = np.linspace(0, 4e-6, 80)
    readout = {"r_hz": 6e7, "contrast": 0.02, "t_read_s": 0.4e-6, "f_pump": 1.0}
    n = 1000
    d = 1.7e6
    lam = n * readout["r_hz"] * readout["t_read_s"] * (
        1 - readout["contrast"] * (1 - 0.5 * (1 + np.cos(2 * np.pi * d * taus))))
    dlam = (n * readout["r_hz"] * readout["t_read_s"] * readout["contrast"]
            * 0.5 * (-2 * np.pi * taus) * np.sin(2 * np.pi * d * taus))
    expected = (dlam**2 / lam).sum()
    fim = fisher_matrix(taus, (d, 1.0), readout, n)  # T2* = 1 s ~ no decay
    assert fim[0, 0] == pytest.approx(expected, rel=1e-4)


def test_sigma_b_conversion():
    assert sigma_b_tesla(28.02e9) == pytest.approx(1.0, rel=1e-12)
    assert sigma_b_tesla(2.802) == pytest.approx(1e-10, rel=1e-9)
