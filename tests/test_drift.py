import numpy as np
import pytest

from nvsim.drift import sample_drift


def test_constant_and_linear():
    t = np.linspace(0, 10, 101)
    rng = np.random.default_rng(0)
    np.testing.assert_allclose(
        sample_drift({"kind": "constant", "value": 3.5}, t, rng), 3.5)
    lin = sample_drift({"kind": "linear", "rate_per_s": -74e3 * 0.002}, t, rng)
    assert lin[0] == 0.0
    assert lin[-1] == pytest.approx(-74e3 * 0.002 * 10)


def test_ou_stationary_std_and_correlation_time():
    t = np.arange(0, 2000.0, 0.5)
    rng = np.random.default_rng(1)
    x = sample_drift({"kind": "ou", "sigma": 2.0, "tau_s": 5.0}, t, rng)
    assert x.std() == pytest.approx(2.0, rel=0.1)
    # autocorrelation at lag tau ~ exp(-1)
    lag = int(5.0 / 0.5)
    ac = np.corrcoef(x[:-lag], x[lag:])[0, 1]
    assert ac == pytest.approx(np.exp(-1), abs=0.1)


def test_one_over_f_psd_slope():
    t = np.linspace(0, 100, 2**14)
    rng = np.random.default_rng(2)
    x = sample_drift({"kind": "one_over_f", "rms": 1e-7, "alpha": 1.0}, t, rng)
    assert x.std() == pytest.approx(1e-7, rel=0.05)
    psd = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(len(t), t[1] - t[0])
    # fit log-log slope over the middle two decades
    m = (freqs > freqs[1] * 10) & (freqs < freqs[-1] / 10)
    slope = np.polyfit(np.log(freqs[m]), np.log(psd[m]), 1)[0]
    assert slope == pytest.approx(-1.0, abs=0.3)


def test_deterministic_given_rng_seed():
    t = np.linspace(0, 10, 100)
    a = sample_drift({"kind": "ou", "sigma": 1, "tau_s": 2},
                     t, np.random.default_rng(7))
    b = sample_drift({"kind": "ou", "sigma": 1, "tau_s": 2},
                     t, np.random.default_rng(7))
    np.testing.assert_array_equal(a, b)
