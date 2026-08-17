import numpy as np
import pytest

from nvsim.pulsed import rabi


def test_rabi_frequency_equals_drive_amplitude():
    f_rabi = 5e6
    t = np.linspace(0, 2e-6, 2001)
    p0 = rabi(f_rabi, t)
    # P0(t) = cos^2(pi * f_rabi * t) on resonance
    np.testing.assert_allclose(p0, np.cos(np.pi * f_rabi * t) ** 2, atol=1e-6)


def test_rabi_scales_linearly_with_amplitude():
    t = np.linspace(0, 4e-6, 4001)
    for f_rabi in (1e6, 2e6, 4e6):
        p0 = rabi(f_rabi, t)
        # first minimum of P0 at t = 1/(2 f_rabi)
        t_pi = t[np.argmin(p0)]
        assert t_pi == pytest.approx(1 / (2 * f_rabi), rel=2e-3)


def test_detuned_rabi_generalized_frequency_and_reduced_contrast():
    f_rabi, delta = 2e6, 1.5e6
    omega_gen = np.hypot(f_rabi, delta)
    t = np.linspace(0, 3e-6, 3001)
    p0 = rabi(f_rabi, t, detuning_hz=delta)
    expected = 1 - (f_rabi / omega_gen) ** 2 * np.sin(np.pi * omega_gen * t) ** 2
    np.testing.assert_allclose(p0, expected, atol=1e-4)


def test_lindblad_preserves_trace_and_damps_rabi():
    t = np.linspace(0, 10e-6, 501)
    p0 = rabi(1e6, t, t1_s=20e-6, t2_s=5e-6)
    assert np.all((p0 >= -1e-9) & (p0 <= 1 + 1e-9))
    # damped toward mixed state: late-time oscillation amplitude well below early
    early = p0[t < 2e-6]
    late = p0[t > 8e-6]
    early_amp = early.max() - early.min()
    late_amp = late.max() - late.min()
    assert late_amp < 0.5 * early_amp
    # analytic Rabi decay rate 3*gamma1/4 + gamma_phi/2 = 1.25e5 /s -> ~0.33 at t~9us
    assert late_amp == pytest.approx(0.35, abs=0.08)
