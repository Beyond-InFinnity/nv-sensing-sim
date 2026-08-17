import numpy as np
import pytest

from nvsim.pulsed import hahn_echo, rabi, ramsey, t2star_from_sigma


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


def _fringe_freq(taus, p0):
    p = p0 - p0.mean()
    spec = np.abs(np.fft.rfft(p))
    freqs = np.fft.rfftfreq(len(taus), taus[1] - taus[0])
    return freqs[np.argmax(spec)]


def test_ramsey_fringe_frequency_equals_detuning():
    delta = 2e6
    taus = np.linspace(0, 8e-6, 4096)
    p0 = ramsey(taus, detuning_hz=delta)
    assert _fringe_freq(taus, p0) == pytest.approx(delta, rel=2e-2)


def test_ramsey_lindblad_envelope_is_exp_tau_over_t2star():
    delta, t2s = 2e6, 3e-6
    taus = np.arange(0, 12e-6, 1 / (4 * delta))  # 4 samples per fringe period
    p0 = ramsey(taus, detuning_hz=delta, t2star_s=t2s)
    envelope = np.abs(p0 - 0.5) * 2
    # at fringe maxima (tau_k = k/delta, cos = 1), envelope = exp(-tau/T2*)
    peaks = envelope[::4]
    taus_pk = taus[::4]
    np.testing.assert_allclose(peaks, np.exp(-taus_pk / t2s), atol=0.02)


def test_ramsey_static_sampling_gives_gaussian_envelope():
    sigma = 100e3
    t2s = t2star_from_sigma(sigma)
    taus = np.linspace(0, 2.5 * t2s, 200)
    p0 = ramsey(taus, detuning_hz=0.0, mode="static",
                sigma_detuning_hz=sigma, n_samples=3000, seed=7)
    np.testing.assert_allclose(
        p0, 0.5 * (1 + np.exp(-((taus / t2s) ** 2))), atol=0.02
    )


def test_echo_removes_static_detuning():
    taus = np.linspace(0, 5e-6, 50)
    for delta in (0.0, 1e6, 3.7e6):
        p0 = hahn_echo(taus, static_detuning_hz=delta)
        np.testing.assert_allclose(p0, 1.0, atol=1e-8)


def test_echo_removes_inhomogeneous_broadening():
    sigma = 200e3
    taus = np.linspace(1e-8, 3e-6, 30)
    echo = hahn_echo(taus, mode="static", sigma_detuning_hz=sigma,
                     n_samples=500, seed=3)
    np.testing.assert_allclose(echo, 1.0, atol=1e-8)


def test_echo_decays_with_t2_over_total_time():
    t2 = 100e-6
    taus = np.linspace(0, 150e-6, 40)
    p0 = hahn_echo(taus, t2_s=t2)
    # envelope exp(-(2 tau)/T2): P0 = (1 + exp(-2 tau/T2))/2
    np.testing.assert_allclose(p0, 0.5 * (1 + np.exp(-2 * taus / t2)), atol=1e-3)
