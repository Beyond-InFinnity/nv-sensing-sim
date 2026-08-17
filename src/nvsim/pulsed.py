"""Pulsed protocols on the driven NV transition, reduced to a two-level system
{|0> = ms=0, |1> = ms=-1} in the MW rotating frame under the RWA
(see docs/PHYSICS.md, "Pulsed two-level reduction"). All inputs in Hz / s;
angular frequencies (omega_*) only inside this module."""
import numpy as np
import qutip

_SM = qutip.destroy(2)          # |0><1|
_SZ = qutip.sigmaz()            # +1 on |0>, -1 on |1> in qutip convention
_P0 = qutip.basis(2, 0) * qutip.basis(2, 0).dag()
_EXC = qutip.basis(2, 1) * qutip.basis(2, 1).dag()
# tight ODE tolerances: invariant tests compare against analytic forms
_SOLVER_OPTS = {"atol": 1e-12, "rtol": 1e-10, "nsteps": 100000}


def _free_h(detuning_hz):
    """Free-evolution H in the rotating frame (angular units): -delta * |1><1|."""
    omega_delta = 2 * np.pi * detuning_hz
    return -omega_delta * _EXC


def _drive_h(rabi_hz, detuning_hz):
    omega_r = 2 * np.pi * rabi_hz
    return _free_h(detuning_hz) + omega_r / 2 * (_SM + _SM.dag())


def _collapse_ops(t1_s, t2_s):
    """T1 as symmetric relaxation (infinite-temperature bath), T2 total coherence
    time -> pure dephasing rate gamma_phi = 1/T2 - 1/(2 T1)."""
    ops = []
    gamma1 = 1 / t1_s if t1_s else 0.0
    if gamma1:
        ops += [np.sqrt(gamma1 / 2) * _SM, np.sqrt(gamma1 / 2) * _SM.dag()]
    if t2_s:
        gamma_phi = 1 / t2_s - gamma1 / 2
        if gamma_phi < -1e-12:
            raise ValueError("t2_s must satisfy T2 <= 2*T1")
        if gamma_phi > 0:
            ops.append(np.sqrt(gamma_phi / 2) * _SZ)
    return ops


def _rx(theta):
    """Ideal (instantaneous) rotation about x by theta."""
    return (-1j * theta / 2 * (_SM + _SM.dag())).expm()


def t2star_from_sigma(sigma_detuning_hz):
    """T2* of the Gaussian free-induction envelope exp(-(tau/T2*)^2) produced by
    Gaussian static detunings of std sigma (Hz): T2* = sqrt(2)/(2 pi sigma)."""
    return np.sqrt(2) / (2 * np.pi * sigma_detuning_hz)


def _ramsey_single(taus_s, detuning_hz, t2star_s):
    psi0 = _rx(np.pi / 2) * qutip.basis(2, 0)
    result = qutip.mesolve(
        _free_h(detuning_hz), psi0, taus_s,
        c_ops=_collapse_ops(None, t2star_s), options=_SOLVER_OPTS,
    )
    out = np.empty(len(taus_s))
    # second pulse about -x: bright fringe at tau=0, P0 = (1 + cos(2 pi delta tau))/2
    half = _rx(-np.pi / 2)
    for i, rho in enumerate(result.states):
        state = half * rho * half.dag() if rho.isoper else half * rho
        out[i] = qutip.expect(_P0, state)
    return out


def ramsey(taus_s, detuning_hz, t2star_s=None, mode="lindblad",
           sigma_detuning_hz=None, n_samples=400, seed=None):
    """P(ms=0) after pi/2 - tau - pi/2 vs free-evolution time tau.

    mode='lindblad': Markovian dephasing at rate 1/T2* (exponential envelope).
    mode='static': average over Gaussian static detunings of std
    sigma_detuning_hz around detuning_hz (Gaussian envelope; the physical
    choice for slow baths, T2* = t2star_from_sigma(sigma)).
    """
    taus = np.asarray(taus_s, dtype=float)
    if mode == "lindblad":
        return _ramsey_single(taus, detuning_hz, t2star_s)
    if mode == "static":
        rng = np.random.default_rng(seed)
        deltas = detuning_hz + sigma_detuning_hz * rng.standard_normal(n_samples)
        return np.mean([_ramsey_single(taus, d, t2star_s) for d in deltas], axis=0)
    raise ValueError(f"unknown mode: {mode}")


def rabi(rabi_hz, times_s, detuning_hz=0.0, t1_s=None, t2_s=None):
    """P(ms=0) under continuous drive, starting from |ms=0>."""
    result = qutip.mesolve(
        _drive_h(rabi_hz, detuning_hz),
        qutip.basis(2, 0),
        np.asarray(times_s, dtype=float),
        c_ops=_collapse_ops(t1_s, t2_s),
        e_ops=[_P0],
        options=_SOLVER_OPTS,
    )
    return np.asarray(result.expect[0])
