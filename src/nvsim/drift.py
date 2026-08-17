"""Slow-drift processes on a wall-clock axis (docs/PHYSICS.md, Drift models).

All processes are zero-mean unless 'constant'; callers scale/offset into
physical units (tesla, Hz, relative power)."""
import numpy as np


def _ou(sigma, tau_s, times_s, rng):
    x = np.empty(len(times_s))
    x[0] = sigma * rng.standard_normal()
    for i in range(1, len(times_s)):
        dt = times_s[i] - times_s[i - 1]
        a = np.exp(-dt / tau_s)
        x[i] = x[i - 1] * a + sigma * np.sqrt(1 - a * a) * rng.standard_normal()
    return x


def _one_over_f(rms, alpha, times_s, rng):
    n = max(len(times_s), 2**12)
    grid = np.linspace(times_s[0], times_s[-1], n)
    amp = np.zeros(n // 2 + 1)
    freqs = np.fft.rfftfreq(n, grid[1] - grid[0])
    amp[1:] = freqs[1:] ** (-alpha / 2)
    phases = rng.uniform(0, 2 * np.pi, len(amp))
    spec = amp * np.exp(1j * phases)
    spec[0] = 0.0
    x = np.fft.irfft(spec, n)
    x *= rms / x.std()
    return np.interp(times_s, grid, x)


def sample_drift(cfg, times_s, rng):
    """One realization of the configured drift process at the given times."""
    t = np.asarray(times_s, dtype=float)
    kind = cfg["kind"]
    if kind == "constant":
        return np.full(len(t), float(cfg["value"]))
    if kind == "linear":
        return cfg["rate_per_s"] * (t - t[0])
    if kind == "ou":
        return _ou(cfg["sigma"], cfg["tau_s"], t, rng)
    if kind == "one_over_f":
        return _one_over_f(cfg["rms"], cfg.get("alpha", 1.0), t, rng)
    raise ValueError(f"unknown drift kind: {kind}")
