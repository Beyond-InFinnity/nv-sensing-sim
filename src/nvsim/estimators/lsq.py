"""Weighted least-squares fit — the universal lab default baseline."""
import numpy as np
from scipy.optimize import curve_fit

from .model import expected_counts


def _fft_delta_guess(counts, taus_s):
    c = counts - counts.mean()
    freqs = np.fft.rfftfreq(len(taus_s), taus_s[1] - taus_s[0])
    return float(freqs[np.argmax(np.abs(np.fft.rfft(c))[1:]) + 1])


def fit_lsq(counts, taus_s, readout_cfg, n_shots, delta_range_hz=(0.2e6, 4e6)):
    counts = np.asarray(counts, dtype=float)
    d0 = np.clip(_fft_delta_guess(counts, taus_s), *delta_range_hz)

    def f(t, delta_hz, t2star_s):
        return expected_counts(t, delta_hz, t2star_s, readout_cfg, n_shots)

    try:
        popt, _ = curve_fit(
            f, np.asarray(taus_s), counts, p0=[d0, 2e-6],
            sigma=np.sqrt(np.maximum(counts, 1.0)), absolute_sigma=True,
            bounds=([delta_range_hz[0], 0.2e-6], [delta_range_hz[1], 50e-6]),
            maxfev=10000)
        return {"delta_hz": float(popt[0]), "t2star_s": float(popt[1]),
                "converged": True}
    except (RuntimeError, ValueError):
        return {"delta_hz": d0, "t2star_s": float("nan"), "converged": False}
