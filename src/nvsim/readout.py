"""Phenomenological spin-dependent readout: fluorescence rate R for ms=0,
R(1-C) for ms=±1; counts are Poisson draws (see docs/PHYSICS.md). The unpumped
fraction (1 - f_pump) is unpolarized (1/3 per ms level) and contributes a
protocol-independent background rate R(1 - 2C/3)."""
import numpy as np


def mean_counts_per_shot(p0, cfg):
    """Mean detected photons in one readout window, given P(ms=0) = p0."""
    p0 = np.asarray(p0, dtype=float)
    r, c, t, fp = cfg["r_hz"], cfg["contrast"], cfg["t_read_s"], cfg["f_pump"]
    polarized = 1 - c * (1 - p0)
    background = 1 - 2 * c / 3
    lam = r * t * (fp * polarized + (1 - fp) * background)
    return lam if lam.ndim else float(lam)


def sample_counts(p0, cfg, n_shots, rng):
    """Total Poisson counts over n_shots readouts, per element of p0.

    Sum of n Poisson(lam) draws == Poisson(n*lam); sampled directly.
    """
    lam = np.atleast_1d(np.asarray(mean_counts_per_shot(p0, cfg), dtype=float))
    return rng.poisson(n_shots * lam)
