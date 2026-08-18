"""Analytic forward model for Ramsey records: the shared model all estimators
and the CRB use. Identical (unit-tested) to the qutip generator's lindblad
path — see docs/PHYSICS.md, 'Pulsed two-level reduction'."""
import numpy as np

from ..readout import mean_counts_per_shot


def ramsey_p0(taus_s, delta_hz, t2star_s):
    """P(ms=0) after pi/2(x) - tau - pi/2(-x): (1 + cos(2 pi delta tau) E)/2."""
    taus = np.asarray(taus_s, dtype=float)
    env = np.exp(-taus / t2star_s) if t2star_s else 1.0
    return 0.5 * (1 + np.cos(2 * np.pi * delta_hz * taus) * env)


def expected_counts(taus_s, delta_hz, t2star_s, readout_cfg, n_shots):
    """Mean total counts per sweep point for the Ramsey protocol."""
    return n_shots * mean_counts_per_shot(
        ramsey_p0(taus_s, delta_hz, t2star_s), readout_cfg)
