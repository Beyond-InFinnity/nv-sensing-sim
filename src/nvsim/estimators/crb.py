"""Poisson Cramér–Rao bound for Ramsey records (docs/PHYSICS.md, Sensitivity)."""
import numpy as np

from ..constants import GAMMA_E_HZ_PER_T
from .model import expected_counts

_REL_STEP = 1e-6


def _grad_lambda(taus_s, theta, readout_cfg, n_shots):
    """Central-difference d(lambda)/d(theta), shape (n_params, n_points)."""
    theta = np.asarray(theta, dtype=float)
    grads = []
    for j in range(len(theta)):
        h = _REL_STEP * max(abs(theta[j]), 1e-12)
        up, dn = theta.copy(), theta.copy()
        up[j] += h
        dn[j] -= h
        grads.append((expected_counts(taus_s, *up, readout_cfg, n_shots)
                      - expected_counts(taus_s, *dn, readout_cfg, n_shots))
                     / (2 * h))
    return np.vstack(grads)


def fisher_matrix(taus_s, theta, readout_cfg, n_shots):
    """Poisson Fisher information: I_jk = sum_i dlam_i/dth_j dlam_i/dth_k / lam_i."""
    lam = expected_counts(taus_s, *theta, readout_cfg, n_shots)
    g = _grad_lambda(taus_s, theta, readout_cfg, n_shots)
    return (g / lam) @ g.T


def crb_sigma_delta(taus_s, theta, readout_cfg, n_shots):
    """CRB standard deviation on delta (Hz), T2* treated as jointly unknown."""
    return float(np.sqrt(np.linalg.inv(
        fisher_matrix(taus_s, theta, readout_cfg, n_shots))[0, 0]))


def sigma_b_tesla(sigma_delta_hz):
    return sigma_delta_hz / GAMMA_E_HZ_PER_T
