"""Sequential Bayesian adaptive Ramsey: 1D grid posterior over delta with
A-optimal (expected-posterior-variance) interrogation-time selection.
T2* is known and fixed (docs/PHYSICS.md, 'Adaptive Ramsey')."""
import numpy as np

from ..readout import mean_counts_per_shot


class DeltaPosterior:
    def __init__(self, delta_range_hz=(0.2e6, 4e6), n_grid=600,
                 t2star_s=1.5e-6, readout_cfg=None):
        self.grid = np.linspace(*delta_range_hz, n_grid)
        self.p = np.full(n_grid, 1.0 / n_grid)
        self.t2star_s = t2star_s
        self.readout_cfg = readout_cfg

    def _lam_grid(self, tau_s, n_shots):
        """Expected batch counts lambda(delta) for every grid delta.

        Vectorized closed form: p0(delta) = (1 + cos(2 pi delta tau) E)/2."""
        env = np.exp(-tau_s / self.t2star_s) if self.t2star_s else 1.0
        p0 = 0.5 * (1 + np.cos(2 * np.pi * self.grid * tau_s) * env)
        return n_shots * mean_counts_per_shot(p0, self.readout_cfg)

    def update(self, counts, tau_s, n_shots):
        lam = self._lam_grid(tau_s, n_shots)
        logp = np.log(self.p) + counts * np.log(lam) - lam
        logp -= logp.max()
        self.p = np.exp(logp)
        self.p /= self.p.sum()

    def mean(self):
        return float((self.p * self.grid).sum())

    def sigma(self):
        m = self.mean()
        return float(np.sqrt((self.p * (self.grid - m) ** 2).sum()))
