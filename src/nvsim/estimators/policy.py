"""Amortized adaptive-Ramsey policy (Phase 3b): vectorized posterior
environment for training a neural policy that maps posterior summaries to
the next interrogation time.

The environment is a batched re-implementation of the Phase 3 sequential
loop: E parallel episodes hold (E, G) posterior grids with exact-Poisson
updates (bitwise-equivalent to adaptive.DeltaPosterior — tested). The
action set is exactly choose_tau's coarsened candidate grid, so policy
and A-optimal lookahead pick from the same menu.

Feature vector per env (n_features = 6 + 32):
  0  log10(posterior sigma / prior span), roughly in [-3.5, 0]
  1  posterior mean, mapped to [0, 1] over the prior range
  2  posterior entropy / log(G)  (1 = flat prior)
  3  fraction of the time budget elapsed
  4  top-2 mode mass ratio p2/p1 (the fringe-aliasing signal; 0 if unimodal)
  5  top-2 mode separation / prior span
  6+ 32-bin max-pooled posterior, normalized to peak 1

Drift (Phase 3b-B): true delta follows exact-discretization OU around its
episode-initial value (drift is a zero-mean addition, as in nvsim.drift);
the posterior prediction step applies the matching *diffusion* kernel
N(0, sigma^2 (1 - a^2)) by Gaussian convolution, without the mean-reversion
term (the reversion mean is unknown to the estimator; at dt/tau ~ 2% per
batch the omission is negligible and stated in the writeup). Mass convolved
past the grid edge is clipped and renormalized."""
import numpy as np
from scipy.ndimage import gaussian_filter1d

from ..readout import mean_counts_per_shot


def tau_candidates(cfg):
    """choose_tau's coarsened candidate grid, reproduced exactly."""
    full = np.geomspace(cfg["tau_min_s"], cfg["tau_max_s"], 60)
    return full[:: max(1, len(full) // 24)]


class VecRamseyEnv:
    def __init__(self, cfg, n_envs, rng, drift=None, n_grid=600):
        self.cfg = cfg
        self.n_envs = n_envs
        self.rng = rng
        self.drift = drift
        lo, hi = cfg["delta_range_hz"]
        self.grid = np.linspace(lo, hi, n_grid)
        self.span = hi - lo
        self.tau_candidates_s = tau_candidates(cfg)
        self.n_features = 6 + 32
        self.step_count = 0

    # -- posterior pieces (mirror adaptive.DeltaPosterior exactly) --

    def _lam_grid(self, tau_s):
        """(E_active, G) expected batch counts for each env's tau."""
        env_dec = np.exp(-tau_s / self.cfg["t2star_s"])
        p0 = 0.5 * (1 + np.cos(2 * np.pi * np.outer(tau_s, self.grid))
                    * env_dec[:, None])
        return self.cfg["n_shots_per_batch"] * mean_counts_per_shot(
            p0, self.cfg["readout"])

    def _moments(self, p):
        mu = p @ self.grid
        var = (p * (self.grid[None, :] - mu[:, None]) ** 2).sum(axis=1)
        return mu, np.sqrt(var)

    def sigma_hz(self):
        return self._moments(self.p)[1]

    def mean_hz(self):
        return self._moments(self.p)[0]

    # -- episode control --

    def reset(self):
        lo, hi = self.cfg["delta_range_hz"]
        self.true_delta_hz = self.rng.uniform(lo, hi, self.n_envs)
        self.base_delta_hz = self.true_delta_hz.copy()
        self.p = np.full((self.n_envs, len(self.grid)), 1.0 / len(self.grid))
        self.elapsed_s = np.zeros(self.n_envs)
        self.done = np.zeros(self.n_envs, bool)
        self.step_count = 0
        return self.features()

    def _diffuse(self, dt_s):
        """OU prediction step on active posteriors: diffusion-only kernel."""
        a = np.exp(-dt_s / self.drift["tau_s"])
        sig_hz = self.drift["sigma_hz"] * np.sqrt(1 - a * a)
        cell = self.grid[1] - self.grid[0]
        for e in np.nonzero(~self.done)[0]:
            if sig_hz[e] > 0.05 * cell:
                self.p[e] = gaussian_filter1d(self.p[e], sig_hz[e] / cell,
                                              mode="constant", cval=0.0)
                self.p[e] /= self.p[e].sum()

    def _drift_truth(self, dt_s):
        """Exact-discretization OU step on true delta (zero-mean around base)."""
        a = np.exp(-dt_s / self.drift["tau_s"])
        x = self.true_delta_hz - self.base_delta_hz
        noise = self.rng.standard_normal(self.n_envs)
        x = x * a + self.drift["sigma_hz"] * np.sqrt(1 - a * a) * noise
        self.true_delta_hz = np.where(self.done, self.true_delta_hz,
                                      self.base_delta_hz + x)

    def step(self, actions):
        actions = np.asarray(actions)
        active = ~self.done
        tau = self.tau_candidates_s[actions]
        t = self.cfg["timing"]
        n_b = self.cfg["n_shots_per_batch"]
        dt = n_b * (t["t_init_s"] + tau + t["t_read_s"] + t["t_dead_s"])

        sigma_before = self.sigma_hz()
        counts = np.zeros(self.n_envs, dtype=np.int64)
        if active.any():
            if self.drift is not None:
                self._drift_truth(np.where(active, dt, 0.0))
                self._diffuse(np.where(active, dt, 0.0))
            idx = np.nonzero(active)[0]
            lam_grid = self._lam_grid(tau[idx])            # (E_a, G)
            env_dec = np.exp(-tau[idx] / self.cfg["t2star_s"])
            p0_true = 0.5 * (1 + np.cos(2 * np.pi * tau[idx]
                                        * self.true_delta_hz[idx]) * env_dec)
            lam_true = n_b * mean_counts_per_shot(p0_true, self.cfg["readout"])
            counts[idx] = self.rng.poisson(lam_true)
            logp = (np.log(self.p[idx]) + counts[idx, None]
                    * np.log(lam_grid) - lam_grid)
            logp -= logp.max(axis=1, keepdims=True)
            p_new = np.exp(logp)
            self.p[idx] = p_new / p_new.sum(axis=1, keepdims=True)
            self.elapsed_s[idx] += dt[idx]

        sigma_after = self.sigma_hz()
        # floor at 1 Hz: a posterior collapsed onto a single grid node has
        # sigma exactly 0 and would give an infinite reward
        reward = np.where(active,
                          np.log(np.maximum(sigma_before, 1.0) ** 2)
                          - np.log(np.maximum(sigma_after, 1.0) ** 2), 0.0)
        self.done = self.elapsed_s >= self.cfg["time_budget_s"]
        self.step_count += 1
        info = {"counts": counts, "sigma_hz": sigma_after,
                "mean_hz": self.mean_hz(), "elapsed_s": self.elapsed_s.copy(),
                "tau_s": tau}
        return self.features(), reward, self.done.copy(), info

    # -- features --

    def mode_mass_ratio(self):
        return _top2_modes(self.p, self.grid, self.span)[0]

    def features(self):
        return compute_features(self.p, self.grid, self.cfg,
                                self.elapsed_s)


def _top2_modes(p, grid, span):
    peaks = np.zeros_like(p, dtype=bool)
    peaks[:, 1:-1] = (p[:, 1:-1] > p[:, :-2]) & (p[:, 1:-1] > p[:, 2:])
    v = np.where(peaks, p, 0.0)
    i1 = v.argmax(axis=1)
    rows = np.arange(p.shape[0])
    v1 = v[rows, i1].copy()
    v[rows, i1] = 0.0
    i2 = v.argmax(axis=1)
    v2 = v[rows, i2]
    ratio = np.where(v1 > 0, v2 / np.maximum(v1, 1e-300), 0.0)
    sep = np.abs(grid[i1] - grid[i2]) / span
    sep = np.where(v2 > 0, sep, 0.0)
    return ratio, sep


def compute_features(p, grid, cfg, elapsed_s):
    """Feature matrix (E, 38) from posterior rows (E, G) — the single
    definition used by both the training env and evaluation-time schedules."""
    p = np.atleast_2d(p)
    elapsed_s = np.atleast_1d(np.asarray(elapsed_s, dtype=float))
    lo, hi = cfg["delta_range_hz"]
    span = hi - lo
    mu = p @ grid
    sig = np.sqrt((p * (grid[None, :] - mu[:, None]) ** 2).sum(axis=1))
    entropy = -(p * np.log(np.maximum(p, 1e-300))).sum(axis=1)
    ratio, sep = _top2_modes(p, grid, span)
    scalars = np.stack([
        np.log10(np.maximum(sig, 1.0) / span),
        (mu - lo) / span,
        entropy / np.log(p.shape[1]),
        elapsed_s / cfg["time_budget_s"],
        ratio,
        sep,
    ], axis=1)
    g = p.shape[1] // 32
    pooled = p[:, : 32 * g].reshape(p.shape[0], 32, g).max(axis=2)
    # a posterior concentrated entirely in the residual cells beyond 32*g
    # pools to all zeros — keep it zeros rather than dividing 0/0 (the
    # scalar features still carry mean/sigma; pooling window unchanged to
    # keep the feature definition identical to the BC dataset)
    pooled = pooled / np.maximum(pooled.max(axis=1, keepdims=True), 1e-300)
    return np.concatenate([scalars, pooled], axis=1)
