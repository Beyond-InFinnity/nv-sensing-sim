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


_GH_NODES, _GH_WEIGHTS = np.polynomial.hermite_e.hermegauss(32)
_GH_WEIGHTS = _GH_WEIGHTS / _GH_WEIGHTS.sum()


def _expected_posterior_variance(posterior, tau_s, n_shots):
    """E_k[Var(delta | k)] with k ~ N(n lam, n lam) marginalized over p(delta).

    For each Gauss-Hermite node z, k(d_true) = n lam(d_true) + z sqrt(n lam)
    is a representative outcome per candidate truth; each branch's updated
    posterior variance is weighted by prior mass p(d_true) and GH weight."""
    lam = posterior._lam_grid(tau_s, n_shots)          # (G,)
    grid, p = posterior.grid, posterior.p
    log_lam = np.log(lam)
    var_exp = 0.0
    for z, w in zip(_GH_NODES, _GH_WEIGHTS):
        k = lam + z * np.sqrt(lam)                      # outcome per true delta
        logl = np.outer(k, log_lam) - lam               # (G_true, G)
        logl -= logl.max(axis=1, keepdims=True)
        q = p * np.exp(logl)
        q /= q.sum(axis=1, keepdims=True)
        mu = q @ grid
        var = (q * (grid - mu[:, None]) ** 2).sum(axis=1)
        var_exp += w * float(p @ var)
    return var_exp


def _coarsened(posterior, n_coarse=200):
    """Decimated copy of the posterior for the utility lookahead only
    (exact updates keep the full grid). Documented in the Phase 3 writeup."""
    step = max(1, len(posterior.grid) // n_coarse)
    c = DeltaPosterior.__new__(DeltaPosterior)
    c.grid = posterior.grid[::step]
    c.p = posterior.p[::step].copy()
    c.p /= c.p.sum()
    c.t2star_s = posterior.t2star_s
    c.readout_cfg = posterior.readout_cfg
    return c


def choose_tau(posterior, tau_grid_s, n_shots, n_coarse=200, n_cand=24):
    """A-optimal next interrogation time (coarsened lookahead)."""
    cand = tau_grid_s[:: max(1, len(tau_grid_s) // n_cand)]
    coarse = _coarsened(posterior, n_coarse)
    evs = [_expected_posterior_variance(coarse, t, n_shots) for t in cand]
    return float(cand[int(np.argmin(evs))])


def _prior_fisher_tau(cfg):
    """Best single tau under the prior: max expected Fisher information."""
    from .crb import fisher_matrix
    taus = np.geomspace(cfg["tau_min_s"], cfg["tau_max_s"], 60)
    grid = np.linspace(*cfg["delta_range_hz"], 40)
    best, best_tau = -1.0, taus[0]
    for t in taus:
        fi = np.mean([
            fisher_matrix(np.array([t]), (d, cfg["t2star_s"]),
                          cfg["readout"], cfg["n_shots_per_batch"])[0, 0]
            for d in grid])
        if fi > best:
            best, best_tau = fi, float(t)
    return best_tau


def make_schedule(kind, cfg):
    """Return schedule(step, posterior) -> tau_s. Fixed kinds ignore the
    posterior; 'adaptive' re-optimizes each batch."""
    if kind == "adaptive":
        tau_grid = np.geomspace(cfg["tau_min_s"], cfg["tau_max_s"], 60)
        return lambda step, post: choose_tau(post, tau_grid,
                                             cfg["n_shots_per_batch"])
    if kind == "fixed_tau":
        tau = _prior_fisher_tau(cfg)
        return lambda step, post: tau
    if kind == "linear_sweep":
        taus = np.linspace(cfg["tau_min_s"], cfg["tau_max_s"],
                           cfg["n_sweep_points"])
        return lambda step, post: float(taus[step % len(taus)])
    if kind == "exp_ladder":
        ladder = []
        t = cfg["tau_min_s"]
        while t <= cfg["tau_max_s"]:
            ladder.append(t)
            t *= 2
        return lambda step, post: float(ladder[step % len(ladder)])
    raise ValueError(f"unknown schedule: {kind}")


def simulate_run(true_delta_hz, kind, cfg, rng):
    """One sequential experiment under a schedule; honest wall-clock cost."""
    from .model import expected_counts

    post = DeltaPosterior(tuple(cfg["delta_range_hz"]),
                          t2star_s=cfg["t2star_s"],
                          readout_cfg=cfg["readout"])
    schedule = make_schedule(kind, cfg)
    t = cfg["timing"]
    n_b = cfg["n_shots_per_batch"]
    out = {"kind": kind, "wall_time_s": [], "sigma_hz": [],
           "abs_err_hz": [], "tau_s": []}
    elapsed, step = 0.0, 0
    while elapsed < cfg["time_budget_s"]:
        tau = schedule(step, post)
        lam = expected_counts([tau], true_delta_hz, cfg["t2star_s"],
                              cfg["readout"], n_b)[0]
        post.update(rng.poisson(lam), tau, n_b)
        elapsed += n_b * (t["t_init_s"] + tau + t["t_read_s"] + t["t_dead_s"])
        step += 1
        out["wall_time_s"].append(elapsed)
        out["sigma_hz"].append(post.sigma())
        out["abs_err_hz"].append(abs(post.mean() - true_delta_hz))
        out["tau_s"].append(tau)
    return out
