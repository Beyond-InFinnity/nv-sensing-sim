"""Bayesian grid posterior over (delta, T2*) with exact Poisson likelihood.

DECISION (Phase 2, recorded in CLAUDE.md): grid, not SMC — 2 parameters on a
400x60 grid is exact, vectorized, and fast per record; SMC buys nothing until
the adaptive/sequential setting of Phase 3.

The delta grid zooms adaptively: if the marginal posterior is narrower than
~5 grid cells it is re-evaluated on a finer grid centered on the posterior
mean (up to 3 refinements). The zoom triggers only for narrow unimodal
posteriors — wide or multimodal (fringe-ambiguous) posteriors keep the full
prior range, so no mode is ever clipped."""
import numpy as np
from scipy.special import gammaln

from .model import expected_counts

_MAX_ZOOMS = 3


def _grid_posterior(counts, taus_s, readout_cfg, n_shots, deltas, t2ss):
    logl = np.empty((len(deltas), len(t2ss)))
    norm = gammaln(counts + 1).sum()
    for k, t2s in enumerate(t2ss):
        lam = np.stack([expected_counts(taus_s, d, t2s, readout_cfg, n_shots)
                        for d in deltas])
        logl[:, k] = (counts * np.log(lam) - lam).sum(axis=1) - norm
    log_evidence = float(np.log(np.exp(logl - logl.max()).sum()) + logl.max())
    post = np.exp(logl - logl.max())
    post /= post.sum()
    return post, log_evidence


def fit_bayes(counts, taus_s, readout_cfg, n_shots,
              delta_range_hz=(0.2e6, 4e6), t2s_range_s=(0.5e-6, 6e-6),
              n_delta=400, n_t2s=60):
    counts = np.asarray(counts, dtype=float)
    t2ss = np.geomspace(*t2s_range_s, n_t2s)
    lo, hi = delta_range_hz
    for _ in range(_MAX_ZOOMS + 1):
        deltas = np.linspace(lo, hi, n_delta)
        spacing = deltas[1] - deltas[0]
        post, log_evidence = _grid_posterior(
            counts, taus_s, readout_cfg, n_shots, deltas, t2ss)
        p_delta = post.sum(axis=1)
        mean = float((p_delta * deltas).sum())
        sigma = float(np.sqrt((p_delta * (deltas - mean) ** 2).sum()))
        if sigma >= 5 * spacing:
            break
        # posterior unresolved on this grid: zoom in around the mean
        half = max(8 * sigma, 5 * spacing)
        lo = max(delta_range_hz[0], mean - half)
        hi = min(delta_range_hz[1], mean + half)
    p_t2s = post.sum(axis=0)
    return {"delta_hz": mean, "delta_sigma_hz": sigma,
            "t2star_s": float((p_t2s * t2ss).sum()),
            "log_evidence": log_evidence}
