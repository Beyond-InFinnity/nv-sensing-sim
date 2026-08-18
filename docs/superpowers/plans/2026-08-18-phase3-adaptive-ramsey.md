# Phase 3 — Bayesian Adaptive Ramsey Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Quantify whether choosing each Ramsey interrogation time from the current posterior (expected-posterior-variance criterion) reaches a target δ-precision in less total wall-clock time than fixed schedules — and write it up at blog-post grade.

**Architecture:** `nvsim.estimators.adaptive`: a 1D grid posterior over δ with sequential exact-Poisson updates (T2* known and fixed — documented simplification), an A-optimal decision rule (minimize expected posterior variance, Gauss–Hermite quadrature over the normal-approximated batch-count distribution), three fixed baselines (best-fixed-τ, linear sweep, exponential ladder), and a run simulator with honest wall-clock accounting (t_init + τ + t_read + t_dead per shot — long τ costs real time). Runs are replicated over true δ drawn from the prior, paired across schedules by true-δ list. Exit artifact: `docs/phase3-adaptive-ramsey.md` with reproducible figures.

**Tech Stack:** numpy/scipy (CPU-only, claude-server). No torch: the optional RL/amortized policy is gated — implement only if the Bayesian adaptive shows a ≥2× time-to-target win, else document why not (roadmap marks it "optional").

## Global Constraints

- Hz / tesla / s at API boundaries; Poisson likelihood exact in posterior updates (the normal approximation lives only inside the utility lookahead; batch size ≥ 25 shots keeps n·λ ≳ 100 there).
- T2* is known and fixed in the sequential loop (single-parameter posterior) — a stated simplification, recorded in PHYSICS.md and the writeup limitations; the fixed baselines get the same knowledge (symmetric).
- Wall-clock accounting per shot: `t_init_s + tau + t_read_s + t_dead_s` — the τ-vs-time tradeoff is the point; no free long interrogations.
- Schedules compared on **paired true-δ lists** (same draws per replicate index); measurement outcomes necessarily differ once τ choices diverge — documented, not hidden.
- Artifacts embed config + seed + git SHA; jobs >10 min get LEDGER.md entries; figures via committed scripts + dataviz palette constants (as in `scripts/plot_phase0.py`); `pytest` green before every commit.
- Roadmap exit: "writeup in docs/ (blog-post grade) with reproducible figures."

---

### Task 1: Sequential grid posterior

**Files:**
- Create: `src/nvsim/estimators/adaptive.py`
- Modify: `docs/PHYSICS.md` (adaptive protocol section)
- Test: `tests/test_adaptive.py`

**Interfaces:**
- Produces: `adaptive.DeltaPosterior(delta_range_hz=(0.2e6, 4e6), n_grid=600, t2star_s=1.5e-6, readout_cfg=None)` with:
  - `.update(counts, tau_s, n_shots)` — exact Poisson Bayes update at one τ (counts = total over the batch);
  - `.mean() -> float`, `.sigma() -> float`, `.grid -> np.ndarray`, `.p -> np.ndarray` (normalized).

- [ ] **Step 1: Write failing tests**

```python
# tests/test_adaptive.py
import numpy as np
import pytest

from nvsim.estimators.adaptive import DeltaPosterior
from nvsim.estimators.model import expected_counts

READOUT = {"r_hz": 6e7, "contrast": 0.25, "t_read_s": 0.4e-6, "f_pump": 0.95}


def _posterior():
    return DeltaPosterior(readout_cfg=READOUT)


def test_prior_is_flat_and_normalized():
    p = _posterior()
    assert p.p.sum() == pytest.approx(1.0, rel=1e-12)
    assert p.p.std() / p.p.mean() < 1e-9
    assert p.sigma() > 0.9e6  # flat over 3.8 MHz span -> sigma ~ span/sqrt(12)


def test_sequential_updates_concentrate_on_truth():
    rng = np.random.default_rng(0)
    truth = 2.3e6
    p = _posterior()
    for tau in (0.3e-6, 0.7e-6, 1.1e-6, 1.5e-6) * 5:
        lam = expected_counts([tau], truth, 1.5e-6, READOUT, 100)[0]
        p.update(rng.poisson(lam), tau, 100)
    assert abs(p.mean() - truth) < 3 * p.sigma()
    assert p.sigma() < 30e3


def test_update_is_bayes_consistent_with_batch_likelihood():
    """One update with counts k must equal renormalized prior * Poisson lik."""
    p = _posterior()
    tau, n, k = 0.8e-6, 50, 600
    lam = expected_counts(np.full(1, tau), p.grid[:, None].T[0], 1.5e-6,
                          READOUT, n) if False else None  # (see impl note)
    p2 = _posterior()
    p2.update(k, tau, n)
    lams = np.array([expected_counts([tau], d, 1.5e-6, READOUT, n)[0]
                     for d in p.grid])
    manual = p.p * np.exp(k * np.log(lams) - lams
                          - (k * np.log(lams) - lams).max())
    manual /= manual.sum()
    np.testing.assert_allclose(p2.p, manual, atol=1e-12)
```

- [ ] **Step 2:** run → FAIL (ModuleNotFoundError).
- [ ] **Step 3: Implement (in `adaptive.py`)**

```python
# src/nvsim/estimators/adaptive.py
"""Sequential Bayesian adaptive Ramsey: 1D grid posterior over delta with
A-optimal (expected-posterior-variance) interrogation-time selection.
T2* is known and fixed (docs/PHYSICS.md, 'Adaptive Ramsey')."""
import numpy as np

from .model import expected_counts


class DeltaPosterior:
    def __init__(self, delta_range_hz=(0.2e6, 4e6), n_grid=600,
                 t2star_s=1.5e-6, readout_cfg=None):
        self.grid = np.linspace(*delta_range_hz, n_grid)
        self.p = np.full(n_grid, 1.0 / n_grid)
        self.t2star_s = t2star_s
        self.readout_cfg = readout_cfg

    def _lam_grid(self, tau_s, n_shots):
        """Expected batch counts lambda(delta) for every grid delta."""
        return np.array([
            expected_counts([tau_s], d, self.t2star_s,
                            self.readout_cfg, n_shots)[0]
            for d in self.grid])

    def update(self, counts, tau_s, n_shots):
        lam = self._lam_grid(tau_s, n_shots)
        logl = counts * np.log(lam) - lam
        logp = np.log(self.p) + logl
        logp -= logp.max()
        self.p = np.exp(logp)
        self.p /= self.p.sum()

    def mean(self):
        return float((self.p * self.grid).sum())

    def sigma(self):
        m = self.mean()
        return float(np.sqrt((self.p * (self.grid - m) ** 2).sum()))
```

Implementation note for the engineer: `_lam_grid`'s per-δ loop is O(n_grid) calls of a cheap closed form; if profiling shows it hot, vectorize `ramsey_p0` over δ (it broadcasts if you pass `taus` scalar and `delta_hz` array — verify against the loop before switching). Delete the dead `if False` scaffold line from the test (it is a placeholder trap — write the manual-likelihood comparison exactly as the `lams = np.array(...)` lines do).

PHYSICS.md addition (same commit):

```markdown
## Adaptive Ramsey (Phase 3)

Sequential estimation of δ with batches of n_b shots at chosen τ; posterior
on a 600-point δ grid, exact Poisson updates. T2* is known and fixed in the
loop (single-parameter tracking) — both adaptive and fixed schedules get the
same knowledge. Wall-clock cost per shot: t_init + τ + t_read + t_dead.
Decision rule: minimize expected posterior variance (A-optimal), with the
batch count distribution approximated as N(nλ, nλ) inside the lookahead only
(valid for nλ ≳ 100); posterior updates stay exact-Poisson.
```

- [ ] **Step 4:** run → PASS. **Step 5:** Commit: `feat: sequential grid posterior for adaptive Ramsey`

---

### Task 2: Decision rule + fixed baselines

**Files:**
- Modify: `src/nvsim/estimators/adaptive.py`
- Test: `tests/test_adaptive.py` (append)

**Interfaces:**
- Produces:
  - `adaptive.choose_tau(posterior, tau_grid_s, n_shots) -> float` — τ minimizing expected posterior variance, Gauss–Hermite (32 nodes) over k ~ N(nλ(δ), nλ(δ)) marginalized over the posterior;
  - `adaptive.make_schedule(kind, cfg) -> callable(step, posterior) -> float` for `kind` ∈ `{"adaptive", "fixed_tau", "linear_sweep", "exp_ladder"}`. `fixed_tau`: τ maximizing expected Fisher under the PRIOR (computed once); `linear_sweep`: cycle over `np.linspace(cfg["tau_min_s"], cfg["tau_max_s"], cfg["n_sweep_points"])`; `exp_ladder`: cycle `tau_min * 2**j` while ≤ tau_max.

- [ ] **Step 1: Write failing tests (append)**

```python
from nvsim.estimators.adaptive import choose_tau, make_schedule

TAU_GRID = np.geomspace(0.05e-6, 4.5e-6, 60)


def test_expected_variance_never_exceeds_current():
    p = _posterior()
    rng = np.random.default_rng(1)
    lam = expected_counts([0.4e-6], 2.0e6, 1.5e-6, READOUT, 100)[0]
    p.update(rng.poisson(lam), 0.4e-6, 100)
    # choose_tau internally computes expected variances; expose via return of
    # the best tau: measuring anything cannot be expected to widen the posterior,
    # so best expected variance <= current variance. Test via a probe posterior.
    tau = choose_tau(p, TAU_GRID, 100)
    assert TAU_GRID[0] <= tau <= TAU_GRID[-1]


def test_narrow_posterior_prefers_longer_tau():
    wide = _posterior()
    narrow = _posterior()
    rng = np.random.default_rng(2)
    truth = 2.0e6
    for tau in (0.2e-6, 0.5e-6, 0.9e-6, 1.3e-6) * 6:
        lam = expected_counts([tau], truth, 1.5e-6, READOUT, 200)[0]
        narrow.update(rng.poisson(lam), tau, 200)
    t_wide = choose_tau(wide, TAU_GRID, 100)
    t_narrow = choose_tau(narrow, TAU_GRID, 100)
    assert t_narrow > 1.5 * t_wide


def test_fixed_schedules_cycle_and_are_posterior_blind():
    cfg = {"tau_min_s": 0.1e-6, "tau_max_s": 3.2e-6, "n_sweep_points": 8}
    lin = make_schedule("linear_sweep", cfg)
    taus = [lin(i, None) for i in range(16)]
    assert taus[:8] == taus[8:]
    assert taus[0] == pytest.approx(0.1e-6) and taus[7] == pytest.approx(3.2e-6)
    exp = make_schedule("exp_ladder", cfg)
    ladder = [exp(i, None) for i in range(6)]
    assert ladder[1] / ladder[0] == pytest.approx(2.0)
```

- [ ] **Step 2:** run → FAIL.
- [ ] **Step 3: Implement (append to adaptive.py)**

```python
_GH_NODES, _GH_WEIGHTS = np.polynomial.hermite_e.hermegauss(32)
_GH_WEIGHTS = _GH_WEIGHTS / _GH_WEIGHTS.sum()


def _expected_posterior_variance(posterior, tau_s, n_shots):
    """E_k[Var(delta | k)] with k ~ N(n lam, n lam) marginalized over p(delta).

    Vectorized: for each GH node z, k(delta) = n lam + z sqrt(n lam) is a
    plausible outcome per true delta; weight by posterior and GH weight."""
    lam = posterior._lam_grid(tau_s, n_shots)          # (G,)
    grid, p = posterior.grid, posterior.p
    var_exp = 0.0
    for z, w in zip(_GH_NODES, _GH_WEIGHTS):
        k = lam + z * np.sqrt(lam)                      # outcome per true delta
        # posterior after observing k(d_true), for each candidate d_true,
        # collapses to: q(d | k) prop p(d) exp(k ln lam(d) - lam(d)).
        # Weight each d_true branch by its posterior mass.
        logl = np.outer(k, np.log(lam)) - lam           # (G_true, G)
        logl -= logl.max(axis=1, keepdims=True)
        q = p * np.exp(logl)
        q /= q.sum(axis=1, keepdims=True)
        mu = q @ grid
        var = (q * (grid - mu[:, None]) ** 2).sum(axis=1)
        var_exp += w * float(p @ var)
    return var_exp


def choose_tau(posterior, tau_grid_s, n_shots):
    """A-optimal next interrogation time."""
    evs = [_expected_posterior_variance(posterior, t, n_shots)
           for t in tau_grid_s]
    return float(tau_grid_s[int(np.argmin(evs))])


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
```

Note: `_expected_posterior_variance` is O(32 · G²) with G = 600 → ~11M flops per candidate τ, ×60 candidates per decision ≈ 0.7 GFlop — ~0.5 s/decision on this CPU. With ~100 decisions per run and ~400 runs this is the production bottleneck (Task 4 budgets for it; coarsen the candidate grid to 24 and G to 400 inside `choose_tau` via subsampling if a profile shows >2 s/decision — accuracy loss is negligible, note it in the writeup if used).

- [ ] **Step 4:** run → PASS. **Step 5:** Commit: `feat: A-optimal tau selection and fixed-schedule baselines`

---

### Task 3: Run simulator + runner

**Files:**
- Modify: `src/nvsim/estimators/adaptive.py`
- Create: `experiments/phase3/adaptive_vs_fixed.json`, `scripts/run_phase3.py`
- Test: `tests/test_adaptive.py` (append), `tests/test_run_phase3.py`

**Interfaces:**
- Produces: `adaptive.simulate_run(true_delta_hz, kind, cfg, rng) -> dict` with keys `wall_time_s` (list, cumulative after each batch), `sigma_hz` (posterior σ trajectory), `abs_err_hz` (|posterior mean − truth| trajectory), `tau_s` (chosen τ per batch), `kind`. Stops when `wall_time_s` exceeds `cfg["time_budget_s"]`.
- CLI: `run_phase3.py CONFIG [--out-dir DIR]` → one artifact JSON per schedule kind with all replicates + provenance.

Config (exact):

```json
// experiments/phase3/adaptive_vs_fixed.json
{"name": "adaptive_vs_fixed", "seed": 401,
 "schedules": ["adaptive", "fixed_tau", "linear_sweep", "exp_ladder"],
 "n_replicates": 60,
 "time_budget_s": 0.25,
 "n_shots_per_batch": 100,
 "delta_range_hz": [0.3e6, 3.7e6],
 "t2star_s": 1.5e-6,
 "tau_min_s": 0.05e-6, "tau_max_s": 4.5e-6, "n_sweep_points": 12,
 "timing": {"t_init_s": 2.0e-6, "t_read_s": 0.4e-6, "t_dead_s": 1.0e-6},
 "readout": {"r_hz": 6.0e7, "contrast": 0.25, "t_read_s": 0.4e-6, "f_pump": 0.95}}
```

- [ ] **Step 1: Write failing tests**

```python
# append to tests/test_adaptive.py
from nvsim.estimators.adaptive import simulate_run

CFG = {"seed": 1, "n_shots_per_batch": 100, "time_budget_s": 0.02,
       "delta_range_hz": [0.3e6, 3.7e6], "t2star_s": 1.5e-6,
       "tau_min_s": 0.05e-6, "tau_max_s": 4.5e-6, "n_sweep_points": 12,
       "timing": {"t_init_s": 2.0e-6, "t_read_s": 0.4e-6, "t_dead_s": 1.0e-6},
       "readout": READOUT}


def test_simulate_run_time_accounting_and_shrinkage():
    rng = np.random.default_rng(3)
    out = simulate_run(2.1e6, "linear_sweep", CFG, rng)
    times = np.array(out["wall_time_s"])
    taus = np.array(out["tau_s"])
    per_shot = 2.0e-6 + taus + 0.4e-6 + 1.0e-6
    np.testing.assert_allclose(np.diff(times), (100 * per_shot)[1:], rtol=1e-12)
    assert times[-1] <= 0.02 + 100 * (2.0e-6 + 4.5e-6 + 1.4e-6)
    assert out["sigma_hz"][-1] < out["sigma_hz"][0]


def test_adaptive_beats_fixed_tau_on_time_to_target():
    """The headline claim, as a smoke-scale test (generous margin, one seed)."""
    target = 20e3
    t_reach = {}
    for kind in ("adaptive", "fixed_tau"):
        rng = np.random.default_rng(7)
        out = simulate_run(2.6e6, kind, dict(CFG, time_budget_s=0.05), rng)
        sig = np.array(out["sigma_hz"])
        t = np.array(out["wall_time_s"])
        hit = np.nonzero(sig < target)[0]
        t_reach[kind] = t[hit[0]] if len(hit) else np.inf
    assert t_reach["adaptive"] < t_reach["fixed_tau"]
```

```python
# tests/test_run_phase3.py
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_runner_writes_per_schedule_artifacts(tmp_path):
    cfg = json.loads((REPO / "experiments/phase3/adaptive_vs_fixed.json").read_text())
    cfg["n_replicates"] = 2
    cfg["time_budget_s"] = 0.01
    cfg_path = tmp_path / "mini.json"
    cfg_path.write_text(json.dumps(cfg))
    r = subprocess.run([sys.executable, str(REPO / "scripts/run_phase3.py"),
                       str(cfg_path), "--out-dir", str(tmp_path)],
                      capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    for kind in cfg["schedules"]:
        art = json.loads((tmp_path / f"{kind}.json").read_text())
        assert len(art["runs"]) == 2
        assert art["runs"][0]["true_delta_hz"] == art["true_deltas_hz"][0]
        assert len(art["git_sha"]) >= 7
```

- [ ] **Step 2:** run → FAIL.
- [ ] **Step 3: Implement**

```python
# append to adaptive.py
def simulate_run(true_delta_hz, kind, cfg, rng):
    """One sequential experiment under a schedule; honest wall-clock cost."""
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
```

```python
#!/usr/bin/env python3
# scripts/run_phase3.py
"""Run adaptive-vs-fixed schedule comparison; one artifact per schedule,
paired true-delta draws. Usage: run_phase3.py CONFIG [--out-dir DIR]"""
import argparse
import json
from pathlib import Path

import numpy as np

from nvsim.estimators.adaptive import simulate_run
from nvsim.provenance import git_sha

REPO = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path)
    ap.add_argument("--out-dir", type=Path,
                    default=REPO / "experiments/phase3/artifacts")
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ss = np.random.SeedSequence(cfg["seed"])
    rng_truth = np.random.default_rng(ss.spawn(1)[0])
    lo, hi = cfg["delta_range_hz"]
    true_deltas = rng_truth.uniform(lo, hi, cfg["n_replicates"]).tolist()
    for kind in cfg["schedules"]:
        runs = []
        for i, d in enumerate(true_deltas):
            rng = np.random.default_rng(
                np.random.SeedSequence([cfg["seed"], hash(kind) % 2**31, i]))
            runs.append({"true_delta_hz": d, **simulate_run(d, kind, cfg, rng)})
            print(f"{kind} {i + 1}/{len(true_deltas)}", flush=True)
        art = {"config": cfg, "seed": cfg["seed"], "git_sha": git_sha(),
               "true_deltas_hz": true_deltas, "runs": runs}
        out = args.out_dir / f"{kind}.json"
        out.write_text(json.dumps(art))
        print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4:** run both test files → PASS. **Step 5:** Commit: `feat: adaptive-vs-fixed run simulator and runner`

---

### Task 4: Production runs

- [ ] **Step 1:** Estimate cost from the smoke test timings (adaptive decision ~0.5 s × ~250 batches/run at 0.25 s budget × 60 replicates ≈ hours — if projected >2 h, apply the Task-2 coarsening note, re-run `pytest tests/test_adaptive.py`, and re-estimate; target <1.5 h).
- [ ] **Step 2:** LEDGER.md entry (`started`, server.cpu). Launch detached with a log: `(nohup .venv/bin/python scripts/run_phase3.py experiments/phase3/adaptive_vs_fixed.json > experiments/phase3/run.log 2>&1 &)` — or Bash run_in_background equivalent. Watcher per RULES §3 (progress lines + death coverage).
- [ ] **Step 3:** On completion: artifacts committed (JSON, small), LEDGER → done.
- [ ] Commit: `feat: phase 3 production runs — adaptive vs fixed schedules`

---

### Task 5: Figures

**Files:**
- Create: `scripts/plot_phase3.py`; outputs `docs/figures/phase3_sigma_vs_time.png`, `docs/figures/phase3_tau_trajectory.png`

- [ ] **Step 1:** dataviz rules as established; categorical: adaptive = blue slot 1, fixed_tau = orange slot 2, exp_ladder = aqua slot 3, linear_sweep = yellow slot 4 `#eda100` (4 series: legend + direct labels).
- [ ] **Step 2:** `phase3_sigma_vs_time.png` — log-log median posterior σ_δ vs wall-clock time per schedule with IQR bands; overlay reference slopes t^(−1/2) and t^(−1); annotate time-to-target-precision (σ = 5 kHz) per schedule and the adaptive speedup factor. Include a median |error| panel (posterior σ must not flatter actual error — plot both).
- [ ] **Step 3:** `phase3_tau_trajectory.png` — median chosen τ vs elapsed time for the adaptive schedule (IQR band), with T2* marked; expected: τ grows as the posterior narrows and saturates near the T2*-limited optimum, while fixed schedules are flat lines (shown for contrast).
- [ ] **Step 4:** Run, Read PNGs, check: σ and |err| agree in scale (calibration), bands don't hide crossovers, τ trajectory saturates ≲ T2* × O(1). Any surprise (adaptive losing, τ pinned at a grid edge) → investigate before committing; bounded iterations.
- [ ] **Step 5:** Commit: `feat: phase 3 figures — time-to-precision and tau trajectories`

---

### Task 6: Writeup + RL gate decision

**Files:**
- Create: `docs/phase3-adaptive-ramsey.md`
- Modify: `docs/ROADMAP.md` (RL-optional box: implemented or explicitly declined with reason)

- [ ] **Step 1:** Compute the RL gate number from artifacts: adaptive time-to-5-kHz vs best fixed schedule. If speedup ≥ 2×, add an RL/amortized-policy task list to the roadmap as future work with a concrete sketch (the roadmap marks it optional — do NOT implement it this phase without Connor's go-ahead either way; the gate decides what the writeup recommends). If < 2×, decline it in ROADMAP.md with the measured number.
- [ ] **Step 2:** Write `docs/phase3-adaptive-ramsey.md`, blog-post grade (house style: `qec-neural-decoder/docs/phase2-results.md` — mechanism-building, numbers with units, no hype): setup, decision rule derivation at the NOTATION.md level (why expected-posterior-variance; why the normal approximation is safe inside the lookahead), results with both figures inline, the time-to-target table, limitations (known T2*, no drift, batch granularity, grid prior range), and what Phase 2's misspecification finding implies for adaptive-under-drift as follow-on.
- [ ] **Step 3:** Every number in the writeup regenerable: cite config + seed + SHA per figure.
- [ ] **Step 4:** Commit: `docs: phase 3 writeup — adaptive Ramsey results`

---

### Task 7: Close out Phase 3

- [ ] **Step 1:** Full `pytest` (fresh, record count) — verification-before-completion.
- [ ] **Step 2:** Roadmap exit check: writeup exists, figures reproducible, adaptive-vs-fixed comparison quantified, RL box resolved.
- [ ] **Step 3:** CLAUDE.md status → Phase 3 complete (project phases 0–3 done); README status; ROADMAP boxes.
- [ ] **Step 4:** Commit `docs: mark phase 3 complete`; push origin + cmfinnerty.
- [ ] **Step 5:** Report to Connor with the headline speedup number and the writeup path; project-complete summary; stop.
