# Phase 2 — Estimators Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Quantify how much sensitivity better estimation buys: least-squares (lab default), Bayesian grid posterior, and a small NN, all on identical Ramsey records, benchmarked against the Poisson Cramér–Rao bound across an SNR range, with the answer in nT/√Hz.

**Architecture:** A `nvsim.estimators` subpackage: `model.py` (analytic Ramsey expected-counts model — the shared forward model, unit-tested identical to the qutip generator), `crb.py` (Poisson Fisher matrix → CRB), `lsq.py`, `bayes.py` (grid posterior — **decision: grid, not SMC**; 2-parameter problems don't need SMC, revisit for Phase 3 adaptive), `nn.py` (torch 1D-CNN, heteroscedastic Gaussian head; torch imported lazily so the base package stays torch-free). Estimation problem: θ = (δ, T2*) from one sweep of Ramsey counts with known calibrated readout; δ → B via γ. Eval harness runs all estimators over the same dataset files (pairing is automatic). NN training runs on the workstation RTX 3070 (torch cuda:0) under homelab-orchestration rules; everything else on claude-server.

**Tech Stack:** Python ≥3.11, numpy/scipy/qutip (base), torch ≥2.3 (`.[ml]` extra, workstation only), pytest.

## Global Constraints

- Hz / tesla / SI at API boundaries. Estimator comparisons run on **identical noise realizations (paired)** — enforced structurally by consuming the same .npz dataset files.
- Poisson likelihood everywhere — no Gaussian approximations in the Bayesian estimator or CRB.
- torch is an `ml` extra; `src/nvsim/` base modules must import without torch installed. NN training only on the workstation RTX 3070 = **torch cuda:0** (nvidia-smi order is REVERSED; MACHINES.md NB).
- Orchestration law (homelab-orchestration/RULES.md): claim the 3070 by editing MACHINES.md **before** launching (§6); LEDGER.md entry for jobs >10 min; detached launch `(nohup <cmd> > <log> 2>&1 < /dev/null &)` (§2); watchers use bracketed pgrep `pgrep -f "[t]rain_nn"` and check results-file AND process-alive (§3); code flows server→workstation via git, results scp back and committed on the server, then `scripts/safe_pull.sh` on the workstation (§5); checkpoint weights to disk immediately after training, before eval (§4); grad-clip 1.0 + lr ≤ 1e-3 + 500-step linear warmup (§4).
- Artifacts embed config + seed + git SHA. Figures from committed scripts reading committed artifacts; dataviz skill; palette constants as in `scripts/plot_phase0.py`.
- Every sensitivity number states its photon budget (PHYSICS.md).
- `pytest` green before every commit. Record the grid-vs-SMC decision in CLAUDE.md "Key technical decisions" in the same commit as `bayes.py`.

---

### Task 1: Analytic forward model (+ fast generator path)

**Files:**
- Create: `src/nvsim/estimators/__init__.py` (empty), `src/nvsim/estimators/model.py`
- Modify: `src/nvsim/experiment.py` (Ramsey branch uses the analytic model), `docs/PHYSICS.md`
- Test: `tests/test_est_model.py`

**Interfaces:**
- Produces: `model.ramsey_p0(taus_s, delta_hz, t2star_s) -> np.ndarray` — ½(1 + cos(2πδτ)·e^(−τ/T2*)); `model.expected_counts(taus_s, delta_hz, t2star_s, readout_cfg, n_shots) -> np.ndarray` (= n_shots · mean_counts_per_shot(p0, readout_cfg)).
- Modifies: `experiment._p0_pulsed` ramsey branch calls `ramsey_p0` per point (same physics, ~100× faster — needed for NN-scale dataset generation).

- [ ] **Step 1: Write failing tests**

```python
# tests/test_est_model.py
import numpy as np
import pytest

from nvsim.estimators.model import expected_counts, ramsey_p0
from nvsim.pulsed import ramsey

READOUT = {"r_hz": 6e7, "contrast": 0.25, "t_read_s": 0.4e-6, "f_pump": 0.95}


def test_analytic_matches_mesolve_generator():
    taus = np.linspace(0, 5e-6, 40)
    for delta, t2s in ((2e6, 1.5e-6), (0.7e6, 3e-6), (-1.2e6, 2e-6)):
        np.testing.assert_allclose(
            ramsey_p0(taus, delta, t2s),
            ramsey(taus, detuning_hz=delta, t2star_s=t2s), atol=1e-8)


def test_analytic_matches_mesolve_no_decay():
    taus = np.linspace(0, 3e-6, 30)
    np.testing.assert_allclose(
        ramsey_p0(taus, 1.5e6, None), ramsey(taus, detuning_hz=1.5e6), atol=1e-8)


def test_expected_counts_positive_and_scaled():
    taus = np.linspace(0, 5e-6, 150)
    lam = expected_counts(taus, 2e6, 1.5e-6, READOUT, 2000)
    assert lam.shape == (150,)
    assert np.all(lam > 0)
    np.testing.assert_allclose(
        expected_counts(taus, 2e6, 1.5e-6, READOUT, 4000), 2 * lam, rtol=1e-12)


def test_experiment_ramsey_unchanged_by_fast_path():
    # regression: the API's ideal curve equals the analytic model
    from nvsim.experiment import run_experiment
    cfg = {"name": "t", "protocol": "ramsey", "seed": 11,
           "sweep": {"min": 0.0, "max": 4e-6, "n_points": 40},
           "n_sweeps": 2, "n_shots": 200,
           "truth": {"detuning_hz": 2e6, "t2star_s": 3e-6},
           "timing": {"t_init_s": 2e-6, "t_read_s": 0.4e-6, "t_dead_s": 1e-6},
           "readout": READOUT, "drift": {}}
    ds = run_experiment(cfg)
    taus = ds["sweep_values"]
    np.testing.assert_allclose(ds["truth"]["p0_ideal"],
                               ramsey_p0(taus, 2e6, 3e-6), atol=1e-8)
```

- [ ] **Step 2:** `pytest tests/test_est_model.py -v` → FAIL (ModuleNotFoundError).
- [ ] **Step 3: Implement**

```python
# src/nvsim/estimators/model.py
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
```

In `experiment.py`, replace the ramsey branch of `_p0_pulsed`:

```python
            elif proto == "ramsey":
                p0[i, j] = ramsey_p0([x], tr["detuning_hz"] + det_shift[i, j],
                                     tr.get("t2star_s"))[0]
```

with `from .estimators.model import ramsey_p0` added to the imports (import at module top is fine — estimators.model has no torch).

PHYSICS.md note (same commit), under "Pulsed two-level reduction": "The Ramsey record generator uses the closed-form P₀ (identical to the mesolve path, unit-tested to 1e-8) for speed; Rabi and echo still integrate the master equation."

- [ ] **Step 4:** `pytest tests/test_est_model.py tests/test_experiment.py -v` → PASS (regenerating Phase 1 artifacts is NOT needed — same physics).
- [ ] **Step 5:** Commit: `feat: analytic Ramsey forward model shared by estimators and generator`

---

### Task 2: Poisson CRB

**Files:**
- Create: `src/nvsim/estimators/crb.py`
- Modify: `docs/PHYSICS.md` (CRB + sensitivity conventions)
- Test: `tests/test_crb.py`

**Interfaces:**
- Produces: `crb.fisher_matrix(taus_s, theta, readout_cfg, n_shots) -> np.ndarray (2,2)` with θ = (delta_hz, t2star_s), Poisson Fisher I_jk = Σ_i (∂λᵢ/∂θⱼ)(∂λᵢ/∂θₖ)/λᵢ via central differences; `crb.crb_sigma_delta(taus_s, theta, readout_cfg, n_shots) -> float` (√[I⁻¹]₀₀, Hz); `crb.sigma_b_tesla(sigma_delta_hz) -> float` (σ_δ/γ).

- [ ] **Step 1: Write failing tests**

```python
# tests/test_crb.py
import numpy as np
import pytest

from nvsim.constants import GAMMA_E_HZ_PER_T
from nvsim.estimators.crb import crb_sigma_delta, fisher_matrix, sigma_b_tesla

READOUT = {"r_hz": 6e7, "contrast": 0.25, "t_read_s": 0.4e-6, "f_pump": 0.95}
TAUS = np.linspace(0, 5e-6, 150)
THETA = (2e6, 1.5e-6)


def test_fisher_symmetric_positive_definite():
    fim = fisher_matrix(TAUS, THETA, READOUT, 2000)
    assert fim.shape == (2, 2)
    np.testing.assert_allclose(fim, fim.T, rtol=1e-9)
    assert np.all(np.linalg.eigvalsh(fim) > 0)


def test_crb_scales_as_one_over_sqrt_n():
    s1 = crb_sigma_delta(TAUS, THETA, READOUT, 200)
    s2 = crb_sigma_delta(TAUS, THETA, READOUT, 20000)
    assert s1 / s2 == pytest.approx(10.0, rel=1e-3)


def test_crb_against_analytic_single_parameter():
    """Gaussian-limit check: for lam_i = A(1 + c cos(2 pi d t_i))/1 with known
    T2*->inf and c<<1, Fisher for delta reduces to
    I = sum_i (dlam/dd)^2/lam_i; compare a hand-computed sum."""
    taus = np.linspace(0, 4e-6, 80)
    readout = {"r_hz": 6e7, "contrast": 0.02, "t_read_s": 0.4e-6, "f_pump": 1.0}
    n = 1000
    d = 1.7e6
    lam = n * readout["r_hz"] * readout["t_read_s"] * (
        1 - readout["contrast"] * (1 - 0.5 * (1 + np.cos(2 * np.pi * d * taus))))
    dlam = (n * readout["r_hz"] * readout["t_read_s"] * readout["contrast"]
            * 0.5 * (-2 * np.pi * taus) * np.sin(2 * np.pi * d * taus))
    expected = (dlam**2 / lam).sum()
    fim = fisher_matrix(taus, (d, 1.0), readout, n)  # T2* = 1 s ~ no decay
    assert fim[0, 0] == pytest.approx(expected, rel=1e-4)


def test_sigma_b_conversion():
    assert sigma_b_tesla(28.02e9) == pytest.approx(1.0, rel=1e-12)
    assert sigma_b_tesla(2.802) == pytest.approx(1e-10, rel=1e-9)
```

- [ ] **Step 2:** run → FAIL.
- [ ] **Step 3: Implement**

```python
# src/nvsim/estimators/crb.py
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
    lam = expected_counts(taus_s, *theta, readout_cfg, n_shots)
    g = _grad_lambda(taus_s, theta, readout_cfg, n_shots)
    return (g / lam) @ g.T


def crb_sigma_delta(taus_s, theta, readout_cfg, n_shots):
    """CRB standard deviation on delta (Hz), T2* treated as jointly unknown."""
    return float(np.sqrt(np.linalg.inv(
        fisher_matrix(taus_s, theta, readout_cfg, n_shots))[0, 0]))


def sigma_b_tesla(sigma_delta_hz):
    return sigma_delta_hz / GAMMA_E_HZ_PER_T
```

PHYSICS.md (same commit), extend "Sensitivity accounting": "Phase 2 computes the Poisson CRB directly from the record model: I_jk = Σᵢ ∂λᵢ/∂θⱼ·∂λᵢ/∂θₖ/λᵢ (θ = (δ, T2*) jointly unknown; readout calibrated/known). σ_B = σ_δ/γ. Sensitivity η = σ_B·√T_total with T_total the record's wall-clock duration from the timing model; the photon budget is always stated."

- [ ] **Step 4:** run → PASS. **Step 5:** Commit: `feat: Poisson Cramér-Rao bound for Ramsey records`

---

### Task 3: Least-squares baseline

**Files:**
- Create: `src/nvsim/estimators/lsq.py`
- Test: `tests/test_lsq.py`

**Interfaces:**
- Produces: `lsq.fit_lsq(counts, taus_s, readout_cfg, n_shots, delta_range_hz=(0.2e6, 4e6)) -> dict` with keys `delta_hz`, `t2star_s`, `converged` (bool). Initial δ from the FFT peak of mean-subtracted counts; weighted least squares with σᵢ = √max(countsᵢ, 1) via `scipy.optimize.curve_fit`.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_lsq.py
import numpy as np
import pytest

from nvsim.estimators.crb import crb_sigma_delta
from nvsim.estimators.lsq import fit_lsq
from nvsim.estimators.model import expected_counts

READOUT = {"r_hz": 6e7, "contrast": 0.25, "t_read_s": 0.4e-6, "f_pump": 0.95}
TAUS = np.linspace(0, 5e-6, 150)


def _record(delta, t2s, n_shots, rng):
    return rng.poisson(expected_counts(TAUS, delta, t2s, READOUT, n_shots))


def test_lsq_recovers_truth_at_high_snr():
    rng = np.random.default_rng(0)
    counts = _record(2.1e6, 1.5e-6, 20000, rng)
    est = fit_lsq(counts, TAUS, READOUT, 20000)
    assert est["converged"]
    assert est["delta_hz"] == pytest.approx(2.1e6, rel=2e-3)
    assert est["t2star_s"] == pytest.approx(1.5e-6, rel=0.1)


def test_lsq_rmse_at_or_above_crb():
    rng = np.random.default_rng(1)
    truth = (2.0e6, 1.5e-6)
    n_shots = 2000
    errs = []
    for _ in range(60):
        est = fit_lsq(_record(*truth, n_shots, rng), TAUS, READOUT, n_shots)
        errs.append(est["delta_hz"] - truth[0])
    rmse = np.sqrt(np.mean(np.square(errs)))
    crb = crb_sigma_delta(TAUS, truth, READOUT, n_shots)
    assert rmse > 0.9 * crb          # cannot beat the bound (stat slack)
    assert rmse < 3.0 * crb          # but should be in its vicinity here


def test_lsq_flags_nonconvergence_gracefully():
    counts = np.zeros(len(TAUS), dtype=int)  # pathological record
    est = fit_lsq(counts, TAUS, READOUT, 10)
    assert est["converged"] in (True, False)  # returns, never raises
```

- [ ] **Step 2:** run → FAIL.
- [ ] **Step 3: Implement**

```python
# src/nvsim/estimators/lsq.py
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
```

- [ ] **Step 4:** run → PASS. **Step 5:** Commit: `feat: weighted least-squares baseline estimator`

---

### Task 4: Bayesian grid posterior

**Files:**
- Create: `src/nvsim/estimators/bayes.py`
- Modify: `CLAUDE.md` (record the grid decision under "Key technical decisions")
- Test: `tests/test_bayes.py`

**Interfaces:**
- Produces: `bayes.fit_bayes(counts, taus_s, readout_cfg, n_shots, delta_range_hz=(0.2e6, 4e6), t2s_range_s=(0.5e-6, 6e-6), n_delta=400, n_t2s=60) -> dict` with keys `delta_hz` (posterior mean), `delta_sigma_hz` (posterior std), `t2star_s`, `log_evidence`. Exact Poisson log-likelihood on a (δ, T2*) grid, flat priors over the ranges, marginalized in the T2* direction.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_bayes.py
import numpy as np
import pytest

from nvsim.estimators.bayes import fit_bayes
from nvsim.estimators.crb import crb_sigma_delta
from nvsim.estimators.model import expected_counts

READOUT = {"r_hz": 6e7, "contrast": 0.25, "t_read_s": 0.4e-6, "f_pump": 0.95}
TAUS = np.linspace(0, 5e-6, 150)


def test_posterior_mean_near_truth_high_snr():
    rng = np.random.default_rng(3)
    counts = rng.poisson(expected_counts(TAUS, 2.1e6, 1.5e-6, READOUT, 20000))
    est = fit_bayes(counts, TAUS, READOUT, 20000)
    assert est["delta_hz"] == pytest.approx(2.1e6, rel=2e-3)


def test_posterior_std_tracks_crb_at_high_snr():
    rng = np.random.default_rng(4)
    truth = (2.0e6, 1.5e-6)
    n_shots = 5000
    sigmas = []
    for _ in range(20):
        counts = rng.poisson(expected_counts(TAUS, *truth, READOUT, n_shots))
        sigmas.append(fit_bayes(counts, TAUS, READOUT, n_shots)["delta_sigma_hz"])
    crb = crb_sigma_delta(TAUS, truth, READOUT, n_shots)
    assert np.mean(sigmas) == pytest.approx(crb, rel=0.25)


def test_posterior_coverage_two_sigma():
    rng = np.random.default_rng(5)
    truth = (1.7e6, 2.0e-6)
    hits = 0
    n_rec = 40
    for _ in range(n_rec):
        counts = rng.poisson(expected_counts(TAUS, *truth, READOUT, 500))
        est = fit_bayes(counts, TAUS, READOUT, 500)
        hits += abs(est["delta_hz"] - truth[0]) < 2 * est["delta_sigma_hz"]
    assert hits / n_rec >= 0.85  # ~95% nominal, small-sample slack
```

- [ ] **Step 2:** run → FAIL.
- [ ] **Step 3: Implement**

```python
# src/nvsim/estimators/bayes.py
"""Bayesian grid posterior over (delta, T2*) with exact Poisson likelihood.

DECISION (Phase 2, recorded in CLAUDE.md): grid, not SMC — 2 parameters on a
400x60 grid is exact, vectorized, and ~ms per record; SMC buys nothing until
the adaptive/sequential setting of Phase 3."""
import numpy as np
from scipy.special import gammaln

from .model import expected_counts


def fit_bayes(counts, taus_s, readout_cfg, n_shots,
              delta_range_hz=(0.2e6, 4e6), t2s_range_s=(0.5e-6, 6e-6),
              n_delta=400, n_t2s=60):
    counts = np.asarray(counts, dtype=float)
    deltas = np.linspace(*delta_range_hz, n_delta)
    t2ss = np.geomspace(*t2s_range_s, n_t2s)
    logl = np.empty((n_delta, n_t2s))
    norm = gammaln(counts + 1).sum()
    for k, t2s in enumerate(t2ss):
        lam = np.stack([expected_counts(taus_s, d, t2s, readout_cfg, n_shots)
                        for d in deltas])
        logl[:, k] = (counts * np.log(lam) - lam).sum(axis=1) - norm
    logl -= logl.max()
    post = np.exp(logl)
    post /= post.sum()
    p_delta = post.sum(axis=1)
    mean = float((p_delta * deltas).sum())
    var = float((p_delta * (deltas - mean) ** 2).sum())
    p_t2s = post.sum(axis=0)
    return {"delta_hz": mean, "delta_sigma_hz": float(np.sqrt(var)),
            "t2star_s": float((p_t2s * t2ss).sum()),
            "log_evidence": float(np.log(np.exp(logl).sum()))}
```

CLAUDE.md, "Key technical decisions" — replace the pending line with:
`- Bayesian inference machinery: grid posterior (400×60 over δ×T2*, exact Poisson likelihood) — decided 2026-08-18. SMC revisited only if Phase 3 adaptive needs sequential updates.`

- [ ] **Step 4:** run → PASS (grid eval ~1–2 s/record; the coverage test ~1 min — mark `@pytest.mark.slow` if it exceeds that budget and keep a fast variant with n_rec=10 unmarked).
- [ ] **Step 5:** Commit: `feat: Bayesian grid posterior estimator; record grid-vs-SMC decision`

---

### Task 5: Paired eval harness (LSQ + Bayes now, NN slot ready)

**Files:**
- Create: `experiments/phase2/ramsey_eval_ladder.json`, `scripts/eval_phase2.py`
- Test: `tests/test_eval_phase2.py`

**Interfaces:**
- Consumes: `run_phase1.py`-style ladder config (Phase 1 runner reused verbatim to generate eval datasets); `fit_lsq`, `fit_bayes`; later `nn.infer` (Task 6) via `--estimators lsq,bayes,nn`.
- Produces: `experiments/phase2/artifacts/estimates_<rung>.json` — per-record estimates for every estimator, plus truth, CRB, and provenance. Each sweep of a rung dataset = one independent record.

Config (exact) — eval datasets are drift-free (estimator-vs-CRB is only meaningful when the model is well-specified; drifted eval is a stated Phase 3 follow-on):

```json
// experiments/phase2/ramsey_eval_ladder.json
{"name": "ramsey_eval_ladder", "kind": "ladder", "vary": "n_shots",
 "values": [20, 66, 200, 660, 2000, 6600, 20000],
 "base": {"protocol": "ramsey", "seed": 202,
   "sweep": {"min": 0.0, "max": 5.0e-6, "n_points": 150},
   "n_sweeps": 200,
   "truth": {"detuning_hz": 2.0e6, "t2star_s": 1.5e-6},
   "timing": {"t_init_s": 2.0e-6, "t_read_s": 0.4e-6, "t_dead_s": 1.0e-6},
   "readout": {"r_hz": 6.0e7, "contrast": 0.25, "t_read_s": 0.4e-6, "f_pump": 0.95},
   "drift": {}}}
```

- [ ] **Step 1: Write failing test**

```python
# tests/test_eval_phase2.py
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_eval_produces_paired_estimates(tmp_path):
    # tiny dataset via the phase-1 runner machinery
    cfg = {"name": "mini", "kind": "ladder", "vary": "n_shots",
           "values": [2000],
           "base": json.loads(
               (REPO / "experiments/phase2/ramsey_eval_ladder.json").read_text()
           )["base"] | {"n_sweeps": 8}}
    cfg_path = tmp_path / "mini.json"
    cfg_path.write_text(json.dumps(cfg))
    r1 = subprocess.run([sys.executable, str(REPO / "scripts/run_phase1.py"),
                         str(cfg_path), "--out-dir", str(tmp_path)],
                        capture_output=True, text=True)
    assert r1.returncode == 0, r1.stderr
    r2 = subprocess.run(
        [sys.executable, str(REPO / "scripts/eval_phase2.py"),
         str(tmp_path / "mini"), "--out-dir", str(tmp_path),
         "--estimators", "lsq,bayes"],
        capture_output=True, text=True)
    assert r2.returncode == 0, r2.stderr
    art = json.loads((tmp_path / "estimates_n_shots_2000.json").read_text())
    assert len(art["records"]) == 8
    for rec in art["records"]:
        assert set(rec) >= {"lsq", "bayes"}          # paired: same record, all estimators
        assert "delta_hz" in rec["lsq"] and "delta_hz" in rec["bayes"]
    assert art["crb_sigma_delta_hz"] > 0
    assert len(art["git_sha"]) >= 7
```

- [ ] **Step 2:** run → FAIL.
- [ ] **Step 3: Implement `scripts/eval_phase2.py`**

```python
#!/usr/bin/env python3
"""Run estimators over every record of Phase-2 eval datasets (paired by
construction: all estimators see the same .npz records).

Usage: eval_phase2.py DATASET_DIR [--out-dir DIR] [--estimators lsq,bayes,nn]"""
import argparse
import json
from pathlib import Path

from nvsim.estimators.bayes import fit_bayes
from nvsim.estimators.crb import crb_sigma_delta
from nvsim.estimators.lsq import fit_lsq
from nvsim.experiment import load_dataset
from nvsim.provenance import git_sha

REPO = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset_dir", type=Path)
    ap.add_argument("--out-dir", type=Path,
                    default=REPO / "experiments/phase2/artifacts")
    ap.add_argument("--estimators", default="lsq,bayes")
    args = ap.parse_args()
    names = args.estimators.split(",")
    if "nn" in names:
        from nvsim.estimators.nn import NNEstimator
        nn_est = NNEstimator.load(REPO / "experiments/phase2/nn_ckpt.pt")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for path in sorted(args.dataset_dir.glob("*.npz")):
        ds = load_dataset(path)
        cfg = ds["config"]
        taus, ro, n = ds["sweep_values"], cfg["readout"], cfg["n_shots"]
        truth = cfg["truth"]
        records = []
        for counts in ds["counts"]:
            rec = {}
            if "lsq" in names:
                rec["lsq"] = fit_lsq(counts, taus, ro, n)
            if "bayes" in names:
                rec["bayes"] = fit_bayes(counts, taus, ro, n)
            if "nn" in names:
                rec["nn"] = nn_est.infer(counts, taus, ro, n)
            records.append(rec)
        art = {"dataset": path.name, "config": cfg, "git_sha": git_sha(),
               "truth": truth,
               "crb_sigma_delta_hz": crb_sigma_delta(
                   taus, (truth["detuning_hz"], truth["t2star_s"]), ro, n),
               "records": records}
        out = args.out_dir / f"estimates_{path.stem}.json"
        out.write_text(json.dumps(art))
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4:** test → PASS. Then generate the real eval ladder and run lsq+bayes over it:
  `python scripts/run_phase1.py experiments/phase2/ramsey_eval_ladder.json --out-dir experiments/phase2/datasets && python scripts/eval_phase2.py experiments/phase2/datasets/ramsey_eval_ladder --estimators lsq,bayes`
  (7 rungs × 200 records × bayes grid ~1 s ≈ ~25 min — **append a LEDGER.md entry** in homelab-orchestration, server-CPU job. Commit datasets only if < ~10 MB total; else commit configs + estimates and note dataset regeneration is deterministic.)
- [ ] **Step 5:** Commit: `feat: paired estimator evaluation harness with CRB reference`

---

### Task 6: NN estimator (torch; training on workstation RTX 3070)

**Files:**
- Create: `src/nvsim/estimators/nn.py`, `scripts/train_nn.py`, `experiments/phase2/nn_train.json`
- Test: `tests/test_nn.py` (CPU smoke; skipped if torch missing)

**Interfaces:**
- Produces: `nn.NNEstimator` with `.infer(counts, taus_s, readout_cfg, n_shots) -> dict` (keys `delta_hz`, `delta_sigma_hz`) and classmethod `.load(path)`; `train_nn.py --config ... --device cuda:0|cpu` writing checkpoint + `nn_train.results.json` (config+seed+SHA+val metrics).
- Model: Conv1d(2→32,k7) → ReLU → Conv1d(32→32,k7,stride2) ×2 → GAP → concat log10(n_shots)/5 → MLP(64) → (δ̂_norm, log σ_norm). Input channels: normalized counts (c/mean(c) − 1) and τ/τ_max. δ normalized to the prior range (0.2–4 MHz). Loss: Gaussian NLL. Training data: generated on-the-fly each epoch from the analytic model + Poisson sampling (seeded), δ ~ U(0.3, 3.8) MHz, T2* ~ U(0.8, 4) µs, n_shots ~ log-U(20, 20000). Held-out eval configs INCLUDE what training never saw: drifted records (1/f B, rms 200 nT) and f_pump 0.90 — the noise-model-overfitting check the roadmap demands.
- Optimizer per RULES.md §4: Adam lr 1e-3, grad-clip 1.0, 500-step linear warmup; checkpoint to disk immediately after training, before any eval.

```json
// experiments/phase2/nn_train.json
{"seed": 301, "epochs": 40, "steps_per_epoch": 250, "batch": 256,
 "lr": 1.0e-3, "warmup_steps": 500, "grad_clip": 1.0,
 "taus": {"min": 0.0, "max": 5.0e-6, "n_points": 150},
 "readout": {"r_hz": 6.0e7, "contrast": 0.25, "t_read_s": 0.4e-6, "f_pump": 0.95},
 "delta_range_hz": [0.3e6, 3.8e6], "t2s_range_s": [0.8e-6, 4.0e-6],
 "n_shots_range": [20, 20000], "n_val": 2000}
```

- [ ] **Step 1: Write CPU smoke tests**

```python
# tests/test_nn.py
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from nvsim.estimators.nn import NNEstimator, RamseyNet, make_batch  # noqa: E402

READOUT = {"r_hz": 6e7, "contrast": 0.25, "t_read_s": 0.4e-6, "f_pump": 0.95}
CFG = {"seed": 1, "taus": {"min": 0.0, "max": 5e-6, "n_points": 150},
       "readout": READOUT, "delta_range_hz": [0.3e6, 3.8e6],
       "t2s_range_s": [0.8e-6, 4.0e-6], "n_shots_range": [20, 20000]}


def test_make_batch_shapes_and_ranges():
    x, aux, y = make_batch(CFG, 32, np.random.default_rng(0))
    assert x.shape == (32, 2, 150) and aux.shape == (32, 1) and y.shape == (32,)
    assert torch.all((y >= 0) & (y <= 1))


def test_loss_decreases_on_tiny_train():
    torch.manual_seed(0)
    net = RamseyNet()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    rng = np.random.default_rng(0)
    losses = []
    for _ in range(60):
        x, aux, y = make_batch(CFG, 64, rng)
        mu, log_sigma = net(x, aux)
        loss = (log_sigma + 0.5 * ((y - mu) / log_sigma.exp()) ** 2).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        losses.append(float(loss))
    assert np.mean(losses[-10:]) < np.mean(losses[:10])


def test_infer_roundtrip(tmp_path):
    net = RamseyNet()
    est = NNEstimator(net, CFG)
    p = tmp_path / "ckpt.pt"
    est.save(p)
    est2 = NNEstimator.load(p)
    taus = np.linspace(0, 5e-6, 150)
    counts = np.random.default_rng(0).poisson(2000 * 24.0, 150)
    out = est2.infer(counts, taus, READOUT, 2000)
    assert set(out) >= {"delta_hz", "delta_sigma_hz"}
    assert 0 < out["delta_hz"] < 5e6
```

- [ ] **Step 2:** On claude-server torch is NOT installed → tests skip; that is the expected local state. Verify: `pytest tests/test_nn.py -v` → "skipped". Full implementation is still test-driven — the tests RUN on the workstation in Step 5.
- [ ] **Step 3: Implement `src/nvsim/estimators/nn.py`**

```python
# src/nvsim/estimators/nn.py
"""1D-CNN Ramsey estimator with heteroscedastic Gaussian head.

torch imported at module top — import this module only under the `.[ml]`
extra; the rest of nvsim stays torch-free."""
import numpy as np
import torch
import torch.nn as nn

from .model import expected_counts


def make_batch(cfg, batch, rng):
    """Sample (theta, n_shots), simulate Poisson records, return tensors."""
    t = cfg["taus"]
    taus = np.linspace(t["min"], t["max"], t["n_points"])
    d_lo, d_hi = cfg["delta_range_hz"]
    deltas = rng.uniform(d_lo, d_hi, batch)
    t2ss = rng.uniform(*cfg["t2s_range_s"], batch)
    n_shots = np.exp(rng.uniform(*np.log(cfg["n_shots_range"]), batch))
    xs = np.empty((batch, 2, len(taus)), dtype=np.float32)
    for i in range(batch):
        lam = expected_counts(taus, deltas[i], t2ss[i], cfg["readout"],
                              n_shots[i])
        c = rng.poisson(lam).astype(np.float64)
        xs[i, 0] = c / max(c.mean(), 1.0) - 1.0
        xs[i, 1] = taus / t["max"]
    aux = (np.log10(n_shots) / 5.0).astype(np.float32)[:, None]
    y = ((deltas - d_lo) / (d_hi - d_lo)).astype(np.float32)
    return torch.from_numpy(xs), torch.from_numpy(aux), torch.from_numpy(y)


class RamseyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(2, 32, 7, padding=3), nn.ReLU(),
            nn.Conv1d(32, 32, 7, stride=2, padding=3), nn.ReLU(),
            nn.Conv1d(32, 32, 7, stride=2, padding=3), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1))
        self.head = nn.Sequential(nn.Linear(33, 64), nn.ReLU(), nn.Linear(64, 2))

    def forward(self, x, aux):
        z = self.conv(x).squeeze(-1)
        out = self.head(torch.cat([z, aux], dim=1))
        return out[:, 0], out[:, 1].clamp(-7, 2)   # mu_norm, log_sigma_norm


class NNEstimator:
    def __init__(self, net, cfg):
        self.net, self.cfg = net.eval(), cfg

    def save(self, path):
        torch.save({"state": self.net.state_dict(), "cfg": self.cfg}, path)

    @classmethod
    def load(cls, path):
        blob = torch.load(path, map_location="cpu", weights_only=False)
        net = RamseyNet()
        net.load_state_dict(blob["state"])
        return cls(net, blob["cfg"])

    def infer(self, counts, taus_s, readout_cfg, n_shots):
        c = np.asarray(counts, dtype=np.float64)
        t = self.cfg["taus"]
        x = np.stack([c / max(c.mean(), 1.0) - 1.0,
                      np.asarray(taus_s) / t["max"]]).astype(np.float32)[None]
        aux = np.array([[np.log10(n_shots) / 5.0]], dtype=np.float32)
        with torch.no_grad():
            mu, log_sigma = self.net(torch.from_numpy(x), torch.from_numpy(aux))
        d_lo, d_hi = self.cfg["delta_range_hz"]
        span = d_hi - d_lo
        return {"delta_hz": float(d_lo + mu.item() * span),
                "delta_sigma_hz": float(np.exp(log_sigma.item()) * span)}
```

- [ ] **Step 4: Implement `scripts/train_nn.py`** (checkpoint IMMEDIATELY post-train, then val):

```python
#!/usr/bin/env python3
"""Train the Ramsey NN estimator. Usage: train_nn.py --config C [--device cpu]"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from nvsim.estimators.nn import NNEstimator, RamseyNet, make_batch
from nvsim.provenance import git_sha

REPO = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text())
    torch.manual_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    dev = torch.device(args.device)
    net = RamseyNet().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=cfg["lr"])
    sched = torch.optim.lr_scheduler.LinearLR(
        opt, start_factor=1e-2, total_iters=cfg["warmup_steps"])
    step = 0
    for epoch in range(cfg["epochs"]):
        for _ in range(cfg["steps_per_epoch"]):
            x, aux, y = make_batch(cfg, cfg["batch"], rng)
            x, aux, y = x.to(dev), aux.to(dev), y.to(dev)
            mu, log_sigma = net(x, aux)
            loss = (log_sigma + 0.5 * ((y - mu) / log_sigma.exp()) ** 2).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), cfg["grad_clip"])
            opt.step(); sched.step(); step += 1
        print(f"[train] epoch {epoch + 1}/{cfg['epochs']} loss {loss:.5f}",
              flush=True)
    ckpt = REPO / "experiments/phase2/nn_ckpt.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    NNEstimator(net.cpu(), cfg).save(ckpt)          # checkpoint BEFORE eval
    print(f"checkpoint {ckpt}", flush=True)

    rng_val = np.random.default_rng(cfg["seed"] + 1)
    x, aux, y = make_batch(cfg, cfg["n_val"], rng_val)
    with torch.no_grad():
        mu, log_sigma = net(x, aux)
    d_lo, d_hi = cfg["delta_range_hz"]
    rmse = float(((mu - y) ** 2).mean().sqrt() * (d_hi - d_lo))
    results = {"config": cfg, "git_sha": git_sha(), "seed": cfg["seed"],
               "val_rmse_delta_hz": rmse, "final_train_loss": float(loss)}
    out = REPO / "experiments/phase2/nn_train.results.json"
    out.write_text(json.dumps(results))
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Workstation run — follow orchestration law exactly, in this order:**
  1. `bash ~/Documents/projects/homelab-orchestration/scripts/status.sh` + read LEDGER.md tail — confirm the 3070 is still free.
  2. **Claim the GPU:** edit MACHINES.md GPU-ownership line to `cuda:0 = RTX 3070 → nv-sensing-sim (claimed 2026-08-18, NN training)`; commit in homelab-orchestration. Append LEDGER.md entry (`started`).
  3. Sync code: mirror the qec bare-repo convention (`git remote -v` in `~/Documents/projects/qec-neural-decoder` shows the pattern; create `cmfinnerty` remote for this repo the same way), push, then on the workstation clone/pull into `~/projects/nv-sensing-sim`, `python3 -m venv .venv && .venv/bin/pip install -e ".[dev,ml]"` (inventory `pip index`/wheel cache first per RULES §1; torch wheel for sm_86 is standard cu12x).
  4. Run `pytest tests/test_nn.py` on the workstation (torch present → smoke tests actually run; must pass before the long job).
  5. Launch detached: `(nohup ./.venv/bin/python scripts/train_nn.py --config experiments/phase2/nn_train.json --device cuda:0 > experiments/phase2/nn_train.log 2>&1 < /dev/null &)`.
  6. Watcher (server side, repeat manually or via Monitor): `ssh cmfinnerty@100.109.23.69 'pgrep -af "[t]rain_nn" ; tail -2 ~/projects/nv-sensing-sim/experiments/phase2/nn_train.log ; ls -la ~/projects/nv-sensing-sim/experiments/phase2/nn_train.results.json 2>/dev/null'` — success = results file exists; death = no process AND no results → report log tail, stop, diagnose (bounded iterations: one designed retry max).
  7. On success: `scp` `nn_ckpt.pt` + `nn_train.results.json` + log to the server repo, commit HERE; run `scripts/safe_pull.sh` on the workstation; LEDGER entry → `done`; **release the 3070** in MACHINES.md (back to `benchmarks/eval; nv-sensing-sim Phase 2+`), commit.
- [ ] **Step 6:** On the server (torch-less): `pytest -q` → NN tests skip, everything else green. Run the NN over the eval ladder — NN inference is CPU-cheap but needs torch: run inference on the workstation (`eval_phase2.py ... --estimators nn` there, scp estimates back) OR install CPU torch here if wheel cache allows (RULES §1: project-local venv, so `pip install -e ".[ml]"` into `.venv` is legal on claude-server; prefer this — no GPU needed for inference).
- [ ] **Step 7:** Commit: `feat: NN Ramsey estimator trained on RTX 3070` (code + results + estimates; checkpoint committed only if < ~5 MB, else note its workstation path per RULES §5).

---

### Task 7: Headline figure + sensitivity numbers

**Files:**
- Create: `scripts/plot_phase2.py`; outputs `docs/figures/phase2_rmse_vs_crb.png` (the headline) and `docs/figures/phase2_nn_generalization.png`

**Interfaces:**
- Consumes: `experiments/phase2/artifacts/estimates_*.json`; palette constants as in `scripts/plot_phase0.py`.

- [ ] **Step 1:** Invoke the dataviz skill (rules already established; categorical: LSQ = slot 2 orange, Bayes = slot 1 blue, NN = slot 3 aqua `#1baf7a`; CRB = dashed ink line).
- [ ] **Step 2:** `phase2_rmse_vs_crb.png` — log-log RMSE(δ̂) [and right-hand σ_B in nT via γ] vs n_shots for the three estimators, CRB dashed; bootstrap (1000 resamples of the 200 paired records) 68% error bars; annotate each estimator's efficiency CRB/RMSE at low and high SNR; sensitivity η = σ_B·√T_record stated for the 2000-shot rung with the full photon budget in the caption.
- [ ] **Step 3:** `phase2_nn_generalization.png` — NN (and Bayes, for reference) RMSE on held-out noise configs: in-distribution vs drifted (1/f 200 nT rms) vs f_pump=0.90 — the explicit noise-model-overfitting test. Generate these two extra eval datasets with the Phase-1 runner (same seeds pattern, `"drift": {"b_field_t": {"kind": "one_over_f", "rms": 2.0e-7}}` and `readout.f_pump: 0.90` variants of the 2000-shot rung).
- [ ] **Step 4:** Run, Read the PNGs, verify: RMSE ≥ CRB everywhere; Bayes hugs CRB at high SNR; LSQ degrades at low SNR (fringe-ambiguity outliers); NN sits between; any surprise → investigate before committing (bounded iterations).
- [ ] **Step 5:** Commit: `feat: phase 2 headline figure — estimator RMSE vs CRB`

---

### Task 8: Close out Phase 2

**Files:**
- Modify: `CLAUDE.md`, `README.md`, `docs/ROADMAP.md`

- [ ] **Step 1:** Full `pytest` on server (NN skipped) AND `pytest tests/test_nn.py` result from the workstation — both recorded (verification-before-completion).
- [ ] **Step 2:** Roadmap exit checklist: LSQ baseline ✓, CRB ✓, Bayesian ✓, NN + held-out noise-config eval ✓, headline figure with paired realizations + error bars + nT/√Hz ✓. Numbers in the report: CRB at each rung, each estimator's efficiency, the sensitivity statement.
- [ ] **Step 3:** Update CLAUDE.md status (Phase 2 complete; 3070 released; Phase 3 awaiting go-ahead), README, ROADMAP boxes.
- [ ] **Step 4:** Commit `docs: mark phase 2 complete`; push. Verify MACHINES.md shows the 3070 released and LEDGER entries closed.
- [ ] **Step 5:** Report to Connor: the headline answer — how much sensitivity better estimation buys, how close each method gets to the CRB, at which SNR — with numbers and units; then **stop** (Phase 3 needs explicit go-ahead).
