# Phase 1 — Realistic Measurement Records Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A config-driven synthetic data generator whose measurement records look like the lab — Poisson photon counts from an explicit budget, ¹⁴N hyperfine + power broadening in ODMR, physically-motivated slow drifts — with every noise knob traceable to a mechanism.

**Architecture:** Three new noise modules layered on the Phase 0 physics core: `readout.py` (spin-dependent fluorescence → Poisson counts), `odmr.py` gains a ¹⁴N/power-broadening spectrum, `drift.py` (linear / Ornstein–Uhlenbeck / 1-over-f processes on a wall-clock axis). `experiment.py` composes them into the virtual-experiment API: JSON config (protocol + true parameters + noise config) → dataset with per-point counts `[n_sweeps, n_points]`, timestamps, embedded truth + provenance, saved as `.npz`. **Record format decision (Connor, 2026-08-17): per-point summed counts + n_shots; per-shot records deferred until a Phase 2 consumer needs them.**

**Tech Stack:** Python ≥3.11, qutip ≥5, numpy, scipy, matplotlib, pytest. No torch.

## Global Constraints

- Frequencies in **Hz** at API boundaries; fields in **tesla**; SI throughout; `omega_*` for internal angular frequencies. (CLAUDE.md)
- Shot noise comes from **Poisson sampling, never additive Gaussian**. (PHYSICS.md)
- A∥(¹⁴N) = −2.16 MHz (already `constants.A_PAR_N14_HZ`); dD/dT = −74 kHz/K (already `constants.DD_DT_HZ_PER_K`).
- Every new physics term/approximation lands in `docs/PHYSICS.md` **in the same commit** as the code.
- Artifacts embed config + seed + git SHA. Datasets are deterministic given (config, seed); **noise sources use separate spawned RNG streams** so changing one knob does not perturb the others' realizations (paired-comparison requirement for Phase 2).
- Figures from committed scripts reading committed artifacts → `docs/figures/`; dataviz skill; palette constants as in `scripts/plot_phase0.py`. Published-trace comparison citations must be **verified against the paper during execution** (as done for Barry in Phase 0), not recalled.
- `pytest` green before every commit; small one-concern commits. CPU-only on claude-server; jobs >10 min get a LEDGER.md entry in homelab-orchestration.
- Roadmap exit: "generator produces datasets across an SNR range with documented, physically-motivated noise; every knob traceable to a physical mechanism."

---

### Task 1: Photon-budget readout

**Files:**
- Create: `src/nvsim/readout.py`
- Modify: `docs/PHYSICS.md` (readout equations, pump-fidelity model)
- Test: `tests/test_readout.py`

**Interfaces:**
- Produces: `readout.mean_counts_per_shot(p0, readout_cfg) -> float | np.ndarray` and `readout.sample_counts(p0, readout_cfg, n_shots, rng) -> np.ndarray[int]` (summed counts per element of `p0`, one draw of `n_shots` shots each).
- `readout_cfg` dict keys: `r_hz` (detected photon rate for ms=0, includes collection efficiency), `contrast` (C), `t_read_s`, `f_pump`.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_readout.py
import numpy as np
import pytest

from nvsim.readout import mean_counts_per_shot, sample_counts

CFG = {"r_hz": 6.0e7, "contrast": 0.25, "t_read_s": 0.4e-6, "f_pump": 1.0}


def test_mean_counts_bright_and_dark_states():
    lam0 = mean_counts_per_shot(1.0, CFG)          # ms=0: R * t_read
    lam1 = mean_counts_per_shot(0.0, CFG)          # ms=±1: R(1-C) * t_read
    assert lam0 == pytest.approx(6.0e7 * 0.4e-6)   # 24 photons/shot
    assert lam1 == pytest.approx(lam0 * (1 - 0.25))


def test_counts_are_poisson_fano_factor_one():
    rng = np.random.default_rng(0)
    counts = np.array([sample_counts(0.5, CFG, 1, rng)[0] for _ in range(20000)])
    fano = counts.var() / counts.mean()
    assert fano == pytest.approx(1.0, abs=0.05)
    assert counts.dtype.kind == "i"


def test_n_shots_sum_scales_mean_and_snr():
    rng = np.random.default_rng(1)
    n = 400
    reps = np.array([sample_counts(1.0, CFG, n, rng)[0] for _ in range(3000)])
    lam = mean_counts_per_shot(1.0, CFG)
    assert reps.mean() == pytest.approx(n * lam, rel=0.01)
    # SNR = mean/std = sqrt(n*lam) for Poisson
    assert reps.mean() / reps.std() == pytest.approx(np.sqrt(n * lam), rel=0.05)


def test_imperfect_pump_compresses_contrast():
    cfg = dict(CFG, f_pump=0.8)
    lam0 = mean_counts_per_shot(1.0, cfg)
    lam1 = mean_counts_per_shot(0.0, cfg)
    ideal0 = mean_counts_per_shot(1.0, CFG)
    # unpumped 1/3-per-state background: both levels move toward 1 - (2/3)C
    assert lam0 < ideal0
    assert (lam0 - lam1) == pytest.approx(0.8 * 0.25 * 6.0e7 * 0.4e-6, rel=1e-9)


def test_vectorized_over_p0():
    rng = np.random.default_rng(2)
    p0 = np.linspace(0, 1, 7)
    out = sample_counts(p0, CFG, 100, rng)
    assert out.shape == (7,)
    means = mean_counts_per_shot(p0, CFG)
    assert means.shape == (7,)
    assert np.all(np.diff(means) > 0)  # brighter with more ms=0 population
```

- [ ] **Step 2:** `pytest tests/test_readout.py -v` → FAIL (ModuleNotFoundError).
- [ ] **Step 3: Implement**

```python
# src/nvsim/readout.py
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
```

PHYSICS.md — replace the "Optical / readout model" bullet list body with the implemented equations (same commit):

```markdown
Implemented (Phase 1, `readout.py`): mean counts per shot
λ(p₀) = R·t_read·[f_pump·(1 − C(1−p₀)) + (1−f_pump)·(1 − 2C/3)],
where R is the *detected* ms=0 photon rate (collection efficiency folded in),
C the fluorescence contrast, p₀ = P(ms=0) at readout. The unpumped fraction
is taken as unpolarized (1/3 per ms level) and coherence-free — a constant
background. Recorded counts over n shots are a single Poisson(n·λ) draw.
```

- [ ] **Step 4:** `pytest tests/test_readout.py -v` → PASS.
- [ ] **Step 5:** Commit: `feat: photon-budget Poisson readout model`

---

### Task 2: ¹⁴N hyperfine triplet + power broadening in ODMR

**Files:**
- Modify: `src/nvsim/odmr.py`, `docs/PHYSICS.md`
- Test: `tests/test_odmr.py` (append)

**Interfaces:**
- Consumes: `hamiltonian.h_gs`, `hamiltonian.transition_frequencies`, `constants.A_PAR_N14_HZ`.
- Produces: `odmr.odmr_spectrum_n14(f_hz, d_hz=D_GS_HZ, e_hz=0.0, b_nv_t=(0,0,0), contrast=0.2, fwhm_hz=1e6, a_par_hz=A_PAR_N14_HZ, saturation=None) -> np.ndarray`. Each electronic transition splits into three lines at `f_trans + ms_target * a_par_hz * mI` (mI ∈ {−1,0,+1}), each carrying `contrast/3`. If `saturation` (dimensionless s) is not None: FWHM → `fwhm_hz·√(1+s)`, contrast → `contrast·s/(1+s)`.

- [ ] **Step 1: Write failing tests (append to tests/test_odmr.py)**

```python
from nvsim.constants import A_PAR_N14_HZ
from nvsim.odmr import odmr_spectrum_n14


def test_n14_triplet_spacing_2p16_mhz():
    bz = 2e-3  # separate the two electronic transitions cleanly
    f = np.linspace(2.80e9, 2.83e9, 60001)  # around f_minus
    s = odmr_spectrum_n14(f, b_nv_t=(0, 0, bz), fwhm_hz=0.4e6)
    dips = _dip_freqs(f, s)
    assert len(dips) == 3
    df = f[1] - f[0]
    assert dips[1] - dips[0] == pytest.approx(abs(A_PAR_N14_HZ), abs=2 * df)
    assert dips[2] - dips[1] == pytest.approx(abs(A_PAR_N14_HZ), abs=2 * df)


def test_triplet_contrast_is_one_third_each():
    f = np.linspace(2.80e9, 2.83e9, 60001)
    s = odmr_spectrum_n14(f, b_nv_t=(0, 0, 2e-3), contrast=0.24, fwhm_hz=0.4e6)
    # well-separated lines: each dip depth ~ contrast/3
    assert 1 - s.min() == pytest.approx(0.24 / 3, rel=0.05)


def test_power_broadening_sqrt_1_plus_s():
    f = np.linspace(2.865e9, 2.875e9, 40001)

    def fwhm_of_center_line(s_par):
        s = odmr_spectrum_n14(f, fwhm_hz=0.5e6, a_par_hz=20e6, saturation=s_par)
        # a_par 20 MHz (unphysical, test-only) isolates the mI=0 line at D
        depth = 1 - s.min()
        half = 1 - depth / 2
        below = f[s < half]
        return below.max() - below.min()

    w1, w4 = fwhm_of_center_line(1.0), fwhm_of_center_line(4.0)
    assert w4 / w1 == pytest.approx(np.sqrt(5) / np.sqrt(2), rel=0.03)


def test_saturation_scales_contrast():
    f = np.linspace(2.85e9, 2.89e9, 20001)
    depths = []
    for s_par in (0.5, 2.0, 8.0):
        s = odmr_spectrum_n14(f, contrast=0.3, saturation=s_par)
        depths.append(1 - s.min())
    ratios = np.array(depths) / (0.3 * np.array([0.5, 2.0, 8.0])
                                 / (1 + np.array([0.5, 2.0, 8.0])))
    np.testing.assert_allclose(ratios, ratios[0], rtol=0.1)  # ∝ s/(1+s)
```

- [ ] **Step 2:** run → FAIL.
- [ ] **Step 3: Implement (append to src/nvsim/odmr.py)**

```python
def odmr_spectrum_n14(f_hz, d_hz=D_GS_HZ, e_hz=0.0, b_nv_t=(0.0, 0.0, 0.0),
                      contrast=0.2, fwhm_hz=1e6, a_par_hz=None, saturation=None):
    """CW-ODMR with the 14N hyperfine triplet and optional power broadening.

    Each electronic transition (toward ms_target = ±1) splits into three lines
    at f_trans + ms_target * A_par * mI, mI in {-1, 0, +1}, contrast/3 each
    (static-hyperfine approximation, docs/PHYSICS.md). saturation = s applies
    FWHM * sqrt(1+s) and contrast * s/(1+s).
    """
    from .constants import A_PAR_N14_HZ
    if a_par_hz is None:
        a_par_hz = A_PAR_N14_HZ
    f = np.asarray(f_hz, dtype=float)
    c_eff, w_eff = contrast, fwhm_hz
    if saturation is not None:
        c_eff = contrast * saturation / (1 + saturation)
        w_eff = fwhm_hz * np.sqrt(1 + saturation)
    hwhm = w_eff / 2
    f_minus, f_plus = transition_frequencies(h_gs(d_hz, e_hz, b_nv_t))
    s = np.ones_like(f)
    for ms_target, f0 in ((-1, f_minus), (+1, f_plus)):
        for mi in (-1, 0, 1):
            fc = f0 + ms_target * a_par_hz * mi
            s -= (c_eff / 3) * hwhm**2 / ((f - fc) ** 2 + hwhm**2)
    return s
```

PHYSICS.md — under "¹⁴N hyperfine (Phase 1+)" append (same commit):

```markdown
Implemented (Phase 1): static-splitting model — each ms=0 → ms=±1 line splits
into three at f_trans ± A∥·mI (sign via ms_target·A∥·mI), contrast/3 each.
Power broadening (cw): FWHM(s) = FWHM₀·√(1+s), contrast(s) = C₀·s/(1+s),
with s the dimensionless saturation parameter (MW power / saturation power).
Cite Dréau et al., Phys. Rev. B 84, 195204 (2011) — verify the exact equation
numbers against the paper before writing figure captions.
```

- [ ] **Step 4:** run → PASS. **Step 5:** Commit: `feat: 14N hyperfine triplet and power broadening in ODMR`

---

### Task 3: Slow-drift processes

**Files:**
- Create: `src/nvsim/drift.py`
- Modify: `docs/PHYSICS.md`
- Test: `tests/test_drift.py`

**Interfaces:**
- Produces: `drift.sample_drift(cfg, times_s, rng) -> np.ndarray` (same length as `times_s`). `cfg["kind"]` ∈:
  - `"constant"`: `{"kind": "constant", "value": v}` → constant array.
  - `"linear"`: `{"kind": "linear", "rate_per_s": r}` → `r * (t - t[0])`.
  - `"ou"`: `{"kind": "ou", "sigma": s, "tau_s": tau}` → stationary Ornstein–Uhlenbeck, std `s`, correlation time `tau` (exact discretization, arbitrary time stamps).
  - `"one_over_f"`: `{"kind": "one_over_f", "rms": a, "alpha": 1.0}` → spectral synthesis on a uniform grid spanning the record, PSD ∝ 1/f^alpha, scaled to rms `a`, interpolated to `times_s`.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_drift.py
import numpy as np
import pytest

from nvsim.drift import sample_drift


def test_constant_and_linear():
    t = np.linspace(0, 10, 101)
    rng = np.random.default_rng(0)
    np.testing.assert_allclose(
        sample_drift({"kind": "constant", "value": 3.5}, t, rng), 3.5)
    lin = sample_drift({"kind": "linear", "rate_per_s": -74e3 * 0.002}, t, rng)
    assert lin[0] == 0.0
    assert lin[-1] == pytest.approx(-74e3 * 0.002 * 10)


def test_ou_stationary_std_and_correlation_time():
    t = np.arange(0, 2000.0, 0.5)
    rng = np.random.default_rng(1)
    x = sample_drift({"kind": "ou", "sigma": 2.0, "tau_s": 5.0}, t, rng)
    assert x.std() == pytest.approx(2.0, rel=0.1)
    # autocorrelation at lag tau ~ exp(-1)
    lag = int(5.0 / 0.5)
    ac = np.corrcoef(x[:-lag], x[lag:])[0, 1]
    assert ac == pytest.approx(np.exp(-1), abs=0.1)


def test_one_over_f_psd_slope():
    t = np.linspace(0, 100, 2**14)
    rng = np.random.default_rng(2)
    x = sample_drift({"kind": "one_over_f", "rms": 1e-7, "alpha": 1.0}, t, rng)
    assert x.std() == pytest.approx(1e-7, rel=0.05)
    psd = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(len(t), t[1] - t[0])
    # fit log-log slope over the middle two decades
    m = (freqs > freqs[1] * 10) & (freqs < freqs[-1] / 10)
    slope = np.polyfit(np.log(freqs[m]), np.log(psd[m]), 1)[0]
    assert slope == pytest.approx(-1.0, abs=0.3)


def test_deterministic_given_rng_seed():
    t = np.linspace(0, 10, 100)
    a = sample_drift({"kind": "ou", "sigma": 1, "tau_s": 2},
                     t, np.random.default_rng(7))
    b = sample_drift({"kind": "ou", "sigma": 1, "tau_s": 2},
                     t, np.random.default_rng(7))
    np.testing.assert_array_equal(a, b)
```

- [ ] **Step 2:** run → FAIL.
- [ ] **Step 3: Implement**

```python
# src/nvsim/drift.py
"""Slow-drift processes on a wall-clock axis (docs/PHYSICS.md, Drift models).

All processes are zero-mean unless 'constant'; callers scale/offset into
physical units (tesla, Hz, relative power)."""
import numpy as np


def _ou(sigma, tau_s, times_s, rng):
    x = np.empty(len(times_s))
    x[0] = sigma * rng.standard_normal()
    for i in range(1, len(times_s)):
        dt = times_s[i] - times_s[i - 1]
        a = np.exp(-dt / tau_s)
        x[i] = x[i - 1] * a + sigma * np.sqrt(1 - a * a) * rng.standard_normal()
    return x


def _one_over_f(rms, alpha, times_s, rng):
    n = max(len(times_s), 2**12)
    grid = np.linspace(times_s[0], times_s[-1], n)
    amp = np.zeros(n // 2 + 1)
    freqs = np.fft.rfftfreq(n, grid[1] - grid[0])
    amp[1:] = freqs[1:] ** (-alpha / 2)
    phases = rng.uniform(0, 2 * np.pi, len(amp))
    spec = amp * np.exp(1j * phases)
    spec[0] = 0.0
    x = np.fft.irfft(spec, n)
    x *= rms / x.std()
    return np.interp(times_s, grid, x)


def sample_drift(cfg, times_s, rng):
    """One realization of the configured drift process at the given times."""
    t = np.asarray(times_s, dtype=float)
    kind = cfg["kind"]
    if kind == "constant":
        return np.full(len(t), float(cfg["value"]))
    if kind == "linear":
        return cfg["rate_per_s"] * (t - t[0])
    if kind == "ou":
        return _ou(cfg["sigma"], cfg["tau_s"], t, rng)
    if kind == "one_over_f":
        return _one_over_f(cfg["rms"], cfg.get("alpha", 1.0), t, rng)
    raise ValueError(f"unknown drift kind: {kind}")
```

PHYSICS.md — new section (same commit):

```markdown
## Drift models (Phase 1)

Wall-clock processes composed into records (`drift.py`), each traceable to a
mechanism:
- Laser/MW power: relative multiplicative drift, Ornstein–Uhlenbeck
  (correlation time ~seconds–minutes) or linear; scales detected rate R
  (laser) or Rabi amplitude Ω (MW, amplitude ∝ √power — configured directly
  as amplitude drift).
- Temperature → D: linear or OU temperature trace T(t), D(t) = D₀ − 74 kHz/K
  × (T − T₀).
- Magnetic background: 1/f^α synthesis (α = 1 default), zero-mean, scaled to
  a target rms in tesla; enters as detuning γ·δB_z(t).
OU uses the exact discretization x_{k+1} = x_k·e^{−Δt/τ} + σ√(1−e^{−2Δt/τ})·ξ.
1/f is synthesized spectrally on a uniform grid over the record and
interpolated; DC bin zeroed.
```

- [ ] **Step 4:** run → PASS. **Step 5:** Commit: `feat: linear, OU, and 1/f drift processes`

---

### Task 4: Virtual-experiment API

**Files:**
- Create: `src/nvsim/experiment.py`, `src/nvsim/provenance.py`
- Modify: `scripts/run_phase0.py` (import `git_sha` from `nvsim.provenance` instead of its local copy), `docs/PHYSICS.md` (timing model + drift→parameter mapping)
- Test: `tests/test_experiment.py`

**Interfaces:**
- Consumes: `odmr_spectrum_n14`, `pulsed.rabi/ramsey/hahn_echo`, `readout.sample_counts/mean_counts_per_shot`, `drift.sample_drift`.
- Produces:
  - `provenance.git_sha() -> str`
  - `experiment.run_experiment(config: dict) -> dict` — dataset dict with keys `config`, `seed`, `git_sha`, `timestamps_s [n_sweeps, n_points]`, `sweep_values` (Hz or s), `counts [n_sweeps, n_points] int64`, `truth` (dict: ideal p0 or spectrum, drift traces, drifted parameter arrays).
  - `experiment.save_dataset(dataset, path)` / `experiment.load_dataset(path) -> dict` — `.npz` with arrays + a JSON-encoded `meta` string (config, seed, git_sha, truth scalars).
- Timing model: shot time `t_shot = timing.t_init_s + t_manip + timing.t_read_s + timing.t_dead_s` where `t_manip` is the sweep variable for pulsed protocols (drive duration / τ / 2τ) and 0 for ODMR; a point's wall-clock timestamp is the cumulative time of its shots; sweeps run back-to-back.
- Drift → parameter mapping (all optional, keyed under `config["drift"]`): `laser_power` (relative, scales counts rate), `mw_amplitude` (relative, scales Ω for rabi), `temperature_k` (via −74 kHz/K into D for ODMR, into detuning for pulsed), `b_field_t` (tesla, via γ into dip positions / detuning).
- RNG streams: `SeedSequence(config["seed"]).spawn(2)` → `rng_drift`, `rng_shot`. Changing readout/photon config must not change drift realizations.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_experiment.py
import numpy as np
import pytest

from nvsim.experiment import load_dataset, run_experiment, save_dataset

RAMSEY_CFG = {
    "name": "t", "protocol": "ramsey", "seed": 11,
    "sweep": {"min": 0.0, "max": 4e-6, "n_points": 40},
    "n_sweeps": 3, "n_shots": 200,
    "truth": {"detuning_hz": 2e6, "t2star_s": 3e-6},
    "timing": {"t_init_s": 2e-6, "t_read_s": 0.4e-6, "t_dead_s": 1e-6},
    "readout": {"r_hz": 6e7, "contrast": 0.25, "t_read_s": 0.4e-6, "f_pump": 0.95},
    "drift": {"b_field_t": {"kind": "one_over_f", "rms": 2e-7, "alpha": 1.0}},
}


def test_dataset_shapes_types_provenance():
    ds = run_experiment(RAMSEY_CFG)
    assert ds["counts"].shape == (3, 40)
    assert ds["counts"].dtype == np.int64
    assert ds["timestamps_s"].shape == (3, 40)
    assert np.all(np.diff(ds["timestamps_s"].ravel()) > 0)  # monotonic wall clock
    assert len(ds["git_sha"]) >= 7
    assert ds["config"] == RAMSEY_CFG


def test_deterministic_given_seed():
    a, b = run_experiment(RAMSEY_CFG), run_experiment(RAMSEY_CFG)
    np.testing.assert_array_equal(a["counts"], b["counts"])


def test_rng_stream_separation_pairing():
    """Changing the photon budget must not change the drift realization."""
    hot = dict(RAMSEY_CFG, readout=dict(RAMSEY_CFG["readout"], r_hz=1.2e8))
    a, b = run_experiment(RAMSEY_CFG), run_experiment(hot)
    np.testing.assert_array_equal(
        a["truth"]["drift_b_field_t"], b["truth"]["drift_b_field_t"])
    assert a["counts"].sum() < b["counts"].sum()  # brighter budget, same noise path


def test_counts_track_ideal_curve():
    cfg = dict(RAMSEY_CFG, n_sweeps=30, n_shots=2000, drift={})
    ds = run_experiment(cfg)
    mean = ds["counts"].mean(axis=0).astype(float)
    p0 = ds["truth"]["p0_ideal"]
    # correlation between measured counts and ideal p0 curve
    assert np.corrcoef(mean, p0)[0, 1] > 0.99


def test_odmr_protocol_and_roundtrip(tmp_path):
    cfg = {
        "name": "o", "protocol": "odmr", "seed": 5,
        "sweep": {"min": 2.80e9, "max": 2.94e9, "n_points": 120},
        "n_sweeps": 2, "n_shots": 300,
        "truth": {"b_nv_t": [0.0, 0.0, 1e-3], "contrast": 0.2,
                  "fwhm_hz": 1e6, "saturation": 2.0},
        "timing": {"t_init_s": 0.0, "t_read_s": 1e-3, "t_dead_s": 0.0},
        "readout": {"r_hz": 6e7, "contrast": 0.2, "t_read_s": 1e-3, "f_pump": 1.0},
        "drift": {},
    }
    ds = run_experiment(cfg)
    assert ds["counts"].shape == (2, 120)
    p = tmp_path / "o.npz"
    save_dataset(ds, p)
    back = load_dataset(p)
    np.testing.assert_array_equal(back["counts"], ds["counts"])
    assert back["config"] == cfg
    assert back["git_sha"] == ds["git_sha"]
```

- [ ] **Step 2:** run → FAIL.
- [ ] **Step 3: Implement**

```python
# src/nvsim/provenance.py
"""Artifact provenance helpers."""
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def git_sha():
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO
    ).stdout.strip()
```

```python
# src/nvsim/experiment.py
"""Virtual-experiment API: config -> measurement-record dataset.

Record format (decided 2026-08-17): per-point summed Poisson counts with
n_shots recorded; per-shot records are a future opt-in. Timing model and
drift->parameter mapping in docs/PHYSICS.md."""
import json

import numpy as np

from .constants import DD_DT_HZ_PER_K, GAMMA_E_HZ_PER_T
from .drift import sample_drift
from .odmr import odmr_spectrum_n14
from .provenance import git_sha
from .pulsed import hahn_echo, rabi, ramsey
from .readout import sample_counts


def _timestamps(cfg, sweep_values):
    """Wall-clock time of each (sweep, point): cumulative shot durations."""
    t = cfg["timing"]
    manip = sweep_values if cfg["protocol"] != "odmr" else np.zeros_like(sweep_values)
    if cfg["protocol"] == "hahn_echo":
        manip = 2 * manip
    point_dur = cfg["n_shots"] * (t["t_init_s"] + manip + t["t_read_s"]
                                  + t["t_dead_s"])
    ends = np.cumsum(np.tile(point_dur, cfg["n_sweeps"]))
    return ends.reshape(cfg["n_sweeps"], len(sweep_values))


def _drift_traces(cfg, times, rng_drift):
    traces = {}
    for key, dcfg in cfg.get("drift", {}).items():
        traces[key] = sample_drift(dcfg, times.ravel(), rng_drift).reshape(times.shape)
    return traces


def _p0_pulsed(cfg, sweep_values, det_shift):
    """Ideal + per-point drifted P(ms=0) for pulsed protocols."""
    tr = cfg["truth"]
    proto = cfg["protocol"]
    n_sweeps, n_points = det_shift.shape
    p0 = np.empty((n_sweeps, n_points))
    for i in range(n_sweeps):
        for j, x in enumerate(sweep_values):
            if proto == "rabi":
                p0[i, j] = rabi(tr["rabi_hz"], [0.0, x],
                                detuning_hz=det_shift[i, j],
                                t1_s=tr.get("t1_s"), t2_s=tr.get("t2_s"))[-1]
            elif proto == "ramsey":
                p0[i, j] = ramsey([0.0, x], tr["detuning_hz"] + det_shift[i, j],
                                  t2star_s=tr.get("t2star_s"))[-1]
            elif proto == "hahn_echo":
                p0[i, j] = hahn_echo([x], static_detuning_hz=det_shift[i, j],
                                     t2_s=tr.get("t2_s"))[0]
    return p0


def run_experiment(cfg):
    ss = np.random.SeedSequence(cfg["seed"]).spawn(2)
    rng_drift, rng_shot = (np.random.default_rng(s) for s in ss)
    sw = cfg["sweep"]
    sweep_values = np.linspace(sw["min"], sw["max"], sw["n_points"])
    times = _timestamps(cfg, sweep_values)
    traces = _drift_traces(cfg, times, rng_drift)

    det_shift = np.zeros(times.shape)
    if "b_field_t" in traces:
        det_shift += GAMMA_E_HZ_PER_T * traces["b_field_t"]
    if "temperature_k" in traces:
        det_shift += DD_DT_HZ_PER_K * traces["temperature_k"]

    tr = cfg["truth"]
    truth = {f"drift_{k}": v for k, v in traces.items()}
    if cfg["protocol"] == "odmr":
        base = odmr_spectrum_n14(
            sweep_values, b_nv_t=tuple(tr["b_nv_t"]), contrast=tr["contrast"],
            fwhm_hz=tr["fwhm_hz"], saturation=tr.get("saturation"))
        truth["spectrum_ideal"] = base
        # dip positions shift with drift: evaluate spectrum at f - shift
        rate_scale = np.empty(times.shape + (len(sweep_values),)[:0])
        s_drifted = np.empty(times.shape)
        for i in range(cfg["n_sweeps"]):
            s_drifted[i] = odmr_spectrum_n14(
                sweep_values - det_shift[i], b_nv_t=tuple(tr["b_nv_t"]),
                contrast=tr["contrast"], fwhm_hz=tr["fwhm_hz"],
                saturation=tr.get("saturation"))
        lam_scale = s_drifted
        r = cfg["readout"]
        lam = r["r_hz"] * r["t_read_s"] * lam_scale
        if "laser_power" in traces:
            lam = lam * (1 + traces["laser_power"])
        counts = rng_shot.poisson(cfg["n_shots"] * lam).astype(np.int64)
    else:
        if cfg["protocol"] == "rabi" and "mw_amplitude" in traces:
            tr = dict(tr)  # per-point Ω handled inside _p0_pulsed via truth copy
        p0 = _p0_pulsed(cfg, sweep_values, det_shift)
        truth["p0_ideal"] = _p0_pulsed(
            cfg, sweep_values, np.zeros((1, len(sweep_values))))[0]
        lam_mult = (1 + traces["laser_power"]) if "laser_power" in traces else 1.0
        counts = np.empty(times.shape, dtype=np.int64)
        for i in range(cfg["n_sweeps"]):
            mult = lam_mult[i] if isinstance(lam_mult, np.ndarray) else 1.0
            base_counts = sample_counts(p0[i], cfg["readout"],
                                        cfg["n_shots"], rng_shot)
            counts[i] = np.round(base_counts * mult).astype(np.int64)
        truth["p0_drifted"] = p0

    return {"config": cfg, "seed": cfg["seed"], "git_sha": git_sha(),
            "timestamps_s": times, "sweep_values": sweep_values,
            "counts": counts, "truth": truth}


def save_dataset(ds, path):
    arrays = {"timestamps_s": ds["timestamps_s"],
              "sweep_values": ds["sweep_values"], "counts": ds["counts"]}
    arrays |= {f"truth_{k}": np.asarray(v) for k, v in ds["truth"].items()}
    meta = json.dumps({"config": ds["config"], "seed": ds["seed"],
                       "git_sha": ds["git_sha"],
                       "truth_keys": list(ds["truth"])})
    np.savez_compressed(path, meta=np.frombuffer(meta.encode(), dtype=np.uint8),
                        **arrays)


def load_dataset(path):
    with np.load(path) as z:
        meta = json.loads(bytes(z["meta"]).decode())
        ds = {"config": meta["config"], "seed": meta["seed"],
              "git_sha": meta["git_sha"],
              "timestamps_s": z["timestamps_s"],
              "sweep_values": z["sweep_values"],
              "counts": z["counts"],
              "truth": {k: z[f"truth_{k}"] for k in meta["truth_keys"]}}
    return ds
```

Implementation notes for the engineer:
- The laser-power multiplier on pulsed counts above multiplies *sampled* counts; that slightly distorts Poisson statistics. If `test_counts_track_ideal_curve` or Fano tests flag it, restructure to scale λ before sampling: compute `lam = mean_counts_per_shot(p0[i], cfg["readout"]) * mult` then `rng_shot.poisson(cfg["n_shots"] * lam)` — that is the physically correct order and the preferred fix (readout exposes `mean_counts_per_shot` for exactly this).
- `mw_amplitude` drift for Rabi: multiply `tr["rabi_hz"]` by `(1 + trace[i, j])` per point inside `_p0_pulsed` (pass the trace in). If this exceeds the task budget, drop `mw_amplitude` from Task 4 and note it in PHYSICS.md as config-reserved.
- Delete the stray `rate_scale` line if unused (it is — leftover; do not ship dead code).

Also in this task: `scripts/run_phase0.py` drops its local `git_sha()` and does `from nvsim.provenance import git_sha`; run `pytest tests/test_run_phase0.py` to confirm no regression.

PHYSICS.md — new section (same commit):

```markdown
## Virtual experiment timing model (Phase 1)

A record is n_sweeps sequential sweeps over n_points settings; each point is
n_shots back-to-back shots of duration t_init + t_manip + t_read + t_dead
(t_manip = drive time / τ / 2τ per protocol, 0 for cw-ODMR). Drift processes
are sampled at each point's wall-clock timestamp. Drift mapping: laser power →
detected rate multiplier; MW amplitude → Ω multiplier (Rabi); temperature →
D via −74 kHz/K → detuning; B background → detuning via γ (pulsed) / line
positions (ODMR). Per-point counts are one Poisson(n_shots·λ) draw; drift is
treated as constant within a point (τ_drift ≫ point duration assumed —
approximations ledger).
```

Append to approximations ledger: `| Drift frozen within a measurement point | experiment.py | drift correlation times approach the per-point duration |`

- [ ] **Step 4:** run `pytest tests/test_experiment.py tests/test_run_phase0.py -v` → PASS.
- [ ] **Step 5:** Commit: `feat: virtual-experiment API with paired RNG streams and npz datasets`

---

### Task 5: Phase 1 configs + runner

**Files:**
- Create: `experiments/phase1/odmr_n14_power.json`, `experiments/phase1/ramsey_snr_ladder.json`, `experiments/phase1/ramsey_drift.json`, `scripts/run_phase1.py`
- Test: `tests/test_run_phase1.py`

**Interfaces:**
- Consumes: `run_experiment`, `save_dataset`.
- Produces: CLI `python scripts/run_phase1.py experiments/phase1/<name>.json [--out-dir DIR]` → `experiments/phase1/artifacts/<name>.npz` (or `<name>/<sub>.npz` for ladder configs, one per SNR rung).

Config contents (exact):

```json
// experiments/phase1/odmr_n14_power.json — triplet + power broadening ladder
{"name": "odmr_n14_power", "kind": "ladder", "vary": "truth.saturation",
 "values": [0.2, 1.0, 5.0, 20.0],
 "base": {"protocol": "odmr", "seed": 21,
   "sweep": {"min": 2.8595e9, "max": 2.8805e9, "n_points": 421},
   "n_sweeps": 8, "n_shots": 2000,
   "truth": {"b_nv_t": [0.0, 0.0, 3.0e-4], "contrast": 0.18,
             "fwhm_hz": 0.35e6, "saturation": 1.0},
   "timing": {"t_init_s": 0.0, "t_read_s": 1.0e-3, "t_dead_s": 0.0},
   "readout": {"r_hz": 6.0e7, "contrast": 0.18, "t_read_s": 1.0e-3, "f_pump": 1.0},
   "drift": {}}}
```

```json
// experiments/phase1/ramsey_snr_ladder.json — same physics, photon budget swept
{"name": "ramsey_snr_ladder", "kind": "ladder", "vary": "n_shots",
 "values": [20, 200, 2000, 20000],
 "base": {"protocol": "ramsey", "seed": 22,
   "sweep": {"min": 0.0, "max": 5.0e-6, "n_points": 150},
   "n_sweeps": 10,
   "truth": {"detuning_hz": 2.0e6, "t2star_s": 1.5e-6},
   "timing": {"t_init_s": 2.0e-6, "t_read_s": 0.4e-6, "t_dead_s": 1.0e-6},
   "readout": {"r_hz": 6.0e7, "contrast": 0.25, "t_read_s": 0.4e-6, "f_pump": 0.95},
   "drift": {}}}
```

```json
// experiments/phase1/ramsey_drift.json — all drift channels on, long record
{"name": "ramsey_drift", "kind": "single",
 "base": {"protocol": "ramsey", "seed": 23,
   "sweep": {"min": 0.0, "max": 5.0e-6, "n_points": 120},
   "n_sweeps": 60, "n_shots": 2000,
   "truth": {"detuning_hz": 2.0e6, "t2star_s": 1.5e-6},
   "timing": {"t_init_s": 2.0e-6, "t_read_s": 0.4e-6, "t_dead_s": 1.0e-6},
   "readout": {"r_hz": 6.0e7, "contrast": 0.25, "t_read_s": 0.4e-6, "f_pump": 0.95},
   "drift": {"b_field_t": {"kind": "one_over_f", "rms": 4.0e-7, "alpha": 1.0},
             "laser_power": {"kind": "ou", "sigma": 0.015, "tau_s": 30.0},
             "temperature_k": {"kind": "linear", "rate_per_s": 0.0005}}}}
```

- [ ] **Step 1: Write failing test**

```python
# tests/test_run_phase1.py
import subprocess
import sys
from pathlib import Path

from nvsim.experiment import load_dataset

REPO = Path(__file__).resolve().parents[1]


def test_ladder_runner_produces_one_dataset_per_rung(tmp_path):
    cfg = REPO / "experiments/phase1/ramsey_snr_ladder.json"
    out = subprocess.run(
        [sys.executable, str(REPO / "scripts/run_phase1.py"), str(cfg),
         "--out-dir", str(tmp_path)],
        capture_output=True, text=True,
    )
    assert out.returncode == 0, out.stderr
    rungs = sorted((tmp_path / "ramsey_snr_ladder").glob("*.npz"))
    assert len(rungs) == 4
    ds = load_dataset(rungs[0])
    assert ds["counts"].shape == (10, 150)
    assert len(ds["git_sha"]) >= 7
```

- [ ] **Step 2:** run → FAIL.
- [ ] **Step 3: Implement `scripts/run_phase1.py`**

```python
#!/usr/bin/env python3
"""Run a Phase 1 virtual experiment (single or ladder config) to .npz datasets.

Usage: run_phase1.py CONFIG [--out-dir DIR]"""
import argparse
import copy
import json
from pathlib import Path

from nvsim.experiment import run_experiment, save_dataset

REPO = Path(__file__).resolve().parent.parent


def set_dotted(cfg, dotted, value):
    node = cfg
    *parents, leaf = dotted.split(".")
    for p in parents:
        node = node[p]
    node[leaf] = value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path)
    ap.add_argument("--out-dir", type=Path,
                    default=REPO / "experiments/phase1/artifacts")
    args = ap.parse_args()
    spec = json.loads(args.config.read_text())
    base = spec["base"]
    if spec["kind"] == "single":
        args.out_dir.mkdir(parents=True, exist_ok=True)
        ds = run_experiment(base)
        out = args.out_dir / f"{spec['name']}.npz"
        save_dataset(ds, out)
        print(f"wrote {out}")
        return
    sub = args.out_dir / spec["name"]
    sub.mkdir(parents=True, exist_ok=True)
    for v in spec["values"]:
        cfg = copy.deepcopy(base)
        set_dotted(cfg, spec["vary"], v)
        ds = run_experiment(cfg)
        tag = spec["vary"].split(".")[-1]
        out = sub / f"{tag}_{v}.npz"
        save_dataset(ds, out)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4:** test → PASS. Then run all three configs for real (default out-dir); check total artifact size stays < ~5 MB (npz compressed) before committing binaries.
- [ ] **Step 5:** Commit: `feat: phase 1 experiment configs and dataset runner`

---

### Task 6: Validation figures (side-by-side with published traces)

**Files:**
- Create: `scripts/plot_phase1.py`; outputs `docs/figures/phase1_odmr_n14.png`, `phase1_snr_ladder.png`, `phase1_drift_record.png`

**Interfaces:**
- Consumes: Task 5 artifacts via `load_dataset`; palette/style constants exactly as in `scripts/plot_phase0.py` (copy the constant block).

- [ ] **Step 1:** Invoke the **dataviz skill**. Verify the published-trace citations before writing captions (fetch the papers as done for Barry in Phase 0):
  - ¹⁴N triplet + power broadening: Dréau et al., Phys. Rev. B 84, 195204 (2011) — confirm which figure shows linewidth/contrast vs MW power; fall back to section number if figures can't be verified.
  - Shot-noise-limited records: Barry et al., RMP 92, 015004 (2020) §III (sensitivity/photon budget context; /tmp/barry.txt may still exist from Phase 0, else re-fetch).
- [ ] **Step 2:** Write `scripts/plot_phase1.py`:
  - `phase1_odmr_n14.png` — measured (Poisson-noisy, sweep-averaged) spectra for the four saturation rungs, offset-stacked, triplet resolved at low s and washed out at high s; annotate FWHM(s) and the 2.16 MHz spacing.
  - `phase1_snr_ladder.png` — Ramsey records at the four photon budgets (counts, per-point error bars = √counts), with the noiseless truth curve overlaid; annotate empirical SNR per rung and the √N scaling (fit in an inset like phase0_rabi).
  - `phase1_drift_record.png` — two-panel: (top) counts raster [sweep × τ] showing fringe phase wander from B/temperature drift; (bottom) the three drift traces vs wall-clock time in physical units (nT, mK·s⁻¹ context, %).
- [ ] **Step 3:** Run; **Read each PNG** and check: triplet spacing, broadening direction, error bars ~√counts, drift visibly non-white. Fix before committing.
- [ ] **Step 4:** Commit: `feat: phase 1 validation figures`

---

### Task 7: Close out Phase 1

**Files:**
- Modify: `CLAUDE.md` (status line), `README.md`, `docs/ROADMAP.md` (check Phase 1 boxes)

- [ ] **Step 1:** Full `pytest` — all green, record the count (use verification-before-completion).
- [ ] **Step 2:** Roadmap exit-criterion checklist, each item → named test or figure: Poisson (not Gaussian) shot noise; triplet + power broadening; three drift mechanisms with units; config-driven API; SNR range demonstrated; every knob traced to a mechanism in PHYSICS.md.
- [ ] **Step 3:** Update CLAUDE.md status ("Phase 1 complete (date); Phase 2 not started — awaiting go-ahead"), README, ROADMAP checkboxes.
- [ ] **Step 4:** Commit `docs: mark phase 1 complete`; push to origin.
- [ ] **Step 5:** Report to Connor — numbers with units, figure paths, honest discrepancies — then **stop** (Phase 2 needs explicit go-ahead, and it's where the grid-vs-SMC Bayesian decision and the RTX 3070 claim happen).
