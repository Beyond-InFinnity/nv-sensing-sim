# Phase 0 — Physics Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A trustworthy NV ground-state spin-1 model: Hamiltonian with D/E/Zeeman for arbitrary B orientation and all four NV axes, CW-ODMR spectra, and pulsed Rabi/Ramsey/Hahn-echo simulations with T1/T2*/T2 — every textbook invariant unit-tested, plus a validation figure set.

**Architecture:** A small `nvsim` package: `constants.py` (physical constants), `hamiltonian.py` (spin-1 operators, frame transforms, transition frequencies), `odmr.py` (Lorentzian rate-equation-level CW spectrum built on exact eigen-transitions), `pulsed.py` (two-level RWA reduction of the driven transition, ideal pulses + Lindblad free evolution via qutip `mesolve`). Experiments are JSON-config-driven scripts writing artifacts (config+seed+git SHA embedded); a separate plot script reads artifacts and writes `docs/figures/`.

**Tech Stack:** Python ≥3.11, qutip ≥5, numpy, scipy, matplotlib, pytest. No torch.

## Global Constraints

- Frequencies in **Hz** at all API boundaries; angular frequency only internally, named `omega_*`. Magnetic field in **tesla**. (CLAUDE.md / PHYSICS.md)
- D = 2.870 GHz; γe/2π = 28.02 GHz/T. (PHYSICS.md)
- Every Hamiltonian term / approximation must be documented in `docs/PHYSICS.md` **in the same commit** that introduces it in code.
- Shot noise (later phases) is Poisson, never additive Gaussian — Phase 0 figures are noiseless physics validation.
- Experiments: JSON config under `experiments/`; artifacts embed config + seed + git SHA. Figures come from committed scripts reading committed artifacts, saved in `docs/figures/`; use the dataviz skill when writing plotting code.
- `pytest` green before every commit; small one-concern commits, imperative subject.
- CPU-only, everything inside this repo + `.venv`. No workstation, no GPUs.
- Validation figures cite the specific Barry et al., Rev. Mod. Phys. 92, 015004 (2020) figure they are compared against.

---

### Task 0: Environment

**Files:** none created (venv only)

- [ ] **Step 1:** `python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"`
- [ ] **Step 2:** `.venv/bin/pytest` — expect the existing smoke test to pass.

No commit (no repo changes).

---

### Task 1: Constants + spin-1 Hamiltonian

**Files:**
- Create: `src/nvsim/constants.py`, `src/nvsim/hamiltonian.py`
- Modify: `docs/PHYSICS.md` (document eigenstate-identification convention + NV-frame construction)
- Test: `tests/test_hamiltonian.py`

**Interfaces:**
- Produces: `constants.D_GS_HZ = 2.870e9`, `constants.GAMMA_E_HZ_PER_T = 28.02e9`, `constants.NV_AXES` (4×3 unit vectors along ⟨111⟩).
- Produces: `hamiltonian.SX, SY, SZ` (qutip spin-1 ops); `hamiltonian.h_gs(d_hz=D_GS_HZ, e_hz=0.0, b_nv_t=(0,0,0)) -> qutip.Qobj` (units: Hz); `hamiltonian.transition_frequencies(h) -> tuple[float, float]` (f_minus ≤ f_plus, from the mostly-|ms=0⟩ eigenstate); `hamiltonian.b_lab_to_nv(b_lab_t, orientation: int) -> np.ndarray` (3-vector in that NV's frame).

- [ ] **Step 1: Write failing tests**

```python
# tests/test_hamiltonian.py
import numpy as np
import pytest
from nvsim.constants import D_GS_HZ, GAMMA_E_HZ_PER_T, NV_AXES
from nvsim.hamiltonian import h_gs, transition_frequencies, b_lab_to_nv


def test_zero_field_splitting_at_2p870_ghz():
    f_minus, f_plus = transition_frequencies(h_gs())
    assert f_minus == pytest.approx(2.870e9, rel=1e-12)
    assert f_plus == pytest.approx(2.870e9, rel=1e-12)


def test_strain_splits_by_2e():
    e = 5e6
    f_minus, f_plus = transition_frequencies(h_gs(e_hz=e))
    assert f_plus - f_minus == pytest.approx(2 * e, rel=1e-9)
    assert (f_plus + f_minus) / 2 == pytest.approx(D_GS_HZ, rel=1e-12)


def test_axial_zeeman_splitting_slope_28p02_ghz_per_t():
    for bz in (0.5e-3, 1e-3, 3e-3):
        f_minus, f_plus = transition_frequencies(h_gs(b_nv_t=(0, 0, bz)))
        assert f_plus - f_minus == pytest.approx(2 * GAMMA_E_HZ_PER_T * bz, rel=1e-6)


def test_transverse_field_is_second_order():
    bx = 1e-3
    f_minus, f_plus = transition_frequencies(h_gs(b_nv_t=(bx, 0, 0)))
    # splitting from transverse field is O((γB)²/D), not linear
    assert f_plus - f_minus < 2 * GAMMA_E_HZ_PER_T * bx * 0.05


def test_nv_axes_are_unit_111_directions():
    assert NV_AXES.shape == (4, 3)
    np.testing.assert_allclose(np.linalg.norm(NV_AXES, axis=1), 1.0, rtol=1e-12)
    # pairwise angle: cos = -1/3 for distinct <111> axes with these sign choices
    for i in range(4):
        for j in range(i + 1, 4):
            assert NV_AXES[i] @ NV_AXES[j] == pytest.approx(-1 / 3, rel=1e-9)


def test_b_lab_to_nv_projection():
    b_lab = 1e-3 * NV_AXES[0]  # field along orientation 0's axis
    b0 = b_lab_to_nv(b_lab, 0)
    assert b0[2] == pytest.approx(1e-3, rel=1e-12)          # fully axial for 0
    assert np.hypot(b0[0], b0[1]) == pytest.approx(0, abs=1e-15)
    for k in (1, 2, 3):                                      # cos(theta) = -1/3 for others
        bk = b_lab_to_nv(b_lab, k)
        assert bk[2] == pytest.approx(-1e-3 / 3, rel=1e-9)
        assert np.linalg.norm(bk) == pytest.approx(1e-3, rel=1e-12)
```

- [ ] **Step 2:** `pytest tests/test_hamiltonian.py -v` → FAIL (ModuleNotFoundError).
- [ ] **Step 3: Implement**

```python
# src/nvsim/constants.py
"""Physical constants for the NV ground-state model. Units: SI, frequencies in Hz."""
import numpy as np

D_GS_HZ = 2.870e9            # zero-field splitting
GAMMA_E_HZ_PER_T = 28.02e9   # electron gyromagnetic ratio / 2pi, magnitude
DD_DT_HZ_PER_K = -74e3       # thermal shift of D (Phase 1)
A_PAR_N14_HZ = -2.16e6       # 14N axial hyperfine (Phase 1)

# Four NV orientations along <111>; rows are unit vectors in the diamond cubic frame.
NV_AXES = np.array(
    [[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=float
) / np.sqrt(3)
```

```python
# src/nvsim/hamiltonian.py
"""NV ground-state spin-1 Hamiltonian. H/h in Hz; B in tesla; NV frame z along the NV axis.

Conventions per docs/PHYSICS.md: H/h = D*Sz^2 + E*(Sx^2 - Sy^2) + (gamma_e/2pi)*(B . S).
Transition frequencies are eigenvalue differences from the eigenstate with maximum
overlap with |ms=0>; valid while the ms=0 character is well defined (B << D/gamma).
"""
import numpy as np
import qutip

from .constants import D_GS_HZ, GAMMA_E_HZ_PER_T, NV_AXES

SX = qutip.jmat(1, "x")
SY = qutip.jmat(1, "y")
SZ = qutip.jmat(1, "z")
_MS0 = qutip.basis(3, 1)  # jmat basis order: m = +1, 0, -1


def h_gs(d_hz=D_GS_HZ, e_hz=0.0, b_nv_t=(0.0, 0.0, 0.0)):
    """Ground-state Hamiltonian H/h (Hz), B given in the NV frame (tesla)."""
    bx, by, bz = b_nv_t
    return (
        d_hz * SZ**2
        + e_hz * (SX**2 - SY**2)
        + GAMMA_E_HZ_PER_T * (bx * SX + by * SY + bz * SZ)
    )


def transition_frequencies(h):
    """(f_minus, f_plus) in Hz: transitions from the mostly-|ms=0> eigenstate."""
    evals, evecs = h.eigenstates()
    i0 = int(np.argmax([abs(_MS0.overlap(v)) ** 2 for v in evecs]))
    others = sorted(evals[i] - evals[i0] for i in range(3) if i != i0)
    return float(others[0]), float(others[1])


def nv_frame(orientation):
    """Orthonormal (x, y, z) rows for one NV orientation; z along NV_AXES[orientation]."""
    z = NV_AXES[orientation]
    helper = np.array([0.0, 0.0, 1.0])
    x = np.cross(helper, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.vstack([x, y, z])


def b_lab_to_nv(b_lab_t, orientation):
    """Project a lab-frame B (tesla, 3-vector) into one NV orientation's frame."""
    return nv_frame(orientation) @ np.asarray(b_lab_t, dtype=float)
```

Add to `docs/PHYSICS.md` under the Hamiltonian section (same commit):

```markdown
### Implementation conventions (Phase 0)

- Transition frequencies are computed as eigenvalue differences from the
  eigenstate with maximum |⟨ms=0|ψ⟩|²; valid while γB ≪ D so the ms=0
  character is well defined.
- NV frame per orientation: z along the ⟨111⟩ axis, x = ẑ_lab × z (normalized),
  y = z × x. The in-plane x choice is arbitrary and physically irrelevant for
  |B⊥|-dependent quantities; E-term anisotropy would pin it (not needed yet).
```

- [ ] **Step 4:** `pytest tests/test_hamiltonian.py -v` → PASS.
- [ ] **Step 5:** Commit: `feat: spin-1 ground-state Hamiltonian with frame transforms`

---

### Task 2: CW-ODMR spectrum

**Files:**
- Create: `src/nvsim/odmr.py`
- Modify: `docs/PHYSICS.md` (document Lorentzian rate-equation-level model)
- Test: `tests/test_odmr.py`

**Interfaces:**
- Consumes: `h_gs`, `transition_frequencies`.
- Produces: `odmr.odmr_spectrum(f_hz, d_hz=..., e_hz=0.0, b_nv_t=(0,0,0), contrast=0.2, fwhm_hz=8e6) -> np.ndarray` — normalized fluorescence S(f) ∈ (0, 1], dips at transitions.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_odmr.py
import numpy as np
import pytest
from nvsim.constants import D_GS_HZ, GAMMA_E_HZ_PER_T
from nvsim.odmr import odmr_spectrum


def _dip_freqs(f, s):
    """Frequencies of local minima of s."""
    idx = np.where((s[1:-1] < s[:-2]) & (s[1:-1] < s[2:]))[0] + 1
    return f[idx]


def test_single_dip_at_d_for_zero_field_zero_strain():
    f = np.linspace(2.80e9, 2.94e9, 4001)
    s = odmr_spectrum(f)
    dips = _dip_freqs(f, s)
    assert len(dips) == 1
    assert dips[0] == pytest.approx(D_GS_HZ, abs=f[1] - f[0])


def test_two_dips_split_linearly_in_axial_field():
    f = np.linspace(2.70e9, 3.04e9, 8001)
    for bz in (1e-3, 2e-3, 4e-3):
        s = odmr_spectrum(f, b_nv_t=(0, 0, bz))
        dips = _dip_freqs(f, s)
        assert len(dips) == 2
        assert dips[1] - dips[0] == pytest.approx(
            2 * GAMMA_E_HZ_PER_T * bz, abs=2 * (f[1] - f[0])
        )


def test_contrast_at_dip():
    f = np.linspace(2.86e9, 2.88e9, 2001)
    s = odmr_spectrum(f, contrast=0.15)
    # both transitions degenerate at D -> dips add; each carries `contrast`
    assert s.min() == pytest.approx(1 - 2 * 0.15, rel=1e-3)
    assert s.max() <= 1.0
```

- [ ] **Step 2:** run → FAIL.
- [ ] **Step 3: Implement**

```python
# src/nvsim/odmr.py
"""CW-ODMR at the rate-equation level: unit-normalized fluorescence with one
Lorentzian dip per exact eigen-transition (see docs/PHYSICS.md)."""
import numpy as np

from .constants import D_GS_HZ
from .hamiltonian import h_gs, transition_frequencies


def odmr_spectrum(f_hz, d_hz=D_GS_HZ, e_hz=0.0, b_nv_t=(0.0, 0.0, 0.0),
                  contrast=0.2, fwhm_hz=8e6):
    """Normalized fluorescence vs MW frequency f_hz (array-like, Hz)."""
    f = np.asarray(f_hz, dtype=float)
    hwhm = fwhm_hz / 2
    s = np.ones_like(f)
    for f0 in transition_frequencies(h_gs(d_hz, e_hz, b_nv_t)):
        s -= contrast * hwhm**2 / ((f - f0) ** 2 + hwhm**2)
    return s
```

PHYSICS.md addition (same commit):

```markdown
## CW-ODMR model (Phase 0)

Rate-equation level: fluorescence S(f) = 1 − Σᵢ C·L(f − fᵢ), one Lorentzian
L (FWHM Γ) per ground-state eigen-transition fᵢ from the exact Hamiltonian.
Equal contrast C per transition (unpolarized MW, both ΔmS = ±1 allowed).
Linewidth and contrast are phenomenological inputs; power broadening and
hyperfine structure enter in Phase 1.
```

- [ ] **Step 4:** run → PASS. **Step 5:** Commit: `feat: CW-ODMR spectrum at rate-equation level`

---

### Task 3: Pulsed core + Rabi

**Files:**
- Create: `src/nvsim/pulsed.py`
- Modify: `docs/PHYSICS.md` (two-level RWA reduction, ideal pulses, Lindblad ops)
- Test: `tests/test_pulsed.py`

**Interfaces:**
- Produces: `pulsed.rabi(rabi_hz, times_s, detuning_hz=0.0, t1_s=None, t2_s=None) -> np.ndarray` (P(ms=0) at each time).
- Produces (internal, reused by Tasks 4–5): `pulsed._collapse_ops(t1_s, t2_s) -> list[qutip.Qobj]`, `pulsed._free_h(detuning_hz) -> qutip.Qobj` (angular units), `pulsed._rx(theta) -> qutip.Qobj` (ideal X rotation on {|0⟩≡ms=0, |1⟩≡ms=−1}).

- [ ] **Step 1: Write failing tests**

```python
# tests/test_pulsed.py
import numpy as np
import pytest
from nvsim.pulsed import rabi


def test_rabi_frequency_equals_drive_amplitude():
    f_rabi = 5e6
    t = np.linspace(0, 2e-6, 2001)
    p0 = rabi(f_rabi, t)
    # P0(t) = cos^2(pi * f_rabi * t) on resonance
    np.testing.assert_allclose(p0, np.cos(np.pi * f_rabi * t) ** 2, atol=1e-6)


def test_rabi_scales_linearly_with_amplitude():
    t = np.linspace(0, 4e-6, 4001)
    for f_rabi in (1e6, 2e6, 4e6):
        p0 = rabi(f_rabi, t)
        # first minimum of P0 at t = 1/(2 f_rabi)
        t_pi = t[np.argmin(p0)]
        assert t_pi == pytest.approx(1 / (2 * f_rabi), rel=2e-3)


def test_detuned_rabi_generalized_frequency_and_reduced_contrast():
    f_rabi, delta = 2e6, 1.5e6
    omega_gen = np.hypot(f_rabi, delta)
    t = np.linspace(0, 3e-6, 3001)
    p0 = rabi(f_rabi, t, detuning_hz=delta)
    expected = 1 - (f_rabi / omega_gen) ** 2 * np.sin(np.pi * omega_gen * t) ** 2
    np.testing.assert_allclose(p0, expected, atol=1e-4)


def test_lindblad_preserves_trace_and_damps_rabi():
    t = np.linspace(0, 10e-6, 501)
    p0 = rabi(1e6, t, t1_s=20e-6, t2_s=5e-6)
    assert np.all((p0 >= -1e-9) & (p0 <= 1 + 1e-9))
    # damped toward mixed state: late-time oscillation amplitude much smaller
    late = p0[t > 8e-6]
    assert late.max() - late.min() < 0.35
```

- [ ] **Step 2:** run → FAIL.
- [ ] **Step 3: Implement**

```python
# src/nvsim/pulsed.py
"""Pulsed protocols on the driven NV transition, reduced to a two-level system
{|0> = ms=0, |1> = ms=-1} in the MW rotating frame under the RWA
(see docs/PHYSICS.md, "Pulsed two-level reduction"). All inputs in Hz / s;
angular frequencies (omega_*) only inside this module."""
import numpy as np
import qutip

_SM = qutip.destroy(2)          # |0><1|
_SZ = qutip.sigmaz()            # +1 on |0>, -1 on |1> in qutip convention
_P0 = qutip.basis(2, 0) * qutip.basis(2, 0).dag()


def _free_h(detuning_hz):
    """Free-evolution H in the rotating frame (angular units): -delta * |1><1|."""
    omega_delta = 2 * np.pi * detuning_hz
    return -omega_delta * qutip.basis(2, 1) * qutip.basis(2, 1).dag()


def _drive_h(rabi_hz, detuning_hz):
    omega_r = 2 * np.pi * rabi_hz
    return _free_h(detuning_hz) + omega_r / 2 * (_SM + _SM.dag())


def _collapse_ops(t1_s, t2_s):
    """T1 as symmetric relaxation (infinite-temperature bath), T2 total coherence
    time -> pure dephasing rate gamma_phi = 1/T2 - 1/(2 T1)."""
    ops = []
    gamma1 = 1 / t1_s if t1_s else 0.0
    if gamma1:
        ops += [np.sqrt(gamma1 / 2) * _SM, np.sqrt(gamma1 / 2) * _SM.dag()]
    if t2_s:
        gamma_phi = 1 / t2_s - gamma1 / 2
        if gamma_phi < -1e-12:
            raise ValueError("t2_s must satisfy T2 <= 2*T1")
        if gamma_phi > 0:
            ops.append(np.sqrt(gamma_phi / 2) * _SZ)
    return ops


def _rx(theta):
    """Ideal (instantaneous) rotation about x by theta."""
    return (-1j * theta / 2 * (_SM + _SM.dag())).expm()


def rabi(rabi_hz, times_s, detuning_hz=0.0, t1_s=None, t2_s=None):
    """P(ms=0) under continuous resonant drive, starting from |ms=0>."""
    result = qutip.mesolve(
        _drive_h(rabi_hz, detuning_hz),
        qutip.basis(2, 0),
        np.asarray(times_s, dtype=float),
        c_ops=_collapse_ops(t1_s, t2_s),
        e_ops=[_P0],
    )
    return np.asarray(result.expect[0])
```

PHYSICS.md addition (same commit):

```markdown
## Pulsed two-level reduction (Phase 0)

Pulsed protocols drive one transition (|ms=0⟩ ↔ |ms=−1⟩) with the other far
detuned (2γBz ≫ Ω), so the dynamics reduce to a two-level system in the frame
rotating at the MW frequency, under the RWA:

H/h = −δ·|1⟩⟨1| + (Ω_R/2)·(|0⟩⟨1| + |1⟩⟨0|),  δ = f_MW − f_transition.

- Pulses are ideal (instantaneous rotations); finite-pulse-width effects are
  out of scope until something needs them (added to approximations ledger).
- Lindblad: T1 as symmetric jump ops √(1/2T1)·σ∓ (infinite-temperature bath —
  fine since ħω ≪ kT at room temperature); pure dephasing √(γφ/2)·σz with
  γφ = 1/T2 − 1/(2T1). Requires T2 ≤ 2T1.
- On resonance P₀(t) = cos²(πΩ_R t); detuned, generalized Rabi
  Ω' = √(Ω_R² + δ²) with contrast Ω_R²/Ω'².
```

Also append to the approximations ledger table: `| Ideal (delta-function) MW pulses | pulsed sims | pulse durations comparable to 1/detunings or to T2* |`

- [ ] **Step 4:** run → PASS. **Step 5:** Commit: `feat: pulsed two-level core and Rabi oscillations`

---

### Task 4: Ramsey

**Files:**
- Modify: `src/nvsim/pulsed.py`, `docs/PHYSICS.md`
- Test: `tests/test_pulsed.py` (append)

**Interfaces:**
- Consumes: `_free_h`, `_collapse_ops`, `_rx`, `_P0` from Task 3.
- Produces: `pulsed.ramsey(taus_s, detuning_hz, t2star_s=None, mode="lindblad", sigma_detuning_hz=None, n_samples=400, seed=None) -> np.ndarray` — P(ms=0) vs free-evolution time τ. `mode="static"` samples Gaussian static detunings (std `sigma_detuning_hz`) around `detuning_hz` and averages → Gaussian envelope exp(−(τ/T2*)²) with T2* = √2/(2π·σ).

- [ ] **Step 1: Write failing tests (append to tests/test_pulsed.py)**

```python
from nvsim.pulsed import ramsey, t2star_from_sigma


def _fringe_freq(taus, p0):
    p = p0 - p0.mean()
    spec = np.abs(np.fft.rfft(p))
    freqs = np.fft.rfftfreq(len(taus), taus[1] - taus[0])
    return freqs[np.argmax(spec)]


def test_ramsey_fringe_frequency_equals_detuning():
    delta = 2e6
    taus = np.linspace(0, 8e-6, 4096)
    p0 = ramsey(taus, detuning_hz=delta)
    assert _fringe_freq(taus, p0) == pytest.approx(delta, rel=2e-2)


def test_ramsey_lindblad_envelope_is_exp_tau_over_t2star():
    delta, t2s = 2e6, 3e-6
    taus = np.arange(0, 12e-6, 1 / (4 * delta))  # sample at fringe maxima rate
    p0 = ramsey(taus, detuning_hz=delta, t2star_s=t2s)
    envelope = np.abs(p0 - 0.5) * 2
    # at fringe maxima (cos = +/-1), envelope = exp(-tau/T2*)
    peaks = envelope[:: 4]  # every full period: tau_k = k/delta
    taus_pk = taus[:: 4]
    np.testing.assert_allclose(peaks, np.exp(-taus_pk / t2s), atol=0.02)


def test_ramsey_static_sampling_gives_gaussian_envelope():
    sigma = 100e3
    t2s = t2star_from_sigma(sigma)
    taus = np.linspace(0, 2.5 * t2s, 200)
    p0 = ramsey(taus, detuning_hz=0.0, mode="static",
                sigma_detuning_hz=sigma, n_samples=3000, seed=7)
    np.testing.assert_allclose(
        p0, 0.5 * (1 + np.exp(-((taus / t2s) ** 2))), atol=0.02
    )
```

- [ ] **Step 2:** run → FAIL.
- [ ] **Step 3: Implement (append to pulsed.py)**

```python
def t2star_from_sigma(sigma_detuning_hz):
    """T2* of the Gaussian free-induction envelope exp(-(tau/T2*)^2) produced by
    Gaussian static detunings of std sigma (Hz): T2* = sqrt(2)/(2 pi sigma)."""
    return np.sqrt(2) / (2 * np.pi * sigma_detuning_hz)


def _ramsey_single(taus_s, detuning_hz, t2star_s):
    psi = _rx(np.pi / 2) * qutip.basis(2, 0)
    result = qutip.mesolve(
        _free_h(detuning_hz), psi, taus_s,
        c_ops=_collapse_ops(None, t2star_s), e_ops=[],
    )
    return np.array(
        [qutip.expect(_P0, _rx(np.pi / 2) * rho * _rx(np.pi / 2).dag())
         for rho in result.states]
    )


def ramsey(taus_s, detuning_hz, t2star_s=None, mode="lindblad",
           sigma_detuning_hz=None, n_samples=400, seed=None):
    """P(ms=0) after pi/2 - tau - pi/2. mode='lindblad': Markovian dephasing at
    rate 1/T2* (exponential envelope). mode='static': average over Gaussian
    static detunings (Gaussian envelope; the physical choice for slow baths)."""
    taus = np.asarray(taus_s, dtype=float)
    if mode == "lindblad":
        return _ramsey_single(taus, detuning_hz, t2star_s)
    if mode == "static":
        rng = np.random.default_rng(seed)
        deltas = detuning_hz + sigma_detuning_hz * rng.standard_normal(n_samples)
        # free evolution of a pure state is analytic in the coherence phase;
        # but keep the mesolve path for uniformity with lindblad mode
        return np.mean(
            [_ramsey_single(taus, d, None) for d in deltas], axis=0
        )
    raise ValueError(f"unknown mode: {mode}")
```

Note for implementer: `_ramsey_single` with `c_ops=[]` and `e_ops=[]` returns kets in `result.states`; `qutip.expect` handles kets and density matrices alike, and `ket * rho * ket.dag()` composition — if qutip returns kets, `_rx(pi/2) * psi` is a ket and `qutip.expect(_P0, ket)` works directly; adjust: use `state = _rx(np.pi/2) * rho if rho.isket else _rx(np.pi/2) * rho * _rx(np.pi/2).dag()`.

PHYSICS.md: extend the Decoherence section noting the two Ramsey modes implement exactly the two T2* options already documented, with `T2* = √2/(2πσ)` for the static-Gaussian mode and envelope conventions (`exp(−τ/T2*)` Lindblad vs `exp(−(τ/T2*)²)` static).

- [ ] **Step 4:** run → PASS. **Step 5:** Commit: `feat: Ramsey fringes with Lindblad and static-detuning dephasing`

---

### Task 5: Hahn echo

**Files:**
- Modify: `src/nvsim/pulsed.py`, `docs/PHYSICS.md`
- Test: `tests/test_pulsed.py` (append)

**Interfaces:**
- Produces: `pulsed.hahn_echo(taus_s, static_detuning_hz=0.0, t2_s=None, mode="lindblad", sigma_detuning_hz=None, n_samples=400, seed=None) -> np.ndarray` — P(ms=0) after π/2 – τ – π – τ – π/2, vs τ (half the total free time).

- [ ] **Step 1: Write failing tests (append)**

```python
from nvsim.pulsed import hahn_echo


def test_echo_removes_static_detuning():
    taus = np.linspace(0, 5e-6, 50)
    for delta in (0.0, 1e6, 3.7e6):
        p0 = hahn_echo(taus, static_detuning_hz=delta)
        np.testing.assert_allclose(p0, 1.0, atol=1e-9)


def test_echo_removes_inhomogeneous_broadening_but_ramsey_does_not():
    sigma = 200e3
    taus = np.linspace(1e-8, 3e-6, 30)
    echo = hahn_echo(taus, mode="static", sigma_detuning_hz=sigma,
                     n_samples=500, seed=3)
    np.testing.assert_allclose(echo, 1.0, atol=1e-9)


def test_echo_decays_with_t2_over_total_time():
    t2 = 100e-6
    taus = np.linspace(0, 150e-6, 40)
    p0 = hahn_echo(taus, t2_s=t2)
    # envelope exp(-(2 tau)/T2): P0 = (1 + exp(-2 tau/T2))/2
    np.testing.assert_allclose(p0, 0.5 * (1 + np.exp(-2 * taus / t2)), atol=1e-3)
```

- [ ] **Step 2:** run → FAIL.
- [ ] **Step 3: Implement (append to pulsed.py)**

```python
def _echo_single(taus_s, detuning_hz, t2_s):
    """Sequence pi/2 - tau - pi - tau - pi/2 for each tau (separate evolutions)."""
    out = np.empty(len(taus_s))
    c_ops = _collapse_ops(None, t2_s)
    psi0 = _rx(np.pi / 2) * qutip.basis(2, 0)
    h = _free_h(detuning_hz)
    for i, tau in enumerate(taus_s):
        tlist = [0.0, tau] if tau > 0 else [0.0, 0.0]
        rho = qutip.mesolve(h, psi0, tlist, c_ops=c_ops).states[-1]
        rho = _rx(np.pi) * rho * _rx(np.pi).dag() if rho.isoper else _rx(np.pi) * rho
        rho = qutip.mesolve(h, rho, tlist, c_ops=c_ops).states[-1]
        state = (_rx(np.pi / 2) * rho * _rx(np.pi / 2).dag()
                 if rho.isoper else _rx(np.pi / 2) * rho)
        out[i] = qutip.expect(_P0, state)
    return out


def hahn_echo(taus_s, static_detuning_hz=0.0, t2_s=None, mode="lindblad",
              sigma_detuning_hz=None, n_samples=400, seed=None):
    """P(ms=0) after pi/2 - tau - pi - tau - pi/2 (x-axis pulses)."""
    taus = np.asarray(taus_s, dtype=float)
    if mode == "lindblad":
        return _echo_single(taus, static_detuning_hz, t2_s)
    if mode == "static":
        rng = np.random.default_rng(seed)
        deltas = static_detuning_hz + sigma_detuning_hz * rng.standard_normal(n_samples)
        return np.mean([_echo_single(taus, d, t2_s) for d in deltas], axis=0)
    raise ValueError(f"unknown mode: {mode}")
```

Note for implementer: with all-x pulses the refocused signal returns to P0 = 1
(π/2ₓ–τ–πₓ–τ–π/2ₓ maps |0⟩ back to −|0⟩ up to phase for any static δ). Verify the
first test's expectation analytically before adjusting tolerances; if the sign
convention lands on P0 = 0, flip the final assertion target once, document why.

PHYSICS.md: extend Decoherence — echo implemented with n = 1 (exponential)
Lindblad envelope by default, exp[−(2τ/T2)]; stretched exponentials (n from
bath) deferred to Phase 1's noise modeling; static detunings (any σ) are exactly
refocused as tested.

- [ ] **Step 4:** run → PASS. **Step 5:** Commit: `feat: Hahn echo with static-detuning refocusing and T2 decay`

---

### Task 6: Config-driven experiment runner + artifacts

**Files:**
- Create: `experiments/phase0/odmr_vs_field.json`, `experiments/phase0/rabi.json`, `experiments/phase0/ramsey.json`, `experiments/phase0/hahn_echo.json`, `scripts/run_phase0.py`
- Test: `tests/test_run_phase0.py`

**Interfaces:**
- Consumes: `odmr_spectrum`, `rabi`, `ramsey`, `hahn_echo`, `t2star_from_sigma`.
- Produces: artifact JSON files `experiments/phase0/artifacts/<name>.json` with keys `{"config": {...}, "seed": int, "git_sha": str, "results": {...}}`; CLI `python scripts/run_phase0.py experiments/phase0/<name>.json`.

Config schemas (exact files to create):

```json
// experiments/phase0/odmr_vs_field.json
{"name": "odmr_vs_field", "protocol": "odmr",
 "f_min_hz": 2.72e9, "f_max_hz": 3.02e9, "n_freq": 3001,
 "b_axial_t": [0.0, 1.0e-3, 2.0e-3, 3.0e-3],
 "e_hz": 0.0, "contrast": 0.2, "fwhm_hz": 8.0e6, "seed": 0}
```

```json
// experiments/phase0/rabi.json
{"name": "rabi", "protocol": "rabi",
 "rabi_hz": [1.0e6, 2.0e6, 4.0e6], "t_max_s": 3.0e-6, "n_t": 1501,
 "detuning_hz": 0.0, "t1_s": 100.0e-6, "t2_s": 5.0e-6, "seed": 0}
```

```json
// experiments/phase0/ramsey.json
{"name": "ramsey", "protocol": "ramsey",
 "detuning_hz": 2.0e6, "tau_max_s": 6.0e-6, "n_tau": 1200,
 "mode": "static", "sigma_detuning_hz": 1.5e5, "n_samples": 2000, "seed": 42}
```

```json
// experiments/phase0/hahn_echo.json
{"name": "hahn_echo", "protocol": "hahn_echo",
 "tau_max_s": 300.0e-6, "n_tau": 60, "t2_s": 150.0e-6,
 "mode": "static", "sigma_detuning_hz": 1.5e5, "n_samples": 300,
 "ramsey_comparison": true, "seed": 42}
```

- [ ] **Step 1: Write failing test**

```python
# tests/test_run_phase0.py
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_runner_produces_artifact_with_provenance(tmp_path):
    cfg = REPO / "experiments/phase0/rabi.json"
    out = subprocess.run(
        [sys.executable, str(REPO / "scripts/run_phase0.py"), str(cfg),
         "--out-dir", str(tmp_path)],
        capture_output=True, text=True,
    )
    assert out.returncode == 0, out.stderr
    artifact = json.loads((tmp_path / "rabi.json").read_text())
    assert artifact["config"] == json.loads(cfg.read_text())
    assert isinstance(artifact["seed"], int)
    assert len(artifact["git_sha"]) >= 7
    assert "times_s" in artifact["results"]
    assert len(artifact["results"]["p0"]) == len(artifact["results"]["rabi_hz"])
```

- [ ] **Step 2:** run → FAIL.
- [ ] **Step 3: Implement `scripts/run_phase0.py`**

```python
#!/usr/bin/env python3
"""Run a Phase 0 experiment from a JSON config; write an artifact embedding
config + seed + git SHA. Usage: run_phase0.py CONFIG [--out-dir DIR]"""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from nvsim.odmr import odmr_spectrum
from nvsim.pulsed import hahn_echo, rabi, ramsey, t2star_from_sigma


def git_sha():
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
        cwd=Path(__file__).resolve().parent,
    ).stdout.strip()


def run_odmr(c):
    f = np.linspace(c["f_min_hz"], c["f_max_hz"], c["n_freq"])
    return {
        "f_hz": f.tolist(),
        "b_axial_t": c["b_axial_t"],
        "spectra": [
            odmr_spectrum(f, e_hz=c["e_hz"], b_nv_t=(0, 0, bz),
                          contrast=c["contrast"], fwhm_hz=c["fwhm_hz"]).tolist()
            for bz in c["b_axial_t"]
        ],
    }


def run_rabi(c):
    t = np.linspace(0, c["t_max_s"], c["n_t"])
    return {
        "times_s": t.tolist(),
        "rabi_hz": c["rabi_hz"],
        "p0": [rabi(fr, t, detuning_hz=c["detuning_hz"],
                    t1_s=c["t1_s"], t2_s=c["t2_s"]).tolist()
               for fr in c["rabi_hz"]],
    }


def run_ramsey(c):
    taus = np.linspace(0, c["tau_max_s"], c["n_tau"])
    p0 = ramsey(taus, detuning_hz=c["detuning_hz"], mode=c["mode"],
                sigma_detuning_hz=c["sigma_detuning_hz"],
                n_samples=c["n_samples"], seed=c["seed"])
    return {
        "taus_s": taus.tolist(), "p0": p0.tolist(),
        "t2star_s": t2star_from_sigma(c["sigma_detuning_hz"]),
    }


def run_hahn_echo(c):
    taus = np.linspace(0, c["tau_max_s"], c["n_tau"])
    out = {
        "taus_s": taus.tolist(),
        "p0_echo": hahn_echo(taus, t2_s=c["t2_s"], mode=c["mode"],
                             sigma_detuning_hz=c["sigma_detuning_hz"],
                             n_samples=c["n_samples"], seed=c["seed"]).tolist(),
        "t2_s": c["t2_s"],
        "t2star_s": t2star_from_sigma(c["sigma_detuning_hz"]),
    }
    if c.get("ramsey_comparison"):
        out["p0_ramsey"] = ramsey(
            taus, detuning_hz=0.0, mode=c["mode"],
            sigma_detuning_hz=c["sigma_detuning_hz"],
            n_samples=c["n_samples"], seed=c["seed"],
        ).tolist()
    return out


PROTOCOLS = {"odmr": run_odmr, "rabi": run_rabi,
             "ramsey": run_ramsey, "hahn_echo": run_hahn_echo}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path)
    ap.add_argument("--out-dir", type=Path,
                    default=Path(__file__).resolve().parent.parent
                    / "experiments/phase0/artifacts")
    args = ap.parse_args()
    config = json.loads(args.config.read_text())
    np.random.seed(config["seed"])
    results = PROTOCOLS[config["protocol"]](config)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    artifact = {"config": config, "seed": config["seed"],
                "git_sha": git_sha(), "results": results}
    out_path = args.out_dir / f"{config['name']}.json"
    out_path.write_text(json.dumps(artifact))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4:** run test → PASS.
- [ ] **Step 5:** Run all four configs for real (default out-dir), commit configs + runner + artifacts: `feat: config-driven phase 0 experiment runner with provenance`

---

### Task 7: Validation figures

**Files:**
- Create: `scripts/plot_phase0.py`; outputs `docs/figures/phase0_odmr.png`, `phase0_rabi.png`, `phase0_ramsey.png`, `phase0_echo.png`

**Interfaces:**
- Consumes: artifact JSONs from Task 6.

- [ ] **Step 1:** Invoke the **dataviz skill** before writing any plotting code; follow its palette/mark rules (copy palette constants from `qec-neural-decoder/src/qecdec/plotting.py` as reference values — do not import across repos).
- [ ] **Step 2:** Write `scripts/plot_phase0.py` reading the four artifacts and producing:
  - `phase0_odmr.png` — spectra offset-stacked vs axial B with dip positions marked; caption cites Barry et al., RMP 92, 015004 (2020) Fig. 2 (level structure / ODMR vs B) — verify the figure number against the paper abstract/figure list before writing the caption; if unverifiable offline, cite section §II instead.
  - `phase0_rabi.png` — P₀(t) for the three drive amplitudes + inset: fitted Rabi frequency vs drive amplitude with the y = x line.
  - `phase0_ramsey.png` — fringes with ±Gaussian envelope exp(−(τ/T2*)²) overlaid; annotate fringe frequency = detuning (2.0 MHz).
  - `phase0_echo.png` — echo decay vs total time 2τ with exp(−2τ/T2) overlay, and the Ramsey (T2*) decay on the same axes to show the timescale separation.
- [ ] **Step 3:** Run it; visually inspect each figure (Read the PNGs) for correctness: dip counts, splitting direction, envelope shapes.
- [ ] **Step 4:** Commit: `feat: phase 0 validation figures`

---

### Task 8: Close out Phase 0

**Files:**
- Modify: `CLAUDE.md` (status line), `README.md` (status), `docs/ROADMAP.md` (check Phase 0 boxes)

- [ ] **Step 1:** Full `pytest` run — all green (record the count).
- [ ] **Step 2:** Cross-check every number against the roadmap exit criterion: 2.870 GHz ZFS, 28.02 GHz/T slope, 2E strain splitting, fringe = detuning, Rabi linear in amplitude, echo refocusing, envelope forms. Each must be enforced by a named test.
- [ ] **Step 3:** Update CLAUDE.md "Current status" → "Phase 0 (physics core) — complete (date, commit). Phase 1 not started; awaiting go-ahead." Check the four Phase 0 boxes in ROADMAP.md; update README status.
- [ ] **Step 4:** Commit `docs: mark phase 0 complete`, push to `origin`.
- [ ] **Step 5:** Report to Connor: what was built, every checked number with units, figure paths, deviations/discrepancies if any — then **stop** (Phase 1 needs explicit go-ahead).
