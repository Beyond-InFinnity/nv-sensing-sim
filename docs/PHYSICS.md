# Physics model notes

Running reference for every convention and approximation in `nvsim`. If a
Hamiltonian term isn't documented here, it doesn't go in the code.

## Ground-state spin Hamiltonian

Spin-1 (S = 1), in the NV frame (z along the NV axis), frequencies in Hz:

H/h = D·Sz² + E·(Sx² − Sy²) + (γe/2π)·(Bx·Sx + By·Sy + Bz·Sz)

- D ≈ 2.870 GHz (zero-field splitting), temperature-dependent: dD/dT ≈ −74 kHz/K
- E: strain/electric-field splitting, typically ~0–10 MHz
- γe/2π = 28.02 GHz/T (electron gyromagnetic ratio; sign convention: note it
  and keep it consistent — we track |γ| and orientation explicitly)
- Four NV orientations along the ⟨111⟩ diamond axes; lab-frame B projects
  differently onto each. `hamiltonian.py` owns the frame transforms.

### Implementation conventions (Phase 0)

- Transition frequencies are computed as eigenvalue differences from the
  eigenstate with maximum |⟨ms=0|ψ⟩|²; valid while γB ≪ D so the ms=0
  character is well defined.
- NV frame per orientation: z along the ⟨111⟩ axis, x = ẑ_lab × z (normalized),
  y = z × x. The in-plane x choice is arbitrary and physically irrelevant for
  |B⊥|-dependent quantities; E-term anisotropy would pin it (not needed yet).

## ¹⁴N hyperfine (Phase 1+)

Axial A∥ ≈ −2.16 MHz coupling to the I = 1 nuclear spin → ODMR triplet.
Model as static splitting first (three sub-ensembles), full tensor-product
Hilbert space (9-dim) only if needed.

## CW-ODMR model (Phase 0)

Rate-equation level: fluorescence S(f) = 1 − Σᵢ C·L(f − fᵢ), one Lorentzian
L (FWHM Γ) per ground-state eigen-transition fᵢ from the exact Hamiltonian.
Equal contrast C per transition (unpolarized MW, both ΔmS = ±1 allowed).
Linewidth and contrast are phenomenological inputs; power broadening and
hyperfine structure enter in Phase 1.

## Optical / readout model

We do NOT simulate the full optical cycle. Readout is phenomenological:

- Spin-dependent fluorescence: rate R for ms=0, R·(1−C) for ms=±1, contrast C.
- A measurement of duration t_read with collection efficiency folded into R
  yields Poisson-distributed counts. Shot noise comes from Poisson sampling,
  never additive Gaussian.
- Optical pumping = state reset to ms=0 with fidelity f_pump.

Implemented (Phase 1, `readout.py`): mean counts per shot
λ(p₀) = R·t_read·[f_pump·(1 − C(1−p₀)) + (1−f_pump)·(1 − 2C/3)],
where R is the *detected* ms=0 photon rate (collection efficiency folded in),
C the fluorescence contrast, p₀ = P(ms=0) at readout. The unpumped fraction
is taken as unpolarized (1/3 per ms level) and coherence-free — a constant
background. Recorded counts over n shots are a single Poisson(n·λ) draw.

## Pulsed two-level reduction (Phase 0)

Pulsed protocols drive one transition (|ms=0⟩ ↔ |ms=−1⟩) with the other far
detuned (2γBz ≫ Ω), so the dynamics reduce to a two-level system in the frame
rotating at the MW frequency, under the RWA:

H/h = −δ·|1⟩⟨1| + (Ω_R/2)·(|0⟩⟨1| + |1⟩⟨0|),  δ = f_MW − f_transition.

- Pulses are ideal (instantaneous rotations); finite-pulse-width effects are
  out of scope until something needs them (see approximations ledger).
- Lindblad: T1 as symmetric jump ops √(1/2T1)·σ∓ (infinite-temperature bath —
  fine since ħω ≪ kT at room temperature); pure dephasing √(γφ/2)·σz with
  γφ = 1/T2 − 1/(2T1). Requires T2 ≤ 2T1.
- On resonance P₀(t) = cos²(πΩ_R t); detuned, generalized Rabi
  Ω' = √(Ω_R² + δ²) with contrast Ω_R²/Ω'².

## Decoherence

Lindblad operators on the spin-1 system:
- T1: relaxation between ms levels (rate 1/T1).
- T2* (Ramsey) via pure dephasing rate; inhomogeneous broadening alternatively
  modeled by sampling static detunings across realizations (more physical for
  ensemble/slow bath — prefer this for Ramsey decay shape).
  Implemented as `ramsey(mode=...)`: `"lindblad"` gives envelope exp(−τ/T2*);
  `"static"` averages Gaussian static detunings (std σ) giving envelope
  exp(−(τ/T2*)²) with T2* = √2/(2πσ). Pulse convention: π/2 about +x, second
  π/2 about −x → bright fringe at τ=0, P₀ = ½(1 + cos 2πδτ · envelope).
- T2 (echo): residual dephasing under echo; decay envelope exp[−(t/T2)^n],
  n from the assumed bath (document choice per experiment).
  Phase 0 implements n = 1 (Markovian Lindblad dephasing at rate 1/T2):
  P₀ = ½(1 + exp(−2τ/T2)) for the π/2ₓ–τ–πₓ–τ–π/2ₓ sequence. Static
  detunings of any σ are exactly refocused (unit-tested). Stretched
  exponentials (n ≠ 1) arrive with Phase 1's noise modeling.

Typical values to default to (bulk, ppb-grade diamond): T1 ~ ms,
T2* ~ 1–5 µs, T2 ~ 100–500 µs. Cite Barry et al., Rev. Mod. Phys. 92, 015004
(2020) for parameter ranges.

## Sensitivity accounting

DC sensitivity η ≈ (1/(γ·C·√(R·t_read))) · (√(t_total)/ (dS/dB slope term)) —
implement the standard shot-noise-limited formulas from Barry et al. §III and
test against their worked examples. All sensitivity numbers must state the
assumed photon budget.

## Approximations ledger

| Approximation | Where | Revisit when |
|---|---|---|
| Phenomenological readout (no 7-level model) | readout | metastable-singlet dynamics matter (high MW/laser power regimes) |
| Rotating wave approximation for MW drive | pulsed sims | Rabi frequencies approach detunings (~100 MHz) |
| Static hyperfine (no nuclear dynamics) | ODMR lineshape | simulating nuclear-spin-assisted protocols |
| Single NV orientation unless stated | most sims | ensemble/vector magnetometry |
| Ideal (delta-function) MW pulses | pulsed sims | pulse durations comparable to 1/detunings or to T2* |
| Two-level reduction of the driven transition | pulsed sims | Ω comparable to the 2γBz splitting, or degenerate transitions at B≈0 |
