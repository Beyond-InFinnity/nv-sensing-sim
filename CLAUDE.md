# CLAUDE.md — nv-sensing-sim

## Project purpose

Simulate NV-center magnetometry end-to-end (spin physics → realistic noisy
measurement records → parameter estimation) and quantify how much sensitivity
improved estimators (Bayesian, NN) recover over least-squares fitting,
benchmarked against the Cramér–Rao bound.

Connor has hands-on NV-diamond magnetometry lab experience (Walsworth group)
and strong DSP background — this project deliberately connects that hardware
experience to computational methods. Realism of the noise model is the point;
a noise model that flatters the estimator is worthless.

## Current status

Phase 0 (physics core) — not started. See docs/ROADMAP.md. Update this line
when a phase completes.

## Hardware context

Physics simulation is trivially cheap (3–9 dim Hilbert spaces) — runs
anywhere. NN estimator training (Phase 2) targets the RTX 3070 / RTX 5050
(8 GB) on the i9-9900 / 64 GB machine.

## Conventions

- Python ≥3.11, package in `src/nvsim/`, editable install (`pip install -e ".[dev]"`).
- Core deps: qutip, numpy, scipy, matplotlib; torch enters at Phase 2. Keep it lean.
- **Units:** SI + explicit. Frequencies in Hz (not rad/s) at API boundaries;
  angular frequency only internally and named `omega_*`. Magnetic field in
  tesla. Document every Hamiltonian term's convention in docs/PHYSICS.md.
- Physics invariants get unit tests: zero-field splitting at 2.87 GHz, Zeeman
  splitting slope 28.02 GHz/T (γ_e/2π = 28.02 GHz/T), trace preservation in
  Lindblad evolution, Ramsey fringe frequency = detuning.
- Every experiment is config-driven with serialized config + seed + git SHA in
  the artifact. Estimator comparisons always run on identical noise
  realizations (paired).
- Plots follow the dataviz skill if available; figures in `docs/figures/`.

## Key technical decisions (append as made)

- Ground-state spin-1 model first (D, E, Zeeman); add ¹⁴N hyperfine when
  Phase 1 requires realistic ODMR lineshapes. Excited-state / optical dynamics
  modeled phenomenologically (contrast + photon budget), not as a full 7-level
  master equation, until something needs it.
- (pending) Bayesian inference machinery: grid vs SMC — decide in Phase 2.

## Commands

```bash
source .venv/bin/activate
pytest
```
