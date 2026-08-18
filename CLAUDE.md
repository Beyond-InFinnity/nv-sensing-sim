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

Phase 2 (estimators) — complete (2026-08-18); Phases 0–1 also complete.
59 unit tests green; headline figure docs/figures/phase2_rmse_vs_crb.png.
RTX 3070 claimed and released same day (NN training, 2 runs). Phase 3
(adaptive sensing) not started — awaiting Connor's go-ahead.
See docs/ROADMAP.md. Update this line when a phase completes.

## Orchestration (multi-machine, multi-session)

Rules/protocols live in the PRIVATE repo `homelab-orchestration`
(`~/Documents/projects/homelab-orchestration` on the server;
`~/projects/homelab-orchestration` on the workstation). Read its RULES.md
before launching anything remote; append job boundaries to LEDGER.md; live
state via `scripts/status.sh`. GPU ownership: qec-neural-decoder owns the
RTX 5050 (torch cuda:1); **this project may use the RTX 3070 (torch cuda:0)
in Phase 2+** — update MACHINES.md when claiming it. Phases 0–1 here are
CPU-only by design and can run on either machine. One Claude session per
repo; cross-session handoff via this status section + the ledger.

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
- Bayesian inference machinery: grid posterior (400×60 over δ×T2*, exact
  Poisson likelihood, adaptive zoom when the posterior is narrower than ~5
  cells) — decided 2026-08-18. SMC revisited only if Phase 3 adaptive needs
  sequential updates.

## Commands

```bash
source .venv/bin/activate
pytest
```
