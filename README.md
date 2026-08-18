# nv-sensing-sim

Simulation of NV-diamond magnetometry with realistic noise, plus ML/Bayesian
signal extraction benchmarked against the Cramér–Rao bound.

**Status: Phases 0–3 complete.** Phase 3 headline: a Bayesian adaptive
Ramsey protocol (sequential grid posterior + A-optimal interrogation-time
selection) reaches 5 kHz posterior uncertainty on the detuning in a median
48 ms of wall-clock sensing time vs 157 ms for the best convergent fixed
schedule — a 3.25× speedup — while the prior-Fisher "best single τ" never
converges at all over a 3.4 MHz prior (fringe aliasing). The adaptive rule
rediscovers the phase-estimation ladder on its own: it starts at τ ≈ 0.12 µs
(unambiguous) and ramps to the T2*-limited optimum ≈1.43 µs. See
`docs/phase3-adaptive-ramsey.md`. Phase 2 headline: on identical Poisson
Ramsey records, weighted least squares and a Bayesian grid posterior both
reach 0.94–1.09× the Cramér–Rao bound at every photon budget tested
(20–20 000 shots/point), i.e. classical fitting is already near-optimal when
the model is well-specified; a small 1D-CNN reaches only 0.2–0.4× CRB but
degrades far more gracefully under model misspecification (1/f field drift:
Bayes loses 2.9×, NN 1.4×). See `docs/figures/phase2_rmse_vs_crb.png`.
Earlier phases: Phase 0: spin-1 ground-state Hamiltonian,
CW-ODMR, and pulsed Rabi/Ramsey/Hahn-echo simulations, every textbook invariant
unit-tested against Barry et al., Rev. Mod. Phys. 92, 015004 (2020). Phase 1:
config-driven virtual-experiment generator — Poisson counts from an explicit
photon budget, ¹⁴N hyperfine + power broadening (Dréau et al., PRB 84, 195204
(2011)), and 1/f / OU / thermal drifts on a wall-clock axis, with paired RNG
streams for estimator comparisons. 68 tests across all phases; figures in
`docs/figures/`. Datasets (.npz) are gitignored by
design; each embeds config + seed + git SHA and regenerates deterministically
via `scripts/run_phase1.py <config>`.

## What this is

NV centers in diamond are a leading room-temperature quantum sensing platform.
This project simulates the NV ground-state spin (QuTiP: Hamiltonian + Lindblad
dynamics), generates realistic noisy measurement records (ODMR spectra, Rabi,
Ramsey, Hahn echo), and then treats **parameter estimation from noisy records**
as the real problem: classical least-squares fitting vs Bayesian inference vs
learned estimators, judged against the Cramér–Rao bound.

The through-line: how much magnetometer sensitivity is left on the table by
naive fitting, and how much of it can better estimators recover?

## Roadmap (summary)

| Phase | Goal | Exit criterion |
|-------|------|----------------|
| 0 | Physics core | NV spin model reproduces textbook ODMR/Rabi/Ramsey/echo behavior |
| 1 | Realistic noise | Synthetic data generator with shot noise, drift, strain, hyperfine structure |
| 2 | Estimators | Bayesian + NN estimators vs least-squares vs CRB across SNR |
| 3 | Adaptive sensing | Adaptive/online protocols (Bayesian experiment design) |

Full detail in [docs/ROADMAP.md](docs/ROADMAP.md); physics model documented in
[docs/PHYSICS.md](docs/PHYSICS.md).

## Layout

```
src/nvsim/       Python package (hamiltonian, dynamics, noise, estimators)
tests/           pytest suite (physics invariants are tested)
notebooks/       exploratory analysis
experiments/     run configs + result artifacts
docs/            roadmap, physics notes, results
data/            generated datasets (gitignored)
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

Hilbert spaces here are tiny (spin-1, plus a nuclear spin at most) — this
project is CPU-friendly; GPUs only enter for NN estimator training (Phase 2).
