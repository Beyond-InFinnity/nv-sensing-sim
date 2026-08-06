# nv-sensing-sim

Simulation of NV-diamond magnetometry with realistic noise, plus ML/Bayesian
signal extraction that beats conventional curve fitting at low SNR.

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
