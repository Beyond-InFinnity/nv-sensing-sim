# Roadmap

Each phase has a checkable exit criterion; results backing it live in
`experiments/` with configs, figures in `docs/figures/`.

## Phase 0 — Physics core

Goal: a trustworthy NV ground-state spin model.

- [x] Spin-1 Hamiltonian: zero-field splitting D = 2.87 GHz, strain E, Zeeman
      term for arbitrary B-field orientation; the four NV crystallographic
      orientations.
- [x] CW-ODMR spectrum simulation (rate-equation or steady-state Lindblad
      level): two dips at zero strain/field, splitting linear in axial B.
- [x] Pulsed protocols via time evolution: Rabi oscillations, Ramsey fringes,
      Hahn echo, with T1/T2* / T2 as Lindblad dephasing/relaxation rates.
- [x] Unit tests for all textbook behaviors (splittings, fringe frequencies,
      echo decay envelopes).

**Exit:** simulated ODMR/Rabi/Ramsey/echo figures that match textbook/review
results (e.g. Barry et al., RMP 2020) qualitatively and in every checkable
number.

## Phase 1 — Realistic measurement records

Goal: a synthetic data generator whose output looks like the lab, not like a
clean curve plus white noise.

- [ ] Photon shot noise from an explicit photon budget (counts/measurement,
      contrast ~1–30%), not additive Gaussian.
- [ ] ¹⁴N hyperfine triplet in ODMR lineshapes; power broadening.
- [ ] Slow drifts: laser/MW power drift, thermal drift of D (−74 kHz/K),
      1/f-style B-field background.
- [ ] Config-driven "virtual experiment" API: protocol + true parameters +
      noise config → measurement record dataset.
- [ ] Validation figure set: side-by-side with published experimental traces.

**Exit:** generator produces datasets across an SNR range with documented,
physically-motivated noise; every knob traceable to a physical mechanism.

## Phase 2 — Estimators (the headline)

Goal: quantify estimator quality for B-field (and D/T2*) extraction.

- [ ] Baseline: least-squares curve fitting (the universal lab default).
- [ ] Cramér–Rao bound computation for each protocol/noise config — the
      yardstick everything is measured against.
- [ ] Bayesian estimator (grid or SMC posterior) on the same records.
- [ ] NN estimator: small 1D-CNN/transformer mapping raw records → parameter
      + uncertainty; trained on the Phase 1 generator, evaluated on held-out
      noise configs (test for noise-model overfitting explicitly).
- [ ] Headline figure: estimator RMSE vs SNR vs CRB, paired realizations,
      error bars; sensitivity (nT/√Hz) implications stated.

**Exit:** a defensible answer to "how much sensitivity does better estimation
buy, and how close to the CRB can each method get, at which SNR regimes?"

## Phase 3 — Adaptive sensing (exploratory)

- [ ] Bayesian adaptive Ramsey: choose next interrogation time from current
      posterior (information-gain criterion); compare total-time-to-target-
      precision vs fixed schedules.
- [ ] Optional: RL-style or amortized policy if the Bayesian version shows
      clear wins.

**Exit:** writeup in `docs/` (blog-post grade) with reproducible figures.

## Non-goals (for now)

- Full 7-level optical master equation, NV ensembles with dipolar
  interactions, nanoscale NMR/AC sensing protocols (XY8 etc.) — revisit after
  Phase 2.
- Fitting real lab data (none available); realism enters through the noise
  model and published-trace comparisons.
