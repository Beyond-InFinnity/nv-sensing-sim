# Phase 3b — Amortized/RL Adaptive-Ramsey Policy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the A-optimal lookahead (~0.1–0.2 s CPU per decision) with a
small neural policy (posterior summary → τ) that (A) matches the Bayesian
adaptive schedule's time-to-precision at real-time inference latency, and
(B) tests whether a nonmyopically-trained policy beats the myopic one-batch
lookahead where myopia can actually hurt: detuning drift during the run.
Two separately-gated experiments; (B) is where new physics insight can
appear, (A) is the engineering payoff either way.

**Architecture:** `nvsim.estimators.policy`: a vectorized batch of Phase 3
posterior environments (E parallel episodes as (E, G) arrays — the grid
update is already closed-form and broadcasts), a feature extractor mapping
each posterior to a fixed vector, a small MLP policy over the same 24-point
log-τ candidate grid the coarsened A-optimal search uses (fair comparison —
identical action set), trained in two stages: behavior cloning on logged
A-optimal decisions, then policy-gradient fine-tuning on episode return.
Drift enters as an OU process on δ_true plus a matching diffusion step on
the posterior (symmetric knowledge: every method gets the same
drift-aware posterior). Exit artifact: `docs/phase3b-amortized-policy.md`.

**Tech Stack:** numpy + torch (already a Phase 2 dependency). NO RL
library (stable-baselines etc.) — the training loop is ~150 lines of
REINFORCE-with-baseline / clipped-surrogate and CLAUDE.md says keep deps
lean. CPU-first: the environment is (E, 600) numpy broadcasting and the
policy is a ~20k-parameter MLP; claim the RTX 3070 via MACHINES.md only if
a measured training run projects > 1 h on CPU (decide at Task 3 Step 1,
not before).

## Global Constraints

- Hz / tesla / s at API boundaries; posterior updates stay exact-Poisson.
- The policy's action set is exactly the 24-candidate log grid from
  `choose_tau`'s coarsened search — any win/loss is about the *rule*, not
  the menu.
- All evaluation runs paired on identical true-δ draws (and identical
  drift paths in 3b-B) across methods, same artifact/provenance scheme as
  Phase 3 (config + seed + git SHA embedded, JSON artifacts committed).
- Wall-clock accounting identical to Phase 3: n_b · (t_init + τ + t_read +
  t_dead) per batch; decision compute reported separately as a measured
  latency table (the point of the phase), never mixed into the sensing axis.
- Jobs > 10 min: LEDGER.md entry, detached nohup launch, watcher covering
  death (RULES §3/§7 — hub restarts kill harness-tracked children).
- Honest-negative rule: if the RL fine-tune does not beat BC, or the
  policy does not beat myopic A-optimal under drift, that is the result —
  report it with numbers, do not iterate past the one designed retrain
  (Phase 2 NN lesson).

**Gates:**
- **3b-A exit:** policy median time-to-5-kHz within **1.2×** of A-optimal
  (paired, 60 replicates) at **< 1 ms** single-decision CPU latency
  (vs ~0.1–0.2 s for the lookahead). Miss → report the measured ratio,
  skip 3b-B RL claims about latency-free adaptivity accordingly.
- **3b-B exit:** measured verdict on nonmyopia-under-drift, either sign.

---

### Task 1: Vectorized posterior environment (+ drift diffusion)

**Files:**
- Create: `src/nvsim/estimators/policy.py`
- Modify: `docs/PHYSICS.md` (drift-aware sequential posterior section)
- Test: `tests/test_policy_env.py`

**Interfaces:**
- Produces: `policy.VecRamseyEnv(cfg, n_envs, rng, drift=None)`:
  - `.reset() -> features (E, F)` — draws true δ per env from the prior,
    resets posteriors to flat (E, G) arrays;
  - `.step(actions (E,) int) -> (features, reward, done, info)` — one batch
    per env at τ = tau_candidates[action]: Poisson counts at each env's
    true δ, exact-Poisson posterior row update, wall-clock advance, done
    when the env's elapsed time exceeds `cfg["time_budget_s"]`;
  - reward per step = Δ(−log σ²) of that env's posterior (dense shaping
    whose telescoped sum is total log-variance reduction — the episode
    objective is the final precision, arrived at by any route);
  - `.tau_candidates_s` — the 24-point log grid; `info` carries per-env
    σ, |err|, elapsed time for evaluation reuse.
  - `drift`: `None` (3b-A) or `{"kind": "ou", "sigma_hz": ..., "tau_s": ...}`
    (3b-B): per step, true δ follows exact-discretization OU (reuse
    `nvsim.drift`), and every posterior row is convolved with the matching
    Gaussian transition kernel (grid convolution, `scipy.ndimage` or FFT)
    **before** the likelihood update. Both the env truth and the posterior
    prediction step use the same OU parameters — symmetric knowledge, as
    T2* was in Phase 3.
- Features (F ≈ 38): log σ (normalized), posterior mean (mapped to
  [0, 1] over the prior range), posterior entropy, fraction of time budget
  elapsed, top-two-mode mass ratio and separation (the aliasing signal —
  Phase 3 showed multimodality is the thing the schedule must manage), and
  a 32-bin max-pooled log-posterior. Document each in the module docstring.

- [ ] **Step 1: Write failing tests** — (a) vectorized env with E = 3 gives
  bitwise-identical σ trajectories to three sequential Phase 3
  `DeltaPosterior` runs fed the same counts and τ (the env is a
  re-implementation; prove equivalence, don't assume it); (b) reward
  telescopes: Σ rewards = log σ₀² − log σ_final² per env; (c) drifted env:
  with `drift` on and zero measurements, posterior σ grows monotonically
  (diffusion only); with drift on and measurements, tracking error stays
  bounded over 3× the OU correlation time.
- [ ] **Step 2:** run → FAIL. **Step 3:** implement. **Step 4:** run → PASS.
- [ ] **Step 5:** Commit: `feat: vectorized posterior env with OU drift for policy training`

---

### Task 2: Features → policy net + behavior cloning

**Files:**
- Modify: `src/nvsim/estimators/policy.py`
- Create: `scripts/train_policy.py` (BC stage), `experiments/phase3b/bc.json`
- Test: `tests/test_policy.py`

**Interfaces:**
- Produces: `policy.PolicyNet(n_features, n_actions=24)` — MLP
  F → 128 → 128 → 24 logits (~20k params); `policy.AmortizedPolicy` with
  `.act(features) -> actions` (argmax at eval), `.save/.load`;
  `policy.collect_bc_dataset(cfg, n_episodes, rng) -> (X, y)` — roll
  episodes where the *teacher* `choose_tau` (coarsened A-optimal) picks
  actions; log (features, teacher action index) pairs.
- BC config: ~400 teacher episodes (≈ 80k decisions at ~208 batches/run;
  teacher cost ~0.15 s/decision → ~3.5 h **teacher-data generation is the
  expensive step** — LEDGER + detached launch), Adam 1e-3, cross-entropy,
  20 epochs, 10% held-out decision accuracy reported.

- [ ] **Step 1: Write failing tests** — shapes/save-load round-trip; BC on a
  tiny synthetic teacher (fixed rule: short τ when σ large, long when
  small) reaches > 90% held-out accuracy in < 30 s (training machinery
  works; the real teacher is production-scale).
- [ ] **Step 2:** run → FAIL. **Step 3:** implement. **Step 4:** run → PASS.
- [ ] **Step 5:** Teacher-data generation as a production job (LEDGER,
  nohup, watcher); then BC training (minutes, CPU). Artifact:
  `experiments/phase3b/bc_policy.pt` (gitignored) +
  `bc_train.results.json` (committed: accuracy, config, SHA).
- [ ] **Step 6:** Commit: `feat: policy net + behavior cloning of A-optimal teacher`

---

### Task 3: RL fine-tuning

**Files:**
- Modify: `src/nvsim/estimators/policy.py`, `scripts/train_policy.py`
  (`--stage rl`), `experiments/phase3b/rl.json`
- Test: `tests/test_policy.py` (append)

**Interfaces:**
- Produces: `policy.finetune_rl(policy, env_cfg, rl_cfg, rng) -> history` —
  clipped-surrogate policy gradient (PPO-style, in-repo): E = 256 parallel
  envs, full-episode rollouts, per-step advantage = reward-to-go minus a
  learned value head (share trunk, add scalar head), entropy bonus 0.01
  annealed, clip 0.2, Adam 3e-4, ~300 iterations. Objective: expected
  total log-variance reduction within the time budget (identical to what
  the myopic rule locally chases — differences are pure nonmyopia).
- [ ] **Step 1:** Measure one iteration's wall time; project total. > 1 h
  CPU → claim RTX 3070 in MACHINES.md first (env stays numpy/CPU; only
  the net moves). Record the decision.
- [ ] **Step 2: Smoke test** — 20 iterations from a *random-init* policy on
  a shrunken env (G = 200, budget 0.02 s): mean episode return strictly
  improves; then assert BC-init return ≥ random-init return at iteration 0
  (BC actually transfers).
- [ ] **Step 3:** Production fine-tune, static env (3b-A), from BC init.
  LEDGER + detached + watcher. Artifact: `rl_policy.pt` (gitignored) +
  `rl_train.results.json` (committed: return curve, config, SHA).
- [ ] **Step 4:** Commit: `feat: PPO-style fine-tuning of amortized policy`

---

### Task 4: Static evaluation — the 3b-A gate

**Files:**
- Create: `scripts/run_phase3b.py`, `experiments/phase3b/policy_vs_aoptimal.json`
- Test: `tests/test_run_phase3b.py` (mini-config subprocess test, as Phase 3)

- [ ] **Step 1:** Extend `make_schedule` with `kind="policy"` (cfg carries a
  checkpoint path) so `simulate_run` — the *Phase 3* simulator, not the
  training env — evaluates it. Same seed-pairing scheme, seed 402; methods:
  `adaptive` (A-optimal teacher), `policy_bc`, `policy_rl`, `exp_ladder`;
  60 fresh paired replicates, budget 0.25 s.
- [ ] **Step 2:** Latency microbenchmark: median single-decision wall time,
  A-optimal vs policy (CPU, batch=1, torch.no_grad), 1000 reps →
  `latency.results.json`. The claim in the writeup is this table.
- [ ] **Step 3:** Run (likely > 10 min because the A-optimal arm re-runs:
  LEDGER + detached + watcher); commit artifacts. **Evaluate gate 3b-A**
  and record the number in the plan checkbox here: time-to-5-kHz ratio
  policy_rl / adaptive = ____ (pass < 1.2), latency = ____ ms.
- [ ] **Step 4:** Commit: `feat: phase 3b static evaluation — policy vs A-optimal`

---

### Task 5: Drift evaluation — the 3b-B question

**Files:**
- Create: `experiments/phase3b/drift.json`
- Modify: `scripts/run_phase3b.py` (drift plumbed through `simulate_run`
  via the Task-1 env pieces)

- [ ] **Step 1:** Drift setting: OU on δ with σ_drift = 100 kHz,
  τ_corr = 50 ms (drift comparable to the target precision within a
  correlation time — the regime where reacquisition vs refinement is a
  real tradeoff; state the choice in the writeup). Metric changes from
  time-to-target to **median tracking error ⟨|δ̂ − δ_true(t)|⟩ over the
  final 80% of a 0.5 s run** (steady-state tracking, not acquisition).
- [ ] **Step 2:** RL fine-tune a *drift* variant from the BC init on the
  drifted env (same rl.json budget; LEDGER etc.). Methods compared, all on
  the same drift-aware posterior: myopic A-optimal, policy_rl_static
  (transfer, no retrain), policy_rl_drift, exp_ladder. Paired δ paths.
- [ ] **Step 3:** Run, commit artifacts, record the verdict: drift-trained
  policy vs myopic A-optimal tracking-error ratio = ____. Either sign is
  a result; no tuning loops beyond the one designed training run.
- [ ] **Step 4:** Commit: `feat: phase 3b drift-tracking evaluation`

---

### Task 6: Figures

**Files:**
- Create: `scripts/plot_phase3b.py` → `docs/figures/phase3b_policy_vs_aoptimal.png`,
  `docs/figures/phase3b_drift_tracking.png`

- [ ] **Step 1:** Figure 1 — static: log-log σ vs wall-clock (Phase 3 style,
  same palette; policy curves vs teacher), inset or annotation: the
  latency table (0.1 s vs < 1 ms) and time-to-target ratios. Figure 2 —
  drift: tracking error vs time with the drift realization shown, plus
  median chosen-τ trajectories (does the drift policy hold τ shorter to
  keep reacquisition headroom? that is the nonmyopia signature to look
  for).
- [ ] **Step 2:** Run, Read PNGs, check calibration and surprises as in
  Phase 3 Task 5; bounded iterations. Commit: `feat: phase 3b figures`

---

### Task 7: Writeup + close out

- [ ] **Step 1:** `docs/phase3b-amortized-policy.md`, house style: why
  amortize (the 100× decision/sensing mismatch), BC-then-RL rationale,
  gate results with the measured numbers, the drift verdict with
  mechanism (what the policy does differently, shown from τ trajectories,
  not asserted), limitations (fixed batch size, known drift parameters,
  known T2*, sim-to-real gap unaddressed).
- [ ] **Step 2:** ROADMAP.md: RL-optional box → implemented, with headline
  numbers; add follow-on non-goals honestly (unknown T2*, real-hardware
  latency path).
- [ ] **Step 3:** Full fresh `pytest` (record count); CLAUDE.md status →
  Phase 3b complete; README status paragraph; release the 3070 in
  MACHINES.md if claimed; close all LEDGER entries.
- [ ] **Step 4:** Push origin + cmfinnerty; report to Connor: both gate
  numbers, latency table, drift verdict, writeup path; stop.
