# Handoff: execute Phase 0 of nv-sensing-sim

You are a Claude Code session on **claude-server**, taking over this project
cold. Everything you need is on disk; this document is the entry point and
the contract. You own this repository exclusively — a parallel session in a
different context owns `qec-neural-decoder`; do not read from or write to
that repo except where this document cites it as reference material.

## 1. Read these first, in order

1. `CLAUDE.md` (this repo) — purpose, conventions, hardware, orchestration.
2. `docs/ROADMAP.md` — your assignment is **Phase 0 exactly as specified
   there**, exit criterion included.
3. `docs/PHYSICS.md` — the physics model contract. Every Hamiltonian term,
   unit convention, and approximation you implement must already be
   documented there, or you add it there first. No undocumented physics in
   code.
4. `~/Documents/projects/homelab-orchestration/RULES.md` and `MACHINES.md`
   (private repo, already cloned) — network-wide operational law. §1
   (shared workstation), §3 (watcher patterns), and §6 (parallel sessions)
   are the ones most likely to bite you.

## 2. Scope and boundaries

- **Do:** Phase 0 of the roadmap — spin-1 ground-state Hamiltonian, CW-ODMR,
  Rabi/Ramsey/Hahn-echo simulations, unit tests for every textbook invariant,
  validation figure set. Stop at the Phase 0 exit criterion and report;
  Phase 1 begins only when Connor says so.
- **Don't:** touch the workstation GPUs (the RTX 5050 is owned by
  qec-neural-decoder; the 3070 is not yours until Phase 2, per MACHINES.md).
  Phase 0 is CPU-only by design and runs entirely on claude-server. If you
  believe you need the workstation's CPU, consult RULES.md §5–6, check
  `homelab-orchestration/scripts/status.sh` and `LEDGER.md` first, and
  prefer not to.
- **Don't** modify `homelab-orchestration` except to append `LEDGER.md`
  entries for any job longer than ~10 minutes.

## 3. Working conventions (established across this homelab, follow exactly)

- **Environment:** project-local venv (`python3 -m venv .venv`,
  `pip install -e ".[dev]"`). Core deps are qutip/numpy/scipy/matplotlib —
  torch is an extra and Phase 0 must not need it. Nothing installed outside
  the repo.
- **Tests are physics invariants**, not formalities. Minimum set (from
  PHYSICS.md): zero-field splitting at 2.870 GHz; Zeeman slope 28.02 GHz/T
  on-axis; Lindblad evolution preserves trace; Ramsey fringe frequency
  equals detuning; Rabi frequency scales linearly with drive amplitude;
  Hahn echo removes static detuning. `pytest` green before every commit.
- **Experiments are config-driven and seeded.** Any generated dataset or
  scan is specified by a JSON config under `experiments/`; the artifact
  embeds config + seed + git SHA. No results that exist only in notebooks.
- **Figures** come from committed scripts reading committed artifacts, saved
  under `docs/figures/`. When writing any plotting code, use the dataviz
  skill; the established palette/ink constants are in
  `qec-neural-decoder/src/qecdec/plotting.py` (reference only — copy the
  constants, don't import across repos). Validation figures for Phase 0
  should be side-by-side comparable with Barry et al., Rev. Mod. Phys. 92,
  015004 (2020) — cite the specific figure you're matching.
- **Units:** Hz (not rad/s) at API boundaries; tesla for fields; SI
  throughout; conventions per PHYSICS.md. If you change or extend a
  convention, PHYSICS.md changes in the same commit.
- **Commits:** small, one concern each, imperative subject. Push to `origin`
  (public GitHub) at milestones. Update the "Current status" line in
  CLAUDE.md when Phase 0 completes — that line is the cross-session ledger.
- **Bounded iterations.** If something won't converge or match after one
  designed attempt + one targeted fix, document the discrepancy honestly
  (numbers, not adjectives) and surface it to Connor rather than thrashing.
  Negative results are kept, dated, and explained — see
  `qec-neural-decoder/docs/phase2-results.md` for the house style.

## 4. Definition of done (restated from ROADMAP.md — verify there)

Phase 0 exits when simulated ODMR, Rabi, Ramsey, and Hahn-echo outputs match
textbook behavior in **every checkable number** (splittings, slopes, fringe
frequencies, decay envelopes), with unit tests enforcing each, and a
validation figure set under `docs/figures/`. When you get there: update
CLAUDE.md status, commit, push, and give Connor a results summary with the
figures — then stop.

## 5. Style notes for reporting to Connor

Lead with what happened and what it means; numbers with units; no hype.
He has a neuroscience/instrumentation background and hands-on NV lab
experience (Walsworth group) — he knows what a real ODMR trace looks like,
so realism claims will be checked. He is self-taught in the quantum
formalism and appreciates explanations that build mechanism rather than
gesture at it; `qec-neural-decoder/docs/NOTATION.md` documents the level he
works at. When he asks "status update," answer with live-checked facts, not
recollection — run the check, then report.
