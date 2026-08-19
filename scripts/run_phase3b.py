#!/usr/bin/env python3
"""Phase 3b evaluation: named methods (schedule kind + optional policy
checkpoint) on paired true-delta draws through the Phase 3 simulator.
Usage: run_phase3b.py CONFIG [--out-dir DIR] [--latency]"""
import argparse
import json
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from nvsim.estimators.adaptive import simulate_run
from nvsim.provenance import git_sha

REPO = Path(__file__).resolve().parent.parent


def _one_run(job):
    """Worker: one (method, replicate) cell. Seeding is per-cell, so results
    are independent of worker count and scheduling order."""
    run_cfg, kind, name, seed, i, d = job
    rng = np.random.default_rng(
        np.random.SeedSequence([seed, zlib.crc32(name.encode()), i]))
    return {"true_delta_hz": d, **simulate_run(d, kind, run_cfg, rng)}


def latency_bench(cfg, n_reps=200):
    """Median single-decision wall time: A-optimal lookahead vs policy."""
    from nvsim.estimators.adaptive import DeltaPosterior, choose_tau
    from nvsim.estimators.model import expected_counts
    from nvsim.estimators.policy import compute_features
    from nvsim.estimators.policy_net import AmortizedPolicy

    post = DeltaPosterior(tuple(cfg["delta_range_hz"]),
                          t2star_s=cfg["t2star_s"], readout_cfg=cfg["readout"])
    rng = np.random.default_rng(0)
    for tau in (0.1e-6, 0.4e-6, 0.9e-6):   # a representative mid-run state
        lam = expected_counts([tau], 2.1e6, cfg["t2star_s"], cfg["readout"],
                              cfg["n_shots_per_batch"])[0]
        post.update(rng.poisson(lam), tau, cfg["n_shots_per_batch"])
    full_grid = np.geomspace(cfg["tau_min_s"], cfg["tau_max_s"], 60)
    out = {}
    times = []
    for _ in range(max(5, n_reps // 40)):   # lookahead is slow; fewer reps
        t0 = time.perf_counter()
        choose_tau(post, full_grid, cfg["n_shots_per_batch"])
        times.append(time.perf_counter() - t0)
    out["aoptimal_s"] = float(np.median(times))
    for name, ckpt in cfg.get("latency_policies", {}).items():
        pol = AmortizedPolicy.load(ckpt)
        f = compute_features(post.p, post.grid, cfg, 0.01)
        times = []
        for _ in range(n_reps):
            t0 = time.perf_counter()
            pol.act(f)
            times.append(time.perf_counter() - t0)
        out[f"{name}_s"] = float(np.median(times))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path)
    ap.add_argument("--out-dir", type=Path,
                    default=REPO / "experiments/phase3b/artifacts")
    ap.add_argument("--latency", action="store_true")
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text())
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.latency:
        res = {"config": cfg, "git_sha": git_sha(),
               **latency_bench(cfg)}
        out = args.out_dir / "latency.results.json"
        out.write_text(json.dumps(res))
        print(json.dumps({k: v for k, v in res.items()
                          if k.endswith("_s")}, indent=1))
        print(f"wrote {out}", flush=True)
        return

    ss = np.random.SeedSequence(cfg["seed"])
    rng_truth = np.random.default_rng(ss.spawn(1)[0])
    lo, hi = cfg["delta_range_hz"]
    true_deltas = rng_truth.uniform(lo, hi, cfg["n_replicates"]).tolist()
    n_workers = cfg.get("n_workers", 1)
    pool = ProcessPoolExecutor(max_workers=n_workers) if n_workers > 1 \
        else None
    try:
        for m in cfg["methods"]:
            run_cfg = {k: v for k, v in cfg.items()
                       if k not in ("methods", "n_workers")}
            if "policy_ckpt" in m:
                run_cfg["policy_ckpt"] = m["policy_ckpt"]
            jobs = [(run_cfg, m["kind"], m["name"], cfg["seed"], i, d)
                    for i, d in enumerate(true_deltas)]
            runs = []
            it = pool.map(_one_run, jobs) if pool else map(_one_run, jobs)
            for i, r in enumerate(it):
                runs.append(r)
                print(f"{m['name']} {i + 1}/{len(true_deltas)}", flush=True)
            art = {"config": cfg, "method": m, "seed": cfg["seed"],
                   "git_sha": git_sha(), "true_deltas_hz": true_deltas,
                   "runs": runs}
            out = args.out_dir / f"{m['name']}.json"
            out.write_text(json.dumps(art))
            print(f"wrote {out}", flush=True)
    finally:
        if pool:
            pool.shutdown()


if __name__ == "__main__":
    main()
