#!/usr/bin/env python3
"""Run adaptive-vs-fixed schedule comparison; one artifact per schedule,
paired true-delta draws. Usage: run_phase3.py CONFIG [--out-dir DIR]"""
import argparse
import json
import zlib
from pathlib import Path

import numpy as np

from nvsim.estimators.adaptive import simulate_run
from nvsim.provenance import git_sha

REPO = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path)
    ap.add_argument("--out-dir", type=Path,
                    default=REPO / "experiments/phase3/artifacts")
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ss = np.random.SeedSequence(cfg["seed"])
    rng_truth = np.random.default_rng(ss.spawn(1)[0])
    lo, hi = cfg["delta_range_hz"]
    true_deltas = rng_truth.uniform(lo, hi, cfg["n_replicates"]).tolist()
    for kind in cfg["schedules"]:
        kind_tag = zlib.crc32(kind.encode())  # stable across runs (hash() is not)
        runs = []
        for i, d in enumerate(true_deltas):
            rng = np.random.default_rng(
                np.random.SeedSequence([cfg["seed"], kind_tag, i]))
            runs.append({"true_delta_hz": d, **simulate_run(d, kind, cfg, rng)})
            print(f"{kind} {i + 1}/{len(true_deltas)}", flush=True)
        art = {"config": cfg, "seed": cfg["seed"], "git_sha": git_sha(),
               "true_deltas_hz": true_deltas, "runs": runs}
        out = args.out_dir / f"{kind}.json"
        out.write_text(json.dumps(art))
        print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
