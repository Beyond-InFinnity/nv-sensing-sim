#!/usr/bin/env python3
"""Run a Phase 1 virtual experiment (single or ladder config) to .npz datasets.

Usage: run_phase1.py CONFIG [--out-dir DIR]"""
import argparse
import copy
import json
from pathlib import Path

from nvsim.experiment import run_experiment, save_dataset

REPO = Path(__file__).resolve().parent.parent


def set_dotted(cfg, dotted, value):
    node = cfg
    *parents, leaf = dotted.split(".")
    for p in parents:
        node = node[p]
    node[leaf] = value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path)
    ap.add_argument("--out-dir", type=Path,
                    default=REPO / "experiments/phase1/artifacts")
    args = ap.parse_args()
    spec = json.loads(args.config.read_text())
    base = spec["base"]
    if spec["kind"] == "single":
        args.out_dir.mkdir(parents=True, exist_ok=True)
        ds = run_experiment(base)
        out = args.out_dir / f"{spec['name']}.npz"
        save_dataset(ds, out)
        print(f"wrote {out}")
        return
    sub = args.out_dir / spec["name"]
    sub.mkdir(parents=True, exist_ok=True)
    for v in spec["values"]:
        cfg = copy.deepcopy(base)
        set_dotted(cfg, spec["vary"], v)
        ds = run_experiment(cfg)
        tag = spec["vary"].split(".")[-1]
        out = sub / f"{tag}_{v}.npz"
        save_dataset(ds, out)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
