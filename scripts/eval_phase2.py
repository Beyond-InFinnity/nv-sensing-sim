#!/usr/bin/env python3
"""Run estimators over every record of Phase-2 eval datasets (paired by
construction: all estimators see the same .npz records).

Usage: eval_phase2.py DATASET_DIR [--out-dir DIR] [--estimators lsq,bayes,nn]"""
import argparse
import json
from pathlib import Path

from nvsim.estimators.bayes import fit_bayes
from nvsim.estimators.crb import crb_sigma_delta
from nvsim.estimators.lsq import fit_lsq
from nvsim.experiment import load_dataset
from nvsim.provenance import git_sha

REPO = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset_dir", type=Path)
    ap.add_argument("--out-dir", type=Path,
                    default=REPO / "experiments/phase2/artifacts")
    ap.add_argument("--estimators", default="lsq,bayes")
    args = ap.parse_args()
    names = args.estimators.split(",")
    if "nn" in names:
        from nvsim.estimators.nn import NNEstimator
        nn_est = NNEstimator.load(REPO / "experiments/phase2/nn_ckpt.pt")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for path in sorted(args.dataset_dir.glob("*.npz")):
        ds = load_dataset(path)
        cfg = ds["config"]
        taus, ro, n = ds["sweep_values"], cfg["readout"], cfg["n_shots"]
        truth = cfg["truth"]
        records = []
        for counts in ds["counts"]:
            rec = {}
            if "lsq" in names:
                rec["lsq"] = fit_lsq(counts, taus, ro, n)
            if "bayes" in names:
                rec["bayes"] = fit_bayes(counts, taus, ro, n)
            if "nn" in names:
                rec["nn"] = nn_est.infer(counts, taus, ro, n)
            records.append(rec)
        art = {"dataset": path.name, "config": cfg, "git_sha": git_sha(),
               "truth": truth,
               "crb_sigma_delta_hz": crb_sigma_delta(
                   taus, (truth["detuning_hz"], truth["t2star_s"]), ro, n),
               "records": records}
        out = args.out_dir / f"estimates_{path.stem}.json"
        out.write_text(json.dumps(art))
        print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
