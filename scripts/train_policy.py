#!/usr/bin/env python3
"""Train the amortized adaptive-Ramsey policy.

Stage bc: roll A-optimal teacher episodes (the expensive step), fit the
policy net by cross-entropy, write checkpoint + results JSON.
Stage rl: PPO-style fine-tune from an init checkpoint (config keys:
init_ckpt, out_tag, rl{...}, optional drift{...}).
Usage: train_policy.py CONFIG --stage bc|rl
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from nvsim.estimators.policy import VecRamseyEnv, tau_candidates
from nvsim.estimators.policy_net import (AmortizedPolicy, collect_bc_dataset,
                                         finetune_rl, train_bc)
from nvsim.provenance import git_sha

REPO = Path(__file__).resolve().parent.parent


def run_rl(cfg, out_dir):
    rng = np.random.default_rng(np.random.SeedSequence([cfg["seed"], 1]))
    pol = AmortizedPolicy.load(cfg["init_ckpt"])
    t0 = time.time()
    hist = finetune_rl(pol, cfg["env"], cfg["rl"], rng,
                       drift=cfg.get("drift"),
                       progress=lambda it, ret: print(
                           f"rl iter {it + 1}/{cfg['rl']['n_iters']} "
                           f"return {ret:.2f}", flush=True))
    tag = cfg.get("out_tag", "rl")
    pol.save(out_dir / f"{tag}_policy.pt")
    results = {"config": cfg, "git_sha": git_sha(),
               "hours": (time.time() - t0) / 3600,
               "mean_return": hist["mean_return"],
               "entropy": hist["entropy"]}
    (out_dir / f"{tag}_train.results.json").write_text(json.dumps(results))
    print(f"wrote {out_dir / f'{tag}_train.results.json'} "
          f"(return {hist['mean_return'][0]:.2f} -> "
          f"{hist['mean_return'][-1]:.2f})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path)
    ap.add_argument("--stage", choices=["bc", "rl"], default="bc")
    ap.add_argument("--out-dir", type=Path,
                    default=REPO / "experiments/phase3b")
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.stage == "rl":
        run_rl(cfg, args.out_dir)
        return
    env_cfg = cfg["env"]
    rng = np.random.default_rng(np.random.SeedSequence([cfg["seed"], 0]))

    t0 = time.time()
    X, y = collect_bc_dataset(
        env_cfg, cfg["n_teacher_episodes"], rng,
        n_workers=cfg.get("n_workers", 1),
        progress=lambda step, ndone: print(
            f"teacher step {step}, episodes finished {ndone}"
            f"/{cfg['n_teacher_episodes']}", flush=True))
    teach_s = time.time() - t0
    np.savez(args.out_dir / "bc_dataset.npz", X=X, y=y)  # gitignored
    print(f"teacher dataset: {len(X)} decisions in {teach_s / 3600:.2f} h",
          flush=True)

    env = VecRamseyEnv(env_cfg, 1, np.random.default_rng(0))
    pol, hist = train_bc(X, y, n_features=env.n_features,
                         n_actions=len(tau_candidates(env_cfg)),
                         epochs=cfg["epochs"], lr=cfg["lr"], seed=cfg["seed"])
    ckpt = args.out_dir / "bc_policy.pt"
    pol.save(ckpt)
    results = {"config": cfg, "git_sha": git_sha(),
               "n_decisions": int(len(X)),
               "teacher_hours": teach_s / 3600,
               "heldout_acc": hist["heldout_acc"],
               "final_train_loss": hist["train_loss"][-1],
               "action_histogram": np.bincount(
                   y, minlength=len(tau_candidates(env_cfg))).tolist()}
    (args.out_dir / "bc_train.results.json").write_text(json.dumps(results))
    print(f"wrote {args.out_dir / 'bc_train.results.json'} "
          f"(heldout_acc {hist['heldout_acc']:.3f})", flush=True)


if __name__ == "__main__":
    main()
