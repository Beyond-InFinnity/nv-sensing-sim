#!/usr/bin/env python3
"""Train the Ramsey NN estimator. Usage: train_nn.py --config C [--device cpu]"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from nvsim.estimators.nn import NNEstimator, RamseyNet, make_batch
from nvsim.provenance import git_sha

REPO = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text())
    torch.manual_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    dev = torch.device(args.device)
    net = RamseyNet().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=cfg["lr"])
    sched = torch.optim.lr_scheduler.LinearLR(
        opt, start_factor=1e-2, total_iters=cfg["warmup_steps"])
    for epoch in range(cfg["epochs"]):
        for _ in range(cfg["steps_per_epoch"]):
            x, aux, y = make_batch(cfg, cfg["batch"], rng)
            x, aux, y = x.to(dev), aux.to(dev), y.to(dev)
            mu, log_sigma = net(x, aux)
            loss = (log_sigma + 0.5 * ((y - mu) / log_sigma.exp()) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), cfg["grad_clip"])
            opt.step()
            sched.step()
        print(f"[train] epoch {epoch + 1}/{cfg['epochs']} loss {loss:.5f}",
              flush=True)
    ckpt = REPO / "experiments/phase2/nn_ckpt.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    NNEstimator(net.cpu(), cfg).save(ckpt)          # checkpoint BEFORE eval
    print(f"checkpoint {ckpt}", flush=True)

    rng_val = np.random.default_rng(cfg["seed"] + 1)
    x, aux, y = make_batch(cfg, cfg["n_val"], rng_val)
    with torch.no_grad():
        mu, log_sigma = net(x, aux)
    d_lo, d_hi = cfg["delta_range_hz"]
    rmse = float(((mu - y) ** 2).mean().sqrt() * (d_hi - d_lo))
    results = {"config": cfg, "git_sha": git_sha(), "seed": cfg["seed"],
               "val_rmse_delta_hz": rmse, "final_train_loss": float(loss)}
    out = REPO / "experiments/phase2/nn_train.results.json"
    out.write_text(json.dumps(results))
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
