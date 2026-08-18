#!/usr/bin/env python3
"""Phase 2 headline figures from experiments/phase2/artifacts/estimates_*.json.

Usage: plot_phase2.py [--artifacts DIR] [--out DIR]
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nvsim.constants import GAMMA_E_HZ_PER_T

REPO = Path(__file__).resolve().parent.parent

# Reference dataviz palette (constants as in scripts/plot_phase0.py)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE = "#2a78d6"    # Bayes (slot 1)
ORANGE = "#eb6834"  # LSQ (slot 2)
AQUA = "#1baf7a"    # NN (slot 3)
EST_COLORS = {"lsq": ORANGE, "bayes": BLUE, "nn": AQUA}
EST_LABELS = {"lsq": "least squares", "bayes": "Bayes (grid)", "nn": "NN (1D-CNN)"}


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, which="both", color=GRID, linewidth=0.6)
    ax.tick_params(colors=MUTED, labelcolor=INK)
    for spine in ax.spines.values():
        spine.set_color(AXIS)


def _footnote(fig, text):
    fig.text(0.01, 0.005, text, fontsize=6.5, color=MUTED)


def _save(fig, out):
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {out}")


def _rmse_and_ci(errors, rng, n_boot=1000):
    """RMSE with bootstrap 68% CI."""
    errors = np.asarray(errors)
    rmse = np.sqrt(np.mean(errors**2))
    boots = [np.sqrt(np.mean(rng.choice(errors, len(errors)) ** 2))
             for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [16, 84])
    return rmse, lo, hi


def load_ladder(art_dir):
    rows = []
    for path in sorted(art_dir.glob("estimates_n_shots_*.json"),
                       key=lambda p: int(p.stem.split("_")[-1])):
        art = json.loads(path.read_text())
        rows.append(art)
    return rows


def plot_headline(art_dir, out):
    rng = np.random.default_rng(0)
    rows = load_ladder(art_dir)
    estimators = [e for e in ("lsq", "bayes", "nn") if e in rows[0]["records"][0]]
    fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    n_shots = np.array([r["config"]["n_shots"] for r in rows], dtype=float)
    crb = np.array([r["crb_sigma_delta_hz"] for r in rows])
    ax.loglog(n_shots, crb, color=SECONDARY, linestyle="--", linewidth=1.4,
              label="Cramér–Rao bound", zorder=2)
    for est in estimators:
        rmses, los, his = [], [], []
        for r in rows:
            truth = r["truth"]["detuning_hz"]
            errs = [rec[est]["delta_hz"] - truth for rec in r["records"]]
            rmse, lo, hi = _rmse_and_ci(errs, rng)
            rmses.append(rmse), los.append(lo), his.append(hi)
        rmses, los, his = map(np.asarray, (rmses, los, his))
        open_marker = est == "lsq"  # LSQ overlaps Bayes at every rung
        ax.errorbar(n_shots, rmses, yerr=[rmses - los, his - rmses],
                    color=EST_COLORS[est], linewidth=1.8, marker="o",
                    markersize=8 if open_marker else 5,
                    markerfacecolor="none" if open_marker else EST_COLORS[est],
                    markeredgecolor=EST_COLORS[est] if open_marker else SURFACE,
                    capsize=2, label=EST_LABELS[est],
                    zorder=3 if open_marker else 4)
        eff_hi = crb[-1] / rmses[-1]
        ax.annotate(f"{EST_LABELS[est]}: CRB/RMSE = {eff_hi:.2f} @ 2e4",
                    (0.02, 0.16 - 0.055 * estimators.index(est)),
                    xycoords="axes fraction", color=EST_COLORS[est], fontsize=7.5)
    ax.set_xlabel("shots per point N", color=INK)
    ax.set_ylabel("RMSE of $\\hat{\\delta}$ (Hz)", color=INK)

    def hz_to_nt(x):
        return x / GAMMA_E_HZ_PER_T * 1e9

    sec = ax.secondary_yaxis(
        "right", functions=(hz_to_nt, lambda b: b * GAMMA_E_HZ_PER_T / 1e9))
    sec.set_ylabel("equivalent $\\sigma_B$ (nT)", color=INK)
    sec.tick_params(colors=MUTED, labelcolor=INK)
    for spine in sec.spines.values():
        spine.set_color(AXIS)
    ax.set_title("Estimator RMSE vs photon budget, against the CRB",
                 color=INK, fontsize=11)
    ax.legend(frameon=False, loc="upper right", fontsize=8.5)
    _footnote(fig, "200 paired records/rung; 68% bootstrap CIs; R = 60 Mcps, "
                   "C = 0.25, t_read = 0.4 µs; δ = 2 MHz, T2* jointly fit")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    _save(fig, out)


def plot_generalization(art_dir, out):
    rng = np.random.default_rng(1)
    names = ["heldout_indist", "heldout_drifted", "heldout_pump090"]
    labels = ["in-distribution", "1/f drift\n(200 nT rms)", "f_pump 0.90\n(trained on 0.95)"]
    arts = [json.loads((art_dir / f"estimates_{n}.json").read_text())
            for n in names]
    estimators = [e for e in ("bayes", "nn") if e in arts[0]["records"][0]]
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    ax.grid(True, axis="y", color=GRID, linewidth=0.6)
    ax.grid(False, axis="x")
    width = 0.32
    xs = np.arange(len(names))
    for k, est in enumerate(estimators):
        rmses, errs_lo, errs_hi = [], [], []
        for art in arts:
            truth = art["truth"]["detuning_hz"]
            errs = [rec[est]["delta_hz"] - truth for rec in art["records"]]
            rmse, lo, hi = _rmse_and_ci(errs, rng)
            rmses.append(rmse), errs_lo.append(rmse - lo), errs_hi.append(hi - rmse)
        pos = xs + (k - (len(estimators) - 1) / 2) * width
        ax.bar(pos, rmses, width * 0.94, color=EST_COLORS[est],
               label=EST_LABELS[est], zorder=3)
        ax.errorbar(pos, rmses, yerr=[errs_lo, errs_hi], fmt="none",
                    ecolor=INK, elinewidth=0.9, capsize=2, zorder=4)
        for x, r in zip(pos, rmses):
            ax.annotate(f"{r:.0f}", (x, r), xytext=(0, 3),
                        textcoords="offset points", ha="center",
                        color=INK, fontsize=7.5)
    crb = arts[0]["crb_sigma_delta_hz"]
    ax.axhline(crb, color=SECONDARY, linestyle="--", linewidth=1.2)
    ax.annotate(f"CRB (well-specified) = {crb:.0f} Hz", (0.40, crb),
                xytext=(0, 5), textcoords="offset points", ha="center",
                color=SECONDARY, fontsize=8,
                xycoords=("axes fraction", "data"))
    ax.set_xticks(xs, labels)
    ax.set_ylabel("RMSE of $\\hat{\\delta}$ (Hz)", color=INK)
    ax.set_title("Held-out noise configs: does the NN overfit its noise model?",
                 color=INK, fontsize=11)
    ax.legend(frameon=False, loc="upper left", fontsize=8.5)
    _footnote(fig, "200 paired records each, n_shots = 2000; 68% bootstrap CIs; "
                   "NN trained only on drift-free f_pump = 0.95 records")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    _save(fig, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", type=Path,
                    default=REPO / "experiments/phase2/artifacts")
    ap.add_argument("--out", type=Path, default=REPO / "docs/figures")
    args = ap.parse_args()
    plot_headline(args.artifacts, args.out / "phase2_rmse_vs_crb.png")
    plot_generalization(args.artifacts, args.out / "phase2_nn_generalization.png")


if __name__ == "__main__":
    main()
