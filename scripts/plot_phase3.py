#!/usr/bin/env python3
"""Phase 3 figures from experiments/phase3/artifacts/<schedule>.json.

Usage: plot_phase3.py [--artifacts DIR] [--out DIR]
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent

# Reference dataviz palette (constants as in scripts/plot_phase0.py)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
COLORS = {"adaptive": "#2a78d6", "fixed_tau": "#eb6834",
          "exp_ladder": "#1baf7a", "linear_sweep": "#eda100"}
LABELS = {"adaptive": "adaptive (A-optimal)", "fixed_tau": "best fixed τ",
          "exp_ladder": "exponential ladder", "linear_sweep": "linear sweep"}
ORDER = ["adaptive", "fixed_tau", "exp_ladder", "linear_sweep"]
TARGET_HZ = 5e3


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


def _interp_runs(art, key, t_grid):
    """Interpolate each run's trajectory onto a common time grid (log-spaced)."""
    vals = []
    for run in art["runs"]:
        t = np.array(run["wall_time_s"])
        v = np.array(run[key])
        vals.append(np.interp(t_grid, t, v, left=np.nan, right=v[-1]))
    return np.array(vals)


def _time_to_target(art, target_hz):
    """Per-run wall time to reach posterior sigma < target; nan if never."""
    out = []
    for run in art["runs"]:
        sig = np.array(run["sigma_hz"])
        t = np.array(run["wall_time_s"])
        hit = np.nonzero(sig < target_hz)[0]
        out.append(t[hit[0]] if len(hit) else np.nan)
    return np.array(out)


def plot_sigma_vs_time(arts, out):
    fig, axes = plt.subplots(2, 1, figsize=(6.4, 7.2), dpi=200, sharex=True)
    fig.patch.set_facecolor(SURFACE)
    t_grid = np.geomspace(2e-3, 0.25, 120)
    for ax, key, ylabel in ((axes[0], "sigma_hz", "posterior σ_δ (Hz)"),
                            (axes[1], "abs_err_hz", "|δ̂ − δ_true| (Hz)")):
        _style(ax)
        for kind in ORDER:
            vals = _interp_runs(arts[kind], key, t_grid)
            med = np.nanmedian(vals, axis=0)
            lo, hi = np.nanpercentile(vals, [25, 75], axis=0)
            ax.loglog(t_grid, med, color=COLORS[kind], linewidth=1.8,
                      label=LABELS[kind])
            ax.fill_between(t_grid, lo, hi, color=COLORS[kind], alpha=0.15,
                            linewidth=0)
        ax.set_ylabel(ylabel, color=INK)
    # reference slopes on the top panel
    ax = axes[0]
    ref_t = np.array([4e-3, 0.2])
    ax.loglog(ref_t, 3.5e4 * (ref_t / ref_t[0]) ** -0.5, color=AXIS,
              linewidth=1, linestyle=":")
    ax.annotate("$t^{-1/2}$", (0.19, 5.3e3), color=MUTED, fontsize=8)
    ax.axhline(TARGET_HZ, color=SECONDARY, linewidth=0.9, linestyle="--")
    ax.annotate(f"target {TARGET_HZ / 1e3:.0f} kHz", (2.1e-2, TARGET_HZ * 1.15),
                color=SECONDARY, fontsize=8)
    tt = {}
    lines = [f"median time to {TARGET_HZ / 1e3:.0f} kHz:"]
    for k in ORDER:
        per_run = _time_to_target(arts[k], TARGET_HZ)
        n_hit = int(np.sum(~np.isnan(per_run)))
        tt[k] = np.nanmedian(per_run) if n_hit > len(per_run) // 2 else np.nan
        if np.isnan(tt[k]):
            lines.append(f"  {LABELS[k]}: never ({n_hit}/{len(per_run)} runs)")
        else:
            lines.append(f"  {LABELS[k]}: {tt[k] * 1e3:.0f} ms")
    ax.annotate("\n".join(lines), (0.02, 0.05), xycoords="axes fraction",
                color=INK, fontsize=8, va="bottom")
    axes[0].legend(frameon=False, loc="upper right", fontsize=8.5)
    axes[0].set_title("Time to precision: adaptive vs fixed Ramsey schedules",
                      color=INK, fontsize=11)
    axes[1].set_xlabel("total wall-clock time (s)", color=INK)
    _footnote(fig, "60 replicates, δ_true ~ U(0.3, 3.7) MHz, paired across "
                   "schedules; bands: IQR; 250 shots/batch; T2* = 1.5 µs known; "
                   "overhead 3.4 µs/shot + τ")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    _save(fig, out)
    return tt


def plot_tau_trajectory(arts, out):
    fig, ax = plt.subplots(figsize=(6.4, 4.4), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    t_grid = np.geomspace(2e-3, 0.25, 120)
    for kind in ORDER:
        vals = _interp_runs(arts[kind], "tau_s", t_grid) * 1e6
        med = np.nanmedian(vals, axis=0)
        if kind == "adaptive":
            lo, hi = np.nanpercentile(vals, [25, 75], axis=0)
            ax.fill_between(t_grid, lo, hi, color=COLORS[kind], alpha=0.15,
                            linewidth=0)
            ax.semilogx(t_grid, med, color=COLORS[kind], linewidth=2,
                        label=LABELS[kind])
        else:
            ax.semilogx(t_grid, med, color=COLORS[kind], linewidth=1.2,
                        alpha=0.8, label=LABELS[kind])
    t2s_us = arts["adaptive"]["config"]["t2star_s"] * 1e6
    ax.axhline(t2s_us, color=SECONDARY, linewidth=0.9, linestyle="--")
    ax.annotate(f"T2* = {t2s_us:g} µs", (2.4e-3, t2s_us * 1.06),
                color=SECONDARY, fontsize=8)
    ax.set_xlabel("total wall-clock time (s)", color=INK)
    ax.set_ylabel("interrogation time τ (µs)", color=INK)
    ax.set_title("The adaptive schedule discovers the T2*-limited optimum",
                 color=INK, fontsize=11)
    ax.legend(frameon=False, loc="center right", fontsize=8.5)
    _footnote(fig, "median chosen τ vs elapsed time; band: adaptive IQR; "
                   "fixed schedules shown as medians of their cycles")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    _save(fig, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", type=Path,
                    default=REPO / "experiments/phase3/artifacts")
    ap.add_argument("--out", type=Path, default=REPO / "docs/figures")
    args = ap.parse_args()
    arts = {k: json.loads((args.artifacts / f"{k}.json").read_text())
            for k in ORDER}
    tt = plot_sigma_vs_time(arts, args.out / "phase3_sigma_vs_time.png")
    plot_tau_trajectory(arts, args.out / "phase3_tau_trajectory.png")
    fixed = [tt[k] for k in ORDER if k != "adaptive" and not np.isnan(tt[k])]
    if fixed:
        print(f"speedup adaptive vs best fixed: {min(fixed) / tt['adaptive']:.2f}x")
    else:
        print("speedup adaptive vs best fixed: n/a (no fixed schedule reached target)")


if __name__ == "__main__":
    main()
