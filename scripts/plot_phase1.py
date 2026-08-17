#!/usr/bin/env python3
"""Phase 1 validation figures from experiments/phase1/artifacts/*.npz.

Usage: plot_phase1.py [--artifacts DIR] [--out DIR]

Comparison targets: 14N triplet + power broadening — Dréau et al., Phys. Rev. B
84, 195204 (2011), Fig. 3(a)-(d), eqs. (10)-(11); photon-budget records —
Barry et al., Rev. Mod. Phys. 92, 015004 (2020), Sec. III.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from nvsim.constants import A_PAR_N14_HZ
from nvsim.experiment import load_dataset
from nvsim.readout import mean_counts_per_shot

REPO = Path(__file__).resolve().parent.parent

# Reference dataviz palette (constants as in scripts/plot_phase0.py)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
RAMP4 = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
SEQ_BLUES = LinearSegmentedColormap.from_list(
    "seq_blues", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
                  "#256abf", "#184f95", "#0d366b"])


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6)
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


def plot_odmr_n14(art_dir, out):
    fig, ax = plt.subplots(figsize=(6.4, 5.0), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    offset = 0.10
    rungs = sorted(art_dir.glob("saturation_*.npz"),
                   key=lambda p: float(p.stem.split("_")[1]))
    for i, path in enumerate(rungs):
        ds = load_dataset(path)
        cfg = ds["config"]
        s_par = cfg["truth"]["saturation"]
        f_mhz = (ds["sweep_values"] - 2.87e9) / 1e6
        scale = cfg["n_shots"] * cfg["readout"]["r_hz"] * cfg["readout"]["t_read_s"]
        y = ds["counts"].mean(axis=0) / scale + i * offset
        ax.plot(f_mhz, y, color=RAMP4[i], linewidth=1.2)
        fwhm_mhz = cfg["truth"]["fwhm_hz"] * np.sqrt(1 + s_par) / 1e6
        ax.annotate(f"s = {s_par:g}  (Γ = {fwhm_mhz:.2f} MHz)",
                    (f_mhz[-1], y[-1]), xytext=(6, 0),
                    textcoords="offset points", va="center",
                    color=INK, fontsize=8.5)
    # 2.16 MHz triplet spacing marked between two dips of the narrowest trace
    ds0 = load_dataset(rungs[0])
    bz = ds0["config"]["truth"]["b_nv_t"][2]
    f_minus_mhz = -28.02e9 * bz / 1e6
    a_mhz = abs(A_PAR_N14_HZ) / 1e6
    y_arrow = 1.012
    ax.annotate("", (f_minus_mhz, y_arrow), (f_minus_mhz + a_mhz, y_arrow),
                arrowprops={"arrowstyle": "<->", "color": SECONDARY,
                            "linewidth": 0.9})
    ax.annotate("2.16 MHz", (f_minus_mhz + a_mhz / 2, y_arrow + 0.004),
                color=SECONDARY, fontsize=8, ha="center")
    ax.set_xlabel("MW detuning from D (MHz)", color=INK)
    ax.set_ylabel("counts (norm., offset for clarity)", color=INK)
    ax.set_title("$^{14}$N triplet vs MW saturation: power broadening washes "
                 "out hyperfine", color=INK, fontsize=11)
    ax.margins(x=0.30)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    _footnote(fig, "Γ(s) = Γ₀√(1+s), C(s) = C₀·s/(1+s)   ·   cf. Dréau et al., "
                   "PRB 84, 195204 (2011), Fig. 3(a)–(d), eqs. (10)–(11)")
    _save(fig, out)


def plot_snr_ladder(art_dir, out):
    rungs = sorted(art_dir.glob("n_shots_*.npz"),
                   key=lambda p: int(p.stem.split("_")[2]))
    fig, axes = plt.subplots(len(rungs), 1, figsize=(6.4, 7.6), dpi=200,
                             sharex=True)
    fig.patch.set_facecolor(SURFACE)
    snrs, budgets = [], []
    for ax, path in zip(axes, rungs):
        _style(ax)
        ds = load_dataset(path)
        cfg = ds["config"]
        n = cfg["n_shots"]
        tau_us = ds["sweep_values"] * 1e6
        expected = n * mean_counts_per_shot(ds["truth"]["p0_ideal"],
                                            cfg["readout"])
        mean_counts = ds["counts"].mean(axis=0)
        err = np.sqrt(mean_counts / ds["counts"].shape[0])
        ax.errorbar(tau_us, mean_counts, yerr=err, color=BLUE, linewidth=0,
                    elinewidth=0.8, marker="o", markersize=2.4, zorder=3)
        ax.plot(tau_us, expected, color=SECONDARY, linewidth=1.2, zorder=2)
        noise = (ds["counts"] - expected).std()
        snr = (expected.max() - expected.min()) / noise
        snrs.append(snr)
        budgets.append(n)
        ax.annotate(f"n_shots = {n:g} → single-sweep SNR = {snr:.1f}",
                    (0.02, 0.83), xycoords="axes fraction", color=INK,
                    fontsize=8.5)
    axes[-1].set_xlabel("free precession time τ (µs)", color=INK)
    axes[len(axes) // 2].set_ylabel("counts per point", color=INK)
    axes[0].set_title("Same physics, photon budget swept: SNR grows as √N",
                      color=INK, fontsize=11)
    inset = axes[0].inset_axes([0.72, 0.16, 0.26, 0.60])
    inset.set_facecolor(SURFACE)
    b = np.asarray(budgets, dtype=float)
    inset.loglog(b, snrs, linestyle="none", marker="o", markersize=5,
                 color=BLUE, markeredgecolor=SURFACE)
    inset.loglog(b, snrs[0] * np.sqrt(b / b[0]), color=AXIS, linewidth=1,
                 linestyle="--")
    inset.set_xlabel("n_shots", color=SECONDARY, fontsize=7)
    inset.set_ylabel("SNR", color=SECONDARY, fontsize=7)
    inset.tick_params(colors=MUTED, labelcolor=SECONDARY, labelsize=6)
    for spine in inset.spines.values():
        spine.set_color(AXIS)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    _footnote(fig, "dots: mean of 10 sweeps ± sem; line: expectation; "
                   "dashed: √N · cf. Barry et al., RMP 92, 015004 (2020) §III")
    _save(fig, out)


def plot_drift_record(path, out):
    ds = load_dataset(path)
    cfg = ds["config"]
    tau_us = ds["sweep_values"] * 1e6
    wall_min = ds["timestamps_s"] / 60.0

    fig = plt.figure(figsize=(6.4, 7.8), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    gs = fig.add_gridspec(4, 1, height_ratios=[2.6, 1, 1, 1], hspace=0.45)

    ax0 = fig.add_subplot(gs[0])
    _style(ax0)
    ax0.grid(False)
    im = ax0.pcolormesh(tau_us, wall_min[:, 0], ds["counts"], cmap=SEQ_BLUES,
                        rasterized=True)
    ax0.set_xlabel("free precession time τ (µs)", color=INK)
    ax0.set_ylabel("wall-clock time (min)", color=INK)
    ax0.set_title("Ramsey record under drift: fringe phase wanders with "
                  "B(t) and T(t)", color=INK, fontsize=11)
    cbar = fig.colorbar(im, ax=ax0, pad=0.01)
    cbar.ax.tick_params(colors=MUTED, labelsize=7, labelcolor=SECONDARY)
    cbar.set_label("counts per point", color=SECONDARY, fontsize=8)

    t_wall = wall_min.ravel()
    panels = [
        ("drift_b_field_t", 1e9, "δB (nT)", "1/f, rms 400 nT"),
        ("drift_temperature_k", 1e3, "δT (mK)", "linear 0.5 mK/s"),
        ("drift_laser_power", 1e2, "laser power (%)", "OU, σ 1.5 %, τ 30 s"),
    ]
    for k, (key, scale, label, note) in enumerate(panels):
        ax = fig.add_subplot(gs[k + 1])
        _style(ax)
        ax.plot(t_wall, ds["truth"][key].ravel() * scale, color=ORANGE,
                linewidth=0.9)
        ax.set_ylabel(label, color=INK, fontsize=8.5)
        ax.annotate(note, (0.99, 0.82), xycoords="axes fraction", ha="right",
                    color=SECONDARY, fontsize=7.5)
        if k < 2:
            ax.tick_params(labelbottom=False)
    ax.set_xlabel("wall-clock time (min)", color=INK)
    _footnote(fig, "60 sweeps × 120 points × 2000 shots; drift → detuning via "
                   "γ = 28.02 GHz/T and dD/dT = −74 kHz/K (docs/PHYSICS.md)")
    _save(fig, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", type=Path,
                    default=REPO / "experiments/phase1/artifacts")
    ap.add_argument("--out", type=Path, default=REPO / "docs/figures")
    args = ap.parse_args()
    plot_odmr_n14(args.artifacts / "odmr_n14_power",
                  args.out / "phase1_odmr_n14.png")
    plot_snr_ladder(args.artifacts / "ramsey_snr_ladder",
                    args.out / "phase1_snr_ladder.png")
    plot_drift_record(args.artifacts / "ramsey_drift.npz",
                      args.out / "phase1_drift_record.png")


if __name__ == "__main__":
    main()
