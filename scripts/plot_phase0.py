#!/usr/bin/env python3
"""Phase 0 validation figures from experiments/phase0/artifacts/*.json.

Usage: plot_phase0.py [--artifacts DIR] [--out DIR]

Comparison targets: Barry et al., Rev. Mod. Phys. 92, 015004 (2020) —
ODMR vs bias field: Figs. 7(d), 13; Ramsey FID: Figs. 7(b), 17;
Hahn echo T2 decay: Figs. 12, 14; Rabi: pseudo-spin-1/2 dynamics, Eq. (9).
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from nvsim.constants import D_GS_HZ, GAMMA_E_HZ_PER_T

REPO = Path(__file__).resolve().parent.parent

# Reference dataviz palette (constants copied from qec-neural-decoder plotting.py)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE = "#2a78d6"    # categorical slot 1
ORANGE = "#eb6834"  # categorical slot 2
# ordinal single-hue blue ramp (light-surface bound: nothing lighter than step 250)
RAMP3 = ["#86b6ef", "#2a78d6", "#104281"]
RAMP4 = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]


def _new_axes(figsize=(6.4, 4.6)):
    fig, ax = plt.subplots(figsize=figsize, dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.tick_params(colors=MUTED, labelcolor=INK)
    for spine in ax.spines.values():
        spine.set_color(AXIS)
    return fig, ax


def _footnote(fig, text):
    fig.text(0.01, 0.005, text, fontsize=6.5, color=MUTED)


def _save(fig, out):
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {out}")


def plot_odmr(artifact, out):
    r = artifact["results"]
    f_ghz = np.asarray(r["f_hz"]) / 1e9
    fig, ax = _new_axes()
    offset = 0.45
    for i, (bz, s) in enumerate(zip(r["b_axial_t"], r["spectra"])):
        y = np.asarray(s) + i * offset
        ax.plot(f_ghz, y, color=RAMP4[i], linewidth=2)
        ax.annotate(f"$B_z$ = {bz * 1e3:g} mT", (f_ghz[-1], y[-1]),
                    xytext=(6, 0), textcoords="offset points",
                    va="center", color=INK, fontsize=9)
        for sign in (-1, +1):  # expected dip positions f = D ± γBz
            f_dip = (D_GS_HZ + sign * GAMMA_E_HZ_PER_T * bz) / 1e9
            ax.plot([f_dip], [1 + i * offset + 0.06], marker="v", markersize=5,
                    color=RAMP4[i], markeredgecolor=SURFACE, markeredgewidth=0.5,
                    linestyle="none")
    ax.set_xlabel("MW frequency (GHz)", color=INK)
    ax.set_ylabel("fluorescence (norm., offset for clarity)", color=INK)
    ax.set_title("CW-ODMR vs axial bias field: dips at $D \\pm \\gamma B_z$",
                 color=INK, fontsize=11)
    ax.margins(x=0.16)
    _footnote(fig, "markers: expected D ± γBz, γ/2π = 28.02 GHz/T   ·   "
                   "cf. Barry et al., RMP 92, 015004 (2020), Figs. 7(d) and 13")
    _save(fig, out)


def _damped_cos(t, f, tau):
    return 0.5 + 0.5 * np.cos(2 * np.pi * f * t) * np.exp(-t / tau)


def plot_rabi(artifact, out):
    r = artifact["results"]
    t_us = np.asarray(r["times_s"]) * 1e6
    fig, ax = _new_axes()
    fitted = []
    for i, (f_drive, p0) in enumerate(zip(r["rabi_hz"], r["p0"])):
        p0 = np.asarray(p0)
        ax.plot(t_us, p0, color=RAMP3[i], linewidth=2,
                label=f"$\\Omega_R$ = {f_drive / 1e6:g} MHz")
        popt, _ = curve_fit(_damped_cos, np.asarray(r["times_s"]), p0,
                            p0=[f_drive, 10e-6])
        fitted.append(popt[0])
    ax.set_xlabel("drive duration (µs)", color=INK)
    ax.set_ylabel("$P(m_s{=}0)$", color=INK)
    ax.set_title("Rabi oscillations: frequency linear in drive amplitude",
                 color=INK, fontsize=11)
    ax.legend(frameon=False, loc="upper right", fontsize=9)

    inset = ax.inset_axes([0.66, 0.13, 0.30, 0.30])
    inset.set_facecolor(SURFACE)
    drives = np.asarray(r["rabi_hz"]) / 1e6
    lim = [0, drives.max() * 1.15]
    inset.plot(lim, lim, color=AXIS, linewidth=1, linestyle="--")
    inset.plot(drives, np.asarray(fitted) / 1e6, linestyle="none", marker="o",
               markersize=6, color=BLUE, markeredgecolor=SURFACE)
    inset.set_xlabel("drive (MHz)", color=SECONDARY, fontsize=7)
    inset.set_ylabel("fit $f_{Rabi}$ (MHz)", color=SECONDARY, fontsize=7)
    inset.tick_params(colors=MUTED, labelcolor=SECONDARY, labelsize=7)
    for spine in inset.spines.values():
        spine.set_color(AXIS)
    _footnote(fig, "damped by T1 = 100 µs, T2 = 5 µs   ·   two-level treatment "
                   "per Barry et al., RMP 92, 015004 (2020), Eq. (9)")
    _save(fig, out)


def plot_ramsey(artifact, out):
    r = artifact["results"]
    tau_us = np.asarray(r["taus_s"]) * 1e6
    p0 = np.asarray(r["p0"])
    t2s_us = r["t2star_s"] * 1e6
    delta_mhz = artifact["config"]["detuning_hz"] / 1e6
    fig, ax = _new_axes()
    env = 0.5 * np.exp(-((tau_us / t2s_us) ** 2))
    ax.plot(tau_us, 0.5 + env, color=MUTED, linewidth=1.2, linestyle="--")
    ax.plot(tau_us, 0.5 - env, color=MUTED, linewidth=1.2, linestyle="--")
    ax.plot(tau_us, p0, color=BLUE, linewidth=1.6)
    ax.annotate(f"fringes at detuning $\\delta$ = {delta_mhz:g} MHz",
                (0.62, 0.90), xycoords="axes fraction", color=INK, fontsize=9)
    ax.annotate(f"envelope $e^{{-(\\tau/T_2^*)^2}}$, $T_2^*$ = {t2s_us:.2f} µs",
                (0.62, 0.84), xycoords="axes fraction", color=SECONDARY, fontsize=9)
    ax.set_xlabel("free precession time τ (µs)", color=INK)
    ax.set_ylabel("$P(m_s{=}0)$", color=INK)
    ax.set_title("Ramsey fringes: frequency = detuning, Gaussian FID envelope",
                 color=INK, fontsize=11)
    _footnote(fig, "static Gaussian detuning bath, σ = 150 kHz   ·   "
                   "cf. Barry et al., RMP 92, 015004 (2020), Figs. 7(b) and 17")
    _save(fig, out)


def plot_echo(echo_artifact, ramsey_artifact, out):
    re_ = echo_artifact["results"]
    rr = ramsey_artifact["results"]
    t2_us = re_["t2_s"] * 1e6
    t2s_us = re_["t2star_s"] * 1e6
    total_echo_us = 2 * np.asarray(re_["taus_s"]) * 1e6
    p_echo = np.asarray(re_["p0_echo"])
    tau_ramsey_us = np.asarray(rr["taus_s"]) * 1e6
    p_ramsey = np.asarray(rr["p0"])

    fig, ax = _new_axes()
    ax.grid(True, which="both", color=GRID, linewidth=0.6)
    mask = tau_ramsey_us > 0
    ax.plot(tau_ramsey_us[mask], p_ramsey[mask], color=ORANGE, linewidth=1.2,
            alpha=0.85, label="Ramsey (fringes at δ = 2 MHz)")
    fine = np.geomspace(5e-3, 20, 400)
    ax.plot(fine, 0.5 + 0.5 * np.exp(-((fine / t2s_us) ** 2)), color=ORANGE,
            linewidth=1.2, linestyle="--")
    m2 = total_echo_us > 0
    ax.plot(total_echo_us[m2], p_echo[m2], color=BLUE, linewidth=2, marker="o",
            markersize=5, markeredgecolor=SURFACE, markeredgewidth=0.8,
            label="Hahn echo")
    fine2 = np.geomspace(1, 620, 400)
    ax.plot(fine2, 0.5 + 0.5 * np.exp(-fine2 / t2_us), color=BLUE,
            linewidth=1.2, linestyle="--")
    ax.set_xscale("log")
    ax.set_xlabel("total free-evolution time (µs)", color=INK)
    ax.set_ylabel("$P(m_s{=}0)$", color=INK)
    ax.set_title(
        f"Echo refocuses inhomogeneous dephasing: $T_2^*$ = {t2s_us:.1f} µs "
        f"→ $T_2$ = {t2_us:g} µs", color=INK, fontsize=11)
    ax.annotate("$e^{-(\\tau/T_2^*)^2}$", (1.6, 0.80), color=ORANGE, fontsize=9)
    ax.annotate("$e^{-2\\tau/T_2}$", (150, 0.72), color=BLUE, fontsize=9)
    ax.legend(frameon=False, loc="lower left", fontsize=9)
    _footnote(fig, "same static bath (σ = 150 kHz) in both protocols; dashed: "
                   "analytic envelopes   ·   cf. Barry et al., RMP 92, 015004 "
                   "(2020), Figs. 12 and 14")
    _save(fig, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", type=Path,
                    default=REPO / "experiments/phase0/artifacts")
    ap.add_argument("--out", type=Path, default=REPO / "docs/figures")
    args = ap.parse_args()

    def load(name):
        return json.loads((args.artifacts / f"{name}.json").read_text())

    plot_odmr(load("odmr_vs_field"), args.out / "phase0_odmr.png")
    plot_rabi(load("rabi"), args.out / "phase0_rabi.png")
    plot_ramsey(load("ramsey"), args.out / "phase0_ramsey.png")
    plot_echo(load("hahn_echo"), load("ramsey"), args.out / "phase0_echo.png")


if __name__ == "__main__":
    main()
