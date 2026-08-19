#!/usr/bin/env python3
"""Phase 3b figures.

Usage: plot_phase3b.py [--artifacts DIR] [--out DIR]
Needs artifacts from run_phase3b.py for both configs (static + drift) and
latency.results.json."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
TARGET_HZ = 5e3

STATIC = ["adaptive", "policy_rl", "policy_bc", "exp_ladder"]
STATIC_COLORS = {"adaptive": "#2a78d6", "policy_rl": "#eb6834",
                 "policy_bc": "#1baf7a", "exp_ladder": "#eda100"}
STATIC_LABELS = {"adaptive": "A-optimal lookahead (teacher)",
                 "policy_rl": "policy (BC + RL)",
                 "policy_bc": "policy (BC only)",
                 "exp_ladder": "exponential ladder"}
DRIFT = ["adaptive", "policy_rl_drift", "policy_static", "exp_ladder"]
DRIFT_COLORS = {"adaptive": "#2a78d6", "policy_rl_drift": "#eb6834",
                "policy_static": "#1baf7a", "exp_ladder": "#eda100"}
DRIFT_LABELS = {"adaptive": "myopic A-optimal",
                "policy_rl_drift": "policy (RL on drift)",
                "policy_static": "policy (BC static, transfer)",
                "exp_ladder": "exponential ladder"}


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
    vals = []
    for run in art["runs"]:
        t = np.array(run["wall_time_s"])
        v = np.array(run[key])
        vals.append(np.interp(t_grid, t, v, left=np.nan, right=v[-1]))
    return np.array(vals)


def _time_to_target(art, target_hz):
    out = []
    for run in art["runs"]:
        sig = np.array(run["sigma_hz"])
        t = np.array(run["wall_time_s"])
        hit = np.nonzero(sig < target_hz)[0]
        out.append(t[hit[0]] if len(hit) else np.nan)
    return np.array(out)


def plot_static(arts, latency, out):
    fig, ax = plt.subplots(figsize=(6.4, 4.6), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    t_grid = np.geomspace(2e-3, 0.25, 120)
    tt = {}
    for k in STATIC:
        vals = _interp_runs(arts[k], "sigma_hz", t_grid)
        med = np.nanmedian(vals, axis=0)
        lo, hi = np.nanpercentile(vals, [25, 75], axis=0)
        ax.loglog(t_grid, med, color=STATIC_COLORS[k], linewidth=1.8,
                  label=STATIC_LABELS[k])
        ax.fill_between(t_grid, lo, hi, color=STATIC_COLORS[k], alpha=0.12,
                        linewidth=0)
        per = _time_to_target(arts[k], TARGET_HZ)
        n_hit = int(np.sum(~np.isnan(per)))
        tt[k] = np.nanmedian(per) if n_hit > len(per) // 2 else np.nan
    ax.axhline(TARGET_HZ, color=SECONDARY, linewidth=0.9, linestyle="--")
    ax.annotate(f"target {TARGET_HZ / 1e3:.0f} kHz",
                (2.1e-2, TARGET_HZ * 1.15), color=SECONDARY, fontsize=8)
    lines = [f"median time to {TARGET_HZ / 1e3:.0f} kHz:"]
    for k in STATIC:
        lines.append(f"  {STATIC_LABELS[k]}: "
                     + ("never" if np.isnan(tt[k]) else f"{tt[k] * 1e3:.0f} ms"))
    if latency:
        lines.append("")
        lines.append("decision latency (CPU):")
        lines.append(f"  lookahead: {latency['aoptimal_s'] * 1e3:.0f} ms")
        lines.append(f"  policy: {latency['policy_rl_s'] * 1e6:.0f} µs")
    ax.annotate("\n".join(lines), (0.02, 0.04), xycoords="axes fraction",
                color=INK, fontsize=7.5, va="bottom")
    ax.set_xlabel("total wall-clock time (s)", color=INK)
    ax.set_ylabel("posterior σ_δ (Hz)", color=INK)
    ax.set_title("Amortized policy vs its A-optimal teacher",
                 color=INK, fontsize=11)
    ax.legend(frameon=False, loc="upper right", fontsize=8)
    _footnote(fig, "60 replicates, paired δ_true; bands: IQR; identical "
                   "24-point τ menu for policy and lookahead; 250 shots/batch")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    _save(fig, out)
    return tt


def plot_drift(arts, out):
    fig, axes = plt.subplots(2, 1, figsize=(6.4, 6.8), dpi=200, sharex=True)
    fig.patch.set_facecolor(SURFACE)
    t_grid = np.linspace(5e-3, 0.5, 160)
    ax = axes[0]
    _style(ax)
    track = {}
    for k in DRIFT:
        vals = _interp_runs(arts[k], "abs_err_hz", t_grid)
        med = np.nanmedian(vals, axis=0)
        lo, hi = np.nanpercentile(vals, [25, 75], axis=0)
        ax.semilogy(t_grid, med, color=DRIFT_COLORS[k], linewidth=1.8,
                    label=DRIFT_LABELS[k])
        ax.fill_between(t_grid, lo, hi, color=DRIFT_COLORS[k], alpha=0.12,
                        linewidth=0)
        mask = t_grid > 0.1  # steady-state: final 80%
        track[k] = float(np.nanmedian(vals[:, mask]))
    sig = arts["adaptive"]["config"]["drift"]["sigma_hz"]
    ax.axhline(sig, color=SECONDARY, linewidth=0.9, linestyle="--")
    ax.annotate(f"σ_drift = {sig / 1e3:.0f} kHz", (0.4, sig * 1.15),
                color=SECONDARY, fontsize=8)
    lines = ["steady-state tracking error (median, t > 0.1 s):"]
    for k in DRIFT:
        lines.append(f"  {DRIFT_LABELS[k]}: {track[k] / 1e3:.1f} kHz")
    ax.annotate("\n".join(lines), (0.02, 0.04), xycoords="axes fraction",
                color=INK, fontsize=7.5, va="bottom")
    ax.set_ylabel("|δ̂ − δ_true(t)| (Hz)", color=INK)
    ax.set_title("Tracking a drifting detuning "
                 "(OU: σ 100 kHz, τ_corr 50 ms)", color=INK, fontsize=11)
    ax.legend(frameon=False, loc="upper right", fontsize=8)

    ax = axes[1]
    _style(ax)
    for k in DRIFT:
        vals = _interp_runs(arts[k], "tau_s", t_grid) * 1e6
        med = np.nanmedian(vals, axis=0)
        lw = 2 if k in ("policy_rl_drift", "adaptive") else 1.1
        ax.plot(t_grid, med, color=DRIFT_COLORS[k], linewidth=lw,
                alpha=0.9, label=DRIFT_LABELS[k])
    t2s_us = arts["adaptive"]["config"]["t2star_s"] * 1e6
    ax.axhline(t2s_us, color=SECONDARY, linewidth=0.9, linestyle="--")
    ax.annotate(f"T2* = {t2s_us:g} µs", (0.02, t2s_us * 1.04),
                color=SECONDARY, fontsize=8)
    ax.set_xlabel("total wall-clock time (s)", color=INK)
    ax.set_ylabel("median chosen τ (µs)", color=INK)
    _footnote(fig, "60 replicates, paired δ_true and drift paths; bands: "
                   "IQR; all methods share the drift-aware posterior "
                   "(diffusion step, PHYSICS.md)")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    _save(fig, out)
    return track


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", type=Path,
                    default=REPO / "experiments/phase3b/artifacts")
    ap.add_argument("--artifacts-drift", type=Path,
                    default=REPO / "experiments/phase3b/artifacts_drift")
    ap.add_argument("--out", type=Path, default=REPO / "docs/figures")
    args = ap.parse_args()

    static_arts = {k: json.loads((args.artifacts / f"{k}.json").read_text())
                   for k in STATIC}
    lat_path = args.artifacts / "latency.results.json"
    latency = json.loads(lat_path.read_text()) if lat_path.exists() else None
    tt = plot_static(static_arts, latency,
                     args.out / "phase3b_policy_vs_aoptimal.png")
    if not np.isnan(tt.get("policy_rl", np.nan)):
        print(f"gate 3b-A: policy_rl / adaptive time-to-target = "
              f"{tt['policy_rl'] / tt['adaptive']:.2f} (pass < 1.2)")

    drift_paths = [args.artifacts_drift / f"{k}.json" for k in DRIFT]
    if all(p.exists() for p in drift_paths):
        drift_arts = {k: json.loads(
            (args.artifacts_drift / f"{k}.json").read_text()) for k in DRIFT}
        track = plot_drift(drift_arts, args.out / "phase3b_drift_tracking.png")
        print(f"gate 3b-B: drift-policy / myopic tracking-error ratio = "
              f"{track['policy_rl_drift'] / track['adaptive']:.2f}")
    else:
        print("drift artifacts not present; skipped drift figure")


if __name__ == "__main__":
    main()
