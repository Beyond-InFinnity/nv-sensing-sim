#!/usr/bin/env python3
"""Run a Phase 0 experiment from a JSON config; write an artifact embedding
config + seed + git SHA. Usage: run_phase0.py CONFIG [--out-dir DIR]"""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from nvsim.odmr import odmr_spectrum
from nvsim.pulsed import hahn_echo, rabi, ramsey, t2star_from_sigma

REPO = Path(__file__).resolve().parent.parent


def git_sha():
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO
    ).stdout.strip()


def run_odmr(c):
    f = np.linspace(c["f_min_hz"], c["f_max_hz"], c["n_freq"])
    return {
        "f_hz": f.tolist(),
        "b_axial_t": c["b_axial_t"],
        "spectra": [
            odmr_spectrum(f, e_hz=c["e_hz"], b_nv_t=(0, 0, bz),
                          contrast=c["contrast"], fwhm_hz=c["fwhm_hz"]).tolist()
            for bz in c["b_axial_t"]
        ],
    }


def run_rabi(c):
    t = np.linspace(0, c["t_max_s"], c["n_t"])
    return {
        "times_s": t.tolist(),
        "rabi_hz": c["rabi_hz"],
        "p0": [rabi(fr, t, detuning_hz=c["detuning_hz"],
                    t1_s=c["t1_s"], t2_s=c["t2_s"]).tolist()
               for fr in c["rabi_hz"]],
    }


def run_ramsey(c):
    taus = np.linspace(0, c["tau_max_s"], c["n_tau"])
    p0 = ramsey(taus, detuning_hz=c["detuning_hz"], mode=c["mode"],
                sigma_detuning_hz=c["sigma_detuning_hz"],
                n_samples=c["n_samples"], seed=c["seed"])
    return {
        "taus_s": taus.tolist(), "p0": p0.tolist(),
        "t2star_s": t2star_from_sigma(c["sigma_detuning_hz"]),
    }


def run_hahn_echo(c):
    taus = np.linspace(0, c["tau_max_s"], c["n_tau"])
    out = {
        "taus_s": taus.tolist(),
        "p0_echo": hahn_echo(taus, t2_s=c["t2_s"], mode=c["mode"],
                             sigma_detuning_hz=c["sigma_detuning_hz"],
                             n_samples=c["n_samples"], seed=c["seed"]).tolist(),
        "t2_s": c["t2_s"],
        "t2star_s": t2star_from_sigma(c["sigma_detuning_hz"]),
    }
    if c.get("ramsey_comparison"):
        out["p0_ramsey"] = ramsey(
            taus, detuning_hz=0.0, mode=c["mode"],
            sigma_detuning_hz=c["sigma_detuning_hz"],
            n_samples=c["n_samples"], seed=c["seed"],
        ).tolist()
    return out


PROTOCOLS = {"odmr": run_odmr, "rabi": run_rabi,
             "ramsey": run_ramsey, "hahn_echo": run_hahn_echo}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path)
    ap.add_argument("--out-dir", type=Path,
                    default=REPO / "experiments/phase0/artifacts")
    args = ap.parse_args()
    config = json.loads(args.config.read_text())
    np.random.seed(config["seed"])
    results = PROTOCOLS[config["protocol"]](config)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    artifact = {"config": config, "seed": config["seed"],
                "git_sha": git_sha(), "results": results}
    out_path = args.out_dir / f"{config['name']}.json"
    out_path.write_text(json.dumps(artifact))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
