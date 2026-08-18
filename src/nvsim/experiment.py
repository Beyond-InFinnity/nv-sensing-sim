"""Virtual-experiment API: config -> measurement-record dataset.

Record format (decided 2026-08-17): per-point summed Poisson counts with
n_shots recorded; per-shot records are a future opt-in. Timing model and
drift->parameter mapping in docs/PHYSICS.md ("Virtual experiment timing
model"). Noise sources draw from separate spawned RNG streams so changing
one knob leaves the others' realizations untouched (paired comparisons)."""
import json

import numpy as np

from .constants import DD_DT_HZ_PER_K, GAMMA_E_HZ_PER_T
from .drift import sample_drift
from .estimators.model import ramsey_p0
from .odmr import odmr_spectrum_n14
from .provenance import git_sha
from .pulsed import hahn_echo, rabi, ramsey
from .readout import mean_counts_per_shot


def _timestamps(cfg, sweep_values):
    """Wall-clock time of each (sweep, point): cumulative shot durations."""
    t = cfg["timing"]
    manip = sweep_values if cfg["protocol"] != "odmr" else np.zeros_like(sweep_values)
    if cfg["protocol"] == "hahn_echo":
        manip = 2 * manip
    point_dur = cfg["n_shots"] * (t["t_init_s"] + manip + t["t_read_s"]
                                  + t["t_dead_s"])
    ends = np.cumsum(np.tile(point_dur, cfg["n_sweeps"]))
    return ends.reshape(cfg["n_sweeps"], len(sweep_values))


def _drift_traces(cfg, times, rng_drift):
    traces = {}
    for key, dcfg in cfg.get("drift", {}).items():
        traces[key] = sample_drift(dcfg, times.ravel(), rng_drift).reshape(times.shape)
    return traces


def _p0_pulsed(cfg, sweep_values, det_shift, omega_mult):
    """P(ms=0) per (sweep, point) with drifted detuning / Rabi amplitude."""
    tr = cfg["truth"]
    proto = cfg["protocol"]
    n_sweeps, n_points = det_shift.shape
    p0 = np.empty((n_sweeps, n_points))
    for i in range(n_sweeps):
        for j, x in enumerate(sweep_values):
            if proto == "rabi":
                p0[i, j] = rabi(tr["rabi_hz"] * omega_mult[i, j], [0.0, x],
                                detuning_hz=det_shift[i, j],
                                t1_s=tr.get("t1_s"), t2_s=tr.get("t2_s"))[-1]
            elif proto == "ramsey":
                # closed form, identical to the mesolve path (tested to 1e-8)
                p0[i, j] = ramsey_p0([x], tr["detuning_hz"] + det_shift[i, j],
                                     tr.get("t2star_s"))[0]
            elif proto == "hahn_echo":
                p0[i, j] = hahn_echo([x], static_detuning_hz=det_shift[i, j],
                                     t2_s=tr.get("t2_s"))[0]
            else:
                raise ValueError(f"unknown protocol: {proto}")
    return p0


def run_experiment(cfg):
    rng_drift, rng_shot = (np.random.default_rng(s)
                           for s in np.random.SeedSequence(cfg["seed"]).spawn(2))
    sw = cfg["sweep"]
    sweep_values = np.linspace(sw["min"], sw["max"], sw["n_points"])
    times = _timestamps(cfg, sweep_values)
    traces = _drift_traces(cfg, times, rng_drift)

    det_shift = np.zeros(times.shape)
    if "b_field_t" in traces:
        det_shift += GAMMA_E_HZ_PER_T * traces["b_field_t"]
    if "temperature_k" in traces:
        det_shift += DD_DT_HZ_PER_K * traces["temperature_k"]
    lam_mult = 1 + traces["laser_power"] if "laser_power" in traces else np.ones(times.shape)

    tr = cfg["truth"]
    r = cfg["readout"]
    truth = {f"drift_{k}": v for k, v in traces.items()}
    if cfg["protocol"] == "odmr":
        truth["spectrum_ideal"] = odmr_spectrum_n14(
            sweep_values, b_nv_t=tuple(tr["b_nv_t"]), contrast=tr["contrast"],
            fwhm_hz=tr["fwhm_hz"], saturation=tr.get("saturation"))
        lam = np.empty(times.shape)
        for i in range(cfg["n_sweeps"]):
            # drift shifts line positions: evaluate the spectrum at f - shift
            s_drifted = odmr_spectrum_n14(
                sweep_values - det_shift[i], b_nv_t=tuple(tr["b_nv_t"]),
                contrast=tr["contrast"], fwhm_hz=tr["fwhm_hz"],
                saturation=tr.get("saturation"))
            lam[i] = r["r_hz"] * r["t_read_s"] * s_drifted
    else:
        omega_mult = (1 + traces["mw_amplitude"] if "mw_amplitude" in traces
                      else np.ones(times.shape))
        p0 = _p0_pulsed(cfg, sweep_values, det_shift, omega_mult)
        truth["p0_ideal"] = _p0_pulsed(
            cfg, sweep_values, np.zeros((1, len(sweep_values))),
            np.ones((1, len(sweep_values))))[0]
        truth["p0_drifted"] = p0
        lam = np.vstack([mean_counts_per_shot(p0[i], r)
                         for i in range(cfg["n_sweeps"])])
    counts = rng_shot.poisson(cfg["n_shots"] * lam * lam_mult).astype(np.int64)

    return {"config": cfg, "seed": cfg["seed"], "git_sha": git_sha(),
            "timestamps_s": times, "sweep_values": sweep_values,
            "counts": counts, "truth": truth}


def save_dataset(ds, path):
    arrays = {"timestamps_s": ds["timestamps_s"],
              "sweep_values": ds["sweep_values"], "counts": ds["counts"]}
    arrays |= {f"truth_{k}": np.asarray(v) for k, v in ds["truth"].items()}
    meta = json.dumps({"config": ds["config"], "seed": ds["seed"],
                       "git_sha": ds["git_sha"],
                       "truth_keys": list(ds["truth"])})
    np.savez_compressed(path, meta=np.frombuffer(meta.encode(), dtype=np.uint8),
                        **arrays)


def load_dataset(path):
    with np.load(path) as z:
        meta = json.loads(bytes(z["meta"]).decode())
        ds = {"config": meta["config"], "seed": meta["seed"],
              "git_sha": meta["git_sha"],
              "timestamps_s": z["timestamps_s"],
              "sweep_values": z["sweep_values"],
              "counts": z["counts"],
              "truth": {k: z[f"truth_{k}"] for k in meta["truth_keys"]}}
    return ds
