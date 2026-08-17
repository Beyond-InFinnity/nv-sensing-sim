import numpy as np
import pytest

from nvsim.experiment import load_dataset, run_experiment, save_dataset

RAMSEY_CFG = {
    "name": "t", "protocol": "ramsey", "seed": 11,
    "sweep": {"min": 0.0, "max": 4e-6, "n_points": 40},
    "n_sweeps": 3, "n_shots": 200,
    "truth": {"detuning_hz": 2e6, "t2star_s": 3e-6},
    "timing": {"t_init_s": 2e-6, "t_read_s": 0.4e-6, "t_dead_s": 1e-6},
    "readout": {"r_hz": 6e7, "contrast": 0.25, "t_read_s": 0.4e-6, "f_pump": 0.95},
    "drift": {"b_field_t": {"kind": "one_over_f", "rms": 2e-7, "alpha": 1.0}},
}


def test_dataset_shapes_types_provenance():
    ds = run_experiment(RAMSEY_CFG)
    assert ds["counts"].shape == (3, 40)
    assert ds["counts"].dtype == np.int64
    assert ds["timestamps_s"].shape == (3, 40)
    assert np.all(np.diff(ds["timestamps_s"].ravel()) > 0)  # monotonic wall clock
    assert len(ds["git_sha"]) >= 7
    assert ds["config"] == RAMSEY_CFG


def test_deterministic_given_seed():
    a, b = run_experiment(RAMSEY_CFG), run_experiment(RAMSEY_CFG)
    np.testing.assert_array_equal(a["counts"], b["counts"])


def test_rng_stream_separation_pairing():
    """Changing the photon budget must not change the drift realization."""
    hot = dict(RAMSEY_CFG, readout=dict(RAMSEY_CFG["readout"], r_hz=1.2e8))
    a, b = run_experiment(RAMSEY_CFG), run_experiment(hot)
    np.testing.assert_array_equal(
        a["truth"]["drift_b_field_t"], b["truth"]["drift_b_field_t"])
    assert a["counts"].sum() < b["counts"].sum()  # brighter budget, same noise path


def test_counts_track_ideal_curve():
    cfg = dict(RAMSEY_CFG, n_sweeps=30, n_shots=2000, drift={})
    ds = run_experiment(cfg)
    mean = ds["counts"].mean(axis=0).astype(float)
    p0 = ds["truth"]["p0_ideal"]
    # correlation between measured counts and ideal p0 curve
    assert np.corrcoef(mean, p0)[0, 1] > 0.99


def test_odmr_protocol_and_roundtrip(tmp_path):
    cfg = {
        "name": "o", "protocol": "odmr", "seed": 5,
        "sweep": {"min": 2.80e9, "max": 2.94e9, "n_points": 120},
        "n_sweeps": 2, "n_shots": 300,
        "truth": {"b_nv_t": [0.0, 0.0, 1e-3], "contrast": 0.2,
                  "fwhm_hz": 1e6, "saturation": 2.0},
        "timing": {"t_init_s": 0.0, "t_read_s": 1e-3, "t_dead_s": 0.0},
        "readout": {"r_hz": 6e7, "contrast": 0.2, "t_read_s": 1e-3, "f_pump": 1.0},
        "drift": {},
    }
    ds = run_experiment(cfg)
    assert ds["counts"].shape == (2, 120)
    p = tmp_path / "o.npz"
    save_dataset(ds, p)
    back = load_dataset(p)
    np.testing.assert_array_equal(back["counts"], ds["counts"])
    assert back["config"] == cfg
    assert back["git_sha"] == ds["git_sha"]
