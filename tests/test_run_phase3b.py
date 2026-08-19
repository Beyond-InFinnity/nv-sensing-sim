import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]


def test_runner_writes_per_method_artifacts(tmp_path):
    from nvsim.estimators.policy import VecRamseyEnv, tau_candidates
    from nvsim.estimators.policy_net import AmortizedPolicy, PolicyNet

    cfg = json.loads(
        (REPO / "experiments/phase3b/policy_vs_aoptimal.json").read_text())
    env = VecRamseyEnv(cfg, 1, np.random.default_rng(0))
    n_act = len(tau_candidates(cfg))
    pol = AmortizedPolicy(PolicyNet(env.n_features, n_act),
                          {"n_features": env.n_features, "n_actions": n_act})
    ckpt = tmp_path / "p.pt"
    pol.save(ckpt)
    cfg["n_replicates"] = 2
    cfg["time_budget_s"] = 0.005
    cfg["methods"] = [{"name": "policy_x", "kind": "policy",
                       "policy_ckpt": str(ckpt)},
                      {"name": "exp_ladder", "kind": "exp_ladder"}]
    cfg_path = tmp_path / "mini.json"
    cfg_path.write_text(json.dumps(cfg))
    r = subprocess.run([sys.executable, str(REPO / "scripts/run_phase3b.py"),
                        str(cfg_path), "--out-dir", str(tmp_path)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    for m in cfg["methods"]:
        art = json.loads((tmp_path / f"{m['name']}.json").read_text())
        assert len(art["runs"]) == 2
        assert art["runs"][0]["true_delta_hz"] == art["true_deltas_hz"][0]
        assert len(art["git_sha"]) >= 7


def test_simulate_run_with_drift_tracks_moving_truth():
    from nvsim.estimators.adaptive import simulate_run

    cfg = json.loads(
        (REPO / "experiments/phase3b/policy_vs_aoptimal.json").read_text())
    cfg = {k: v for k, v in cfg.items() if k != "methods"}
    cfg["time_budget_s"] = 0.05
    cfg["drift"] = {"kind": "ou", "sigma_hz": 100e3, "tau_s": 50e-3}
    out = simulate_run(2.0e6, "exp_ladder", cfg, np.random.default_rng(9))
    truth = np.array(out["true_delta_hz_t"])
    assert truth.std() > 0                      # truth actually moved
    err = np.array(out["abs_err_hz"])
    n = len(err)
    assert np.median(err[n // 2:]) < 100e3      # tracked within sigma_drift
