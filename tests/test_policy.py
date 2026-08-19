import numpy as np
import pytest

from nvsim.estimators.policy import VecRamseyEnv, tau_candidates
from nvsim.estimators.policy_net import (AmortizedPolicy, PolicyNet,
                                         collect_bc_dataset, train_bc)

READOUT = {"r_hz": 6e7, "contrast": 0.25, "t_read_s": 0.4e-6, "f_pump": 0.95}
CFG = {"n_shots_per_batch": 250, "time_budget_s": 0.004,
       "delta_range_hz": [0.3e6, 3.7e6], "t2star_s": 1.5e-6,
       "tau_min_s": 0.05e-6, "tau_max_s": 4.5e-6,
       "timing": {"t_init_s": 2.0e-6, "t_read_s": 0.4e-6, "t_dead_s": 1.0e-6},
       "readout": READOUT}


def test_policy_net_shapes_and_roundtrip(tmp_path):
    import torch
    net = PolicyNet(38, 30)
    x = torch.randn(7, 38)
    logits, value = net(x)
    assert logits.shape == (7, 30)
    assert value.shape == (7,)
    pol = AmortizedPolicy(net, {"n_features": 38, "n_actions": 30})
    a = pol.act(np.random.default_rng(0).standard_normal((5, 38)))
    assert a.shape == (5,) and a.dtype.kind == "i"
    assert np.all((a >= 0) & (a < 30))
    path = tmp_path / "pol.pt"
    pol.save(path)
    pol2 = AmortizedPolicy.load(path)
    np.testing.assert_array_equal(
        pol.act(np.zeros((3, 38))), pol2.act(np.zeros((3, 38))))


def test_bc_learns_a_simple_rule():
    """Training machinery sanity: a threshold-on-feature-0 teacher must be
    learnable to >90% held-out accuracy in seconds."""
    rng = np.random.default_rng(1)
    X = rng.standard_normal((2000, 38)).astype(np.float32)
    y = np.where(X[:, 0] > 0, 3, 17).astype(np.int64)
    pol, hist = train_bc(X, y, n_features=38, n_actions=30,
                         epochs=40, lr=1e-3, seed=0)
    assert hist["heldout_acc"] > 0.9


def test_collect_bc_dataset_labels_match_teacher_menu():
    rng = np.random.default_rng(2)
    X, y = collect_bc_dataset(CFG, n_episodes=2, rng=rng)
    n_cand = len(tau_candidates(CFG))
    env = VecRamseyEnv(CFG, 1, np.random.default_rng(0))
    assert X.shape[1] == env.n_features
    assert X.shape[0] == y.shape[0] > 0
    assert np.all((y >= 0) & (y < n_cand))
    # early decisions on a near-flat posterior must be short taus (Phase 3
    # finding: the teacher opens at ~0.12 us to keep the fringe unambiguous)
    first = tau_candidates(CFG)[y[0]]
    assert first < 0.5e-6


def test_rl_finetune_improves_return_from_random_init():
    """Smoke: 15 PPO iterations on a shrunken env must improve mean episode
    return from a random-init policy (machinery works end-to-end)."""
    import torch
    from nvsim.estimators.policy_net import finetune_rl
    torch.manual_seed(0)
    cfg = dict(CFG, time_budget_s=0.008)
    n_act = len(tau_candidates(cfg))
    env_probe = VecRamseyEnv(cfg, 1, np.random.default_rng(0), n_grid=200)
    net = PolicyNet(env_probe.n_features, n_act)
    pol = AmortizedPolicy(net, {"n_features": env_probe.n_features,
                                "n_actions": n_act})
    rl_cfg = {"n_envs": 32, "n_iters": 15, "lr": 3e-4, "clip": 0.2,
              "entropy_coef": 0.01, "value_coef": 0.5, "update_epochs": 3,
              "minibatch": 256, "n_grid": 200, "seed": 5}
    hist = finetune_rl(pol, cfg, rl_cfg, np.random.default_rng(5))
    early = np.mean(hist["mean_return"][:3])
    late = np.mean(hist["mean_return"][-3:])
    assert late > early


def test_policy_schedule_kind_in_simulate_run(tmp_path):
    from nvsim.estimators.adaptive import simulate_run
    from nvsim.estimators.policy_net import PolicyNet
    env = VecRamseyEnv(CFG, 1, np.random.default_rng(0))
    n_act = len(tau_candidates(CFG))
    pol = AmortizedPolicy(PolicyNet(env.n_features, n_act),
                          {"n_features": env.n_features, "n_actions": n_act})
    ckpt = tmp_path / "p.pt"
    pol.save(ckpt)
    cfg = dict(CFG, policy_ckpt=str(ckpt), time_budget_s=0.003)
    out = simulate_run(2.0e6, "policy", cfg, np.random.default_rng(3))
    cand = tau_candidates(CFG)
    for tau in out["tau_s"]:
        assert np.min(np.abs(cand - tau)) < 1e-15
    assert out["sigma_hz"][-1] < out["sigma_hz"][0]


def test_rl_stays_finite_on_drift_env():
    """Regression: drift-env fine-tuning must not NaN (crash 2026-08-19).
    Aggressive lr to provoke instability; guards must keep params finite."""
    import torch
    from nvsim.estimators.policy_net import finetune_rl
    torch.manual_seed(1)
    cfg = dict(CFG, time_budget_s=0.01)
    n_act = len(tau_candidates(cfg))
    env = VecRamseyEnv(cfg, 1, np.random.default_rng(0), n_grid=200)
    net = PolicyNet(env.n_features, n_act)
    pol = AmortizedPolicy(net, {"n_features": env.n_features,
                                "n_actions": n_act})
    rl_cfg = {"n_envs": 16, "n_iters": 10, "lr": 3e-3, "clip": 0.2,
              "entropy_coef": 0.01, "value_coef": 0.5, "update_epochs": 4,
              "minibatch": 64, "n_grid": 200, "seed": 6}
    drift = {"kind": "ou", "sigma_hz": 100e3, "tau_s": 50e-3}
    hist = finetune_rl(pol, cfg, rl_cfg, np.random.default_rng(6),
                       drift=drift)
    assert np.all(np.isfinite(hist["mean_return"]))
    for p in net.parameters():
        assert torch.isfinite(p).all()
