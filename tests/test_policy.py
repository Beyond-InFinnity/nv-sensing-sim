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
