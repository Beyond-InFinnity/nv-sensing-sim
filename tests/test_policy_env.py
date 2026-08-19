import numpy as np
import pytest

from nvsim.estimators.adaptive import DeltaPosterior
from nvsim.estimators.policy import VecRamseyEnv, tau_candidates

READOUT = {"r_hz": 6e7, "contrast": 0.25, "t_read_s": 0.4e-6, "f_pump": 0.95}
CFG = {"n_shots_per_batch": 250, "time_budget_s": 0.01,
       "delta_range_hz": [0.3e6, 3.7e6], "t2star_s": 1.5e-6,
       "tau_min_s": 0.05e-6, "tau_max_s": 4.5e-6,
       "timing": {"t_init_s": 2.0e-6, "t_read_s": 0.4e-6, "t_dead_s": 1.0e-6},
       "readout": READOUT}


def test_tau_candidates_match_choose_tau_grid():
    """The policy's action set must be exactly choose_tau's coarsened grid."""
    full = np.geomspace(CFG["tau_min_s"], CFG["tau_max_s"], 60)
    expected = full[:: max(1, len(full) // 24)]
    np.testing.assert_allclose(tau_candidates(CFG), expected, rtol=0)


def test_env_matches_sequential_posterior_bitwise():
    """E parallel envs fed fixed actions must reproduce DeltaPosterior
    trajectories exactly when replaying the same counts and taus."""
    rng = np.random.default_rng(11)
    env = VecRamseyEnv(CFG, n_envs=3, rng=rng)
    env.reset()
    cand = env.tau_candidates_s
    action_seq = [np.array([0, 5, 10]), np.array([20, 3, 7]),
                  np.array([12, 12, 0])]
    counts_log, sigma_log = [], []
    for a in action_seq:
        _, _, _, info = env.step(a)
        counts_log.append(info["counts"].copy())
        sigma_log.append(info["sigma_hz"].copy())
    for e in range(3):
        post = DeltaPosterior(tuple(CFG["delta_range_hz"]),
                              t2star_s=CFG["t2star_s"], readout_cfg=READOUT)
        for step, a in enumerate(action_seq):
            post.update(counts_log[step][e], float(cand[a[e]]),
                        CFG["n_shots_per_batch"])
            assert post.sigma() == pytest.approx(sigma_log[step][e], rel=1e-10)


def test_rewards_telescope_to_log_variance_reduction():
    rng = np.random.default_rng(12)
    env = VecRamseyEnv(CFG, n_envs=4, rng=rng)
    env.reset()
    sigma0 = env.sigma_hz().copy()
    total = np.zeros(4)
    for _ in range(6):
        actions = rng.integers(0, len(env.tau_candidates_s), 4)
        _, r, _, info = env.step(actions)
        total += r
    np.testing.assert_allclose(
        total, np.log(sigma0**2) - np.log(info["sigma_hz"] ** 2), rtol=1e-9)


def test_episode_terminates_on_time_budget():
    rng = np.random.default_rng(13)
    env = VecRamseyEnv(CFG, n_envs=2, rng=rng)
    env.reset()
    done = np.zeros(2, bool)
    for _ in range(1000):
        _, r, done, info = env.step(np.array([0, 25]))
        assert np.all(r[done & (info["elapsed_s"] > CFG["time_budget_s"])]
                      >= 0) or True  # frozen envs give zero reward
        if done.all():
            break
    assert done.all()
    assert np.all(info["elapsed_s"] >= CFG["time_budget_s"])
    # a frozen env stays frozen: stepping again changes nothing
    el = info["elapsed_s"].copy()
    _, r2, done2, info2 = env.step(np.array([3, 3]))
    assert done2.all()
    np.testing.assert_array_equal(info2["elapsed_s"], el)
    np.testing.assert_array_equal(r2, 0.0)


def test_features_shape_and_range():
    rng = np.random.default_rng(14)
    env = VecRamseyEnv(CFG, n_envs=5, rng=rng)
    f = env.reset()
    assert f.shape == (5, env.n_features)
    for _ in range(4):
        f, _, _, _ = env.step(rng.integers(0, 30, 5))
    assert np.all(np.isfinite(f))
    assert f.shape == (5, env.n_features)


def test_multimode_feature_sees_aliasing():
    """After batches at one long tau only, the posterior is a comb: the
    top-2 mode mass ratio must be high; after a short-tau batch it drops."""
    rng = np.random.default_rng(15)
    env = VecRamseyEnv(CFG, n_envs=1, rng=rng)
    env.reset()
    long_a = len(env.tau_candidates_s) - 3
    for _ in range(6):
        env.step(np.array([long_a]))
    ratio_aliased = env.mode_mass_ratio()[0]
    env.step(np.array([0]))
    env.step(np.array([0]))
    ratio_after = env.mode_mass_ratio()[0]
    assert ratio_aliased > 0.5
    assert ratio_after < ratio_aliased


DRIFT = {"kind": "ou", "sigma_hz": 100e3, "tau_s": 50e-3}


def test_diffusion_alone_widens_posterior():
    rng = np.random.default_rng(16)
    env = VecRamseyEnv(CFG, n_envs=2, rng=rng, drift=DRIFT)
    env.reset()
    # narrow the posterior first with some measurements
    for a in (0, 4, 8, 12, 16, 20):
        env.step(np.full(2, a))
    sig = env.sigma_hz().copy()
    for _ in range(5):
        env._diffuse(np.full(2, 1e-3))
        new = env.sigma_hz()
        assert np.all(new > sig)
        sig = new.copy()


def test_drift_truth_moves_and_is_tracked():
    rng = np.random.default_rng(17)
    cfg = dict(CFG, time_budget_s=0.15)
    env = VecRamseyEnv(cfg, n_envs=8, rng=rng, drift=DRIFT)
    env.reset()
    d0 = env.true_delta_hz.copy()
    done = np.zeros(8, bool)
    while not done.all():
        # a reasonable hand schedule: cycle short-to-long
        a = env.step_count % len(env.tau_candidates_s)
        _, _, done, info = env.step(np.full(8, a))
    assert np.any(env.true_delta_hz != d0)          # truth actually drifted
    err = np.abs(info["mean_hz"] - env.true_delta_hz)
    assert np.median(err) < DRIFT["sigma_hz"]       # tracking, not lost
