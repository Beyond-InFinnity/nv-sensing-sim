import numpy as np
import pytest

torch = pytest.importorskip("torch")

from nvsim.estimators.nn import NNEstimator, RamseyNet, make_batch  # noqa: E402

READOUT = {"r_hz": 6e7, "contrast": 0.25, "t_read_s": 0.4e-6, "f_pump": 0.95}
CFG = {"seed": 1, "taus": {"min": 0.0, "max": 5e-6, "n_points": 150},
       "readout": READOUT, "delta_range_hz": [0.3e6, 3.8e6],
       "t2s_range_s": [0.8e-6, 4.0e-6], "n_shots_range": [20, 20000]}


def test_make_batch_shapes_and_ranges():
    x, aux, y = make_batch(CFG, 32, np.random.default_rng(0))
    assert x.shape == (32, 2, 150) and aux.shape == (32, 1) and y.shape == (32,)
    assert torch.all((y >= 0) & (y <= 1))


def test_loss_decreases_on_tiny_train():
    torch.manual_seed(0)
    net = RamseyNet()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    rng = np.random.default_rng(0)
    losses = []
    for _ in range(60):
        x, aux, y = make_batch(CFG, 64, rng)
        mu, log_sigma = net(x, aux)
        loss = (log_sigma + 0.5 * ((y - mu) / log_sigma.exp()) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        losses.append(float(loss))
    assert np.mean(losses[-10:]) < np.mean(losses[:10])


def test_infer_roundtrip(tmp_path):
    net = RamseyNet()
    est = NNEstimator(net, CFG)
    p = tmp_path / "ckpt.pt"
    est.save(p)
    est2 = NNEstimator.load(p)
    taus = np.linspace(0, 5e-6, 150)
    counts = np.random.default_rng(0).poisson(2000 * 24.0, 150)
    out = est2.infer(counts, taus, READOUT, 2000)
    assert set(out) >= {"delta_hz", "delta_sigma_hz"}
    assert 0 < out["delta_hz"] < 5e6
