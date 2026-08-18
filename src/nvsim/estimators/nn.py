"""1D-CNN Ramsey estimator with heteroscedastic Gaussian head.

torch imported at module top — import this module only under the `.[ml]`
extra; the rest of nvsim stays torch-free."""
import numpy as np
import torch
import torch.nn as nn

from .model import expected_counts


def make_batch(cfg, batch, rng):
    """Sample (theta, n_shots), simulate Poisson records, return tensors."""
    t = cfg["taus"]
    taus = np.linspace(t["min"], t["max"], t["n_points"])
    d_lo, d_hi = cfg["delta_range_hz"]
    deltas = rng.uniform(d_lo, d_hi, batch)
    t2ss = rng.uniform(*cfg["t2s_range_s"], batch)
    n_shots = np.exp(rng.uniform(*np.log(cfg["n_shots_range"]), batch))
    xs = np.empty((batch, 2, len(taus)), dtype=np.float32)
    for i in range(batch):
        lam = expected_counts(taus, deltas[i], t2ss[i], cfg["readout"],
                              n_shots[i])
        c = rng.poisson(lam).astype(np.float64)
        xs[i, 0] = c / max(c.mean(), 1.0) - 1.0
        xs[i, 1] = taus / t["max"]
    aux = (np.log10(n_shots) / 5.0).astype(np.float32)[:, None]
    y = ((deltas - d_lo) / (d_hi - d_lo)).astype(np.float32)
    return torch.from_numpy(xs), torch.from_numpy(aux), torch.from_numpy(y)


class RamseyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(2, 32, 7, padding=3), nn.ReLU(),
            nn.Conv1d(32, 32, 7, stride=2, padding=3), nn.ReLU(),
            nn.Conv1d(32, 32, 7, stride=2, padding=3), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1))
        self.head = nn.Sequential(nn.Linear(33, 64), nn.ReLU(), nn.Linear(64, 2))

    def forward(self, x, aux):
        z = self.conv(x).squeeze(-1)
        out = self.head(torch.cat([z, aux], dim=1))
        return out[:, 0], out[:, 1].clamp(-7, 2)   # mu_norm, log_sigma_norm


class NNEstimator:
    def __init__(self, net, cfg):
        self.net, self.cfg = net.eval(), cfg

    def save(self, path):
        torch.save({"state": self.net.state_dict(), "cfg": self.cfg}, path)

    @classmethod
    def load(cls, path):
        blob = torch.load(path, map_location="cpu", weights_only=False)
        net = RamseyNet()
        net.load_state_dict(blob["state"])
        return cls(net, blob["cfg"])

    def infer(self, counts, taus_s, readout_cfg, n_shots):
        c = np.asarray(counts, dtype=np.float64)
        t = self.cfg["taus"]
        x = np.stack([c / max(c.mean(), 1.0) - 1.0,
                      np.asarray(taus_s) / t["max"]]).astype(np.float32)[None]
        aux = np.array([[np.log10(n_shots) / 5.0]], dtype=np.float32)
        with torch.no_grad():
            mu, log_sigma = self.net(torch.from_numpy(x), torch.from_numpy(aux))
        d_lo, d_hi = self.cfg["delta_range_hz"]
        span = d_hi - d_lo
        return {"delta_hz": float(d_lo + mu.item() * span),
                "delta_sigma_hz": float(np.exp(log_sigma.item()) * span)}
