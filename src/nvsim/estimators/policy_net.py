"""Neural policy for adaptive Ramsey (Phase 3b).

torch imported at module top — import this module only when training or
evaluating a policy; the env (policy.py) and the rest of nvsim stay
torch-free (same convention as estimators/nn.py)."""
import numpy as np
import torch
import torch.nn as nn

from .policy import VecRamseyEnv, tau_candidates


class PolicyNet(nn.Module):
    """MLP trunk with action logits and a scalar value head (for RL)."""

    def __init__(self, n_features, n_actions):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(n_features, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU())
        self.pi = nn.Linear(128, n_actions)
        self.v = nn.Linear(128, 1)

    def forward(self, x):
        z = self.trunk(x)
        return self.pi(z), self.v(z).squeeze(-1)


class AmortizedPolicy:
    def __init__(self, net, cfg):
        self.net = net.eval()
        self.cfg = cfg

    def act(self, features):
        """Greedy action per row; features (E, F) numpy -> (E,) int."""
        with torch.no_grad():
            logits, _ = self.net(torch.as_tensor(features, dtype=torch.float32))
        return logits.argmax(dim=1).numpy().astype(np.int64)

    def save(self, path):
        torch.save({"state": self.net.state_dict(), "cfg": self.cfg}, path)

    @classmethod
    def load(cls, path):
        blob = torch.load(path, map_location="cpu", weights_only=False)
        net = PolicyNet(blob["cfg"]["n_features"], blob["cfg"]["n_actions"])
        net.load_state_dict(blob["state"])
        return cls(net, blob["cfg"])


def _teacher_action(env, row):
    """choose_tau (the Phase 3 A-optimal rule) on one env row's posterior."""
    from .adaptive import DeltaPosterior, choose_tau
    post = DeltaPosterior.__new__(DeltaPosterior)
    post.grid = env.grid
    post.p = env.p[row]
    post.t2star_s = env.cfg["t2star_s"]
    post.readout_cfg = env.cfg["readout"]
    full_grid = np.geomspace(env.cfg["tau_min_s"], env.cfg["tau_max_s"], 60)
    tau = choose_tau(post, full_grid, env.cfg["n_shots_per_batch"])
    return int(np.argmin(np.abs(env.tau_candidates_s - tau)))


def collect_bc_dataset(cfg, n_episodes, rng, drift=None, progress=None):
    """Roll episodes with the A-optimal teacher; log (features, action)."""
    X, y = [], []
    env = VecRamseyEnv(cfg, n_envs=n_episodes, rng=rng, drift=drift)
    feats = env.reset()
    while not env.done.all():
        actions = np.zeros(env.n_envs, dtype=np.int64)
        for e in np.nonzero(~env.done)[0]:
            actions[e] = _teacher_action(env, e)
            X.append(feats[e])
            y.append(actions[e])
        feats, _, _, _ = env.step(actions)
        if progress:
            progress(env.step_count, int(env.done.sum()))
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


def train_bc(X, y, n_features, n_actions, epochs=20, lr=1e-3, seed=0,
             batch_size=512, heldout_frac=0.1):
    """Behavior cloning: cross-entropy on (features, teacher action)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X))
    n_hold = max(1, int(heldout_frac * len(X)))
    hold, train = perm[:n_hold], perm[n_hold:]
    Xt = torch.as_tensor(X[train]), torch.as_tensor(y[train])
    Xh = torch.as_tensor(X[hold]), torch.as_tensor(y[hold])
    net = PolicyNet(n_features, n_actions)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    history = {"train_loss": [], "heldout_acc": 0.0}
    for _ in range(epochs):
        order = torch.randperm(len(train))
        total = 0.0
        for i in range(0, len(order), batch_size):
            idx = order[i:i + batch_size]
            logits, _ = net(Xt[0][idx])
            loss = loss_fn(logits, Xt[1][idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss) * len(idx)
        history["train_loss"].append(total / len(order))
    with torch.no_grad():
        logits, _ = net(Xh[0])
        history["heldout_acc"] = float((logits.argmax(1) == Xh[1])
                                       .float().mean())
    return AmortizedPolicy(net, {"n_features": n_features,
                                 "n_actions": n_actions}), history
