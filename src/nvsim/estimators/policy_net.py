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


def _teacher_action_row(p_row, grid, cfg):
    """choose_tau (the Phase 3 A-optimal rule) on one posterior row.

    Module-level pure function so a process pool can run rows in parallel
    (each decision is ~0.6 s of lookahead; rows are independent)."""
    from .adaptive import DeltaPosterior, choose_tau
    post = DeltaPosterior.__new__(DeltaPosterior)
    post.grid = grid
    post.p = p_row
    post.t2star_s = cfg["t2star_s"]
    post.readout_cfg = cfg["readout"]
    full_grid = np.geomspace(cfg["tau_min_s"], cfg["tau_max_s"], 60)
    tau = choose_tau(post, full_grid, cfg["n_shots_per_batch"])
    return int(np.argmin(np.abs(tau_candidates(cfg) - tau)))


def _teacher_action(env, row):
    return _teacher_action_row(env.p[row], env.grid, env.cfg)


def collect_bc_dataset(cfg, n_episodes, rng, drift=None, progress=None,
                       n_workers=1):
    """Roll episodes with the A-optimal teacher; log (features, action)."""
    from concurrent.futures import ProcessPoolExecutor
    X, y = [], []
    env = VecRamseyEnv(cfg, n_envs=n_episodes, rng=rng, drift=drift)
    feats = env.reset()
    pool = (ProcessPoolExecutor(max_workers=n_workers)
            if n_workers > 1 else None)
    try:
        while not env.done.all():
            active = np.nonzero(~env.done)[0]
            actions = np.zeros(env.n_envs, dtype=np.int64)
            if pool is not None:
                acts = list(pool.map(
                    _teacher_action_row,
                    (env.p[e] for e in active),
                    (env.grid for _ in active),
                    (env.cfg for _ in active), chunksize=4))
            else:
                acts = [_teacher_action(env, e) for e in active]
            for e, a in zip(active, acts):
                actions[e] = a
                X.append(feats[e])
                y.append(a)
            feats, _, _, _ = env.step(actions)
            if progress:
                progress(env.step_count, int(env.done.sum()))
    finally:
        if pool is not None:
            pool.shutdown()
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


def finetune_rl(policy, env_cfg, rl_cfg, rng, drift=None, progress=None):
    """PPO-style fine-tuning of an AmortizedPolicy on VecRamseyEnv.

    Full-episode rollouts from n_envs parallel environments; per-step
    advantage = reward-to-go minus the learned value head; clipped
    surrogate + value loss + linearly annealed entropy bonus. In-repo on
    purpose (~100 lines) — no RL library dependency."""
    torch.manual_seed(rl_cfg.get("seed", 0))
    net = policy.net.train()
    opt = torch.optim.Adam(net.parameters(), lr=rl_cfg["lr"])
    history = {"mean_return": [], "entropy": []}
    for it in range(rl_cfg["n_iters"]):
        env = VecRamseyEnv(env_cfg, rl_cfg["n_envs"], rng, drift=drift,
                           n_grid=rl_cfg.get("n_grid", 600))
        feats = env.reset()
        obs, acts, logps, rews, masks = [], [], [], [], []
        while not env.done.all():
            active = ~env.done
            with torch.no_grad():
                logits, _ = net(torch.as_tensor(feats, dtype=torch.float32))
                dist = torch.distributions.Categorical(logits=logits)
                a = dist.sample()
                lp = dist.log_prob(a)
            obs.append(feats.copy())
            acts.append(a.numpy())
            logps.append(lp.numpy())
            masks.append(active.copy())
            feats, r, _, _ = env.step(a.numpy())
            rews.append(r)
        rews = np.array(rews)                       # (T, E)
        rtg = np.flip(np.cumsum(np.flip(rews, 0), 0), 0).copy()
        m = np.array(masks)
        history["mean_return"].append(float(rews.sum(axis=0).mean()))

        flat = m.reshape(-1)
        O = torch.as_tensor(np.array(obs).reshape(-1, env.n_features)[flat],
                            dtype=torch.float32)
        A_ = torch.as_tensor(np.array(acts).reshape(-1)[flat])
        LP = torch.as_tensor(np.array(logps).reshape(-1)[flat],
                             dtype=torch.float32)
        R = torch.as_tensor(rtg.reshape(-1)[flat], dtype=torch.float32)
        # normalize returns per iteration: bounds the value target so the
        # value loss cannot blow up on long-episode (drift) returns
        R_mu, R_sd = R.mean(), R.std() + 1e-8
        Rn = (R - R_mu) / R_sd
        ent_coef = rl_cfg["entropy_coef"] * (1 - it / rl_cfg["n_iters"])
        for _ in range(rl_cfg["update_epochs"]):
            perm = torch.randperm(len(O))
            for i in range(0, len(perm), rl_cfg["minibatch"]):
                idx = perm[i:i + rl_cfg["minibatch"]]
                logits, value = net(O[idx])
                dist = torch.distributions.Categorical(logits=logits)
                adv = Rn[idx] - value.detach()
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                # clamp the log-ratio: exp() overflow across update epochs
                # was the NaN source in the drift run (2026-08-19)
                log_ratio = torch.clamp(dist.log_prob(A_[idx]) - LP[idx],
                                        -20.0, 20.0)
                ratio = torch.exp(log_ratio)
                clipped = torch.clamp(ratio, 1 - rl_cfg["clip"],
                                      1 + rl_cfg["clip"])
                pg = -torch.min(ratio * adv, clipped * adv).mean()
                vloss = torch.nn.functional.smooth_l1_loss(value, Rn[idx])
                ent = dist.entropy().mean()
                loss = (pg + rl_cfg["value_coef"] * vloss - ent_coef * ent)
                if not torch.isfinite(loss):
                    history.setdefault("skipped_minibatches", 0)
                    history["skipped_minibatches"] += 1
                    continue
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
        history["entropy"].append(float(ent))
        if progress:
            progress(it, history["mean_return"][-1])
    net.eval()
    return history


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
