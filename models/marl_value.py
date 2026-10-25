"""
Shared value-based (Q-learning) multi-agent machinery for Methodologies 1 and 3.

Discrete per-switch action set (D = max out-degree):
    0          keep        : no change (delay-based softmin split)
    1 .. D     avoid k     : +step on outgoing link k's cost residual (shift traffic away)
    D+1 .. 2D  prefer k    : -step on outgoing link k's cost residual (pull traffic onto it)
Actions that refer to a link the switch does not have are masked.

Used by
  methodology_1_marl/independent_dqn.py  -> independent DQN per switch (IQL)
  methodology_1_marl/tabular_q.py        -> tabular Q-learning per switch
  methodology_3_ctde/qmix_maddpg_engine  -> QMIX (same agent nets + monotonic mixer)
IQL and QMIX share this loop line for line; only the TD target differs
(per-agent vs mixed through Q_tot), isolating the cost of non-stationarity.
"""
import copy, csv, os, random, time
import numpy as np
import torch
import torch.nn as nn

from environments.flow_splitting_env import FlowSplittingEnv
from models.decentralized_forwarder import GroupedMLP


VALUE_DEFAULTS = dict(iters=6000, n_envs=8, episode_len=32, gamma=0.9, batch=128, buffer=60_000,
                      lr=5e-4, hidden=64, eps_start=1.0, eps_end=0.05, eps_frac=0.6, tau=0.01,
                      warmup=200, load_range=(0.5, 1.05), eval_every=1000, eval_steps=64,
                      scenario="pareto", seed=0, max_grad_norm=1.0, mixer_embed=32, action_step=0.75)


class DiscreteActions:
    def __init__(self, env, step=0.75):
        """step = size of the cost change for avoid (+step) / prefer (-step)."""
        self.n, self.D = env.n, env.act_dim
        self.A = 1 + 2 * self.D
        tab = np.zeros((self.A, self.D), np.float32)
        for k in range(self.D):
            tab[1 + k, k] = step
            tab[1 + self.D + k, k] = -step
        self.table = tab
        present = env.slot_mask                                      # (n, D)
        self.valid = np.ones((self.n, self.A), bool)
        self.valid[:, 1:1 + self.D] = present
        self.valid[:, 1 + self.D:] = present
        self.valid_t = torch.from_numpy(self.valid)

    def to_continuous(self, a):                                      # a: (n,) ints -> (n, D)
        return self.table[np.asarray(a)]

    def random(self, rng):
        return np.array([rng.choice(np.nonzero(v)[0]) for v in self.valid])

    def masked(self, q):                                             # q (..., n, A)
        return q.masked_fill(~self.valid_t, -1e9)


class QAgents(nn.Module):
    """One Q-network PER SWITCH (independent weights, batched): Q_v(o_v, .)."""

    def __init__(self, n, obs_dim, n_actions, hidden=64):
        super().__init__()
        self.net = GroupedMLP(n, obs_dim, hidden, n_actions)

    def forward(self, obs):                                          # (B, n, obs) -> (B, n, A)
        return self.net(obs)


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def evaluate_policy(G, act_fn, cases, steps, scenario="pareto"):
    """act_fn(env) -> continuous (n, D) action matrix."""
    infos = []
    for load, seed in cases:
        env = FlowSplittingEnv(G, load=load, seed=seed, scenario=scenario)
        for _ in range(steps):
            _, _, info = env.step_matrix(act_fn(env))
            infos.append(info)
    return {k: float(np.mean([i[k] for i in infos])) for k in infos[0]}


def greedy_fn(agents, acts):
    def act(env):
        X, _ = env.observe_all()
        with torch.no_grad():
            q = acts.masked(agents(torch.from_numpy(X)[None]))[0]
        return acts.to_continuous(q.argmax(-1).numpy())
    return act


class Replay:
    def __init__(self, cap, n, od, sd):
        self.X = np.zeros((cap, n, od), np.float32); self.X2 = np.zeros_like(self.X)
        self.S = np.zeros((cap, sd), np.float32); self.S2 = np.zeros_like(self.S)
        self.A = np.zeros((cap, n), np.int64); self.R = np.zeros(cap, np.float32)
        self.cap, self.i, self.full = cap, 0, False

    def add(self, x, s, a, r, x2, s2):
        j = self.i
        self.X[j], self.S[j], self.A[j], self.R[j], self.X2[j], self.S2[j] = x, s, a, r, x2, s2
        self.i = (j + 1) % self.cap; self.full |= self.i == 0

    def sample(self, b, rng):
        idx = rng.integers(0, self.cap if self.full else self.i, b)
        return tuple(torch.from_numpy(z[idx]) for z in (self.X, self.S, self.A, self.R, self.X2, self.S2))


def train_value(G, out_dir, tag, make_mixer=None, **over):
    """Independent DQN (make_mixer=None) or QMIX (make_mixer(env, cfg) -> mixer).
    Double-DQN targets, soft target updates, team reward differenced against a
    counterfactual OSPF network on identical traffic (unbiased variance reduction)."""
    cfg = dict(VALUE_DEFAULTS); cfg.update(over)
    os.makedirs(out_dir, exist_ok=True)
    set_seed(cfg["seed"]); torch.set_num_threads(1)
    rng = np.random.default_rng(cfg["seed"])
    envs = [FlowSplittingEnv(G, seed=i, scenario=cfg["scenario"], shadow_baseline=True) for i in range(cfg["n_envs"])]
    e0 = envs[0]
    acts = DiscreteActions(e0, cfg["action_step"])
    agents = QAgents(e0.n, e0.obs_dim, acts.A, cfg["hidden"])
    mixer = make_mixer(e0, cfg) if make_mixer else None
    agents_t = copy.deepcopy(agents)
    mixer_t = copy.deepcopy(mixer) if mixer is not None else None
    params = list(agents.parameters()) + (list(mixer.parameters()) if mixer is not None else [])
    opt = torch.optim.Adam(params, lr=cfg["lr"])
    buf = Replay(cfg["buffer"], e0.n, e0.obs_dim, e0.state_dim)
    tsteps = np.zeros(cfg["n_envs"], int)

    def new_ep(e):
        e.reset(load=float(rng.uniform(*cfg["load_range"])), seed=int(rng.integers(1 << 31)))
    for e in envs:
        new_ep(e)
    obs = [e.observe_all() for e in envs]
    val_cases = [(float(l), 900 + i) for i, l in enumerate(np.linspace(*cfg["load_range"], 6))]
    f = open(os.path.join(out_dir, f"{tag}_training_curve.csv"), "w", newline="")
    w = csv.writer(f)
    w.writerow(["iter", "env_steps", "wall_clock_s", "train_reward", "train_loss_ratio", "td_loss", "epsilon", "val_reward"])
    best, t0, INF, tl = -1e9, time.time(), [], []
    for it in range(1, cfg["iters"] + 1):
        eps = max(cfg["eps_end"], cfg["eps_start"] - (cfg["eps_start"] - cfg["eps_end"]) * it / (cfg["eps_frac"] * cfg["iters"]))
        X = torch.from_numpy(np.stack([o[0] for o in obs]))
        with torch.no_grad():
            greedy = acts.masked(agents(X)).argmax(-1).numpy()                     # (K, n)
        for k, e in enumerate(envs):
            a = greedy[k].copy()
            explore = rng.random(e0.n) < eps
            if explore.any():
                a[explore] = acts.random(rng)[explore]
            _, r, info = e.step_matrix(acts.to_continuous(a))
            o2 = e.observe_all()
            buf.add(obs[k][0], obs[k][1], a, r, o2[0], o2[1])                       # time-limit only: bootstrap
            INF.append(info)
            tsteps[k] += 1
            if tsteps[k] >= cfg["episode_len"]:
                new_ep(e); tsteps[k] = 0; o2 = e.observe_all()
            obs[k] = o2
        if it > cfg["warmup"]:
            Xb, Sb, Ab, Rb, X2b, S2b = buf.sample(cfg["batch"], rng)
            q = agents(Xb).gather(-1, Ab[..., None]).squeeze(-1)                   # (B, n)
            with torch.no_grad():
                a2 = acts.masked(agents(X2b)).argmax(-1, keepdim=True)             # double DQN: online picks
                q2 = agents_t(X2b).gather(-1, a2).squeeze(-1)                       # target evaluates
            if mixer is None:                                                       # IQL: each agent alone
                y = Rb[:, None] + cfg["gamma"] * q2
                loss = ((q - y) ** 2).mean()
            else:                                                                   # QMIX: through Q_tot(s)
                with torch.no_grad():
                    y = Rb + cfg["gamma"] * mixer_t(q2, S2b)
                loss = ((mixer(q, Sb) - y) ** 2).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, cfg["max_grad_norm"]); opt.step()
            with torch.no_grad():
                pairs = list(zip(agents.parameters(), agents_t.parameters()))
                if mixer is not None:
                    pairs += list(zip(mixer.parameters(), mixer_t.parameters()))
                for p, pt in pairs:
                    pt.mul_(1 - cfg["tau"]).add_(cfg["tau"] * p)
            tl.append(loss.item())
        if it % 100 == 0 or it == cfg["iters"]:
            vr = ""
            if it % cfg["eval_every"] == 0 or it == cfg["iters"]:
                vr = evaluate_policy(G, greedy_fn(agents, acts), val_cases, cfg["eval_steps"], cfg["scenario"])["reward"]
                if vr > best:
                    best = vr; torch.save(agents.state_dict(), os.path.join(out_dir, f"{tag}_agents_best.pt"))
            w.writerow([it, it * cfg["n_envs"], round(time.time() - t0, 1), np.mean([i["reward_raw"] for i in INF]),
                        np.mean([i["loss_ratio"] for i in INF]), np.mean(tl) if tl else "", round(eps, 4), vr]); f.flush()
            if it % 1000 == 0:
                print(f"[{tag}] it {it} R {np.mean([i['reward_raw'] for i in INF]):+.4f} td {np.mean(tl) if tl else 0:.4f} "
                      f"eps {eps:.2f} val {vr} {time.time()-t0:.0f}s", flush=True)
            INF, tl = [], []
    torch.save(agents.state_dict(), os.path.join(out_dir, f"{tag}_agents_last.pt"))
    f.close()
    return best
