"""
Shared on-policy multi-agent PPO loop (used by Methodologies 1 and 3).

The critic is passed in by the caller:
  methodology_1_marl/independent_marl_agent.py -> LocalCritics        (IPPO)
  methodology_3_ctde/qmix_maddpg_engine.py     -> CentralValueCritic  (MAPPO)

Everything else (actors, data, reward, hyper-parameters, seeds) is identical,
so any difference in stability / speed / final reward is attributable to the
critic seeing (or not seeing) the global state - i.e. to non-stationarity.
"""
import csv, os, random, time
import numpy as np
import torch

from environments.flow_splitting_env import FlowSplittingEnv
from models.decentralized_forwarder import DecentralizedForwarder

DEFAULTS = dict(updates=250, n_envs=16, rollout_len=32, episode_len=32, gamma=0.9, lam=0.9,
                lr_actor=1e-3, lr_critic=1e-3, epochs=4, minibatch=128, clip=0.2,
                max_grad_norm=0.5, hidden=64, critic_hidden=256, load_range=(0.5, 1.05),
                eval_every=25, eval_steps=64, scenario="pareto", seed=0)


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


class Slots:
    def __init__(self, G, cfg, rng):
        self.G, self.cfg, self.rng = G, cfg, rng
        self.envs = [FlowSplittingEnv(G, seed=i, scenario=cfg["scenario"], shadow_baseline=True)
                     for i in range(cfg["n_envs"])]
        self.t = np.zeros(cfg["n_envs"], int)
        for e in self.envs:
            self._new(e)

    def _new(self, e):
        e.reset(load=float(self.rng.uniform(*self.cfg["load_range"])), seed=int(self.rng.integers(1 << 31)))

    def observe(self):
        X, S = zip(*[e.observe_all() for e in self.envs])
        return torch.from_numpy(np.stack(X)), torch.from_numpy(np.stack(S).astype(np.float32))


def build(G, cfg, make_critic):
    """make_critic(env, cfg) -> critic module with forward(obs, state) -> (B, n) values."""
    env = FlowSplittingEnv(G)
    actor = DecentralizedForwarder(env.n, env.obs_dim, env.act_dim, cfg["hidden"])
    return env, actor, make_critic(env, cfg)


def evaluate(actor, G, cases, steps, scenario="pareto"):
    infos = []
    for load, seed in cases:
        env = FlowSplittingEnv(G, load=load, seed=seed, scenario=scenario)
        for _ in range(steps):
            X, _ = env.observe_all()
            with torch.no_grad():
                A = actor(torch.from_numpy(X)[None])[0].numpy()
            _, _, info = env.step_matrix(A)
            infos.append(info)
    return {k: float(np.mean([i[k] for i in infos])) for k in infos[0]}


def train(G, make_critic, out_dir, tag, **over):
    cfg = dict(DEFAULTS); cfg.update(over)
    os.makedirs(out_dir, exist_ok=True)
    set_seed(cfg["seed"]); torch.set_num_threads(1)
    rng = np.random.default_rng(cfg["seed"])
    env0, actor, crit = build(G, cfg, make_critic)
    mask = torch.from_numpy(env0.slot_mask.astype(np.float32))            # (n, D)
    opt_a = torch.optim.Adam(actor.parameters(), lr=cfg["lr_actor"])
    opt_c = torch.optim.Adam(crit.parameters(), lr=cfg["lr_critic"])
    slots = Slots(G, cfg, rng)
    K, T, n = cfg["n_envs"], cfg["rollout_len"], env0.n
    val_cases = [(float(l), 900 + i) for i, l in enumerate(np.linspace(*cfg["load_range"], 6))]
    f = open(os.path.join(out_dir, f"{tag}_training_curve.csv"), "w", newline="")
    w = csv.writer(f)
    w.writerow(["update", "env_steps", "wall_clock_s", "train_reward", "train_loss_ratio", "train_delay_ms",
                "pi_loss", "v_loss", "explained_var", "approx_kl", "clip_frac", "val_reward"])
    best, t0 = -1e9, time.time()
    for u in range(1, cfg["updates"] + 1):
        frac = 1 - (u - 1) / cfg["updates"]
        for g in opt_a.param_groups: g["lr"] = cfg["lr_actor"] * frac
        for g in opt_c.param_groups: g["lr"] = cfg["lr_critic"] * frac
        OB, ST, AC, LP, V, R, NV, DN, INF = [], [], [], [], [], [], [], [], []
        X, S = slots.observe()
        for _ in range(T):
            with torch.no_grad():
                d = actor.dist(X)
                a = d.sample()
                lp = (d.log_prob(a) * mask).sum(-1)                         # (K, n)
                v = crit(X, S)
            rs, dn = [], []
            for k, e in enumerate(slots.envs):
                _, r, info = e.step_matrix(a[k].numpy())
                INF.append(info); rs.append(r)
                slots.t[k] += 1
                dn.append(slots.t[k] >= cfg["episode_len"])
            X2, S2 = slots.observe()
            with torch.no_grad():
                nv = crit(X2, S2)
            OB.append(X); ST.append(S); AC.append(a); LP.append(lp); V.append(v)
            R.append(torch.tensor(rs, dtype=torch.float32)); NV.append(nv); DN.append(torch.tensor(dn, dtype=torch.float32))
            for k, d_ in enumerate(dn):
                if d_:
                    slots._new(slots.envs[k]); slots.t[k] = 0
            X, S = slots.observe() if any(dn) else (X2, S2)
        V, NV, R, DN = torch.stack(V), torch.stack(NV), torch.stack(R), torch.stack(DN)   # (T,K,n) / (T,K)
        adv = torch.zeros_like(V); last = torch.zeros(K, n)
        for t in reversed(range(T)):
            delta = R[t][:, None] + cfg["gamma"] * NV[t] - V[t]
            last = delta + cfg["gamma"] * cfg["lam"] * (1 - DN[t])[:, None] * last
            adv[t] = last
        ret = adv + V
        ev = float(1 - (ret - V).var() / (ret.var() + 1e-8))
        flat = lambda x: x.reshape(T * K, *x.shape[2:])
        OB, ST, AC, LP, ADV, RET = map(flat, (torch.stack(OB), torch.stack(ST), torch.stack(AC), torch.stack(LP), adv, ret))
        ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-8)
        stats = []
        for _ in range(cfg["epochs"]):
            perm = torch.randperm(T * K)
            for i in range(0, T * K, cfg["minibatch"]):
                idx = perm[i:i + cfg["minibatch"]]
                d = actor.dist(OB[idx])
                lp = (d.log_prob(AC[idx]) * mask).sum(-1)
                ratio = torch.exp(lp - LP[idx])
                A_ = ADV[idx]
                pi = -torch.min(ratio * A_, torch.clamp(ratio, 1 - cfg["clip"], 1 + cfg["clip"]) * A_).mean()
                vl = ((crit(OB[idx], ST[idx]) - RET[idx]) ** 2).mean()
                opt_a.zero_grad(); pi.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), cfg["max_grad_norm"]); opt_a.step()
                opt_c.zero_grad(); vl.backward()
                torch.nn.utils.clip_grad_norm_(crit.parameters(), cfg["max_grad_norm"]); opt_c.step()
                with torch.no_grad():
                    stats.append((pi.item(), vl.item(), (LP[idx] - lp).mean().item(),
                                  ((ratio - 1).abs() > cfg["clip"]).float().mean().item()))
        st = np.mean(stats, 0)
        vr = ""
        if u % cfg["eval_every"] == 0 or u == cfg["updates"]:
            vr = evaluate(actor, G, val_cases, cfg["eval_steps"], cfg["scenario"])["reward"]
            if vr > best:
                best = vr; torch.save(actor.state_dict(), os.path.join(out_dir, f"{tag}_actor_best.pt"))
        tr = np.mean([i["reward_raw"] for i in INF])
        w.writerow([u, u * T * K, round(time.time() - t0, 1), tr, np.mean([i["loss_ratio"] for i in INF]),
                    np.mean([i["avg_delay_ms"] for i in INF]), st[0], st[1], ev, st[2], st[3], vr]); f.flush()
        if u % 10 == 0 or u == 1:
            print(f"[{tag}] upd {u:4d} R {tr:+.4f} EV {ev:+.2f} kl {st[2]:.4f} clip {st[3]:.2f} val {vr} "
                  f"{time.time()-t0:.0f}s", flush=True)
    torch.save(actor.state_dict(), os.path.join(out_dir, f"{tag}_actor_last.pt"))
    f.close()
    return best
