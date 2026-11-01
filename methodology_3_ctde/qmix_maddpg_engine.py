"""
Methodology 3: Centralised Training with Decentralised Execution (CTDE).

* MAPPO  - decentralised Gaussian actors + centralised V(s) critic
           (shared loop models/marl_onpolicy.py + CentralValueCritic).
* MADDPG - decentralised deterministic actors + centralised Q(s, a_1..a_n)
           critic, off-policy with replay and target networks.

QMIX: factorises a *discrete* joint Q-function through a monotonic mixer, so it
runs on a discrete per-switch action set (keep / avoid link k / prefer link k,
see models/marl_value.py) and is compared against Methodology 1's independent
DQN on that same action set. MAPPO / MADDPG keep the continuous action space.

At execution both critics are discarded: each switch runs only its own actor
on its local observation.
"""
import copy, csv, os, time
import numpy as np
import torch

from environments.flow_splitting_env import FlowSplittingEnv
from models.decentralized_forwarder import DecentralizedForwarder
from methodology_3_ctde.centralized_critic import CentralValueCritic, CentralQCritic
from models.marl_onpolicy import train as train_onpolicy, evaluate, set_seed


def make_central_critic(env, cfg):
    return CentralValueCritic(env.n, env.state_dim, cfg["critic_hidden"])


def train_mappo(G, out_dir, tag="mappo", **over):
    return train_onpolicy(G, make_central_critic, out_dir, tag, **over)


def make_qmix_mixer(env, cfg):
    from methodology_3_ctde.qmix_mixer import QMixer
    return QMixer(env.n, env.state_dim, cfg["mixer_embed"])


def train_qmix(G, out_dir, tag="qmix", **over):
    """QMIX on the discrete per-switch action set (models/marl_value.py). Identical to
    Methodology 1's independent DQN except that TD targets go through the mixer."""
    from models.marl_value import train_value
    return train_value(G, out_dir, tag, make_mixer=make_qmix_mixer, **over)


MADDPG_DEFAULTS = dict(iters=5000, n_envs=8, episode_len=32, gamma=0.9, tau=0.01, batch=256,
                       buffer=100_000, lr_actor=1e-4, lr_critic=1e-3, hidden=64, critic_hidden=256,
                       noise_start=0.3, noise_end=0.05, warmup=50, load_range=(0.5, 1.05),
                       eval_every=500, eval_steps=64, scenario="pareto", seed=0, max_grad_norm=1.0,
                       # TD3-style stabilisers (vanilla MADDPG diverged: actor exploited Q errors
                       # and pushed every link cost to the +-3 bounds within 500 iterations)
                       twin=True, policy_delay=2, actor_warmup=1000, target_noise=0.2,
                       target_noise_clip=0.5, act_reg=0.05)


class Replay:
    def __init__(self, cap, n, od, sd, D):
        self.X = np.zeros((cap, n, od), np.float32); self.X2 = np.zeros_like(self.X)
        self.S = np.zeros((cap, sd), np.float32); self.S2 = np.zeros_like(self.S)
        self.A = np.zeros((cap, n, D), np.float32); self.R = np.zeros(cap, np.float32)
        self.cap, self.i, self.full = cap, 0, False

    def add(self, x, s, a, r, x2, s2):
        j = self.i
        self.X[j], self.S[j], self.A[j], self.R[j], self.X2[j], self.S2[j] = x, s, a, r, x2, s2
        self.i = (j + 1) % self.cap; self.full |= self.i == 0

    def sample(self, b, rng):
        idx = rng.integers(0, self.cap if self.full else self.i, b)
        t = lambda z: torch.from_numpy(z[idx])
        return t(self.X), t(self.S), t(self.A), t(self.R), t(self.X2), t(self.S2)


def train_maddpg(G, out_dir, tag="maddpg", **over):
    cfg = dict(MADDPG_DEFAULTS); cfg.update(over)
    os.makedirs(out_dir, exist_ok=True)
    set_seed(cfg["seed"]); torch.set_num_threads(1)
    rng = np.random.default_rng(cfg["seed"])
    envs = [FlowSplittingEnv(G, seed=i, scenario=cfg["scenario"], shadow_baseline=True) for i in range(cfg["n_envs"])]
    e0 = envs[0]
    n, D = e0.n, e0.act_dim
    mask_np = e0.slot_mask.astype(np.float32)
    mask = torch.from_numpy(mask_np)
    actor = DecentralizedForwarder(n, e0.obs_dim, D, cfg["hidden"], deterministic=True)
    critic = CentralQCritic(e0.state_dim, n * D, cfg["critic_hidden"])
    critic2 = CentralQCritic(e0.state_dim, n * D, cfg["critic_hidden"]) if cfg["twin"] else None
    actor_t, critic_t = copy.deepcopy(actor), copy.deepcopy(critic)
    critic2_t = copy.deepcopy(critic2) if critic2 is not None else None
    opt_a = torch.optim.Adam(actor.parameters(), lr=cfg["lr_actor"])
    c_params = list(critic.parameters()) + (list(critic2.parameters()) if critic2 is not None else [])
    opt_c = torch.optim.Adam(c_params, lr=cfg["lr_critic"])
    buf = Replay(cfg["buffer"], n, e0.obs_dim, e0.state_dim, D)
    tsteps = np.zeros(cfg["n_envs"], int)

    def new_ep(e):
        e.reset(load=float(rng.uniform(*cfg["load_range"])), seed=int(rng.integers(1 << 31)))
    for e in envs:
        new_ep(e)
    val_cases = [(float(l), 900 + i) for i, l in enumerate(np.linspace(*cfg["load_range"], 6))]
    f = open(os.path.join(out_dir, f"{tag}_training_curve.csv"), "w", newline="")
    w = csv.writer(f)
    w.writerow(["iter", "env_steps", "wall_clock_s", "train_reward", "train_loss_ratio", "q_loss", "actor_q", "noise", "val_reward"])
    best, t0, INF, ql, aq = -1e9, time.time(), [], [], []
    obs = [e.observe_all() for e in envs]
    for it in range(1, cfg["iters"] + 1):
        frac = (it - 1) / cfg["iters"]
        sigma = cfg["noise_start"] + (cfg["noise_end"] - cfg["noise_start"]) * frac
        X = torch.from_numpy(np.stack([o[0] for o in obs]))
        with torch.no_grad():
            A = actor(X)
        A = (A + sigma * torch.randn_like(A)).clamp(-3, 3) * mask
        for k, e in enumerate(envs):
            x2, r, info = e.step_matrix(A[k].numpy())
            o2 = e.observe_all()
            buf.add(obs[k][0], obs[k][1], A[k].numpy(), r, o2[0], o2[1])   # time-limit only: always bootstrap
            INF.append(info)
            tsteps[k] += 1
            if tsteps[k] >= cfg["episode_len"]:
                new_ep(e); tsteps[k] = 0; o2 = e.observe_all()
            obs[k] = o2
        if it > cfg["warmup"]:
            Xb, Sb, Ab, Rb, X2b, S2b = buf.sample(cfg["batch"], rng)
            with torch.no_grad():
                a2 = actor_t(X2b)
                if cfg["target_noise"] > 0:          # target policy smoothing
                    a2 = (a2 + (cfg["target_noise"] * torch.randn_like(a2)).clamp(
                        -cfg["target_noise_clip"], cfg["target_noise_clip"])).clamp(-3, 3)
                a2 = (a2 * mask).flatten(1)
                q2 = critic_t(S2b, a2)
                if critic2_t is not None:           # clipped double-Q
                    q2 = torch.min(q2, critic2_t(S2b, a2))
                y = Rb + cfg["gamma"] * q2
            lq = ((critic(Sb, Ab.flatten(1)) - y) ** 2).mean()
            if critic2 is not None:
                lq = lq + ((critic2(Sb, Ab.flatten(1)) - y) ** 2).mean()
            opt_c.zero_grad(); lq.backward()
            torch.nn.utils.clip_grad_norm_(c_params, cfg["max_grad_norm"]); opt_c.step()
            ql.append(lq.item())
            if it > cfg["actor_warmup"] and it % cfg["policy_delay"] == 0:
                a_pi = actor(Xb) * mask
                la = -critic(Sb, a_pi.flatten(1)).mean() + cfg["act_reg"] * (a_pi ** 2).mean()
                opt_a.zero_grad(); la.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), cfg["max_grad_norm"]); opt_a.step()
                aq.append(-la.item())
                with torch.no_grad():
                    pairs = list(zip(actor.parameters(), actor_t.parameters())) + \
                            list(zip(critic.parameters(), critic_t.parameters()))
                    if critic2 is not None:
                        pairs += list(zip(critic2.parameters(), critic2_t.parameters()))
                    for p, pt in pairs:
                        pt.mul_(1 - cfg["tau"]).add_(cfg["tau"] * p)
        if it % 100 == 0 or it == cfg["iters"]:
            vr = ""
            if it % cfg["eval_every"] == 0 or it == cfg["iters"]:
                vr = evaluate(actor, G, val_cases, cfg["eval_steps"], cfg["scenario"])["reward"]
                if vr > best:
                    best = vr; torch.save(actor.state_dict(), os.path.join(out_dir, f"{tag}_actor_best.pt"))
            tr = np.mean([i["reward_raw"] for i in INF])
            w.writerow([it, it * cfg["n_envs"], round(time.time() - t0, 1), tr,
                        np.mean([i["loss_ratio"] for i in INF]), np.mean(ql) if ql else "", np.mean(aq) if aq else "",
                        sigma, vr]); f.flush()
            if it % 500 == 0:
                print(f"[{tag}] it {it} R {tr:+.4f} qloss {np.mean(ql) if ql else 0:.4f} noise {sigma:.2f} "
                      f"val {vr} {time.time()-t0:.0f}s", flush=True)
            INF, ql, aq = [], [], []
    torch.save(actor.state_dict(), os.path.join(out_dir, f"{tag}_actor_last.pt"))
    f.close()
    return best
