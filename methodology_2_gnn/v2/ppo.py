"""
PPO for graph-structured routing (Methodology 2, v2).

Fixes vs. the v1 trainer (train_gnn_policy.py):
  * log-probs are kept per link and per graph (v1 summed over the whole
    100-step batch into ONE scalar, so the PPO ratio was meaningless);
  * K parallel environments on different random graphs per update
    (v1: one graph, one episode per update; `graphs_per_rollout` unused);
  * minibatched epochs, truncation-aware GAE bootstrap, seeding, CSV logging,
    best-checkpoint selection on a held-out *synthetic* validation set
    (never on the real test topologies).

Multi-agent view: every directed link's cost is one agent's action; agents
share parameters (the GNN) and a team reward. The PPO ratio is clipped per
agent (MAPPO-style), which is stable with ~100+ agents per graph.
"""
import csv, os, time, random
import numpy as np
import torch

from te_sim import TESimulator
from policies import logprob_per_graph


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


class EnvSlot:
    """One running environment; resamples graph/load/seed when an episode ends."""

    def __init__(self, sampler, ep_len, rng, use_shadow=True):
        self.sampler, self.ep_len, self.rng, self.use_shadow = sampler, ep_len, rng, use_shadow
        self.new_episode()

    def new_episode(self):
        G, load = self.sampler(self.rng)
        seed = int(self.rng.integers(1 << 31))
        self.sim = TESimulator(G, load=load, seed=seed)
        # Counterfactual "shadow" network: same graph, same traffic seed, routed
        # by OSPF. Its reward depends only on exogenous traffic, never on the
        # agent's actions, so subtracting it is an unbiased baseline that
        # cancels most burst-induced reward noise (common random numbers).
        self.shadow = TESimulator(G, load=load, seed=seed) if self.use_shadow else None
        self.P_sp = self.shadow.routing_matrix(self.shadow.ospf_costs(), "sp") if self.use_shadow else None
        self.obs = self.sim.observe()
        self.t = 0


def collect(model, slots, steps):
    buf = dict(obs=[], act=[], logp=[], val=[], rew=[], done=[], last_val=None, infos=[])
    for _ in range(steps):
        obs = [s.obs for s in slots]
        with torch.no_grad():
            mu, eg, val = model(obs)
            std = model.last_log_std.exp()
            a = mu + std * torch.randn_like(mu)
            lp = torch.distributions.Normal(mu, std).log_prob(a)
        acts, lps = [], []
        rews, dones = [], []
        for i, s in enumerate(slots):
            m = eg == i
            ai, lpi = a[m], lp[m]
            acts.append(ai); lps.append(lpi)
            cost, gam = s.sim.decode_action(ai.numpy())
            obs_next, r, info = s.sim.step(cost, gamma=gam)
            s.obs, s.t = obs_next, s.t + 1
            buf["infos"].append(info)
            if s.shadow is not None:
                _, r_sh, _ = s.shadow.step(P=s.P_sp)
                r = r - r_sh
            rews.append(r)
            done = s.t >= s.ep_len
            dones.append(done)
        buf["obs"].append(obs); buf["act"].append(acts); buf["logp"].append(lps)
        buf["val"].append(val.numpy()); buf["rew"].append(np.array(rews)); buf["done"].append(np.array(dones))
        # value of the state reached (for truncation bootstrap) before resetting
        trunc_obs = [s.obs for s in slots]
        with torch.no_grad():
            _, _, v_next = model(trunc_obs)
        buf.setdefault("next_val", []).append(v_next.numpy())
        for s, d in zip(slots, dones):
            if d:
                s.new_episode()
    return buf


def gae(buf, gamma, lam):
    R = np.stack(buf["rew"]); V = np.stack(buf["val"]); NV = np.stack(buf["next_val"])
    D = np.stack(buf["done"]).astype(float)
    T, K = R.shape
    adv = np.zeros_like(R); last = np.zeros(K)
    for t in reversed(range(T)):
        # time-limit truncation: bootstrap from next state's value, but do not
        # carry the GAE trace across the episode boundary
        delta = R[t] + gamma * NV[t] - V[t]
        last = delta + gamma * lam * (1 - D[t]) * last
        adv[t] = last
    return adv, adv + V


def ppo_update(model, opt, buf, adv, ret, cfg):
    T, K = adv.shape
    flat = [(t, k) for t in range(T) for k in range(K)]
    advn = (adv - adv.mean()) / (adv.std() + 1e-8)
    stats = []
    for _ in range(cfg["epochs"]):
        random.shuffle(flat)
        for i in range(0, len(flat), cfg["minibatch"]):
            mb = flat[i:i + cfg["minibatch"]]
            obs = [buf["obs"][t][k] for t, k in mb]
            act = torch.cat([buf["act"][t][k] for t, k in mb])
            old = torch.cat([buf["logp"][t][k] for t, k in mb])
            A_g = torch.tensor([advn[t, k] for t, k in mb], dtype=torch.float32)
            R_g = torch.tensor([ret[t, k] for t, k in mb], dtype=torch.float32)
            mu, eg, v = model(obs)
            dist = torch.distributions.Normal(mu, model.last_log_std.exp())
            lp = dist.log_prob(act)
            ratio = torch.exp(lp - old)
            A_e = A_g[eg]                                       # team advantage per agent
            s1, s2 = ratio * A_e, torch.clamp(ratio, 1 - cfg["clip"], 1 + cfg["clip"]) * A_e
            # average per graph first so big graphs don't dominate
            per_edge = -torch.min(s1, s2)
            n_e = torch.zeros(len(mb)).index_add_(0, eg, torch.ones_like(per_edge))
            pi_loss = (torch.zeros(len(mb)).index_add_(0, eg, per_edge) / n_e).mean()
            v_loss = ((v - R_g) ** 2).mean()
            ent = dist.entropy().mean()
            loss = pi_loss + cfg["vf_coef"] * v_loss - cfg["ent_coef"] * ent
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.actor_parameters(), cfg["max_grad_norm"])
            torch.nn.utils.clip_grad_norm_(model.critic_parameters(), cfg["max_grad_norm"])
            opt.step()
            with torch.no_grad():
                kl = (old - lp).mean().item()
                clipfrac = ((ratio - 1).abs() > cfg["clip"]).float().mean().item()
            stats.append((pi_loss.item(), v_loss.item(), ent.item(), kl, clipfrac))
    return np.mean(stats, 0)


def evaluate(model, cases, steps, deterministic=True):
    """cases: list of (graph, load, seed). Returns mean reward and metric dict."""
    infos = []
    for G, load, seed in cases:
        sim = TESimulator(G, load=load, seed=seed)
        obs = sim.observe()
        for _ in range(steps):
            with torch.no_grad():
                mu, _, _ = model([obs], with_value=False)
            cost, gam = sim.decode_action(mu.numpy())
            obs, r, info = sim.step(cost, gamma=gam)
            infos.append(info)
    return {k: float(np.mean([i[k] for i in infos])) for k in infos[0]}


def train(model, sampler, cfg, out_dir, val_cases=None, tag="gnn"):
    os.makedirs(out_dir, exist_ok=True)
    set_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    slots = [EnvSlot(sampler, cfg["episode_len"], rng, cfg.get("shadow_baseline", True)) for _ in range(cfg["n_envs"])]
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    log_path = os.path.join(out_dir, f"{tag}_training_curve.csv")
    f = open(log_path, "w", newline="")
    w = csv.writer(f)
    w.writerow(["update", "env_steps", "wall_clock_s", "train_reward", "train_loss_ratio", "train_delay_ms",
                "pi_loss", "v_loss", "entropy", "approx_kl", "clip_frac", "log_std", "val_reward"])
    t0, best, env_steps = time.time(), -1e9, 0
    for u in range(1, cfg["updates"] + 1):
        for g in opt.param_groups:                       # linear LR decay
            g["lr"] = cfg["lr"] * (1 - (u - 1) / cfg["updates"])
        buf = collect(model, slots, cfg["rollout_len"])
        env_steps += cfg["rollout_len"] * cfg["n_envs"]
        adv, ret = gae(buf, cfg["gamma"], cfg["lam"])
        st = ppo_update(model, opt, buf, adv, ret, cfg)
        tr = float(np.mean([i["reward"] for i in buf["infos"]]))     # raw reward, not differenced
        tl = float(np.mean([i["loss_ratio"] for i in buf["infos"]]))
        td = float(np.mean([i["avg_delay_ms"] for i in buf["infos"]]))
        vr = ""
        if val_cases and (u % cfg["eval_every"] == 0 or u == cfg["updates"]):
            vr = evaluate(model, val_cases, cfg["eval_steps"])["reward"]
            if vr > best:
                best = vr
                torch.save(model.state_dict(), os.path.join(out_dir, f"{tag}_best.pt"))
        w.writerow([u, env_steps, round(time.time() - t0, 1), tr, tl, td, *np.round(st, 5),
                    float(model.log_std.item()), vr]); f.flush()
        if u % 10 == 0 or u == 1:
            print(f"[{tag}] upd {u:4d} | R {tr:+.4f} loss {100*tl:.2f}% d {td:.1f}ms | "
                  f"kl {st[3]:.4f} clip {st[4]:.2f} std {model.log_std.exp().item():.3f} | "
                  f"val {vr if vr == '' else round(vr, 4)} | {time.time()-t0:.0f}s", flush=True)
    torch.save(model.state_dict(), os.path.join(out_dir, f"{tag}_last.pt"))
    f.close()
    return best
