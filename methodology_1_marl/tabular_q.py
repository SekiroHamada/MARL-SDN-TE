"""
Methodology 1 (classic variant): independent TABULAR Q-learning per switch.

The "Q-learning" baseline from the project spec, and the per-topology tabular
agent that Methodology 2's generalisation gap is measured against.

Local state of switch v (discretised from its 1-hop observation):
    hot link  : which outgoing link has the highest utilisation (or "none" if all < 0.3)
    hot level : max outgoing utilisation in {<0.5, 0.5-0.9, >=0.9}
    queued    : own mean outgoing queue occupancy > 0.1
-> (D+1) x 3 x 2 states, 1 + 2D discrete actions (models/marl_value.DiscreteActions).
A Q-table can only exist for one fixed topology: this is exactly the
"curse of dimensionality / no transfer" limitation that motivates the GNN.
"""
import csv, os, time
import numpy as np

from environments.flow_splitting_env import FlowSplittingEnv, NODE_F, EDGE_F
from models.marl_value import DiscreteActions, evaluate_policy, set_seed

TAB_DEFAULTS = dict(iters=8000, n_envs=8, episode_len=32, gamma=0.9, alpha=0.1, eps_start=1.0,
                    eps_end=0.05, eps_frac=0.6, load_range=(0.5, 1.05), eval_every=1000,
                    eval_steps=64, scenario="pareto", seed=0, action_step=0.75)
SLOT = EDGE_F + NODE_F + 1          # per-slot block in the local observation
UTIL = 2                            # index of utilisation inside the link features


def discretise(X, D):
    """X (n, obs_dim) local observations -> (n,) state ids."""
    slots = X[:, NODE_F:NODE_F + D * SLOT].reshape(len(X), D, SLOT)
    util = np.where(slots[..., -1] > 0, slots[..., UTIL], -1.0)
    hot = util.argmax(1)
    mx = util.max(1)
    hot = np.where(mx < 0.3, D, hot)
    level = np.digitize(mx, [0.5, 0.9])
    queued = (X[:, 0] > 0.1).astype(int)
    return (hot * 3 + level) * 2 + queued


def n_states(D):
    return (D + 1) * 3 * 2


def greedy_tab(Q, acts):
    def act(env):
        X, _ = env.observe_all()
        s = discretise(X, acts.D)
        q = np.where(acts.valid, Q[np.arange(acts.n), s], -1e9)
        return acts.to_continuous(q.argmax(1))
    return act


def train_tabular(G, out_dir, tag="tabq", **over):
    cfg = dict(TAB_DEFAULTS); cfg.update(over)
    os.makedirs(out_dir, exist_ok=True)
    set_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    envs = [FlowSplittingEnv(G, seed=i, scenario=cfg["scenario"], shadow_baseline=True) for i in range(cfg["n_envs"])]
    acts = DiscreteActions(envs[0], cfg["action_step"])
    n, A, D = acts.n, acts.A, acts.D
    Q = np.zeros((n, n_states(D), A))
    tsteps = np.zeros(len(envs), int)

    def new_ep(e):
        e.reset(load=float(rng.uniform(*cfg["load_range"])), seed=int(rng.integers(1 << 31)))
    for e in envs:
        new_ep(e)
    S = [discretise(e.observe_all()[0], D) for e in envs]
    val_cases = [(float(l), 900 + i) for i, l in enumerate(np.linspace(*cfg["load_range"], 6))]
    f = open(os.path.join(out_dir, f"{tag}_training_curve.csv"), "w", newline="")
    w = csv.writer(f)
    w.writerow(["iter", "env_steps", "wall_clock_s", "train_reward", "train_loss_ratio", "epsilon", "states_visited", "val_reward"])
    best, t0, INF = -1e9, time.time(), []
    idx = np.arange(n)
    for it in range(1, cfg["iters"] + 1):
        eps = max(cfg["eps_end"], cfg["eps_start"] - (cfg["eps_start"] - cfg["eps_end"]) * it / (cfg["eps_frac"] * cfg["iters"]))
        for k, e in enumerate(envs):
            s = S[k]
            q = np.where(acts.valid, Q[idx, s], -1e9)
            a = q.argmax(1)
            ex = rng.random(n) < eps
            if ex.any():
                a[ex] = acts.random(rng)[ex]
            _, r, info = e.step_matrix(acts.to_continuous(a))
            INF.append(info)
            s2 = discretise(e.observe_all()[0], D)
            target = r + cfg["gamma"] * np.where(acts.valid, Q[idx, s2], -1e9).max(1)
            Q[idx, s, a] += cfg["alpha"] * (target - Q[idx, s, a])
            tsteps[k] += 1
            if tsteps[k] >= cfg["episode_len"]:
                new_ep(e); tsteps[k] = 0; s2 = discretise(e.observe_all()[0], D)
            S[k] = s2
        if it % 100 == 0 or it == cfg["iters"]:
            vr = ""
            if it % cfg["eval_every"] == 0 or it == cfg["iters"]:
                vr = evaluate_policy(G, greedy_tab(Q, acts), val_cases, cfg["eval_steps"], cfg["scenario"])["reward"]
                if vr > best:
                    best = vr; np.save(os.path.join(out_dir, f"{tag}_qtable_best.npy"), Q)
            w.writerow([it, it * cfg["n_envs"], round(time.time() - t0, 1), np.mean([i["reward_raw"] for i in INF]),
                        np.mean([i["loss_ratio"] for i in INF]), round(eps, 4), int((Q != 0).any(-1).sum()), vr]); f.flush()
            if it % 1000 == 0:
                print(f"[{tag}] it {it} R {np.mean([i['reward_raw'] for i in INF]):+.4f} eps {eps:.2f} val {vr} "
                      f"{time.time()-t0:.0f}s", flush=True)
            INF = []
    np.save(os.path.join(out_dir, f"{tag}_qtable_last.npy"), Q)
    f.close()
    return best
