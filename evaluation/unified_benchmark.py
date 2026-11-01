"""
Unified cross-methodology benchmark (Week 4/5 milestone: all 9 target metrics).

Every method runs on the same simulator, topology, load, traffic scenario and
traffic seed (common random numbers), so differences are due to routing only.

  Baselines : OSPF, ECMP, ECMP-hop, LP-oracle (mean TM)
  M1        : IPPO (independent learners)                 results_marl/ippo_*
  M2        : GNN zero-shot / GNN specialist / MLP specialist  methodology_2_gnn/v2/results
  M3        : MAPPO, MADDPG (CTDE)                         results_marl/{mappo,maddpg}_*

    python evaluation/unified_benchmark.py            # writes results_unified/*.csv
"""
import os as _os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")   # one BLAS thread per process: parallel jobs otherwise thrash (measured ~10x slowdown)

import glob, json, os, sys, time
import numpy as np, pandas as pd, torch, yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.append(os.path.join(ROOT, "methodology_2_gnn", "v2"))   # append: its baselines.py must not shadow the package
from environments.topologies import load_named, exists_exact
from environments.flow_splitting_env import FlowSplittingEnv
from baselines.te_baselines import Baseline
from models.decentralized_forwarder import DecentralizedForwarder
from policies import GNNPolicy, MLPPolicy             # methodology_2_gnn/v2

M2 = os.path.join(ROOT, "methodology_2_gnn", "v2", "results")
MARL = os.path.join(ROOT, "results_marl")
OUT = os.path.join(ROOT, "results_unified")
METRICS = ["reward", "throughput_mbps", "offered_mbps", "loss_ratio", "avg_delay_ms", "mlu", "util_std", "jain"]
TOPOLOGIES = ["nsfnet", "Geant2012"]
SCENARIOS = [("pareto", 0.5), ("pareto", 0.8), ("pareto", 0.95), ("pareto", 1.1), ("microburst", 0.9)]
SEEDS = [101, 102, 103, 104, 105]
STEPS = 200


class BaselineAgent:
    family = "Baseline"

    def __init__(self, name):
        self.name, self.b = name, Baseline(name)
        self.lat = []

    def act(self, env):
        return dict(P=self.b.act(env.sim)["P"])


class GraphAgent:
    def __init__(self, name, model, family):
        self.name, self.model, self.family, self.lat = name, model.eval(), family, []

    def act(self, env):
        o = env.sim.observe()
        t0 = time.perf_counter()
        with torch.no_grad():
            mu, _, _ = self.model([o], with_value=False)
        self.lat.append((time.perf_counter() - t0) * 1e3)
        cost, gam = env.sim.decode_action(mu.numpy())
        return dict(cost=cost, gamma=gam)


class SwitchAgents:
    def __init__(self, name, actor, family):
        self.name, self.actor, self.family, self.lat = name, actor.eval(), family, []

    def act(self, env):
        X, _ = env.observe_all()
        t0 = time.perf_counter()
        with torch.no_grad():
            A = self.actor(torch.from_numpy(X)[None])[0].numpy()
        self.lat.append((time.perf_counter() - t0) * 1e3)
        return dict(A=A)


def methods_for(topo, G):
    env = FlowSplittingEnv(G)
    ms = [BaselineAgent(b) for b in ["OSPF", "ECMP", "ECMP-hop", "LP-oracle(mean TM)"]]
    cfg = yaml.safe_load(open(os.path.join(ROOT, "methodology_2_gnn", "v2", "config_v2.yaml")))
    for p in sorted(glob.glob(os.path.join(M2, "zero_shot_seed*", "gnn_best.pt"))):
        m = GNNPolicy(**cfg["gnn"]); m.load_state_dict(torch.load(p))
        s = os.path.basename(os.path.dirname(p)).replace("zero_shot_", "")
        ms.append(GraphAgent(f"M2 GNN zero-shot ({s})", m, "M2"))
    p = os.path.join(M2, f"specialist_gnn_{topo}", "gnn_best.pt")
    if exists_exact(p):
        m = GNNPolicy(**cfg["gnn"]); m.load_state_dict(torch.load(p)); ms.append(GraphAgent("M2 GNN specialist", m, "M2"))
    p = os.path.join(M2, f"specialist_mlp_{topo}", "mlp_best.pt")
    if exists_exact(p):
        m = MLPPolicy(G.number_of_nodes(), 2 * G.number_of_edges()); m.load_state_dict(torch.load(p))
        ms.append(GraphAgent("M2 MLP specialist", m, "M2"))
    ms += load_switch_agents(G, topo, ["ippo", "iql", "tabq", "mappo", "maddpg", "qmix"])
    return ms


FAMILY = dict(ippo="M1", iql="M1", tabq="M1", mappo="M3", maddpg="M3", qmix="M3")
LABEL = dict(ippo="IPPO", iql="IQL-DQN", tabq="Tabular-Q", mappo="MAPPO", maddpg="MADDPG", qmix="QMIX")


class DiscreteSwitchAgents:
    """Per-switch discrete-action policy (IQL / QMIX / tabular): greedy on its own Q."""

    def __init__(self, name, act_fn, family):
        self.name, self.act_fn, self.family, self.lat = name, act_fn, family, []

    def act(self, env):
        t0 = time.perf_counter()
        A = self.act_fn(env)
        self.lat.append((time.perf_counter() - t0) * 1e3)
        return dict(A=A)


def load_switch_agents(G, topo, algos):
    """Every trained M1/M3 checkpoint for this topology, named e.g. 'M1 IQL-DQN (seed0)'."""
    from models.marl_value import DiscreteActions, QAgents, greedy_fn
    from methodology_1_marl.tabular_q import greedy_tab
    env = FlowSplittingEnv(G)
    out = []
    for algo in algos:
        for d in sorted(glob.glob(os.path.join(MARL, f"{algo}_{topo}_seed*"))):
            seed = os.path.basename(d).split("_")[-1]
            name = f"{FAMILY[algo]} {LABEL[algo]} ({seed})"
            meta = os.path.join(d, "meta.json")
            step = json.load(open(meta)).get("overrides", {}).get("action_step", 0.75) if os.path.exists(meta) else 0.75
            if algo in ("ippo", "mappo", "maddpg"):
                p = os.path.join(d, f"{algo}_actor_best.pt")
                if not os.path.exists(p):
                    continue
                a = DecentralizedForwarder(env.n, env.obs_dim, env.act_dim, deterministic=(algo == "maddpg"))
                a.load_state_dict(torch.load(p))
                out.append(SwitchAgents(name, a, FAMILY[algo]))
            elif algo in ("iql", "qmix"):
                p = os.path.join(d, f"{algo}_agents_best.pt")
                if not os.path.exists(p):
                    continue
                acts = DiscreteActions(env, step)
                q = QAgents(env.n, env.obs_dim, acts.A); q.load_state_dict(torch.load(p)); q.eval()
                out.append(DiscreteSwitchAgents(name, greedy_fn(q, acts), FAMILY[algo]))
            else:
                p = os.path.join(d, f"{algo}_qtable_best.npy")
                if not os.path.exists(p):
                    continue
                acts = DiscreteActions(env, step)
                out.append(DiscreteSwitchAgents(name, greedy_tab(np.load(p), acts), FAMILY[algo]))
    return out


def run(method, G, scenario, load, seed, steps=STEPS):
    env = FlowSplittingEnv(G, load=load, seed=seed, scenario=scenario)
    if hasattr(method, "b"):
        method.b.reset(env.sim)
    rows = []
    for _ in range(steps):
        a = method.act(env)
        if "P" in a:
            _, _, info = env.sim.step(P=a["P"])
        elif "A" in a:
            _, _, info = env.step_matrix(a["A"])
        else:
            _, _, info = env.sim.step(a["cost"], "softmin", gamma=a.get("gamma"))
        rows.append(info)
    return {k: float(np.mean([r[k] for r in rows])) for k in METRICS}


def training_cost():
    """Convergence (env steps / wall-clock to reach 90% of the best validation
    improvement over the first evaluation) for every trained model."""
    rows = []
    def conv(df, xcol, vcol, tcol):
        v = df.dropna(subset=[vcol])
        if v.empty:
            return None
        v0, vb = v[vcol].iloc[0], v[vcol].max()
        hit = v[v[vcol] >= v0 + 0.9 * (vb - v0)].iloc[0]
        return dict(best_val=float(vb), steps_to_90pct=int(hit[xcol]), wall_s_to_90pct=float(hit[tcol]),
                    total_wall_s=float(df[tcol].iloc[-1]), total_env_steps=int(df[xcol].iloc[-1]))
    for f in glob.glob(os.path.join(M2, "*", "*_training_curve.csv")):
        d = pd.read_csv(f); r = conv(d, "env_steps", "val_reward", "wall_clock_s")
        if r: rows.append(dict(run=os.path.basename(os.path.dirname(f)), family="M2", **r))
    for f in glob.glob(os.path.join(MARL, "*", "*_training_curve.csv")):
        d = pd.read_csv(f); r = conv(d, "env_steps", "val_reward", "wall_clock_s")
        run_ = os.path.basename(os.path.dirname(f))
        if r: rows.append(dict(run=run_, family="M1" if run_.startswith("ippo") else "M3", **r))
    return pd.DataFrame(rows)


def main(topologies=TOPOLOGIES, tag=""):
    os.makedirs(OUT, exist_ok=True)
    torch.set_num_threads(1)
    out = []
    for topo in topologies:
        G = load_named(topo)
        sig = FlowSplittingEnv(G).signalling_bytes()
        for m in methods_for(topo, G):
            for scen, load in SCENARIOS:
                for seed in SEEDS:
                    r = run(m, G, scen, load, seed)
                    r.update(topology=topo, method=m.name, family=m.family, scenario=scen, load=load, seed=seed)
                    out.append(r)
            lat = float(np.median(m.lat)) if m.lat else np.nan
            for r in out:
                if r["method"] == m.name and r["topology"] == topo:
                    r["inference_ms_per_decision"] = lat
                    r["signalling_bytes_per_interval"] = (sig["decentralised_bytes"] if m.family in ("M1", "M3")
                                                          else sig["centralised_bytes"] if m.family == "M2"
                                                          else 0)
            print(f"{topo:10s} {m.name:28s} done", flush=True)
        pd.DataFrame(out).to_csv(os.path.join(OUT, f"unified_per_seed{tag}.csv"), index=False)
    if tag:
        return
    aggregate(pd.DataFrame(out))


def aggregate(df):
    df["loss_pct"] = 100 * df.loss_ratio
    agg = df.groupby(["topology", "scenario", "load", "family", "method"]).agg(
        reward=("reward", "mean"), reward_sd=("reward", "std"), throughput_mbps=("throughput_mbps", "mean"),
        loss_pct=("loss_pct", "mean"), delay_ms=("avg_delay_ms", "mean"), mlu=("mlu", "mean"),
        util_std=("util_std", "mean"), jain=("jain", "mean"),
        inference_ms=("inference_ms_per_decision", "first"),
        signalling_bytes=("signalling_bytes_per_interval", "first")).reset_index()
    agg.round(5).to_csv(os.path.join(OUT, "unified_summary.csv"), index=False)
    training_cost().round(3).to_csv(os.path.join(OUT, "training_cost.csv"), index=False)
    print(agg.to_string())


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--topologies", nargs="*", default=TOPOLOGIES)
    ap.add_argument("--tag", default="", help="write a partial per-seed CSV (for parallel runs)")
    ap.add_argument("--merge", action="store_true", help="merge partial CSVs and aggregate")
    a = ap.parse_args()
    if a.merge:
        parts = sorted(glob.glob(os.path.join(OUT, "unified_per_seed_*.csv")))
        df = pd.concat([pd.read_csv(f) for f in parts])
        df.to_csv(os.path.join(OUT, "unified_per_seed.csv"), index=False)
        aggregate(df)
    else:
        main(a.topologies, a.tag)
