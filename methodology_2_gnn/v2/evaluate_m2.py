"""
Methodology 2 (v2) evaluation: zero-shot transfer, generalisation gap,
broad real-WAN suite, and link-failure ablation. Writes CSVs to results/tables.

    python methodology_2_gnn/v2/evaluate_m2.py --part primary
    python methodology_2_gnn/v2/evaluate_m2.py --part suite
    python methodology_2_gnn/v2/evaluate_m2.py --part failure

Every method runs on the same simulator with the same traffic seeds
(common random numbers), so differences are due to routing only.
"""
import os as _os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")   # one BLAS thread per process: parallel jobs otherwise thrash (measured ~10x slowdown)

import argparse, glob, os, sys, time, yaml, json
import numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import topologies as T
from te_sim import TESimulator
from baselines import Baseline, BASELINES
from policies import GNNPolicy, MLPPolicy

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
TAB = os.path.join(RES, "tables")
METRICS = ["reward", "throughput_mbps", "offered_mbps", "loss_ratio", "avg_delay_ms", "mlu",
           "util_std", "jain"]


class LearnedAgent:
    def __init__(self, name, model):
        self.name, self.model = name, model.eval()
        self.lat = []

    def reset(self, sim):
        pass

    def act(self, sim, obs):
        t0 = time.perf_counter()
        with torch.no_grad():
            mu, _, _ = self.model([obs], with_value=False)
        self.lat.append((time.perf_counter() - t0) * 1e3)
        cost, gam = sim.decode_action(mu.numpy())
        return dict(cost=cost, gamma=gam)


def run_episode(agent, G, load, seed, steps, fail_at=None, fail_links=()):
    sim = TESimulator(G, load=load, seed=seed)
    agent.reset(sim)
    obs = sim.observe()
    rows = []
    for t in range(steps):
        if fail_at is not None and t == fail_at:
            sim.fail_links(fail_links)
            obs = sim.observe()
        a = agent.act(sim, obs)
        if "P" in a:
            obs, r, info = sim.step(P=a["P"])
        else:
            obs, r, info = sim.step(a["cost"], "softmin", gamma=a.get("gamma"))
        info["t"] = t
        rows.append(info)
    return rows


def summarize(rows):
    return {k: float(np.mean([r[k] for r in rows])) for k in METRICS}


def load_gnn(path, cfg):
    m = GNNPolicy(**cfg["gnn"]); m.load_state_dict(torch.load(path)); return m


def zero_shot_ckpts():
    return sorted(glob.glob(os.path.join(RES, "zero_shot_seed*", "gnn_best.pt")))


class TabularSpecialist:
    """Per-topology TABULAR Q-learning agents (Methodology 1's tabular_q.py, trained on this
    topology: results_marl/tabq_<topology>_seed0). The tabular baseline of the M2 gap."""

    def __init__(self, G, qpath, step=0.75):
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if root not in sys.path:
            sys.path.append(root)
        from environments.flow_splitting_env import FlowSplittingEnv
        from models.marl_value import DiscreteActions
        from methodology_1_marl.tabular_q import greedy_tab
        self.name, self.lat = "Tabular specialist", []
        self.env = FlowSplittingEnv(G)
        self.policy = greedy_tab(np.load(qpath), DiscreteActions(self.env, step))

    def reset(self, sim):
        pass

    def act(self, sim, obs):
        self.env.sim = sim                                  # same graph -> same link order
        t0 = time.perf_counter()
        A = self.policy(self.env)
        self.lat.append((time.perf_counter() - t0) * 1e3)
        return dict(cost=self.env.cost_from_matrix(A))


def agents_for(G, name, cfg, include_specialists=True):
    agents = [Baseline(b) for b in BASELINES]
    for p in zero_shot_ckpts():
        s = os.path.basename(os.path.dirname(p)).replace("zero_shot_", "")
        agents.append(LearnedAgent(f"GNN zero-shot ({s})", load_gnn(p, cfg)))
    if include_specialists:
        p = os.path.join(RES, f"specialist_gnn_{name}", "gnn_best.pt")
        if T.exists_exact(p):
            agents.append(LearnedAgent("GNN specialist", load_gnn(p, cfg)))
        p = os.path.join(RES, f"specialist_mlp_{name}", "mlp_best.pt")
        if T.exists_exact(p):
            m = MLPPolicy(G.number_of_nodes(), 2 * G.number_of_edges()); m.load_state_dict(torch.load(p))
            agents.append(LearnedAgent("MLP specialist", m))
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        for sub in ("m2_tabular_specialists", ""):       # own folder first; Geant2012 reuses the M1 run
            p = os.path.join(root, "results_marl", sub, f"tabq_{name}_seed0", "tabq_qtable_best.npy")
            if T.exists_exact(p):
                agents.append(TabularSpecialist(G, p))
                break
    return agents


def part_primary(cfg, topologies):
    out = []
    for name in topologies:
        G = T.load_named(name)
        for ag in agents_for(G, name, cfg):
            for load in cfg["eval"]["loads"]:
                rows = []
                for seed in cfg["eval"]["seeds"]:
                    s = summarize(run_episode(ag, G, load, seed, cfg["eval"]["steps"]))
                    s.update(topology=name, nodes=G.number_of_nodes(), links=G.number_of_edges(),
                             method=ag.name, load=load, seed=seed)
                    rows.append(s)
                out += rows
            lat = getattr(ag, "lat", [])
            if lat:
                for r in out:
                    if r["method"] == ag.name and r["topology"] == name:
                        r["inference_ms"] = float(np.median(lat))
                ag.lat = []
            print(f"  {name:10s} {ag.name:24s} R@0.95 = "
                  f"{np.mean([r['reward'] for r in out if r['method']==ag.name and r['topology']==name and r['load']==0.95]):+.4f}", flush=True)
    df = pd.DataFrame(out)
    df.to_csv(os.path.join(TAB, "m2_primary_per_seed.csv"), index=False)
    return df


def part_suite(cfg, loads=(0.8, 0.95), seeds=(201, 202), steps=100):
    files = sorted(glob.glob(os.path.join(T.ZOO_DIR, "*.graphml")))
    out = []
    for i, f in enumerate(files):
        G = T.load_topology_zoo(f)
        name = os.path.splitext(os.path.basename(f))[0]
        agents = [Baseline(b) for b in ["OSPF", "ECMP-hop", "Softmin(static)", "LP-oracle(mean TM)"]]
        agents += [LearnedAgent(f"GNN zero-shot ({os.path.basename(os.path.dirname(p)).replace('zero_shot_', '')})",
                                load_gnn(p, cfg)) for p in zero_shot_ckpts()]
        for ag in agents:
            for load in loads:
                for seed in seeds:
                    s = summarize(run_episode(ag, G, load, seed, steps))
                    s.update(topology=name, nodes=G.number_of_nodes(), links=G.number_of_edges(),
                             method=ag.name, load=load, seed=seed)
                    out.append(s)
        print(f"[{i+1}/{len(files)}] {name} ({G.number_of_nodes()} nodes)", flush=True)
    df = pd.DataFrame(out)
    df.to_csv(os.path.join(TAB, "m2_zoo_suite_per_seed.csv"), index=False)
    return df


def part_failure(cfg, topology="Geant2012", load=0.8, trials=20, steps=200, fail_at=100):
    G = T.load_named(topology)
    rng = np.random.default_rng(2024)
    probe = TESimulator(G, load=load, seed=0)
    agents = [Baseline(b) for b in ["OSPF", "ECMP-hop", "Softmin(static)"]]
    agents += [a for a in agents_for(G, topology, cfg) if isinstance(a, LearnedAgent)]
    out = []
    for trial in range(trials):
        k = int(rng.integers(1, 4))
        while True:
            links = rng.choice(G.number_of_edges(), size=k, replace=False).tolist()
            if probe.connected_after(links):
                break
        seed = 3000 + trial
        for ag in agents:
            ctrl = run_episode(ag, G, load, seed, steps)
            fail = run_episode(ag, G, load, seed, steps, fail_at=fail_at, fail_links=links)
            c, f = summarize(ctrl[fail_at:]), summarize(fail[fail_at:])
            out.append(dict(trial=trial, links_failed=k, method=ag.name,
                            **{f"ctrl_{m}": c[m] for m in METRICS}, **{f"fail_{m}": f[m] for m in METRICS}))
        print(f"  trial {trial}: {k} link(s) failed", flush=True)
    df = pd.DataFrame(out)
    df.to_csv(os.path.join(TAB, "m2_failure_per_trial.csv"), index=False)
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["primary", "suite", "failure"], required=True)
    ap.add_argument("--topologies", nargs="*", default=["Abilene", "Nsfnet", "Geant2012", "nsfnet", "fattree_k4"])
    args = ap.parse_args()
    torch.set_num_threads(1)
    os.makedirs(TAB, exist_ok=True)
    cfg = yaml.safe_load(open(os.path.join(HERE, "config_v2.yaml")))
    {"primary": lambda: part_primary(cfg, args.topologies), "suite": lambda: part_suite(cfg),
     "failure": lambda: part_failure(cfg)}[args.part]()
