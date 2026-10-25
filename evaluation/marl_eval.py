"""Shared evaluation helper for the switch-level methods (M1 / M3): trained actors vs
OSPF / ECMP / ECMP-hop / LP oracle, same traffic seeds, every scenario."""
import glob, os
import numpy as np, pandas as pd, torch

from environments.flow_splitting_env import FlowSplittingEnv
from evaluation.unified_benchmark import BaselineAgent, load_switch_agents, run, SCENARIOS, SEEDS, MARL

BASELINES = ["OSPF", "ECMP", "ECMP-hop", "LP-oracle(mean TM)"]


def load_actors(G, topo, algos):
    return load_switch_agents(G, topo, algos)


def evaluate(G, topo, algos, out_csv, scenarios=SCENARIOS, seeds=SEEDS):
    torch.set_num_threads(1)
    methods = [BaselineAgent(b) for b in BASELINES] + load_actors(G, topo, algos)
    if len(methods) == len(BASELINES):
        raise SystemExit(f"no trained {algos} checkpoints for '{topo}' in {MARL} - train first")
    rows = []
    for m in methods:
        for scen, load in scenarios:
            for seed in seeds:
                r = run(m, G, scen, load, seed)
                r.update(method=m.name, algo=m.name.split(" (")[0], scenario=scen, load=load, seed=seed)
                rows.append(r)
        print(f"  {m.name:22s} done", flush=True)
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    df["loss_pct"] = 100 * df.loss_ratio
    summ = df.groupby(["scenario", "load", "algo"]).agg(
        reward=("reward", "mean"), loss_pct=("loss_pct", "mean"), delay_ms=("avg_delay_ms", "mean"),
        mlu=("mlu", "mean"), jain=("jain", "mean"), throughput_mbps=("throughput_mbps", "mean")).reset_index()
    summ.round(4).to_csv(out_csv.replace(".csv", "_summary.csv"), index=False)
    print(summ.pivot_table(index="algo", columns=["scenario", "load"], values="reward").round(3).to_string())
    print("\nloss %"); print(summ.pivot_table(index="algo", columns=["scenario", "load"], values="loss_pct").round(2).to_string())
    return summ
