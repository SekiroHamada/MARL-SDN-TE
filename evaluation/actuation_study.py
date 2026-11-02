"""
Actuation-realism study (reviewer fixes #4 and #5).

Part A - per-step LP re-optimisation. Two extra references next to the
         mean-matrix LP oracle, on the same traffic as the unified benchmark:
           LP-reactive(last TM)        re-solved every step on the matrix measured
                                       in the previous interval (deployable)
           LP-clairvoyant(current TM)  re-solved every step on the matrix about to
                                       be routed (upper bound, not deployable)

Part B - control-interval sensitivity. Every trained policy is re-evaluated with
         its decision HELD for k simulator steps (k x 10 ms). A new decision is only
         installed every k steps, as a real controller limited by OpenFlow group
         re-installation (229-439 ms measured under Mininet) would do. We also count
         forwarding-rule churn exactly as the Mininet installer does: a
         (switch, destination) select group is re-installed when its next-hop set
         changes or any bucket weight moves by more than 2%.

    python evaluation/actuation_study.py --part lp
    python evaluation/actuation_study.py --part interval
    python evaluation/actuation_study.py --part deadband   # installer hysteresis at 250 ms

Part C - installer-side hysteresis (dead-band). At a 250 ms interval, a
         destination's rules are only replaced when some group would move a
         weight by more than delta; delta = 0 is the plain installer.
Writes results_unified/actuation_*.csv
"""
import os as _os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")

import argparse, os, sys, time
import numpy as np, pandas as pd, torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import unified_benchmark as ub
from baselines.te_baselines import PER_STEP_LP
from environments.actuation import groups_changed, deadband

OUT = ub.OUT
METRICS = ub.METRICS
INTERVALS = [1, 2, 5, 10, 25, 50]          # x 10 ms -> 10 ms ... 500 ms


def split_tensor(method, env):
    """The split tensor P[t, v, u] that `method` would install now."""
    a = method.act(env)
    if "P" in a:
        return a["P"]
    if "A" in a:
        return env.sim.routing_matrix(env.cost_from_matrix(a["A"]), "softmin")
    return env.sim.routing_matrix(a["cost"], "softmin", gamma=a.get("gamma"))


def run_held(method, G, scenario, load, seed, hold, steps=ub.STEPS, delta=0.0):
    env = ub.FlowSplittingEnv(G, load=load, seed=seed, scenario=scenario)
    if hasattr(method, "b"):
        method.b.reset(env.sim)
    rows, churn, P, t0 = [], [], None, time.perf_counter()
    for t in range(steps):
        if t % hold == 0:
            P_new = deadband(P, split_tensor(method, env), delta)
            churn.append(groups_changed(P, P_new))
            P = P_new
        _, _, info = env.sim.step(P=P)
        rows.append(info)
    r = {k: float(np.mean([x[k] for x in rows])) for k in METRICS}
    r["groups_per_decision"] = float(np.mean(churn[1:])) if len(churn) > 1 else 0.0
    r["groups_per_second"] = r["groups_per_decision"] / (hold * 0.01)
    r["wall_s"] = time.perf_counter() - t0
    return r


def _resume(path):
    """Resumable runs: per-seed rows are appended; finished (topology, method) pairs are skipped."""
    if os.path.exists(path):
        d = pd.read_csv(path)
        return d.to_dict("records"), set(zip(d.topology, d.method))
    return [], set()


def part_lp(topologies):
    path = os.path.join(OUT, "actuation_lp_per_seed.csv")
    out, done = _resume(path)
    for topo in topologies:
        G = ub.load_named(topo)
        for name in PER_STEP_LP:
            if (topo, name) in done:
                continue
            m = ub.BaselineAgent(name)
            for scen, load in ub.SCENARIOS:
                for seed in ub.SEEDS:
                    r = run_held(m, G, scen, load, seed, hold=1)
                    r.update(topology=topo, method=name, scenario=scen, load=load, seed=seed)
                    out.append(r)
                print(f"{topo:10s} {name:28s} {scen} {load} done", flush=True)
            pd.DataFrame(out).to_csv(path, index=False)
    summarize_lp(pd.DataFrame(out))


def summarize_lp(df):
    df["loss_pct"] = 100 * df.loss_ratio
    agg = df.groupby(["topology", "scenario", "load", "method"]).agg(
        reward=("reward", "mean"), reward_sd=("reward", "std"), loss_pct=("loss_pct", "mean"),
        delay_ms=("avg_delay_ms", "mean"), mlu=("mlu", "mean"),
        groups_per_decision=("groups_per_decision", "mean")).reset_index()
    agg.round(5).to_csv(os.path.join(OUT, "actuation_lp_summary.csv"), index=False)
    print(agg.to_string())


KEEP = ("OSPF", "ECMP-hop", "LP-oracle(mean TM)", "M1 IPPO", "M1 IQL-DQN", "M2 GNN zero-shot",
        "M2 GNN specialist", "M3 MAPPO", "M3 QMIX")


def part_interval(topologies, scenario="pareto", load=0.95, intervals=INTERVALS):
    path = os.path.join(OUT, "actuation_interval_per_seed.csv")
    out, done = _resume(path)
    for topo in topologies:
        G = ub.load_named(topo)
        ms = [m for m in ub.methods_for(topo, G) if m.name.startswith(KEEP)]
        ms.append(ub.BaselineAgent("LP-reactive(last TM)"))
        for m in ms:
            if (topo, m.name) in done:
                continue
            for k in intervals:
                for seed in ub.SEEDS:
                    r = run_held(m, G, scenario, load, seed, hold=k)
                    r.update(topology=topo, method=m.name, family=m.family, hold_steps=k,
                             interval_ms=10 * k, scenario=scenario, load=load, seed=seed)
                    out.append(r)
            print(f"{topo:10s} {m.name:28s} done", flush=True)
            pd.DataFrame(out).to_csv(path, index=False)
    summarize_interval(pd.DataFrame(out))


def summarize_interval(df):
    df["loss_pct"] = 100 * df.loss_ratio
    df["algo"] = df["method"].str.replace(r" \(seed\d+\)$", "", regex=True)
    per = df.groupby(["topology", "algo", "method", "interval_ms"]).agg(
        reward=("reward", "mean"), loss_pct=("loss_pct", "mean"),
        groups_per_decision=("groups_per_decision", "mean"),
        groups_per_second=("groups_per_second", "mean")).reset_index()
    agg = per.groupby(["topology", "algo", "interval_ms"]).agg(
        reward=("reward", "mean"), reward_seed_sd=("reward", "std"), loss_pct=("loss_pct", "mean"),
        groups_per_decision=("groups_per_decision", "mean"),
        groups_per_second=("groups_per_second", "mean")).reset_index()
    agg.round(5).to_csv(os.path.join(OUT, "actuation_interval_summary.csv"), index=False)
    print(agg.to_string())


DELTAS = [0.0, 0.05, 0.1, 0.2, 0.3]
DB_KEEP = ("M1 IPPO", "M2 GNN zero-shot", "M3 MAPPO")


def part_deadband(topologies, scenario="pareto", load=0.95, hold=25, deltas=DELTAS):
    """Installer-side hysteresis at a 250 ms interval: reward vs rule churn."""
    path = os.path.join(OUT, "actuation_deadband_per_seed.csv")
    out, done = _resume(path)
    for topo in topologies:
        G = ub.load_named(topo)
        ms = [m for m in ub.methods_for(topo, G) if m.name.startswith(DB_KEEP)]
        ms.append(ub.BaselineAgent("LP-reactive(last TM)"))
        for m in ms:
            if (topo, m.name) in done:
                continue
            for d in deltas:
                for seed in ub.SEEDS:
                    r = run_held(m, G, scenario, load, seed, hold=hold, delta=d)
                    r.update(topology=topo, method=m.name, family=m.family, hold_steps=hold, delta=d,
                             scenario=scenario, load=load, seed=seed)
                    out.append(r)
            print(f"{topo:10s} {m.name:28s} done", flush=True)
            pd.DataFrame(out).to_csv(path, index=False)
    summarize_deadband(pd.DataFrame(out))


def summarize_deadband(df):
    df["loss_pct"] = 100 * df.loss_ratio
    df["algo"] = df["method"].str.replace(r" \(seed\d+\)$", "", regex=True)
    per = df.groupby(["topology", "algo", "method", "delta"]).agg(
        reward=("reward", "mean"), loss_pct=("loss_pct", "mean"),
        groups_per_decision=("groups_per_decision", "mean")).reset_index()
    agg = per.groupby(["topology", "algo", "delta"]).agg(
        reward=("reward", "mean"), reward_seed_sd=("reward", "std"), loss_pct=("loss_pct", "mean"),
        groups_per_decision=("groups_per_decision", "mean")).reset_index()
    agg.round(5).to_csv(os.path.join(OUT, "actuation_deadband_summary.csv"), index=False)
    print(agg.to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["lp", "interval", "deadband"], required=True)
    ap.add_argument("--topologies", nargs="*", default=ub.TOPOLOGIES)
    a = ap.parse_args()
    torch.set_num_threads(1)
    os.makedirs(OUT, exist_ok=True)
    dict(lp=part_lp, interval=part_interval, deadband=part_deadband)[a.part](a.topologies)
