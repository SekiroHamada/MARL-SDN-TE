"""
Live demo of the three methodologies (for a presentation).

Each part prints a short explanation, then runs the trained policies against the
baselines on identical traffic, live, and prints a verdict. Nothing is trained
here: it loads the checkpoints produced by scripts/train_all.sh.

    python scripts/demo.py --part env     # the simulator and the traffic  (~20 s)
    python scripts/demo.py --part m1      # independent learners           (~60 s)
    python scripts/demo.py --part m2      # zero-shot GNN transfer         (~90 s)
    python scripts/demo.py --part m3      # CTDE vs independent            (~60 s)
    python scripts/demo.py --part all     # everything, in order

Options:  --topology nsfnet  --load 0.95  --steps 150  --unseen Abilene
"""
import os as _os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")

import argparse, glob, json, os, sys, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.append(os.path.join(ROOT, "methodology_2_gnn", "v2"))

from environments.topologies import load_named                       # noqa: E402
from environments.flow_splitting_env import FlowSplittingEnv         # noqa: E402
from environments.te_sim import TESimulator                          # noqa: E402

W = 78


def rule(ch="-"):
    print(ch * W, flush=True)


def title(t):
    print()
    rule("=")
    print(f"  {t}")
    rule("=")


def say(*lines):
    for ln in lines:
        print(f"  {ln}" if ln else "", flush=True)
    print(flush=True)


def pause(msg="[Enter] to continue "):
    if sys.stdin.isatty():
        try:
            input(f"\n  {msg}")
        except (EOFError, KeyboardInterrupt):
            print()


# ---------------------------------------------------------------- run helpers
def build_methods(G, topo, want):
    """want: subset of family names, e.g. {'Baseline','M1'}. Returns [(label, agent)]."""
    import evaluation.unified_benchmark as UB
    out = []
    for m in UB.methods_for(topo, G):
        if m.family not in want:
            continue
        name = m.name
        if "(seed" in name and "seed0" not in name:      # one seed is enough for a demo
            continue
        out.append((name.replace(" (seed0)", ""), m))
    return out


def race(G, methods, load, steps, scenario="pareto", every=30, seed=101):
    """Run every method on IDENTICAL traffic and print progress live."""
    import evaluation.unified_benchmark as UB
    envs = {lbl: FlowSplittingEnv(G, load=load, seed=seed, scenario=scenario)
            for lbl, _ in methods}
    acc = {lbl: [] for lbl, _ in methods}
    hdr = "  " + "step".ljust(6) + "".join(l[:17].ljust(19) for l, _ in methods)
    print(hdr)
    print("  " + " " * 6 + "".join("loss%   MLU".ljust(19) for _ in methods))
    rule()
    for t in range(steps):
        for lbl, m in methods:
            env = envs[lbl]
            a = m.act(env)
            if "P" in a:
                _, _, info = env.sim.step(P=a["P"])
            elif "A" in a:
                _, _, info = env.step_matrix(a["A"])
            else:
                _, _, info = env.sim.step(a["cost"], "softmin", gamma=a.get("gamma"))
            acc[lbl].append(info)
        if (t + 1) % every == 0 or t == steps - 1:
            line = "  " + f"{t+1:<6d}"
            for lbl, _ in methods:
                i = acc[lbl][-1]
                line += f"{max(0.0, 100*i["loss_ratio"]):5.2f}  {i['mlu']:5.2f}".ljust(19)
            print(line, flush=True)
            time.sleep(0.15)
    rule()
    return {lbl: {k: float(np.mean([r[k] for r in rows]))
                  for k in ("reward", "loss_ratio", "avg_delay_ms", "mlu", "throughput_mbps")}
            for lbl, rows in acc.items()}


def table(res, order=None, base="OSPF"):
    keys = order or list(res)
    print("  " + "method".ljust(26) + "reward   loss%   delay   MLU    vs OSPF")
    rule()
    b = res.get(base, {}).get("reward")
    for k in keys:
        r = res[k]
        delta = "" if b is None or k == base else f"  {r['reward']-b:+.3f}"
        star = "  <-- best" if k == max(keys, key=lambda x: res[x]["reward"]) else ""
        print(f"  {k[:25].ljust(26)}{r['reward']:+.3f}  {max(0.0, 100*r['loss_ratio']):5.2f}  "
              f"{r['avg_delay_ms']:6.1f}  {r['mlu']:5.2f}{delta}{star}")
    rule()


# ---------------------------------------------------------------- part: env
def part_env(a):
    title("THE SHARED ENVIRONMENT  (everything below is measured in here)")
    G = load_named(a.topology)
    say(f"Topology '{a.topology}': {G.number_of_nodes()} switches, {G.number_of_edges()} links.",
        "Every link: 100 Mbps capacity, 100-packet queue.",
        "One simulation step = one 10 ms control interval.")
    say("Traffic is GENERATED, not replayed: a gravity matrix modulated by",
        "heavy-tailed Pareto ON/OFF bursts (shape 1.5) and correlated hotspots.",
        "It does NOT depend on what the agent does, so every method below sees",
        "byte-identical traffic. That is what makes the comparison fair.")
    sim = TESimulator(G, load=a.load, seed=101)
    say(f"Offered load dial = {a.load} of network capacity. Watch it vary per step:")
    print("  " + "step".ljust(6) + "offered Mbps   busiest link util   loss %   (routing: OSPF)")
    rule()
    off = []
    for t in range(14):
        _, _, info = sim.step(sim.ospf_costs(), "sp")
        off.append(info["offered_mbps"])
        print(f"  {t+1:<6d}{info['offered_mbps']:10.1f}   {info['mlu']:14.2f}   "
              f"{max(0.0, 100*info["loss_ratio"]):6.2f}", flush=True)
        time.sleep(0.12)
    rule()
    say(f"Offered traffic swung between {min(off):.0f} and {max(off):.0f} Mbps "
        f"({max(off)/max(1e-9, np.mean(off)):.2f}x the mean):",
        "that burstiness is the whole problem. Static routing cannot react to it,",
        "and you can see the busiest link going over 1.00 and dropping packets.")
    say("The action every methodology controls is the same: a cost residual per link.",
        "Raising a link's cost pushes traffic off it; lowering it pulls traffic on.",
        "The split fractions come from a softmin over those costs, which is",
        "provably loop-free. OSPF and ECMP are special cases of the same rule.")


# ---------------------------------------------------------------- part: m1
def part_m1(a):
    title("METHODOLOGY 1  -  INDEPENDENT LEARNERS (no controller)")
    say("Every switch is its own agent. It sees ONLY its own links (1-hop) and",
        "sets ONLY its own link costs. No central controller exists, at training",
        "time or at run time. All agents share one team reward.",
        "",
        "Three algorithms, deliberately spanning the spectrum:",
        "   IPPO       - deep policy gradient, continuous actions",
        "   IQL-DQN    - deep Q-learning, discrete actions (keep/avoid/prefer)",
        "   Tabular-Q  - classical Q-learning, a lookup table, no neural net",
        "",
        "The hard part: non-stationarity. From one switch's point of view the",
        "environment keeps changing, because every other switch is learning too.")
    pause()
    G = load_named(a.topology)
    ms = build_methods(G, a.topology, {"Baseline", "M1"})
    ms = [(l, m) for l, m in ms if l in ("OSPF", "ECMP-hop", "M1 IPPO", "M1 IQL-DQN", "M1 Tabular-Q")]
    say(f"Running {len(ms)} methods on '{a.topology}' at load {a.load}, identical traffic:")
    res = race(G, ms, a.load, a.steps)
    table(res, [l for l, _ in ms])


# ---------------------------------------------------------------- part: m2
def part_m2(a):
    title("METHODOLOGY 2  -  ONE GNN THAT TRANSFERS TO ANY NETWORK")
    say("One graph neural network reads the WHOLE network and emits every link",
        "cost in a single forward pass. Its weights are shared across all nodes",
        "and edges, so the same model runs on a graph of ANY size.",
        "",
        "It was trained on 400 RANDOM SYNTHETIC graphs and has never seen a real",
        "network. We now hand it real ones, with no retraining and no fine-tuning.")
    pause()
    G = load_named(a.topology)
    ms = build_methods(G, a.topology, {"Baseline", "M2"})
    ms = [(l, m) for l, m in ms if "MLP" not in l and l in
          ("OSPF", "ECMP-hop", "LP-oracle(mean TM)", "M2 GNN zero-shot")]
    say(f"Network 1: '{a.topology}' ({G.number_of_nodes()} nodes) - never seen in training.")
    res = race(G, ms, a.load, a.steps)
    table(res, [l for l, _ in ms])
    pause()
    G2 = load_named(a.unseen)
    say(f"Network 2: '{a.unseen}' ({G2.number_of_nodes()} nodes) - a DIFFERENT SIZE.",
        "The exact same weights file. Nothing is reloaded or retrained.",
        "A fixed-width MLP cannot even accept this input; the GNN can.")
    ms2 = build_methods(G2, a.unseen, {"Baseline", "M2"})
    ms2 = [(l, m) for l, m in ms2 if l in ("OSPF", "ECMP-hop", "M2 GNN zero-shot")]
    res2 = race(G2, ms2, a.load, a.steps)
    table(res2, [l for l, _ in ms2])
    say("This is the paper's headline (mean of 3 training seeds): 81% of a",
        "topology-specific model's gain over OSPF is retained with zero training on",
        "the target, and the policy beats OSPF on 83% of 98 real wide-area networks.")


# ---------------------------------------------------------------- part: m3
def part_m3(a):
    title("METHODOLOGY 3  -  CENTRALISED TRAINING, DECENTRALISED EXECUTION")
    say("The actors are IDENTICAL to Methodology 1: at run time each switch still",
        "decides alone from its own 1-hop view. The ONLY difference is that during",
        "TRAINING the critic is allowed to see the global state.",
        "",
        "Two matched pairs, so the comparison is controlled:",
        "   IPPO  vs  MAPPO   (local critics  vs  one central value critic)",
        "   IQL   vs  QMIX    (independent Q  vs  monotonic mixed Q)",
        "Same actor, same actions, same hyper-parameters, same traffic.",
        "In the code this is literally one argument: make_local_critic -> make_central_critic.")
    pause()
    G = load_named(a.topology)
    ms = build_methods(G, a.topology, {"Baseline", "M1", "M3"})
    want = ("OSPF", "M1 IPPO", "M3 MAPPO", "M1 IQL-DQN", "M3 QMIX", "M3 MADDPG")
    ms = [(l, m) for l, m in ms if l in want]
    ms.sort(key=lambda x: want.index(x[0]))
    res = race(G, ms, a.load, a.steps)
    table(res, [l for l, _ in ms])
    d = res.get("M1 IPPO", {}).get("reward"), res.get("M3 MAPPO", {}).get("reward")



def show_training_stats(topo):
    f = os.path.join(ROOT, "results_unified", f"nonstationarity_summary_{topo}.csv")
    if not os.path.exists(f):
        say("(run scripts/evaluate_all.sh to generate the non-stationarity table)")
        return
    import pandas as pd
    d = pd.read_csv(f).set_index("algo")
    print("  " + "algo".ljust(10) + "best val   seed spread   critic explained variance")
    rule()
    for k in ("ippo", "mappo", "iql", "qmix"):
        if k not in d.index:
            continue
        r = d.loc[k]
        ev = r.get("critic_explained_var", float("nan"))
        ev = "   n/a" if ev != ev else f"{ev:6.2f}"
        print(f"  {k.ljust(10)}{r['best_val_mean']:.3f}      {r['final_val_sd']:.3f}"
              f"         {ev}")
    rule()
    say("MAPPO's critic explains far more return variance than IPPO's, and its",
        "seeds agree ~4x more closely. Centralisation buys STABILITY, not quality.")


# ---------------------------------------------------------------- main
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["env", "m1", "m2", "m3", "all"], default="all")
    ap.add_argument("--topology", default="nsfnet")
    ap.add_argument("--unseen", default="Abilene", help="second topology for the M2 transfer demo")
    ap.add_argument("--load", type=float, default=0.95)
    ap.add_argument("--steps", type=int, default=150)
    a = ap.parse_args()
    import torch
    torch.set_num_threads(1)
    parts = ["env", "m1", "m2", "m3"] if a.part == "all" else [a.part]
    for i, p in enumerate(parts):
        {"env": part_env, "m1": part_m1, "m2": part_m2, "m3": part_m3}[p](a)
        if i < len(parts) - 1:
            pause("[Enter] for the next methodology ")
    print()
