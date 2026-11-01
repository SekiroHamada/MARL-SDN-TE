"""
Print the Mininet / Open vSwitch validation results as a table, grouped by
methodology. Use this when you want to SHOW the packet-level result without
re-running the emulation (which needs two terminals and root).

    python scripts/show_mininet.py
    python scripts/show_mininet.py --csv results_mininet/nsfnet_runs.csv
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W = 94

FAMILY = [("OSPF", "Baseline"), ("ECMP", "Baseline"), ("ECMP-hop", "Baseline"),
          ("LP-oracle", "Baseline"), ("M1 ", "M1  independent learners"),
          ("M2 ", "M2  graph policy (zero-shot)"), ("M3 ", "M3  CTDE")]


def family(name):
    for pre, fam in FAMILY:
        if name.startswith(pre):
            return fam
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.join(ROOT, "results_mininet", "nsfnet_runs.csv"))
    a = ap.parse_args()
    if not os.path.exists(a.csv):
        print(f"not found: {a.csv}\nRun the emulation first (mininet_validation/README.md).")
        return
    d = pd.read_csv(a.csv)
    d["family"] = d.method.map(family)

    print()
    print("=" * W)
    print("  PACKET-LEVEL VALIDATION  -  Mininet + Open vSwitch, real UDP traffic")
    print("=" * W)
    n_runs, n_m, n_s = len(d), d.method.nunique(), d.seed.nunique()
    pk = d.packets_sent.sum() if "packets_sent" in d else float("nan")
    print(f"  {n_runs} runs = {n_m} methods x {d.load.nunique()} loads x {n_s} traffic seeds"
          f"   |   {pk/1e6:.1f} million UDP packets sent in total")
    print("  OpenFlow 1.3 select groups carry the split weights; loss and delay are")
    print("  measured from receiver-side sequence numbers and timestamps.")
    print()

    for load in sorted(d.load.unique()):
        sub = d[d.load == load]
        g = sub.groupby(["family", "method"]).agg(
            loss=("loss_pct", "mean"), loss_sd=("loss_pct", "std"),
            delay=("delay_ms", "mean"), mlu=("mlu", "mean"),
            reward=("reward", "mean"),
            decision=("decision_ms", "mean"),
            install=("install_ms", "mean"),
            groups=("groups_modified_per_interval", "mean")).reset_index()
        g = g.sort_values("loss")
        print("-" * W)
        print(f"  OFFERED LOAD {load:.2f}")
        print("-" * W)
        print(f"  {'method':26s}{'family':30s}{'loss %':>9s}{'delay ms':>10s}"
              f"{'MLU':>7s}{'groups/int':>12s}")
        best = g.iloc[0].method
        for _, r in g.iterrows():
            sd = f" ±{r.loss_sd:.2f}" if r.loss_sd == r.loss_sd else ""
            print(f"  {r.method[:25]:26s}{r.family:30s}{r.loss:6.2f}{sd:<3s}"
                  f"{r.delay:10.1f}{r.mlu:7.2f}{r.groups:12.1f}"
                  + ("  <-- best" if r.method == best else ""))
        print()

    # the finding that only emulation can show
    pg = d[d.method.str.contains("IPPO|MAPPO|GNN", regex=True)]
    vb = d[d.method.str.contains("IQL|QMIX", regex=True)]
    if len(pg) and len(vb):
        print("=" * W)
        print("  WHAT ONLY THE EMULATION SHOWS")
        print("=" * W)
        print(f"  Installing new group weights costs "
              f"{d.install_ms[d.install_ms > 0].min():.0f}-{d.install_ms.max():.0f} ms per interval,")
        print(f"  far more than the policy's own decision time "
              f"({d.decision_ms.min():.1f}-{d.decision_ms.max():.1f} ms).")
        print(f"  Forwarding-state churn, groups rewritten per interval:")
        print(f"    policy-gradient methods (IPPO / MAPPO / GNN) : "
              f"{pg.groups_modified_per_interval.mean():.1f}")
        print(f"    value-based methods (IQL / QMIX)             : "
              f"{vb.groups_modified_per_interval.mean():.1f}"
              f"   ({vb.groups_modified_per_interval.mean() / max(1e-9, pg.groups_modified_per_interval.mean()):.1f}x more)")
        print("  Their greedy discrete action set flips between neighbouring choices.")
        print("  A fluid simulator cannot expose this cost at all.")
        print()


if __name__ == "__main__":
    main()
