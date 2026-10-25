"""Methodology 1 - train independent learners (no central critic, local observations only).

    --algo ippo  : independent PPO, continuous split weights (actor + local critic per switch)
    --algo iql   : independent DQN per switch, discrete actions (keep / avoid / prefer a link)
    --algo tabq  : independent TABULAR Q-learning per switch, discrete actions

    python methodology_1_marl/train_m1.py --algo ippo --topology nsfnet --seeds 0 1 2
    python methodology_1_marl/train_m1.py --algo iql  --topology nsfnet --seeds 0 1 2
    python methodology_1_marl/train_m1.py --algo tabq --topology nsfnet --seeds 0 1 2

Outputs: results_marl/<algo>_<topology>_seed<k>/  (training curve CSV, checkpoint, meta.json)
"""
import os as _os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")   # one BLAS thread per process (parallel jobs otherwise thrash)

import argparse, json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from environments.topologies import load_named
from methodology_1_marl.independent_marl_agent import train_ippo
from methodology_1_marl.independent_dqn import train_iql
from methodology_1_marl.tabular_q import train_tabular

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=["ippo", "iql", "tabq"], default="ippo")
    ap.add_argument("--topology", default="nsfnet")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--updates", type=int, default=None, help="IPPO: PPO updates (default 250)")
    ap.add_argument("--iters", type=int, default=None, help="IQL / tabular: iterations (default 6000 / 8000)")
    ap.add_argument("--scenario", default="pareto", choices=["poisson", "pareto", "microburst"])
    ap.add_argument("--out", default=os.path.join(ROOT, "results_marl"))
    a = ap.parse_args()
    G = load_named(a.topology)
    for s in a.seeds:
        over = dict(seed=s, scenario=a.scenario)
        if a.updates and a.algo == "ippo": over["updates"] = a.updates
        if a.iters and a.algo != "ippo": over["iters"] = a.iters
        out = os.path.join(a.out, f"{a.algo}_{a.topology}_seed{s}")
        fn = dict(ippo=train_ippo, iql=train_iql, tabq=train_tabular)[a.algo]
        best = fn(G, out, tag=a.algo, **over)
        json.dump(dict(algo=a.algo, topology=a.topology, seed=s, best_val_reward=best, overrides=over),
                  open(os.path.join(out, "meta.json"), "w"), indent=1)
        print(f"seed {s}: best validation reward {best:.4f}")
