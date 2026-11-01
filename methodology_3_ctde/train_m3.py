"""Methodology 3 - train CTDE agents: MAPPO (central V critic), MADDPG (central Q critic)
or QMIX (monotonic mixer over per-switch Q-values, discrete actions).
The critic is used only in training; each switch executes its own actor on local state.

    python methodology_3_ctde/train_m3.py --algo mappo  --topology nsfnet --seeds 0 1 2
    python methodology_3_ctde/train_m3.py --algo maddpg --topology nsfnet --seeds 0 1 2
    python methodology_3_ctde/train_m3.py --algo qmix   --topology nsfnet --seeds 0 1 2

Outputs: results_marl/<algo>_<topology>_seed<k>/
"""
import os as _os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")   # one BLAS thread per process (parallel jobs otherwise thrash)

import argparse, json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from environments.topologies import load_named
from methodology_3_ctde.qmix_maddpg_engine import train_mappo, train_maddpg, train_qmix

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=["mappo", "maddpg", "qmix"], required=True)
    ap.add_argument("--topology", default="nsfnet")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--updates", type=int, default=None, help="MAPPO PPO updates (default 250)")
    ap.add_argument("--iters", type=int, default=None, help="MADDPG / QMIX iterations (default 5000 / 6000)")
    ap.add_argument("--scenario", default="pareto", choices=["poisson", "pareto", "microburst"])
    ap.add_argument("--out", default=os.path.join(ROOT, "results_marl"))
    a = ap.parse_args()
    G = load_named(a.topology)
    fn = dict(mappo=train_mappo, maddpg=train_maddpg, qmix=train_qmix)[a.algo]
    for s in a.seeds:
        over = dict(seed=s, scenario=a.scenario)
        if a.updates and a.algo == "mappo": over["updates"] = a.updates
        if a.iters and a.algo in ("maddpg", "qmix"): over["iters"] = a.iters
        out = os.path.join(a.out, f"{a.algo}_{a.topology}_seed{s}")
        best = fn(G, out, tag=a.algo, **over)
        json.dump(dict(algo=a.algo, topology=a.topology, seed=s, best_val_reward=best, overrides=over),
                  open(os.path.join(out, "meta.json"), "w"), indent=1)
        print(f"seed {s}: best validation reward {best:.4f}")
