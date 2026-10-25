"""Train Methodology 1 (IPPO / IQL / tabular Q) or Methodology 3 (MAPPO / MADDPG / QMIX) on one topology.

    python scripts/train_marl.py --algo ippo   --topology nsfnet --seed 0
    python scripts/train_marl.py --algo mappo  --topology nsfnet --seed 0
    python scripts/train_marl.py --algo maddpg --topology nsfnet --seed 0

Outputs: results_marl/<algo>_<topology>_seed<k>/  (training curve CSV, best/last actor).
"""
import os as _os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")   # one BLAS thread per process: parallel jobs otherwise thrash (measured ~10x slowdown)

import argparse, json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from environments.topologies import load_named
from methodology_1_marl.independent_marl_agent import train_ippo
from methodology_1_marl.independent_dqn import train_iql
from methodology_1_marl.tabular_q import train_tabular
from methodology_3_ctde.qmix_maddpg_engine import train_mappo, train_maddpg, train_qmix

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=["ippo", "iql", "tabq", "mappo", "maddpg", "qmix"], required=True)
    ap.add_argument("--topology", default="nsfnet")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--updates", type=int, default=None, help="PPO updates (ippo/mappo)")
    ap.add_argument("--iters", type=int, default=None, help="MADDPG / IQL / QMIX / tabular iterations")
    ap.add_argument("--out", default=os.path.join(ROOT, "results_marl"))
    a = ap.parse_args()
    G = load_named(a.topology)
    out = os.path.join(a.out, f"{a.algo}_{a.topology}_seed{a.seed}")
    over = dict(seed=a.seed)
    if a.updates: over["updates"] = a.updates
    if a.iters: over["iters"] = a.iters
    fn = dict(ippo=train_ippo, iql=train_iql, tabq=train_tabular, mappo=train_mappo,
              maddpg=train_maddpg, qmix=train_qmix)[a.algo]
    best = fn(G, out, tag=a.algo, **over)
    json.dump(dict(algo=a.algo, topology=a.topology, seed=a.seed, best_val_reward=best, overrides=over),
              open(os.path.join(out, "meta.json"), "w"), indent=1)
    print("best validation reward", best)
