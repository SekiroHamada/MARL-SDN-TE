"""Train ONE GNN policy on synthetic graphs only (zero-shot to every real WAN).

    python methodology_2_gnn/v2/train_zero_shot.py --seed 0
"""
import os as _os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")   # one BLAS thread per process: parallel jobs otherwise thrash (measured ~10x slowdown)

import argparse, os, sys, yaml, json
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import topologies as T
from policies import GNNPolicy, count_params
from ppo import train

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--updates", type=int, default=None)
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    args = ap.parse_args()
    cfg = yaml.safe_load(open(os.path.join(HERE, "config_v2.yaml")))
    torch.set_num_threads(1)
    pcfg = dict(cfg["ppo"]); pcfg["seed"] = args.seed
    if args.updates: pcfg["updates"] = args.updates
    pool = T.synthetic_pool(cfg["data"]["synthetic_pool"], tuple(cfg["data"]["node_range"]), seed=1000 + args.seed)
    val_graphs = T.synthetic_pool(cfg["data"]["synthetic_val"], tuple(cfg["data"]["node_range"]), seed=99_999)
    lo, hi = cfg["train_loads"]
    vrng = np.random.default_rng(7)
    val_cases = [(g, float(vrng.uniform(lo, hi)), 500 + i) for i, g in enumerate(val_graphs)]

    def sampler(rng):
        return pool[int(rng.integers(len(pool)))], float(rng.uniform(lo, hi))

    model = GNNPolicy(**cfg["gnn"])
    print(f"GNN params: {count_params(model):,} | train pool {len(pool)} synthetic graphs "
          f"(mean deg {np.mean([2*g.number_of_edges()/g.number_of_nodes() for g in pool]):.2f})")
    out = os.path.join(args.out, f"zero_shot_seed{args.seed}")
    best = train(model, sampler, pcfg, out, val_cases=val_cases, tag="gnn")
    json.dump({"best_val_reward": best, "config": cfg, "seed": args.seed}, open(os.path.join(out, "meta.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
