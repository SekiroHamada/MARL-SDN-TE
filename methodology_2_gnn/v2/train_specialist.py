"""Train a per-topology specialist (the 'trained-on-target' ceiling).

    python methodology_2_gnn/v2/train_specialist.py --model mlp --topology Geant2012
    python methodology_2_gnn/v2/train_specialist.py --model gnn --topology Abilene

The specialist sees ONLY the target topology (random loads + traffic matrices,
traffic seeds disjoint from the evaluation seeds). Comparing the zero-shot GNN
against it gives the generalisation gap the project asks for.
"""
import os as _os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")   # one BLAS thread per process: parallel jobs otherwise thrash (measured ~10x slowdown)

import argparse, os, sys, yaml, json
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import topologies as T
from policies import GNNPolicy, MLPPolicy, count_params
from ppo import train

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["gnn", "mlp"], required=True)
    ap.add_argument("--topology", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--updates", type=int, default=200)
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    ap.add_argument("--node-actions", action="store_true", help="ablation: per-node split temperature")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(os.path.join(HERE, "config_v2.yaml")))
    torch.set_num_threads(1)
    G = T.load_named(args.topology)
    lo, hi = cfg["train_loads"]
    pcfg = dict(cfg["ppo"]); pcfg.update(seed=args.seed, updates=args.updates)
    vrng = np.random.default_rng(11)
    val_cases = [(G, float(l), 900 + i) for i, l in enumerate(np.linspace(lo, hi, 8))]
    if args.model == "gnn":
        model = GNNPolicy(**cfg["gnn"], node_actions=args.node_actions)
    else:
        model = MLPPolicy(G.number_of_nodes(), 2 * G.number_of_edges())
    print(f"{args.model.upper()} specialist on {args.topology} ({G.number_of_nodes()} nodes, "
          f"{G.number_of_edges()} links) params {count_params(model):,}")
    out = os.path.join(args.out, f"specialist_{args.model}_{args.topology}{args.tag}")
    best = train(model, lambda rng: (G, float(rng.uniform(lo, hi))), pcfg, out, val_cases=val_cases, tag=args.model)
    json.dump({"best_val_reward": best, "topology": args.topology, "model": args.model},
              open(os.path.join(out, "meta.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
