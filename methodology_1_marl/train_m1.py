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
    ap.add_argument("--hold", type=int, default=1, help="IPPO: decision interval in 10 ms steps (actuation-aware)")
    ap.add_argument("--churn-penalty", type=float, default=0.0, help="IPPO: reward cost per share of groups re-installed")
    ap.add_argument("--init", default=None, help="IPPO: fine-tune from this run folder's best actor")
    ap.add_argument("--suffix", default="", help="appended to the run folder name")
    ap.add_argument("--lr", type=float, default=None, help="IPPO: actor and critic learning rate")
    ap.add_argument("--n-envs", type=int, default=None)
    ap.add_argument("--eval-every", type=int, default=None)
    ap.add_argument("--eval-steps", type=int, default=None, help="validation decisions per case")
    ap.add_argument("--rollout", type=int, default=None, help="IPPO: decisions per rollout and per episode")
    a = ap.parse_args()
    G = load_named(a.topology)
    for s in a.seeds:
        over = dict(seed=s, scenario=a.scenario)
        if a.updates and a.algo == "ippo": over["updates"] = a.updates
        if a.iters and a.algo != "ippo": over["iters"] = a.iters
        if a.algo == "ippo" and (a.hold > 1 or a.churn_penalty > 0 or a.init):
            over.update(hold=a.hold, churn_penalty=a.churn_penalty)
            if a.lr: over.update(lr_actor=a.lr, lr_critic=a.lr)
            if a.n_envs: over["n_envs"] = a.n_envs
            if a.eval_every: over["eval_every"] = a.eval_every
            if a.eval_steps: over["eval_steps"] = a.eval_steps
            if a.rollout: over.update(rollout_len=a.rollout, episode_len=a.rollout)
            if a.init:
                over["init_actor"] = os.path.join(a.init.replace("{seed}", str(s)), f"{a.algo}_actor_best.pt")
        out = os.path.join(a.out, f"{a.algo}_{a.topology}_seed{s}{a.suffix}")
        fn = dict(ippo=train_ippo, iql=train_iql, tabq=train_tabular)[a.algo]
        best = fn(G, out, tag=a.algo, **over)
        json.dump(dict(algo=a.algo, topology=a.topology, seed=s, best_val_reward=best, overrides=over),
                  open(os.path.join(out, "meta.json"), "w"), indent=1)
        print(f"seed {s}: best validation reward {best:.4f}")
