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
    ap.add_argument("--hold", type=int, default=1, help="MAPPO: decision interval in 10 ms steps (actuation-aware)")
    ap.add_argument("--churn-penalty", type=float, default=0.0, help="MAPPO: reward cost per share of groups re-installed")
    ap.add_argument("--init", default=None, help="MAPPO: fine-tune from this run folder's best actor")
    ap.add_argument("--suffix", default="", help="appended to the run folder name")
    ap.add_argument("--lr", type=float, default=None, help="MAPPO: actor and critic learning rate")
    ap.add_argument("--n-envs", type=int, default=None)
    ap.add_argument("--eval-every", type=int, default=None)
    ap.add_argument("--eval-steps", type=int, default=None, help="validation decisions per case")
    ap.add_argument("--rollout", type=int, default=None, help="MAPPO: decisions per rollout and per episode")
    a = ap.parse_args()
    G = load_named(a.topology)
    fn = dict(mappo=train_mappo, maddpg=train_maddpg, qmix=train_qmix)[a.algo]
    for s in a.seeds:
        over = dict(seed=s, scenario=a.scenario)
        if a.updates and a.algo == "mappo": over["updates"] = a.updates
        if a.iters and a.algo in ("maddpg", "qmix"): over["iters"] = a.iters
        if a.algo == "mappo" and (a.hold > 1 or a.churn_penalty > 0 or a.init):
            over.update(hold=a.hold, churn_penalty=a.churn_penalty)
            if a.lr: over.update(lr_actor=a.lr, lr_critic=a.lr)
            if a.n_envs: over["n_envs"] = a.n_envs
            if a.eval_every: over["eval_every"] = a.eval_every
            if a.eval_steps: over["eval_steps"] = a.eval_steps
            if a.rollout: over.update(rollout_len=a.rollout, episode_len=a.rollout)
            if a.init:
                over["init_actor"] = os.path.join(a.init.replace("{seed}", str(s)), f"{a.algo}_actor_best.pt")
        out = os.path.join(a.out, f"{a.algo}_{a.topology}_seed{s}{a.suffix}")
        best = fn(G, out, tag=a.algo, **over)
        json.dump(dict(algo=a.algo, topology=a.topology, seed=s, best_val_reward=best, overrides=over),
                  open(os.path.join(out, "meta.json"), "w"), indent=1)
        print(f"seed {s}: best validation reward {best:.4f}")
