"""
Evaluate the actuation-aware fine-tuned policies (reviewer fix #5).

Each *_act checkpoint (scripts/train_actuation.sh: decisions held 250 ms,
churn penalty 0.1) is evaluated exactly like the control-interval study
(actuation_study.py, Pareto bursts at 95% load, the same five traffic seeds)
at a 10 ms and a 250 ms interval, and compared with the original policy rows
already in actuation_interval_per_seed.csv.

    python evaluation/actuation_finetune.py
Writes results_unified/actuation_finetune_per_seed.csv and _summary.csv
"""
import os as _os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")

import glob, os, sys
import pandas as pd, torch, yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import unified_benchmark as ub
from actuation_study import run_held, _resume

HOLDS = [1, 25]
SCEN, LOAD = "pareto", 0.95


def act_models(topo, G):
    env = ub.FlowSplittingEnv(G)
    cfg = yaml.safe_load(open(os.path.join(ROOT, "methodology_2_gnn", "v2", "config_v2.yaml")))
    out = []
    done = lambda p: os.path.exists(os.path.join(os.path.dirname(p), "meta.json"))   # finished runs only
    for p in filter(done, sorted(glob.glob(os.path.join(ub.M2, "zero_shot_seed[0-9]_act", "gnn_best.pt")))):
        s = os.path.basename(os.path.dirname(p)).split("_")[2]
        m = ub.GNNPolicy(**cfg["gnn"]); m.load_state_dict(torch.load(p))
        out.append(ub.GraphAgent(f"M2 GNN zero-shot ({s})", m, "M2"))
    for algo in ("ippo", "mappo"):
        for p in filter(done, sorted(glob.glob(os.path.join(ub.MARL, f"{algo}_{topo}_seed[0-9]_act", f"{algo}_actor_best.pt")))):
            s = os.path.basename(os.path.dirname(p)).split("_")[-2]
            a = ub.DecentralizedForwarder(env.n, env.obs_dim, env.act_dim)
            a.load_state_dict(torch.load(p))
            out.append(ub.SwitchAgents(f"{ub.FAMILY[algo]} {ub.LABEL[algo]} ({s})", a, ub.FAMILY[algo]))
    return out


def main():
    path = os.path.join(ub.OUT, "actuation_finetune_per_seed.csv")
    out, done = _resume(path)
    for topo in ub.TOPOLOGIES:
        G = ub.load_named(topo)
        for m in act_models(topo, G):
            if (topo, m.name) in done:
                continue
            for k in HOLDS:
                for seed in ub.SEEDS:
                    r = run_held(m, G, SCEN, LOAD, seed, hold=k)
                    r.update(topology=topo, method=m.name, family=m.family, hold_steps=k,
                             interval_ms=10 * k, scenario=SCEN, load=LOAD, seed=seed)
                    out.append(r)
            print(f"{topo:10s} {m.name:28s} done", flush=True)
            pd.DataFrame(out).to_csv(path, index=False)
    summarize(pd.DataFrame(out))


def summarize(ft):
    orig = pd.read_csv(os.path.join(ub.OUT, "actuation_interval_per_seed.csv"))
    orig = orig[orig.interval_ms.isin([10 * k for k in HOLDS])]
    orig = orig[orig.method.str.match(r"(M1 IPPO|M3 MAPPO|M2 GNN zero-shot) \(seed\d\)$")]
    orig["variant"], ft["variant"] = "original", "actuation-aware"
    df = pd.concat([orig, ft], ignore_index=True)
    df["loss_pct"] = 100 * df.loss_ratio
    df["algo"] = df["method"].str.replace(r" \(seed\d+\)$", "", regex=True)
    df["seed_id"] = df["method"].str.extract(r"\((seed\d)\)$")[0]
    # only (topology, algo, seed) for which a fine-tuned model exists, so both variants are paired
    have = set(zip(ft.topology, ft.method))
    df = df[[(t, m) in have for t, m in zip(df.topology, df.method)]]
    per = df.groupby(["topology", "algo", "variant", "seed_id", "interval_ms"]).agg(
        reward=("reward", "mean"), loss_pct=("loss_pct", "mean"),
        groups_per_decision=("groups_per_decision", "mean"),
        groups_per_second=("groups_per_second", "mean")).reset_index()
    agg = per.groupby(["topology", "algo", "variant", "interval_ms"]).agg(
        n_seeds=("seed_id", "nunique"), reward=("reward", "mean"), reward_seed_sd=("reward", "std"),
        loss_pct=("loss_pct", "mean"), groups_per_decision=("groups_per_decision", "mean"),
        groups_per_second=("groups_per_second", "mean")).reset_index()
    agg.round(5).to_csv(os.path.join(ub.OUT, "actuation_finetune_summary.csv"), index=False)
    print(agg.to_string())


if __name__ == "__main__":
    torch.set_num_threads(1)
    if len(sys.argv) > 1 and sys.argv[1] == "--summarize":
        summarize(pd.read_csv(os.path.join(ub.OUT, "actuation_finetune_per_seed.csv")))
    else:
        main()
