"""
How much does non-stationarity cost? (Methodology 1 vs Methodology 3)

Two controlled pairs, each identical except for what the learner conditions on:
  IPPO (M1) vs MAPPO (M3)  - local critics V_v(o_v) vs central critic V(s)
  IQL  (M1) vs QMIX  (M3)  - per-agent TD targets vs targets mixed through Q_tot(s) We report, over training seeds:
  * best validation reward (mean, sd)          -> final quality and its variance
  * seed-to-seed sd of the validation curve    -> training instability
  * env steps / wall-clock to 90% of best      -> convergence speed
  * critic explained variance of returns       -> how predictable the target is
    from what each critic sees (local critics chase a moving target)
  * deployed-policy reward on the unified benchmark (if available)
MADDPG (off-policy CTDE) is reported alongside.

    python evaluation/nonstationarity_analysis.py --topology nsfnet
"""
import os as _os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")   # one BLAS thread per process: parallel jobs otherwise thrash (measured ~10x slowdown)

import argparse, glob, os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MARL = os.path.join(ROOT, "results_marl")
OUT = os.path.join(ROOT, "results_unified")
COL = {"ippo": "#eb6834", "mappo": "#2a78d6", "maddpg": "#1baf7a",
       "iql": "#eb6834", "qmix": "#2a78d6", "tabq": "#a3a29c"}
LABEL = {"ippo": "M1 IPPO (local critics)", "mappo": "M3 MAPPO (central critic)", "maddpg": "M3 MADDPG (central Q)",
         "iql": "M1 IQL-DQN (independent Q)", "qmix": "M3 QMIX (monotonic mixer)", "tabq": "M1 Tabular-Q"}
# The two controlled comparisons (identical code except the critic / TD target):
PAIRS = {"policy_gradient": ["ippo", "mappo", "maddpg"], "value_based": ["iql", "qmix", "tabq"]}


def main(topo):
    os.makedirs(OUT, exist_ok=True)
    rows, curves = [], {}
    for algo in PAIRS["policy_gradient"] + PAIRS["value_based"]:
        runs = sorted(glob.glob(os.path.join(MARL, f"{algo}_{topo}_seed*", f"{algo}_training_curve.csv")))
        vals = []
        for f in runs:
            d = pd.read_csv(f)
            x = "env_steps"
            v = d.dropna(subset=["val_reward"])
            v0, vb = v.val_reward.iloc[0], v.val_reward.max()
            hit = v[v.val_reward >= v0 + 0.9 * (vb - v0)].iloc[0]
            r = dict(algo=algo, seed=f.split("seed")[-1].split("/")[0], best_val=vb, final_val=v.val_reward.iloc[-1],
                     steps_to_90pct=int(hit[x]), wall_s_to_90pct=float(hit.wall_clock_s),
                     total_wall_s=float(d.wall_clock_s.iloc[-1]))
            if "explained_var" in d:
                r["critic_explained_var"] = float(d.explained_var.iloc[len(d) // 2:].mean())
                r["approx_kl"] = float(d.approx_kl.mean())
            rows.append(r)
            vals.append(v.set_index(x).val_reward)
        if vals:
            curves[algo] = pd.concat(vals, axis=1)
    df = pd.DataFrame(rows)
    if df.empty:
        print("no runs found"); return
    df.round(4).to_csv(os.path.join(OUT, f"nonstationarity_per_seed_{topo}.csv"), index=False)
    summ = df.groupby("algo").agg(n_seeds=("seed", "count"), best_val_mean=("best_val", "mean"),
                                  best_val_sd=("best_val", "std"), final_val_mean=("final_val", "mean"),
                                  final_val_sd=("final_val", "std"), steps_to_90pct=("steps_to_90pct", "mean"),
                                  wall_s_to_90pct=("wall_s_to_90pct", "mean"), total_wall_s=("total_wall_s", "mean"),
                                  **({"critic_explained_var": ("critic_explained_var", "mean")} if "critic_explained_var" in df else {}))
    summ["curve_seed_sd"] = [float(curves[a].std(axis=1).mean()) if a in curves and curves[a].shape[1] > 1 else np.nan
                             for a in summ.index]
    summ.round(4).to_csv(os.path.join(OUT, f"nonstationarity_summary_{topo}.csv"))
    print(summ.round(4).to_string())

    groups = [(k, [a for a in v if a in curves]) for k, v in PAIRS.items()]
    groups = [(k, v) for k, v in groups if v]
    fig, axes = plt.subplots(1, len(groups), figsize=(5.4 * len(groups), 3.1), dpi=160, squeeze=False)
    for ax, (title, algos) in zip(axes[0], groups):
        for a in algos:
            c = curves[a]
            m, s_ = c.mean(axis=1), c.std(axis=1).fillna(0)
            ax.plot(m.index / 1e3, m, color=COL[a], lw=2, label=LABEL[a])
            ax.fill_between(m.index / 1e3, m - s_, m + s_, color=COL[a], alpha=0.15, lw=0)
        ax.set_xlabel("environment steps (thousands)"); ax.set_ylabel("validation reward")
        ax.set_title(f"{title.replace('_', ' ')} on {topo} (mean ± sd over seeds)", fontsize=8)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        ax.grid(color="#e4e3df", lw=0.6); ax.legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, f"nonstationarity_curves_{topo}.png")); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--topology", default="nsfnet")
    main(ap.parse_args().topology)
