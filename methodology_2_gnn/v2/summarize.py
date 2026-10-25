"""Aggregate evaluation CSVs into report tables + figures (results/tables, results/figures)."""
import glob, os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results"); TAB = os.path.join(RES, "tables"); FIG = os.path.join(RES, "figures")
os.makedirs(FIG, exist_ok=True)
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "figure.dpi": 160})


def ci95(x):
    x = np.asarray(x, float)
    return 1.96 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0


def zs_label(df):
    return [m for m in df.method.unique() if m.startswith("GNN zero-shot")]


def primary():
    p = os.path.join(TAB, "m2_primary_per_seed.csv")
    if not os.path.exists(p):
        return
    df = pd.read_csv(p)
    df["loss_pct"] = 100 * df.loss_ratio
    agg = df.groupby(["topology", "load", "method"]).agg(
        reward=("reward", "mean"), reward_ci95=("reward", ci95), loss_pct=("loss_pct", "mean"),
        delay_ms=("avg_delay_ms", "mean"), throughput_mbps=("throughput_mbps", "mean"),
        mlu=("mlu", "mean"), util_std=("util_std", "mean"), jain=("jain", "mean"),
        inference_ms=("inference_ms", "mean") if "inference_ms" in df else ("reward", "size")).reset_index()
    agg.round(4).to_csv(os.path.join(TAB, "m2_primary_summary.csv"), index=False)

    # generalisation gap: zero-shot vs specialist on the same topology/load/seeds
    rows = []
    zs = zs_label(df)
    for (topo, load), g in df.groupby(["topology", "load"]):
        R = g.groupby("method").reward.mean()
        if "OSPF" not in R:
            continue
        for z in zs:
            for spec in ["GNN specialist", "MLP specialist", "Tabular specialist"]:
                if spec not in R or z not in R:
                    continue
                gain_spec = R[spec] - R["OSPF"]
                rows.append(dict(topology=topo, load=load, zero_shot=z, specialist=spec,
                                 R_ospf=R["OSPF"], R_zero_shot=R[z], R_specialist=R[spec],
                                 gap_abs=R[spec] - R[z],
                                 gap_pct_of_specialist=100 * (R[spec] - R[z]) / abs(R[spec]),
                                 improvement_retained_pct=100 * (R[z] - R["OSPF"]) / gain_spec if abs(gain_spec) > 1e-3 else np.nan))
    if rows:
        pd.DataFrame(rows).round(4).to_csv(os.path.join(TAB, "m2_generalization_gap.csv"), index=False)

    # figure: reward vs load per topology
    methods = ["OSPF", "ECMP-hop", "LP-oracle(mean TM)", "Tabular specialist", "MLP specialist", "GNN specialist"] + zs[:1]
    colors = {"OSPF": MUTED, "ECMP-hop": "#a3a29c", "LP-oracle(mean TM)": VIOLET, "MLP specialist": ORANGE,
              "Tabular specialist": "#e87ba4",
              "GNN specialist": AQUA}
    topos = list(dict.fromkeys(df.topology))
    fig, axes = plt.subplots(1, len(topos), figsize=(2.6 * len(topos), 2.6), sharey=False)
    for ax, t in zip(np.atleast_1d(axes), topos):
        for m in methods:
            g = agg[(agg.topology == t) & (agg.method == m)]
            if g.empty:
                continue
            c = BLUE if m.startswith("GNN zero") else colors.get(m, MUTED)
            lw = 2.2 if m.startswith("GNN zero") else 1.4
            ls = "--" if m.startswith("LP") else "-"
            ax.plot(g.load, g.reward, ls, color=c, lw=lw, marker="o", ms=3.5,
                    label=("GNN zero-shot" if m.startswith("GNN zero") else m))
        ax.set_title(t, color=INK, fontsize=9); ax.set_xlabel("offered load (OSPF MLU)")
    np.atleast_1d(axes)[0].set_ylabel("mean reward per step")
    h, l = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(l), frameon=False, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "reward_vs_load.png"), bbox_inches="tight"); plt.close(fig)


def suite():
    p = os.path.join(TAB, "m2_zoo_suite_per_seed.csv")
    if not os.path.exists(p):
        return
    df = pd.read_csv(p)
    zs = zs_label(df)
    per = df.groupby(["topology", "nodes", "load", "method"]).agg(reward=("reward", "mean"),
                                                                   loss=("loss_ratio", "mean"),
                                                                   delay=("avg_delay_ms", "mean")).reset_index()
    wide = per.pivot_table(index=["topology", "nodes", "load"], columns="method", values="reward").reset_index()
    out = []
    for load, g in wide.groupby("load"):
        for m in [c for c in wide.columns if c not in ("topology", "nodes", "load")]:
            d = g[m] - g["OSPF"]
            lossm = per[(per.load == load) & (per.method == m)].loss.mean()
            out.append(dict(load=load, method=m, mean_reward=g[m].mean(), mean_delta_vs_ospf=d.mean(),
                            ci95=ci95(d), win_rate_vs_ospf_pct=100 * (d > 0).mean(),
                            mean_loss_pct=100 * lossm, n_topologies=len(g)))
    pd.DataFrame(out).round(4).to_csv(os.path.join(TAB, "m2_zoo_suite_summary.csv"), index=False)
    if zs:
        z = zs[0]
        fig, ax = plt.subplots(figsize=(5.2, 3.0))
        for load, c in [(0.8, "#8fb8ea"), (0.95, BLUE)]:
            g = wide[wide.load == load]
            ax.scatter(g.nodes, g[z] - g["OSPF"], s=14, color=c, label=f"load {load}", edgecolor="white", lw=0.5)
        ax.axhline(0, color=MUTED, lw=1)
        ax.set_xlabel("nodes in unseen real topology"); ax.set_ylabel("reward gain vs OSPF")
        ax.set_title(f"Zero-shot GNN on {wide.topology.nunique()} Topology Zoo WANs", color=INK, fontsize=9)
        ax.legend(frameon=False)
        fig.tight_layout(); fig.savefig(os.path.join(FIG, "zoo_suite_gain.png")); plt.close(fig)


def failure():
    p = os.path.join(TAB, "m2_failure_per_trial.csv")
    if not os.path.exists(p):
        return
    df = pd.read_csv(p)
    df["d_reward"] = df.fail_reward - df.ctrl_reward
    df["d_loss_pct"] = 100 * (df.fail_loss_ratio - df.ctrl_loss_ratio)
    s = df.groupby("method").agg(post_failure_reward=("fail_reward", "mean"), control_reward=("ctrl_reward", "mean"),
                                 delta_reward=("d_reward", "mean"), delta_ci95=("d_reward", ci95),
                                 post_failure_loss_pct=("fail_loss_ratio", lambda x: 100 * x.mean()),
                                 delta_loss_pct=("d_loss_pct", "mean"), trials=("trial", "nunique")).reset_index()
    s.round(4).to_csv(os.path.join(TAB, "m2_failure_summary.csv"), index=False)


def curves():
    fs = sorted(glob.glob(os.path.join(RES, "zero_shot_seed*", "gnn_training_curve.csv")))
    if not fs:
        return
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    for i, f in enumerate(fs):
        d = pd.read_csv(f)
        v = d.dropna(subset=["val_reward"])
        ax.plot(d.env_steps / 1e3, d.train_reward.rolling(20, min_periods=1).mean(), color="#a3a29c", lw=1,
                label="train (rolling mean)" if i == 0 else None)
        ax.plot(v.env_steps / 1e3, v.val_reward, color=BLUE, lw=2, marker="o", ms=3,
                label="validation (unseen synthetic)" if i == 0 else None)
    base = os.path.join(TAB, "m2_val_baselines.csv")
    if os.path.exists(base):
        for _, r in pd.read_csv(base).iterrows():
            ax.axhline(r.reward, color=VIOLET if "LP" in r.method else MUTED, lw=1, ls="--")
            ax.text(ax.get_xlim()[1], r.reward, " " + r.method, va="center", fontsize=7, color=MUTED)
    ax.set_xlabel("environment steps (thousands)"); ax.set_ylabel("mean reward per step")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "training_curve.png"), bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    primary(); suite(); failure(); curves()
    print("tables:", sorted(os.listdir(TAB))); print("figures:", sorted(os.listdir(FIG)))
