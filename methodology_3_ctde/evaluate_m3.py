"""Methodology 3 evaluation - MAPPO, MADDPG and QMIX vs their M1 counterparts and the
baselines, plus the non-stationarity analysis over training seeds:
    IPPO (M1) vs MAPPO (M3)     - policy gradient, continuous actions
    IQL-DQN (M1) vs QMIX (M3)   - value-based, discrete actions

    python methodology_3_ctde/evaluate_m3.py --topology nsfnet
-> results_marl/tables/m3_<topology>.csv, *_summary.csv
-> results_unified/nonstationarity_summary_<topology>.csv + curves figure
"""
import os as _os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")   # one BLAS thread per process (parallel jobs otherwise thrash)

import argparse, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from environments.topologies import load_named
from evaluation.marl_eval import evaluate
from evaluation import nonstationarity_analysis

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--topology", default="nsfnet")
    t = ap.parse_args().topology
    evaluate(load_named(t), t, ["ippo", "iql", "mappo", "maddpg", "qmix"], os.path.join(ROOT, "results_marl", "tables", f"m3_{t}.csv"))
    print("\n=== non-stationarity: IPPO (M1) vs MAPPO / MADDPG (M3) ===")
    nonstationarity_analysis.main(t)
