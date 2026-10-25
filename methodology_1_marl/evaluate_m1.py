"""Methodology 1 evaluation - IPPO, independent DQN and tabular Q vs OSPF / ECMP / LP oracle
on the topology they were trained on.
Covers the M1 validation tests: packet loss vs ECMP at 80%+ load (Pareto bursts) and the
micro-burst stress scenario.

    python methodology_1_marl/evaluate_m1.py --topology nsfnet
-> results_marl/tables/m1_<topology>.csv and m1_<topology>_summary.csv
"""
import os as _os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")   # one BLAS thread per process (parallel jobs otherwise thrash)

import argparse, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from environments.topologies import load_named
from evaluation.marl_eval import evaluate

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--topology", default="nsfnet")
    t = ap.parse_args().topology
    evaluate(load_named(t), t, ["ippo", "iql", "tabq"], os.path.join(ROOT, "results_marl", "tables", f"m1_{t}.csv"))
