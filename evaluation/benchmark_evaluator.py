# evaluation/benchmark_evaluator.py
"""
Week-1 milestone: OSPF / ECMP baseline table on the shared topologies at 50/80/95% load.

The Sprint-0 version produced these numbers from `random.uniform(...)` formulas
(no routing was simulated). This version routes real bursty traffic through the
shared simulator (environments/te_sim.py) with the same output columns.

    python evaluation/benchmark_evaluator.py      # -> week1_baseline_metrics.csv
"""
import os as _os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")   # one BLAS thread per process: parallel jobs otherwise thrash (measured ~10x slowdown)

import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from environments.topologies import load_named
from environments.te_sim import TESimulator
from baselines.te_baselines import Baseline
from evaluation.metrics_logger import MetricsLogger


def run_milestone_evaluator(steps=200, seeds=(1, 2, 3)):
    logger = MetricsLogger()
    for topo_name in ["nsfnet", "geant", "fattree_k4"]:
        G = load_named(topo_name)
        for load in [50, 80, 95]:
            for algo in ["OSPF", "ECMP"]:
                infos = []
                for seed in seeds:
                    sim = TESimulator(G, load=load / 100, seed=seed)
                    b = Baseline(algo)
                    for _ in range(steps):
                        infos.append(sim.step(P=b.act(sim)["P"])[2])
                logger.log_step(algo, topo_name, load,
                                float(np.mean([i["throughput_mbps"] for i in infos])),
                                float(np.mean([i["avg_delay_ms"] for i in infos])),
                                100 * float(np.mean([i["loss_ratio"] for i in infos])))
    print("\n--- Week 1 Milestone: Baseline Benchmark Results (simulated) ---")
    df = logger.export_to_csv("week1_baseline_metrics.csv")
    print(df.to_string())
    return df


if __name__ == "__main__":
    run_milestone_evaluator()
