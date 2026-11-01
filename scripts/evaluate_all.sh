#!/bin/bash
# Re-run every evaluation after training (writes all tables and figures).
#   bash scripts/evaluate_all.sh
set -e
cd "$(dirname "$0")/.."
python3 -W ignore methodology_1_marl/evaluate_m1.py --topology nsfnet
python3 -W ignore methodology_1_marl/evaluate_m1.py --topology Geant2012
python3 -W ignore methodology_3_ctde/evaluate_m3.py --topology nsfnet          # + non-stationarity analysis
python3 -W ignore methodology_3_ctde/evaluate_m3.py --topology Geant2012
python3 -W ignore methodology_2_gnn/v2/evaluate_m2.py --part primary
python3 -W ignore methodology_2_gnn/v2/evaluate_m2.py --part failure
python3 -W ignore methodology_2_gnn/v2/evaluate_m2.py --part suite
python3 -W ignore methodology_2_gnn/v2/summarize.py
python3 -W ignore evaluation/unified_benchmark.py --topologies nsfnet --tag _nsfnet &
python3 -W ignore evaluation/unified_benchmark.py --topologies Geant2012 --tag _geant
wait
python3 -W ignore evaluation/unified_benchmark.py --merge > /dev/null
python3 -W ignore evaluation/benchmark_evaluator.py
echo "ALL EVALUATIONS DONE - see results_marl/tables, methodology_2_gnn/v2/results/tables, results_unified/"
