#!/bin/bash
# Train EVERYTHING for the paper (all methods, 3 training seeds), with N parallel workers.
# Resumable: a run is skipped if its meta.json exists (and was not cut short).
#   bash scripts/train_all.sh          # 2 workers (one per CPU core); use: bash scripts/train_all.sh 4
set -u
cd "$(dirname "$0")/.."
WORKERS=${1:-2}
J=results_marl/jobs_all.txt
mkdir -p results_marl/logs methodology_2_gnn/v2/results/logs
: > "$J"; rm -f "$J.lock"
L2=methodology_2_gnn/v2/results
add() {   # add <run_dir> <command...>   (skip finished runs; redo runs marked early_stopped)
  local dir=$1; shift
  if [ -f "$dir/meta.json" ] && ! grep -q early_stopped "$dir/meta.json"; then return; fi
  echo "rm -rf $dir; $* > ${dir}.log 2>&1" >> "$J"
}
# ---- Methodology 2: zero-shot GNN (3 seeds) + per-topology specialists (GNN, MLP)
for s in 0 1 2; do add $L2/zero_shot_seed$s python3 -W ignore methodology_2_gnn/v2/train_zero_shot.py --seed $s; done
for t in Geant2012 Abilene Nsfnet; do for m in gnn mlp; do
  add $L2/specialist_${m}_$t python3 -W ignore methodology_2_gnn/v2/train_specialist.py --model $m --topology $t
done; done
# ---- Methodology 1 (IPPO, IQL-DQN, tabular Q) and Methodology 3 (MAPPO, MADDPG, QMIX): 2 topologies x 3 seeds
for t in nsfnet Geant2012; do for s in 0 1 2; do
  for a in ippo iql tabq;     do add results_marl/${a}_${t}_seed$s python3 -W ignore methodology_1_marl/train_m1.py --algo $a --topology $t --seeds $s; done
  for a in mappo maddpg qmix; do add results_marl/${a}_${t}_seed$s python3 -W ignore methodology_3_ctde/train_m3.py --algo $a --topology $t --seeds $s; done
done; done
# ---- Tabular specialists for the Methodology 2 generalisation gap (Geant2012 reuses the M1 run).
# Own sub-folder: on case-insensitive drives (WSL /mnt/c) 'tabq_Nsfnet_seed0' (Topology-Zoo NSFNet)
# and 'tabq_nsfnet_seed0' (team NSFNet, M1) would be the SAME folder.
TS=results_marl/m2_tabular_specialists
mkdir -p $TS
for t in Abilene Nsfnet; do add $TS/tabq_${t}_seed0 python3 -W ignore methodology_1_marl/train_m1.py --algo tabq --topology $t --seeds 0 --out $TS; done
echo "$(wc -l < "$J") training jobs queued -> $J  (workers: $WORKERS)"
[ "$(wc -l < "$J")" -eq 0 ] && { echo "nothing to train - everything is done"; exit 0; }
LOGS=()
for w in $(seq 1 "$WORKERS"); do
  LOGS+=("results_marl/logs/train_all_worker$w.log")
  : > "${LOGS[-1]}"
  bash scripts/job_queue.sh "$PWD/$J" 0 "worker$w" > "${LOGS[-1]}" 2>&1 &
done
echo "running... progress: tail -f results_marl/logs/train_all_worker*.log   (each run's own log: <run_dir>.log)"
wait
echo "jobs finished OK: $(cat "${LOGS[@]}" | grep -c ' OK$')"
if grep -h "FAILED" "${LOGS[@]}"; then echo "some runs FAILED - check their .log files, then re-run: bash scripts/train_all.sh"
elif [ -s "$J" ]; then echo "jobs left unrun in $J - re-run: bash scripts/train_all.sh"
else echo "ALL TRAINING DONE"; fi
