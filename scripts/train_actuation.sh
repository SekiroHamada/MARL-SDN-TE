#!/bin/bash
# Actuation-aware fine-tuning (reviewer fix #5): every policy-gradient model is
# fine-tuned from its converged checkpoint with decisions held for 250 ms
# (25 x 10 ms, the order of the measured OpenFlow re-install time) and a reward
# penalty of 0.1 per unit share of forwarding groups re-installed per decision.
#   bash scripts/train_actuation.sh [workers] [filter-regex]
# Resumable: a run whose folder already has meta.json is skipped.
set -u
cd "$(dirname "$0")/.."
WORKERS=${1:-2}; FILTER=${2:-.}
HOLD=25; PEN=0.1
COMMON="--hold $HOLD --churn-penalty $PEN --lr 3e-4 --n-envs 8 --rollout 16 --eval-every 10 --suffix _act"
M2=methodology_2_gnn/v2/results
J=$(mktemp)
add() {   # add <run_dir> <command...>
  local dir=$1; shift
  [ -f "$dir/meta.json" ] && return
  echo "$dir" | grep -Eq "$FILTER" || return
  echo "$* > ${dir}.log 2>&1" >> "$J"
}
# longest jobs first (GNN), so the workers finish together
for s in 0 1 2; do
  add $M2/zero_shot_seed${s}_act python3 -W ignore methodology_2_gnn/v2/train_zero_shot.py --seed $s --updates 30 $COMMON --eval-steps 6 --init $M2/zero_shot_seed{seed}/gnn_best.pt
done
for t in Geant2012 nsfnet; do for s in 0 1 2; do
  add results_marl/ippo_${t}_seed${s}_act  python3 -W ignore methodology_1_marl/train_m1.py --algo ippo  --topology $t --seeds $s --updates 30 $COMMON --eval-steps 8 --init "results_marl/ippo_${t}_seed{seed}"
  add results_marl/mappo_${t}_seed${s}_act python3 -W ignore methodology_3_ctde/train_m3.py --algo mappo --topology $t --seeds $s --updates 30 $COMMON --eval-steps 8 --init "results_marl/mappo_${t}_seed{seed}"
done; done
echo "$(wc -l < $J) runs to do"
xargs -P "$WORKERS" -I{} bash -c "{}" < "$J"
rm -f "$J"
echo "ACTUATION TRAINING DONE"
