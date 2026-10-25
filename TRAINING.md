# Training and evaluation guide (complete project)

Everything runs on CPU. Run every command from the repository root (the folder
that contains `README.md`). Times are for one core of a typical laptop; WSL on
`/mnt/c/...` is slower — see step 0.

## 0. One-time setup

```bash
# (recommended on WSL) work inside the Linux file system: faster and case-sensitive
cp -r MARL-SDN-TE ~/ && cd ~/MARL-SDN-TE

cd "/mnt/c/Users/SYED SAAD ALI/Desktop/VIT STUDY/Material/5th SEM/CN Theory/Project/codebase"

python3 -m venv marl-sdn && source marl-sdn/bin/activate
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r methodology_2_gnn/v2/requirements.txt

# smoke test (seconds): every method trains for a moment
python methodology_1_marl/train_m1.py --algo iql  --topology nsfnet --seeds 9 --iters 200 --out /tmp/t
python methodology_3_ctde/train_m3.py --algo qmix --topology nsfnet --seeds 9 --iters 200 --out /tmp/t
python methodology_2_gnn/v2/train_zero_shot.py --seed 9 --updates 2 --out /tmp/t
```

## 1. Train everything — one command

```bash
bash scripts/train_all.sh        # 2 parallel workers; use `bash scripts/train_all.sh 4` on a 4+ core machine
```
* Queues every run that is missing and skips every run that is already finished,
  so you can stop it (Ctrl-C) and start it again later — it resumes.
* The zip already contains finished runs (M2 seed 0 and specialists, M1 IPPO and
  M3 MAPPO/MADDPG on NSFNet x 3 seeds and GÉANT seed 0). `train_all.sh` trains
  only what is missing (~29 runs) and redoes the two runs that were cut short
  (M2 zero-shot seed 1, Abilene GNN specialist).
* Progress: `tail -f results_marl/logs/train_all_worker*.log`. Each run also has its own log
  next to its folder, e.g. `results_marl/qmix_nsfnet_seed1.log`.
* Expected total: **~6–10 hours with 2 workers** (GÉANT runs dominate). Leave it
  overnight. To retrain *everything* from scratch, delete `results_marl/*_seed*`
  and `methodology_2_gnn/v2/results/{zero_shot,specialist}_*` first.

### What it trains (and how to run any piece by hand)

| Methodology | Method | Command (one seed) | Approx. time |
|---|---|---|---|
| M2 | GNN zero-shot (synthetic graphs only) | `python methodology_2_gnn/v2/train_zero_shot.py --seed 1` | 35–60 min |
| M2 | GNN / MLP specialist (one topology) | `python methodology_2_gnn/v2/train_specialist.py --model gnn --topology Geant2012` | 5–45 min |
| M1 | IPPO (independent actor-critic) | `python methodology_1_marl/train_m1.py --algo ippo --topology nsfnet --seeds 1` | 5 min (GÉANT 15–25) |
| M1 | IQL-DQN (independent Q-networks) | `python methodology_1_marl/train_m1.py --algo iql --topology nsfnet --seeds 1` | 4 min (GÉANT 10–15) |
| M1 | Tabular Q-learning | `python methodology_1_marl/train_m1.py --algo tabq --topology nsfnet --seeds 1` | 4 min (GÉANT 10) |
| M3 | MAPPO (central V critic) | `python methodology_3_ctde/train_m3.py --algo mappo --topology nsfnet --seeds 1` | 5 min (GÉANT 15–25) |
| M3 | MADDPG (central Q critic) | `python methodology_3_ctde/train_m3.py --algo maddpg --topology nsfnet --seeds 1` | 3 min (GÉANT 6–10) |
| M3 | QMIX (monotonic mixer) | `python methodology_3_ctde/train_m3.py --algo qmix --topology nsfnet --seeds 1` | 5 min (GÉANT 12–18) |

Topology names are case-sensitive: `nsfnet` = team 14-node file (M1/M3 are trained
on it), `Nsfnet` = Topology Zoo 13-node NSFNet, `Geant2012`, `Abilene`.
Tabular agents are also trained on `Abilene` and `Nsfnet` because they are the
tabular baseline of the M2 generalisation gap.

## 2. Evaluate everything — one command

```bash
bash scripts/evaluate_all.sh     # ~1.5–2 h
```
Writes:
| Output | Contents |
|---|---|
| `results_marl/tables/m1_<topo>*.csv` | M1 (IPPO, IQL, tabular) vs OSPF/ECMP/LP |
| `results_marl/tables/m3_<topo>*.csv` | M3 (MAPPO, MADDPG, QMIX) vs M1 vs baselines |
| `results_unified/nonstationarity_summary_<topo>.csv` + `_curves_<topo>.png` | IPPO vs MAPPO and IQL vs QMIX over 3 seeds |
| `methodology_2_gnn/v2/results/tables/` | zero-shot, generalisation gap (vs GNN / MLP / tabular specialists), 98-WAN suite, link failures |
| `methodology_2_gnn/v2/results/figures/` | reward-vs-load, suite gains, training curve |
| `results_unified/unified_summary.csv`, `training_cost.csv` | all methods, all 9 metrics, convergence |
| `week1_baseline_metrics.csv` (repo root) | Week-1 OSPF/ECMP table |

## 3. Mininet validation (emulated SDN)

See `mininet_validation/README.md`. Two terminals:
```bash
python mininet_validation/policy_server.py --topology nsfnet                 # terminal 1 (venv)
sudo mn -c && sudo python3 mininet_validation/run_mininet.py --check         # terminal 2
sudo python3 mininet_validation/run_mininet.py --methods OSPF ECMP-hop \
  "M2 GNN zero-shot (seed0)" "M1 IPPO (seed0)" "M1 IQL-DQN (seed0)" \
  "M3 MAPPO (seed0)" "M3 QMIX (seed0)" --loads 0.8 0.95 --seeds 101 102 103
```
`python mininet_validation/policy_server.py --list` prints every available method name.

## 4. After the runs
Send the new `results_*` tables (or the whole folder) to update `REPORT.md`: the
numbers in it are from the runs available when it was written (M1/M3 value-based
methods and the extra seeds were added afterwards).
