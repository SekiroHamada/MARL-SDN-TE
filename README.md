# MARL-SDN-TE

Multi-agent reinforcement learning for traffic engineering in software-defined networks.
Three methodologies are trained and compared against classical routing on the same
flow-level simulator, and the best policies are validated on an emulated SDN data plane
(Mininet + Open vSwitch).

| | Methodology | Algorithms | Code |
|---|---|---|---|
| **M1** | Decentralised independent MARL (one agent per switch, local view only) | IPPO, IQL-DQN, Tabular Q-learning | `methodology_1_marl/` |
| **M2** | Topology-agnostic GNN policy (one model for any graph, zero-shot transfer) | GNN zero-shot, GNN / MLP / tabular specialists | `methodology_2_gnn/v2/` |
| **M3** | Centralised training, decentralised execution (CTDE) | MAPPO, MADDPG (TD3 fixes), QMIX | `methodology_3_ctde/` |
| Baselines | Classical and oracle routing | OSPF, ECMP, ECMP-hop, static softmin, LP oracle on the mean traffic matrix | `baselines/` |

Full results and discussion: **[REPORT.md](REPORT.md)**.

## Repository layout
```
environments/        te_sim.py (flow-level simulator), flow_splitting_env.py (per-switch MARL env),
                     bursty_traffic_gen.py, topologies.py, gym_env.py (Gymnasium wrapper)
baselines/           OSPF / ECMP / ECMP-hop / softmin / LP oracle
models/              shared networks: per-switch forwarder, on-policy and value-based MARL loops
methodology_1_marl/  M1 trainers and evaluator
methodology_2_gnn/   v2/ = current GNN methodology (v1 kept for history, see V1_NOTICE.md)
methodology_3_ctde/  M3 trainers, centralised critic, QMIX mixer, evaluator
evaluation/          unified cross-methodology benchmark, non-stationarity analysis, metrics
mininet_validation/  policy server + Mininet runner (OpenFlow select groups, tc-shaped links)
scripts/             train_all.sh, evaluate_all.sh, demos
data/topologies/     NSFNet, GEANT, Abilene, fat-tree k=4, Topology Zoo graphs
results_marl/        M1/M3 checkpoints, training curves, tables
methodology_2_gnn/v2/results/  M2 checkpoints, tables, figures, ablations
results_unified/     cross-methodology tables and figures
results_mininet/     emulation results
```

## Setup
Python 3.10+ (tested with 3.13), CPU only.
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
Mininet validation additionally needs Mininet and Open vSwitch on Linux (see
`mininet_validation/README.md`).

## Reproduce
All commands run from the repository root. Trained checkpoints are included, so the
evaluation and demo steps work without retraining.
```bash
python scripts/demo.py                 # quick look: trained policies vs baselines (~1 min)
python scripts/demo_gym.py             # the same policies through the Gymnasium API

bash scripts/train_all.sh 2            # retrain everything, 3 seeds, 2 parallel workers (several hours)
bash scripts/evaluate_all.sh           # regenerate every table and figure

# Mininet (two terminals)
python mininet_validation/policy_server.py --topology nsfnet
sudo python3 mininet_validation/run_mininet.py
python scripts/show_mininet.py         # print the emulation results
```
Single methods: `methodology_1_marl/train_m1.py --algo ippo|iql|tabq`,
`methodology_3_ctde/train_m3.py --algo mappo|maddpg|qmix`,
`methodology_2_gnn/v2/train_zero_shot.py`, `methodology_2_gnn/v2/train_specialist.py`
(each folder's README has the options).

## Headline results
Mean episode reward over 5 traffic seeds x 3 training seeds (higher is better), Pareto
traffic at 95% load:

| Method | NSFNet | GEANT |
|---|---|---|
| OSPF | -0.009 | 0.276 |
| ECMP-hop | 0.294 | 0.244 |
| LP oracle (mean TM) | 0.331 | 0.412 |
| M1 IPPO | **0.296** | **0.410** |
| M2 GNN zero-shot | 0.289 | 0.388 |
| M2 GNN specialist | – | 0.408 |
| M3 MAPPO | 0.273 | **0.410** |
| M3 QMIX | 0.172 | 0.318 |

On the Mininet emulation (NSFNet, 95% load) the zero-shot GNN cut packet loss from
9.2% (OSPF) to 3.9% and delay from 36.9 ms to 30.5 ms. See [REPORT.md](REPORT.md) for
all scenarios, metrics, generalisation, failure resilience and training cost.
