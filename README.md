# MARL-SDN-TE — Multi-Agent RL for Packet Routing (Flow Splitting)

Team 10 · Project 6 · SDN · SDG 9 / SDG 12

Three routing methodologies, all trained and evaluated on **one shared,
action-dependent simulator** with identical topologies, traffic and metrics:

| | Methodology | Code | Trained on |
|---|---|---|---|
| M1 | Decentralised independent MARL: IPPO (actor + local critic), IQL-DQN and tabular Q-learning, one learner per switch on local state | `methodology_1_marl/` | one topology |
| M2 | Inductive GNN policy (GINE + edge updates, PPO) that transfers **zero-shot** to unseen topologies | `methodology_2_gnn/v2/` | synthetic graphs only |
| M3 | CTDE: MAPPO (central V critic), MADDPG (central Q critic), QMIX (monotonic mixer); decentralised execution | `methodology_3_ctde/` | one topology |

Baselines on the same physics: OSPF (Dijkstra on delay), ECMP (delay ties),
ECMP-hop (hop-count ties) and an LP oracle (min-MLU multi-commodity flow on the
mean traffic matrix; a reference that needs global traffic knowledge).

See **`TRAINING.md`** for step-by-step training/evaluation (`bash scripts/train_all.sh`,
`bash scripts/evaluate_all.sh`) and **`REPORT.md`** for the code review, fixes and results.

---

## Repository layout

```
environments/
  te_sim.py                 shared flow-level simulator (softmin hop-by-hop splitting,
                            finite FIFO buffers, loss propagation, delay model)
  bursty_traffic_gen.py     gravity TM x Pareto ON/OFF + correlated hot-spots; scenarios
                            poisson / pareto / microburst
  flow_splitting_env.py     multi-agent Dec-POMDP wrapper (one agent per switch) for M1/M3
  topologies.py             Topology Zoo loader, team JSON loader, WAN-like synthetic graphs
  shared_env_contract.py    Sprint-0 contract (kept for reference; its step() is random — do not use)
baselines/
  te_baselines.py           OSPF / ECMP / ECMP-hop / LP oracle on the shared simulator
  baseline_ospf_ecmp.py     Sprint-0 path computation (unchanged)
models/                     shared by M1 and M3
  decentralized_forwarder.py   per-switch actors (independent weights, batched)
  marl_onpolicy.py             PPO loop shared by IPPO (M1) and MAPPO (M3); critic is passed in
  marl_value.py                Q-learning loop shared by IQL-DQN (M1) and QMIX (M3); discrete per-switch actions
  gnn_policy_encoder.py        re-export of the M2 GNN (Sprint-0 file name)
methodology_1_marl/         M1 only: independent learners (IPPO)
  independent_marl_agent.py    train_ippo()
  local_critic.py              LocalCritics: one critic per switch on local observation
  independent_dqn.py           train_iql(): independent DQN per switch
  tabular_q.py                 independent tabular Q-learning (also the M2 tabular baseline)
  train_m1.py / evaluate_m1.py
methodology_3_ctde/         M3 only: CTDE
  qmix_maddpg_engine.py        train_mappo(), train_maddpg(), train_qmix()
  centralized_critic.py        CentralValueCritic (MAPPO), CentralQCritic (MADDPG)
  qmix_mixer.py                QMixer: monotonic hypernetwork mixer
  train_m3.py / evaluate_m3.py
methodology_2_gnn/
  v2/                       the working Methodology 2 (see its config_v2.yaml)
  *.py (v1)                 original files, kept for history — see REPORT.md §1 for why
                            their results are not valid
evaluation/
  unified_benchmark.py      all methods x topologies x scenarios -> 9 metrics
  marl_eval.py              shared evaluation helper for M1/M3
  nonstationarity_analysis.py   M1 vs M3 over seeds
  benchmark_evaluator.py    Week-1 OSPF/ECMP table (now simulated; the Sprint-0 version used random formulas)
mininet_validation/          emulated-SDN validation of trained policies (Mininet + OVS + OpenFlow groups)
  policy_server.py             controller decisions (venv): telemetry -> trained policy -> split weights
  run_mininet.py               data plane (sudo): topology, traffic, counters, OpenFlow group install
  traffic_agent.py             per-host bursty UDP sender/receiver (loss + one-way delay)
scripts/
  train_marl.py             single CLI for M1 or M3:  --algo ippo|mappo|maddpg (same as the per-folder scripts)
  train_all.sh              train every method x seed (resumable, parallel workers)
  evaluate_all.sh           run every evaluation
  job_queue.sh              tiny job-queue worker used by train_all.sh
data/topologies/
  nsfnet.json geant.json abilene.json fattree_k4.json   team Sprint-0 graphs
  topologyzoo/              98 real WANs (one per network family) — held-out zero-shot set
results_marl/               M1/M3 checkpoints + training curves
methodology_2_gnn/v2/results/   M2 checkpoints, curves, tables, figures
results_unified/            cross-methodology tables and figures
```

## Setup

```bash
python3 -m venv marl-sdn && source marl-sdn/bin/activate
pip install -r methodology_2_gnn/v2/requirements.txt
```
CPU is enough. Every script is run from the repository root.

## Reproduce

```bash
# Methodology 2
python methodology_2_gnn/v2/train_zero_shot.py --seed 0          # ~35 min on 1 CPU core
python methodology_2_gnn/v2/train_zero_shot.py --seed 1
python methodology_2_gnn/v2/train_specialist.py --model gnn --topology Geant2012
python methodology_2_gnn/v2/train_specialist.py --model mlp --topology Geant2012
python methodology_2_gnn/v2/evaluate_m2.py --part primary         # zero-shot + generalisation gap
python methodology_2_gnn/v2/evaluate_m2.py --part suite           # 98 Topology Zoo WANs
python methodology_2_gnn/v2/evaluate_m2.py --part failure         # link-failure ablation
python methodology_2_gnn/v2/summarize.py                          # tables + figures

# Methodology 1 (independent MARL)
python methodology_1_marl/train_m1.py --topology nsfnet --seeds 0 1 2
python methodology_1_marl/evaluate_m1.py --topology nsfnet

# Methodology 3 (CTDE)
python methodology_3_ctde/train_m3.py --algo mappo  --topology nsfnet --seeds 0 1 2
python methodology_3_ctde/train_m3.py --algo maddpg --topology nsfnet --seeds 0 1 2
python methodology_3_ctde/evaluate_m3.py --topology nsfnet     # + M1-vs-M3 non-stationarity

# Validate in Mininet (two terminals; see mininet_validation/README.md)
python mininet_validation/policy_server.py --topology nsfnet          # terminal 1, venv
sudo python3 mininet_validation/run_mininet.py                        # terminal 2

# Everything side by side
python evaluation/unified_benchmark.py
```

## Simulator in one paragraph

Each control interval (10 ms) a demand matrix is drawn (gravity base, smooth
background + Pareto ON/OFF flows + correlated source hot-spots; mean scaled so
OSPF's max link utilisation equals the target load). Every directed link gets a
cost; for each destination, a node splits traffic over neighbours strictly
closer to it under those costs with weights `exp(-gamma * excess cost)`
(loop-free by construction; OSPF and ECMP are special cases). Link loads come
from one batched linear solve per destination; FIFO queues (100 packets) fill
and overflow; drops propagate downstream; delay = propagation + backlog + M/M/1
waiting. Reward `R = a*Throughput/Offered - b*Delay/RefDelay - g*Loss` with the
team's `configs/shared_env_config.py` weights (1, 0.5, 2).
