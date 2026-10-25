# Methodology 1 — Decentralised Independent MARL

Each switch is an **independent learner** seeing only local, 1-hop information:
no parameter sharing, no central critic. Three learners:
* **IPPO** — policy gradient, continuous split weights, own actor + own critic per switch.
* **IQL-DQN** — independent deep Q-network per switch, discrete actions
  (keep / avoid outgoing link k / prefer outgoing link k).
* **Tabular Q-learning** — the spec's "Q-learning": a Q-table per switch over a
  small discretised local state (hottest link, its utilisation level, queueing).

## Files in this folder (M1 only)
| File | Purpose |
|---|---|
| `independent_marl_agent.py` | `train_ippo()` — plugs `LocalCritics` into the shared PPO loop |
| `local_critic.py` | `LocalCritics`: one value network per switch, V_v(o_v), local observation only |
| `independent_dqn.py` | `train_iql()` — plugs independent per-switch Q-nets into the shared Q-learning loop |
| `tabular_q.py` | tabular Q-learning: state discretiser, trainer, greedy policy |
| `train_m1.py` | CLI: `--algo ippo|iql|tabq`, one or more seeds |
| `evaluate_m1.py` | CLI: all M1 learners vs OSPF / ECMP / ECMP-hop / LP oracle (Pareto loads + micro-burst stress) |

## Shared files M1 uses (also used by M3 and/or M2)
| File | Purpose |
|---|---|
| `models/decentralized_forwarder.py` | Per-switch actor networks (independent weights, batched) |
| `models/marl_onpolicy.py` | PPO loop shared with M3's MAPPO; the critic is passed in |
| `models/marl_value.py` | Q-learning loop + discrete action set shared with M3's QMIX |
| `environments/flow_splitting_env.py` | Dec-POMDP: one agent per switch, local obs, team reward |
| `environments/bursty_traffic_gen.py` | Poisson / Pareto / micro-burst traffic presets |
| `environments/te_sim.py` | The network simulator (all methodologies) |
| `baselines/te_baselines.py` | OSPF / ECMP / LP oracle on the same simulator |

## Run (from the repository root)
```bash
python methodology_1_marl/train_m1.py --algo ippo --topology nsfnet --seeds 0 1 2   # ~5 min per seed
python methodology_1_marl/train_m1.py --algo iql  --topology nsfnet --seeds 0 1 2   # ~4 min per seed
python methodology_1_marl/train_m1.py --algo tabq --topology nsfnet --seeds 0 1 2   # ~4 min per seed
python methodology_1_marl/evaluate_m1.py --topology nsfnet                 # uses results_marl/ippo_nsfnet_seed*
```
Trained models: `results_marl/<algo>_<topology>_seed<k>/`.
