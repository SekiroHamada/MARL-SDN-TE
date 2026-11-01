# Methodology 3 — Centralised Training, Decentralised Execution (CTDE)

Same per-switch actors as Methodology 1, but trained with a **central critic**
that sees the whole network. The critic is thrown away at execution: each
switch still runs only its own actor on its local observation.

* **MAPPO** — central value critic V(s), on-policy (same PPO loop as M1 IPPO,
  so M1 vs MAPPO isolates the cost of non-stationarity).
* **MADDPG** — central Q critic Q(s, a₁…aₙ), off-policy with replay, plus
  TD3-style stabilisers (twin critics, target smoothing, delayed actor,
  action regularisation) — vanilla MADDPG diverged.
* **QMIX** — per-switch Q-networks combined by a monotonic mixer whose weights are
  generated from the global state (hypernetworks). Runs on the discrete action set
  (keep / avoid link k / prefer link k) and is the value-based counterpart of
  M1's IQL-DQN: same agent networks and training loop, only the TD target differs.

## Files in this folder (M3 only)
| File | Purpose |
|---|---|
| `qmix_maddpg_engine.py` | `train_mappo()`, `train_maddpg()` (replay, TD3 fixes), `train_qmix()` |
| `qmix_mixer.py` | `QMixer`: monotonic mixing network with hypernetworks |
| `centralized_critic.py` | `CentralValueCritic` (MAPPO) and `CentralQCritic` (MADDPG) |
| `train_m3.py` | CLI: `--algo mappo|maddpg|qmix`, one or more seeds |
| `evaluate_m3.py` | CLI: MAPPO / MADDPG / QMIX vs their M1 counterparts and baselines + non-stationarity analysis |

## Shared files M3 uses
`models/decentralized_forwarder.py`, `models/marl_onpolicy.py`, `models/marl_value.py`,
`environments/flow_splitting_env.py`, `environments/bursty_traffic_gen.py`,
`environments/te_sim.py`, `baselines/te_baselines.py`,
`evaluation/nonstationarity_analysis.py`.

## Run (from the repository root)
```bash
python methodology_3_ctde/train_m3.py --algo mappo  --topology nsfnet --seeds 0 1 2   # ~5 min per seed
python methodology_3_ctde/train_m3.py --algo maddpg --topology nsfnet --seeds 0 1 2   # ~3 min per seed
python methodology_3_ctde/train_m3.py --algo qmix   --topology nsfnet --seeds 0 1 2   # ~5 min per seed
python methodology_3_ctde/evaluate_m3.py --topology nsfnet     # needs M1 IPPO runs too, for the comparison
```
Trained models: `results_marl/{mappo,maddpg,qmix}_<topology>_seed<k>/`.
