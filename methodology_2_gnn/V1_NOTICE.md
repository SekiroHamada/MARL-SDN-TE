# Methodology 2 — v1 files (kept for history only)

The original files in this folder (`train_gnn_policy.py`, `gnn_actor_critic.py`,
`zero_shot_evaluator.py`, `failure_resilience_test.py`, `mlp_baseline_agent.py`, …)
run, but **their results are not valid**:

1. `environments/shared_env_contract.py::step()` ignores the action and draws
   throughput/delay/loss from `random.uniform`. Measured on GEANT, 100-step
   episodes: all-zeros 7125±94, random 7132±93, trained GNN 7118±95,
   untrained GNN 7128±104 — the pure-noise expectation is 7125. The reported
   0.38% generalisation gap (~27 reward units) is sampling noise.
2. `gnn_actor_critic.py` sums log-probs over the whole batch into one scalar,
   so the PPO ratio is not per-sample; one action per *node* (not per link)
   and no per-node softmax despite the paper describing one.
3. `mlp_baseline_agent.py`: `loss = -torch.tensor(reward, requires_grad=True)`
   has no path to the network — the MLP never trains.
4. `failure_resilience_test.py` prints "successfully adapted" unconditionally.
5. Synthetic training graphs (mean degree 7.8) do not cover the test WANs
   (degree 2.5–3.2); `models/gnn_policy_encoder.py` in the repo is empty.

Use **`v2/`** — see the top-level `REPORT.md`.
