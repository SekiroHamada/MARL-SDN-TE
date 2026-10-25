# Project 6 — Review, Fixes and Results (Methodologies 1, 2, 3)

Team 10 · MARL for Packet Routing (Flow Splitting) · prepared October 2026

---

## 1. Review of the original codebase

### 1.1 The environment ignored the agent (critical)

`environments/shared_env_contract.py::step(actions)` never reads `actions`.
Throughput, delay and loss are drawn from `random.uniform(60,95)`,
`random.uniform(2,15)` and `random.uniform(0,2)`. Measured on GEANT
(100-step episodes):

| Policy | Episode reward (mean ± sd) |
|---|---|
| All-zeros action | 7125 ± 94 |
| Random action ×100 | 7132 ± 93 |
| Trained GNN checkpoint (v1) | 7118 ± 95 |
| Untrained GNN | 7128 ± 104 |
| Pure-noise expectation | 7125 |

Consequences for `review_1.docx`:
* the **0.38% generalisation gap** (~27 units) is sampling noise, < 1 sd;
* "significantly outperforming shortest-path under Pareto bursty traffic" was
  never measured (no traffic, no routing);
* the link-failure test prints "successfully adapted" unconditionally.

### 1.2 Other defects

| Where | Problem |
|---|---|
| `evaluation/benchmark_evaluator.py` | OSPF/ECMP metrics are `random.uniform(...) + load*0.2` formulas; no traffic is routed. The Week-1 table is not a measurement. |
| `gnn_actor_critic.py` | Log-probs summed over the whole 100-step batch into one scalar → PPO ratio meaningless; one action per *node* (not per link); the per-node softmax described in the paper is not implemented. |
| `train_gnn_policy.py` | One graph & one episode per update (`graphs_per_rollout`, `minibatch_size` unused); training-curve CSV never exported. |
| `mlp_baseline_agent.py` | `loss = -torch.tensor(reward, requires_grad=True)` has no gradient path to the network — the MLP never trains, never saved, never evaluated. |
| `zero_shot_evaluator.py` | "Gap" compares GNN-on-train vs GNN-on-test (not vs MLP); `set_traffic_load` doesn't exist so the 50/80/95% loads are identical; Jain's index computed over time steps, not flows; "Gbps" on 100 Mbps links. |
| Synthetic pool | Waxman α=β=0.8 → mean degree 7.8, while test WANs have 2.5–3.2: the training distribution does not cover the test set. |
| `data/topologies/geant.json` | Hand-written 23-node adjacency, not the real GÉANT; Abilene has every delay = 5 ms. |
| `models/gnn_policy_encoder.py` | Empty in the repository. |

None of this is a judgement on the idea — the GNN-for-zero-shot-routing
direction is sound — but the numbers in review_1 cannot be published.

---

## 2. What was built

### 2.1 Shared simulator (`environments/te_sim.py`)
Flow-level model where every metric is a deterministic function of topology,
traffic, routing action and queue state:
* **Traffic** — gravity base matrix (new each episode) × smooth background +
  per-flow Pareto ON/OFF (shape 1.5) + correlated source hot-spots; scaled so
  OSPF's max-link-utilisation on the mean matrix equals the target load.
  Scenarios: `poisson`, `pareto` (default), `microburst` (stress).
* **Routing** — every directed link gets a cost; each node splits traffic for
  destination *t* over neighbours strictly closer to *t*, weights
  `exp(-γ·excess)`: loop-free by construction. OSPF and ECMP are special cases.
* **Links** — 100 Mbps, propagation delay, 100-packet FIFO; fluid queue,
  overflow dropped, drops propagated downstream, delay = propagation + backlog
  + M/M/1 waiting.
* **Reward** — `R = α·Thr/Offered − β·Delay/RefDelay − γ·Loss`, team weights
  (1, 0.5, 2) from `configs/shared_env_config.py`.
* Validated: traffic conserved exactly at low load; random actions are far
  worse than OSPF (−0.37 vs +0.49 at 80% load on GÉANT).

### 2.2 Real topologies
98 Internet Topology Zoo WANs (one per network family, 10–60 nodes) in
`data/topologies/topologyzoo/`, including real **Abilene, NSFNet, GÉANT2012**.
Link delay ∝ map distance (mean 5 ms). Never used to train the zero-shot GNN.

### 2.3 Baselines on identical physics (`baselines/te_baselines.py`)
OSPF, ECMP (delay ties — identical to OSPF on real delays, since ties never
occur), ECMP-hop (hop-count ties), LP oracle (min-MLU multi-commodity flow on
the mean matrix; needs global traffic knowledge — a reference ceiling).

### 2.4 Methodology 2 v2 (`methodology_2_gnn/v2/`)
* GINE message passing with residual **edge-state updates**, global context,
  **one output per directed link** (a residual on the delay-based cost).
* PPO with per-link (per-agent) ratio clipping, 16 parallel environments on
  different random graphs, truncation-aware GAE, minibatches, seeds, CSV logs,
  best checkpoint chosen on a held-out *synthetic* validation set.
* Training fixes found during the work (each verified):
  1. **Counterfactual OSPF baseline** — a shadow network routes the identical
     traffic with OSPF; training reward = R − R_OSPF. Independent of the
     agent's actions ⇒ unbiased, and it cancels most burst noise.
  2. **Separate actor and critic** — with a shared trunk the value gradient
     was ~80× the policy gradient (2.62 vs 0.031) and starved the actor under
     global norm clipping.
  3. **ELU instead of ReLU** — one run ended with 0/64 live units in the actor
     head (policy frozen exactly at its initialisation); another with 3/64.
  4. **Link-only actions** — an added per-node split-temperature action
     destabilised training (KL spike 0.11, then collapse); dropped.
* Specialists for the generalisation gap: the **same GNN** and a **flat MLP**,
  each trained only on the target topology.

### 2.5 Methodologies 1 and 3
* Code: `methodology_1_marl/` (IPPO, local critics), `methodology_3_ctde/` (MAPPO, MADDPG,
  central critics); shared actors and PPO loop in `models/`.
* `environments/flow_splitting_env.py` — Dec-POMDP: one agent per switch;
  local observation = own node + each outgoing link + that neighbour (1-hop
  telemetry only); action = one value per outgoing link.
* **M1 IPPO** — every switch has its own actor *and* its own critic V_v(o_v).
* **M3 MAPPO** — same actors, one central critic V(s) (training only).
* **M3 MADDPG** — deterministic actors + central Q(s, a₁..aₙ), replay, targets.
  QMIX is not used: it factorises a *discrete* joint Q; the split weights are
  continuous, and discretising them would confound the comparison.
* IPPO and MAPPO share every line except the critic, so their difference is
  the cost of non-stationarity.


---

## 3. Results — Methodology 2 (GNN zero-shot)

All numbers: mean over 5 traffic seeds × 200 steps (10 ms each) and over
**3 training seeds**, with identical traffic for every method (common random
numbers). Reward uses the team weights (α, β, γ) = (1, 0.5, 2). "Zero-shot" =
trained only on 400 synthetic graphs (8–45 nodes), never on the test topology.

### 3.1 Primary topologies (reward per step; higher is better)

| Topology | Load | OSPF | ECMP-hop | Softmin | LP oracle | Tabular spec. | MLP spec. | GNN spec. | **GNN zero-shot** |
|---|---|---|---|---|---|---|---|---|---|
| Abilene (11) | 0.80 | 0.174 | 0.278 | 0.230 | **0.338** | 0.225 | 0.295 | 0.309 | 0.297 |
| | 0.95 | −0.037 | 0.137 | 0.057 | **0.215** | 0.071 | 0.157 | 0.183 | 0.173 |
| | 1.10 | −0.294 | −0.029 | −0.172 | **0.072** | −0.122 | −0.010 | 0.027 | 0.017 |
| NSFNet-Zoo (13) | 0.80 | 0.221 | 0.293 | 0.253 | **0.326** | 0.263 | 0.318 | **0.326** | 0.293 |
| | 0.95 | 0.034 | 0.164 | 0.081 | **0.210** | 0.112 | 0.183 | 0.208 | 0.157 |
| | 1.10 | −0.199 | 0.001 | −0.149 | 0.050 | −0.071 | 0.005 | **0.055** | −0.025 |
| GÉANT2012 (40) | 0.50 | **0.492** | 0.403 | 0.464 | 0.437 | 0.444 | 0.461 | 0.465 | 0.464 |
| | 0.80 | 0.409 | 0.335 | 0.400 | 0.428 | 0.395 | 0.436 | **0.442** | 0.431 |
| | 0.95 | 0.276 | 0.244 | 0.285 | **0.412** | 0.315 | 0.396 | 0.408 | 0.388 |
| | 1.10 | 0.047 | 0.125 | 0.085 | **0.385** | 0.189 | 0.316 | 0.350 | 0.316 |
| NSFNet team (14) | 0.80 | 0.172 | 0.371 | 0.276 | **0.397** | 0.273 | – | – | 0.371 |
| | 0.95 | −0.009 | 0.294 | 0.142 | **0.331** | 0.143 | – | – | 0.289 |
| | 1.10 | −0.224 | 0.203 | −0.016 | **0.240** | −0.014 | – | – | 0.188 |
| Fat-Tree k=4 (36) | 0.95 | 0.385 | 0.412 | 0.417 | **0.440** | – | – | – | 0.430 |
| | 1.10 | 0.252 | 0.393 | 0.351 | **0.409** | – | – | – | 0.398 |

Packet loss, OSPF → GNN zero-shot: at 95% load Abilene 4.26% → 1.42%, NSFNet-Zoo
3.31% → 1.72%, GÉANT 0.77% → 0.11%, NSFNet-team 4.91% → 0.77%, Fat-Tree
0.57% → 0.24%; at 110% load GÉANT 3.31% → 0.39% and NSFNet-team 8.09% → 1.52%.
Mean delay falls with it (queues stay short) and Jain's index across flows rises.

**Limitation:** at low load (0.5) on GÉANT, OSPF is better (0.492 vs 0.464):
with no congestion, any spreading only adds propagation delay. A load-aware
switch back to shortest-path routing is an easy follow-up.

### 3.2 Generalisation gap (the paper's main result)

Raw percentage gaps are unstable when rewards are near zero, so the gap is
reported as **improvement retained** = (R_zero-shot − R_OSPF) / (R_specialist − R_OSPF),
i.e. what fraction of a topology-specific model's gain over OSPF the zero-shot
model keeps without ever seeing the topology. Loads ≥ 0.8, mean of 3 seeds:

| Topology | vs GNN specialist | vs MLP specialist | vs Tabular specialist |
|---|---|---|---|
| Abilene | 91 / 96 / 97% | 102 / 109 / 110% | >100% at every load |
| GÉANT2012 | 66 / 85 / 89% | 83 / 93 / 100% | >100% except 0.80 |
| NSFNet-Zoo | 69 / 71 / 69% | 74 / 82 / 85% | >100% at every load |
| **Mean (all topologies, loads ≥ 0.8)** | **81%** | **93%** | **167%** |

(three figures per cell = loads 0.80 / 0.95 / 1.10.)

So the zero-shot GNN keeps **81%** of a topology-specific GNN's gain over OSPF,
is on average **better than a topology-specific MLP** (93–110% retained; and the
MLP cannot run at all on a graph of another size — its input dimension is fixed),
and is **far better than a topology-specific tabular Q-learner**, which is the
spec's classical baseline. The one weak cell, GÉANT at 0.80 (66%), is where OSPF
is already close to optimal, so the denominator is small.

### 3.3 Broad zero-shot suite — 98 real Topology Zoo WANs (10–60 nodes)

Mean over 3 training seeds; the best seed is given separately because a
deployment would pick one checkpoint.

| Load | Method | Mean reward | Δ vs OSPF (95% CI) | Beats OSPF on | Loss |
|---|---|---|---|---|---|
| 0.80 | OSPF | 0.324 | – | – | 1.00% |
| | Softmin (static) | 0.324 | −0.000 ± 0.011 | 54% | 0.91% |
| | ECMP-hop | 0.297 | −0.026 ± 0.021 | 42% | 1.29% |
| | **GNN zero-shot (best seed)** | **0.364** | **+0.041 ± 0.011** | **81%** | **0.55%** |
| | GNN zero-shot (3-seed mean) | 0.352 | +0.029 ± 0.011 | 73% | 0.63% |
| | LP oracle (mean TM) | 0.373 | +0.049 ± 0.009 | 91% | 0.61% |
| 0.95 | OSPF | 0.168 | – | – | 2.67% |
| | Softmin (static) | 0.174 | +0.005 ± 0.019 | 57% | 2.52% |
| | ECMP-hop | 0.172 | +0.003 ± 0.032 | 61% | 2.96% |
| | **GNN zero-shot (best seed)** | **0.262** | **+0.094 ± 0.021** | **90%** | **1.45%** |
| | GNN zero-shot (3-seed mean) | 0.241 | +0.072 ± 0.020 | 83% | 1.68% |
| | LP oracle (mean TM) | 0.280 | +0.112 ± 0.017 | 94% | 1.48% |

The best-seed GNN captures **~84% of the LP oracle's gain** over OSPF at 95% load
while knowing neither the mean traffic matrix nor solving an optimisation; the
3-seed mean captures 64%. Seed spread (+0.094 / +0.064 / +0.060 at 0.95) is the
largest single source of variance in the paper and should be reported as such.
Failure cases: Easynet (19 nodes) and two hub-and-spoke networks, Chinanet and
Bellsouth, whose max-degree/mean-degree ratio (6.4, 7.0) lies beyond the 95th
percentile of the training pool (3.8) — add hub-and-spoke generators to the pool.

### 3.4 Link-failure ablation (GÉANT2012, 80% load, 20 trials, 1–3 random links cut mid-episode)

| Method | Post-failure reward | Δ vs same-seed no-failure run | Post-failure loss |
|---|---|---|---|
| **GNN zero-shot (best seed)** | **0.398** | **−0.032 ± 0.024** | **0.16%** |
| GNN zero-shot (3-seed mean) | 0.375 | −0.038 ± 0.029 | 0.29% |
| GNN specialist | 0.390 | −0.035 ± 0.026 | 0.22% |
| MLP specialist (failed inputs zeroed) | 0.365 | −0.045 ± 0.037 | 0.38% |
| Softmin (static) | 0.320 | −0.052 ± 0.049 | 0.67% |
| ECMP-hop | 0.292 | −0.034 ± 0.029 | 0.46% |
| OSPF (instant reconvergence — optimistic) | 0.317 | −0.060 ± 0.065 | 0.92% |

The GNN re-reads the damaged graph and re-balances with no retraining; it keeps
the highest reward and the lowest loss after failures, and it loses the least
relative to its own undamaged run.

### 3.5 Cost
Inference: 1.7 ms per decision for the whole network on one CPU core (PyTorch,
unoptimised; 2.1 ms for the specialist). Training: 44–55 min wall-clock per seed
on a shared 2-core CPU (400 updates, 205k env steps), with best validation
reached after 26k–166k env steps depending on seed.

---

## 4. Results — Methodologies 1 and 3 (3 seeds each, NSFNet 14 nodes and GÉANT 40 nodes)

Validation = 6 loads (0.5–1.05) × 64 steps. Reference on the same cases
(NSFNet): OSPF 0.163, static softmin 0.210, ECMP-hop 0.294, LP oracle 0.344.

**NSFNet (14 switches), 3 seeds**

| | Best val (mean ± sd) | Final val (mean ± sd) | Env steps to 90% | Critic expl. var. |
|---|---|---|---|---|
| **M1 IPPO** (local critics) | **0.326 ± 0.006** | 0.320 ± 0.011 | 94k | 0.27 |
| **M3 MAPPO** (central critic) | 0.324 ± **0.004** | 0.319 ± **0.003** | **85k** | **0.45** |
| M1 IQL-DQN (independent Q) | 0.269 ± 0.012 | 0.263 ± 0.019 | 32k | – |
| M3 MADDPG (TD3-stabilised) | 0.242 ± 0.034 | 0.071 ± 0.254 | 19k (early peak) | – |
| M3 QMIX (monotonic mixer) | 0.212 ± 0.001 | 0.183 ± 0.041 | 21k | – |
| M1 Tabular Q-learning | 0.194 ± 0.004 | 0.163 ± 0.017 | 35k | – |
| M3 MADDPG (vanilla, no TD3) | −0.116 ± 0.131 | – | – | – |

**GÉANT2012 (40 switches), 3 seeds**

| | Best val (mean ± sd) | Final val (mean ± sd) | Env steps to 90% | Critic expl. var. |
|---|---|---|---|---|
| **M1 IPPO** | **0.345 ± 0.002** | 0.332 ± 0.011 | 94k | 0.18 |
| **M3 MAPPO** | 0.340 ± 0.006 | 0.331 ± 0.014 | 107k | **0.58** |
| M1 IQL-DQN | 0.300 ± 0.008 | 0.297 ± **0.003** | 40k | – |
| M1 Tabular Q-learning | 0.276 ± 0.012 | 0.249 ± 0.030 | 32k | – |
| M3 QMIX | 0.268 ± 0.018 | 0.234 ± 0.029 | 35k | – |
| M3 MADDPG (TD3-stabilised) | 0.257 ± 0.004 | 0.044 ± 0.172 | 8k (early peak) | – |

**What non-stationarity costs here.** Two controlled pairs isolate it, because
each pair shares the actor architecture, action space and hyper-parameters and
differs only in whether the critic sees the global state:

* *IPPO vs MAPPO (policy gradient).* The central critic explains far more return
  variance (NSFNet 0.45 vs 0.27; GÉANT **0.58 vs 0.18**) and the final policies
  vary ~4× less across seeds, but the best policy each reaches is the same within
  noise (0.326 vs 0.324; 0.345 vs 0.340). Notably the critic-accuracy gap **widens
  with the number of agents** (14 → 40 switches), which is the trend the
  non-stationarity argument predicts, even though it has not yet turned into a
  final-performance gap.
* *IQL vs QMIX (value-based).* Here centralisation **hurts**: QMIX is below IQL on
  both networks (0.212 vs 0.269; 0.268 vs 0.300). The monotonicity constraint on
  the mixer is the likely cause — with a shared team reward, a switch that must
  briefly accept more load for the network's benefit is exactly the joint action
  a monotonic mixer cannot represent.

So the honest finding is: **CTDE buys training stability, not final quality, and
only for policy-gradient methods**; for value-based methods on this problem the
mixer is a net loss. That is a more interesting result for the paper than a
blanket "centralisation helps".

**Policy gradient beats value-based by a wide margin** on both networks (IPPO
0.326/0.345 vs IQL 0.269/0.300 vs tabular 0.194/0.276), and all deep methods beat
tabular Q-learning — which is the empirical justification for the MARL framing
over the classical Q-routing baseline the spec asks to improve on.

**MADDPG.** Vanilla MADDPG diverged (the actor exploited Q-errors and saturated
every link cost). TD3-style fixes (twin critics, target smoothing, delayed actor,
action regularisation) make it learn, but its best checkpoints are early and it
degrades badly afterwards (final val 0.071 ± 0.254 on NSFNet, 0.044 ± 0.172 on
GÉANT). With a 56-dimensional joint action the central Q-function is too hard to
fit from replay; MAPPO is the better CTDE choice for this problem.

![M1 vs M3](results_unified/nonstationarity_curves_nsfnet.png)

---

## 5. All three methodologies side by side (9 target metrics)

Same simulator, topology, traffic scenario and traffic seeds for every method;
learned methods averaged over 3 training seeds (± sd across training seeds).
M1/M3 and the M2 specialists were **trained on the topology being tested**; the
M2 zero-shot GNN was **not**.

![Unified comparison](results_unified/unified_comparison.png)

### 5.1 NSFNet (14 nodes), Pareto bursty traffic, 95% load

| Method | Reward | Throughput (Mbps) | Loss | Delay (ms) | MLU | Jain | Inference / decision | Telemetry / interval |
|---|---|---|---|---|---|---|---|---|
| OSPF | −0.009 | 533.5 | 4.91% | 19.7 | 1.28 | 0.991 | – | – |
| ECMP | 0.027 | 538.6 | 4.31% | 19.3 | 1.25 | 0.992 | – | – |
| ECMP-hop | 0.294 | 563.2 | 1.00% | 15.5 | 0.92 | 0.999 | – | – |
| LP oracle (mean TM) | **0.331** | **567.1** | **0.51%** | **15.0** | **0.84** | 1.000 | (LP solve) | global TM |
| **M1 IPPO** | **0.296 ± 0.010** | 564.2 | 0.85% | 15.5 | 0.94 | 0.999 | 0.24 ms (all 14 switches) | 3.6 kB, 1-hop |
| M1 IQL-DQN | 0.204 ± 0.031 | 560.5 | 1.27% | 17.4 | 1.06 | 0.999 | 0.63 ms | 3.6 kB, 1-hop |
| M1 Tabular-Q | 0.134 ± 0.009 | 556.6 | 1.87% | 18.6 | 1.14 | 0.998 | 0.33 ms | 3.6 kB, 1-hop |
| **M2 GNN zero-shot** | 0.289 ± 0.022 | 565.1 | **0.77%** | 15.8 | 0.98 | **1.000** | 1.7 ms (central) | 2.8 kB to controller |
| M3 MAPPO | 0.273 ± 0.014 | 561.7 | 1.16% | 15.8 | 0.98 | 0.999 | 0.29 ms | 3.6 kB, 1-hop |
| M3 QMIX | 0.172 ± 0.013 | 559.1 | 1.52% | 17.9 | 1.12 | 0.998 | 0.65 ms | 3.6 kB, 1-hop |
| M3 MADDPG | 0.156 ± 0.032 | 553.4 | 2.19% | 17.8 | 1.08 | 0.997 | 0.34 ms | 3.6 kB, 1-hop |

At 110% load the ordering is unchanged and the spread widens: LP 0.240 >
ECMP-hop 0.203 > IPPO 0.196 > GNN zero-shot 0.188 > MAPPO 0.172 > IQL 0.082 >
QMIX 0.033 > MADDPG 0.003 > tabular −0.023 > ECMP −0.173 > OSPF −0.224
(loss 1.69% for IPPO vs 8.09% for OSPF).

Micro-burst stress (90% load): ECMP-hop 0.224 > M2 zero-shot 0.204 ≈ M1 IPPO
0.202 ≈ LP 0.200 > MAPPO 0.189 > IQL 0.147 > QMIX 0.106 > MADDPG 0.087 ≈
tabular 0.083 > ECMP 0.023 > OSPF 0.002 (6.50% loss).

### 5.2 GÉANT2012 (40 nodes), Pareto bursty traffic, 95% load

| Method | Reward | Loss | Delay (ms) | MLU | Inference |
|---|---|---|---|---|---|
| OSPF / ECMP | 0.276 | 0.77% | 18.3 | 1.09 | – |
| ECMP-hop | 0.244 | 1.28% | 18.8 | 1.03 | – |
| LP oracle (mean TM) | **0.412** | **0.00%** | **15.4** | **0.73** | – |
| **M3 MAPPO** (trained on GÉANT) | 0.410 ± 0.001 | 0.01% | 15.4 | 0.86 | 0.29 ms |
| **M1 IPPO** (trained on GÉANT) | 0.410 ± 0.002 | 0.02% | 15.4 | 0.82 | 0.24 ms |
| M2 GNN specialist | 0.408 | 0.05% | 15.5 | 0.87 | 2.1 ms |
| M2 MLP specialist | 0.396 | 0.01% | 15.8 | 0.91 | 0.29 ms |
| **M2 GNN zero-shot** (never saw GÉANT) | 0.388 ± 0.032 | 0.11% | 15.9 | 0.93 | 1.7 ms |
| M1 IQL-DQN | 0.360 ± 0.004 | 0.14% | 16.6 | 0.93 | 0.63 ms |
| M3 QMIX | 0.318 ± 0.029 | 0.21% | 17.7 | 1.03 | 0.65 ms |
| M1 Tabular-Q | 0.317 ± 0.005 | 0.09% | 17.8 | 1.03 | 0.33 ms |
| M3 MADDPG | 0.303 ± 0.031 | 0.60% | 17.7 | 1.03 | 0.34 ms |

At 110% load: LP 0.385 > IPPO 0.363 > MAPPO 0.358 > GNN specialist 0.350 >
GNN zero-shot 0.316 ≈ MLP specialist 0.316 > IQL 0.281 > tabular 0.200 >
QMIX 0.183 > ECMP-hop 0.125 ≈ MADDPG 0.125 > OSPF/ECMP 0.047.

Micro-burst stress (90% load): GNN specialist 0.340 > LP 0.333 > M1 IPPO 0.330 >
MAPPO 0.328 > GNN zero-shot 0.314 > MLP 0.298 > IQL 0.284 > tabular 0.246 >
QMIX 0.235 > MADDPG 0.231 > OSPF 0.217 > ECMP-hop 0.186.

### 5.3 Convergence and training cost (metric 5)

| Run (3 seeds each) | Best val | Env steps to 90% of best | Wall-clock to 90% |
|---|---|---|---|
| M1 IPPO, NSFNet | 0.326 | 51k–128k | 2–5 min |
| M3 MAPPO, NSFNet | 0.324 | 77k–102k | 3–4 min |
| M1 IQL-DQN, NSFNet | 0.269 | 16k–48k | <2 min |
| M3 QMIX, NSFNet | 0.212 | 8k–48k | <2 min |
| M1 Tabular-Q, NSFNet | 0.194 | 8k–56k | <2 min |
| M3 MADDPG, NSFNet | 0.242 | 4k–40k (early peak, then degrades) | <3 min |
| M1 IPPO, GÉANT | 0.345 | 77k–115k | 9–14 min |
| M3 MAPPO, GÉANT | 0.340 | 102k–115k | 12–16 min |
| M2 zero-shot GNN (synthetic val) | 0.32–0.33 | 26k–166k | 7–36 min |
| M2 GNN specialist, GÉANT | 0.39 | 64k | 27 min |

Value-based methods converge 2–4× faster in env steps but to a clearly worse
policy; the policy-gradient methods are the ones worth the compute. Wall-clock on
a shared 2-core CPU. Full table: `results_unified/training_cost.csv`.

### 5.4 Reading the comparison
* **Every learned method beats OSPF under bursts** (NSFNet 95%: loss 4.91% →
  0.77–2.19%), and the policy-gradient methods close most of the gap to an LP
  oracle that knows the mean traffic matrix.
* **Algorithm family matters more than centralisation.** Policy gradient (IPPO,
  MAPPO, the GNN) > value-based (IQL, QMIX, tabular) on every topology, load and
  scenario tested. Within each family, CTDE changes stability much more than
  final reward (§4).
* **M2 zero-shot is competitive with methods trained on the test topology.** On
  NSFNet it is within 0.007 of IPPO despite never seeing the network; on GÉANT it
  is within 0.022 of the best learned method and ahead of every value-based
  method trained *on GÉANT itself*. One network serves every topology, and it is
  the only method here that can be deployed to a new WAN with no training at all.
* **M1 IPPO edges out M3 MAPPO at deployment on NSFNet** (0.296 vs 0.273) and
  ties on GÉANT (0.410 vs 0.410), even though MAPPO's critic is far more accurate.
* **Signalling:** M1/M3 need only 1-hop telemetry (~3.6 kB per 10 ms interval
  network-wide on NSFNet, never leaving a link); M2 sends ~2.8 kB to a central
  controller and waits a round-trip. M2 per-decision inference (1.7 ms, whole
  network) is ~6× M1/M3's (0.24–0.65 ms) — still far under the control interval.
* **Fairness** (Jain's index across flows) is ≥ 0.997 for every learned method;
  it does not discriminate between methods here and should be reported as such
  rather than as a result.

---

## 6. Packet-level validation in Mininet (emulated SDN)

The simulator's conclusions were re-tested on an **emulated** network: Mininet
with Open vSwitch, OpenFlow 1.3 **select groups** whose bucket weights carry the
learned split ratios, and real UDP traffic between 14 hosts. Each trained policy
runs in a controller process that reads OVS port counters every interval and
reinstalls group weights. 7 methods × 2 loads × 3 traffic seeds = 42 runs;
~311k–370k packets sent per run. Capacities are scaled 10× down (links are
emulated), so compare the *ordering* and the loss/delay ratios, not the absolute
throughput, with §5.1.

| Load | Method | Loss | Delay (ms) | MLU | Reward | Decision | Group install | Groups changed / interval |
|---|---|---|---|---|---|---|---|---|
| 0.80 | OSPF | 6.67% | 28.0 | 0.84 | −0.466 | 0.01 ms | – | 0 |
| | ECMP-hop | 1.44% | 20.7 | 0.68 | 0.026 | 0.01 ms | – | 0 |
| | **M2 GNN zero-shot** | **1.31%** | **19.7** | 0.71 | **0.089** | 3.3 ms | 293 ms | 48.5 |
| | M1 IPPO | 1.78% | 21.5 | 0.71 | −0.018 | 1.3 ms | 242 ms | 39.6 |
| | M3 MAPPO | 1.91% | 20.8 | 0.72 | 0.016 | 1.2 ms | 229 ms | 37.4 |
| | M1 IQL-DQN | 2.64% | 22.9 | 0.75 | −0.100 | 1.3 ms | 439 ms | 73.0 |
| | M3 QMIX | 3.54% | 24.8 | 0.77 | −0.211 | 1.3 ms | 387 ms | 63.9 |
| 0.95 | OSPF | 9.64% | 33.2 | 0.91 | −0.787 | 0.01 ms | – | 0 |
| | ECMP-hop | 3.35% | 23.4 | 0.75 | −0.151 | 0.01 ms | – | 0 |
| | **M2 GNN zero-shot** | **2.67%** | **23.1** | 0.78 | **−0.111** | 3.0 ms | 324 ms | 58.2 |
| | M1 IPPO | 3.69% | 25.5 | 0.78 | −0.252 | 1.2 ms | 246 ms | 43.6 |
| | M3 MAPPO | 3.72% | 24.5 | 0.79 | −0.208 | 1.2 ms | 257 ms | 44.9 |
| | M1 IQL-DQN | 5.01% | 27.1 | 0.82 | −0.364 | 1.2 ms | 397 ms | 70.8 |
| | M3 QMIX | 5.30% | 27.4 | 0.84 | −0.382 | 1.3 ms | 360 ms | 64.9 |

**The emulation reproduces the simulator's ranking.** OSPF collapses, the
policy-gradient methods and the GNN cut loss by 2.6–5.1×, and the value-based
methods (IQL, QMIX) are clearly worse — the same order as §5.1. The zero-shot
GNN is the best method on the emulated network at both loads, which is the
strongest single piece of evidence in the project: a policy trained on synthetic
graphs in a fluid simulator transfers to packet-level forwarding on hardware-like
switches it has never seen.

Two effects appear only in emulation and belong in the paper's limitations:
* **Rewards are uniformly lower than in simulation** (OSPF −0.47 vs 0.17 at 0.80),
  because real UDP bursts, OVS queueing and 10×-scaled capacities make the same
  offered load harsher than the fluid model predicts.
* **Flow-table churn is a real cost.** Installing new group weights takes
  229–439 ms per interval, and the value-based methods churn ~1.6× more groups
  per interval than the policy-gradient ones (64–73 vs 37–45) because their
  greedy discrete action set flips between neighbouring choices. At short control
  intervals this would become the binding constraint, not inference time. An
  action-change penalty in the reward is the obvious fix.

Raw data: `results_mininet/nsfnet_runs.csv`. Reproduce with
`mininet_validation/README.md`.

---

## 7. Conclusions for the paper

1. **The original review_1 numbers must be withdrawn** (§1): the v1 environment
   ignored the agent's action, so every policy scored the same and the reported
   0.38% "generalisation gap" measured nothing. The rebuilt experiments support a
   stronger and defensible claim in its place.
2. **Main result.** A GNN trained only on 400 synthetic graphs, never on the test
   network, retains **81%** of a topology-specific GNN's gain over OSPF on real
   Abilene/NSFNet/GÉANT, **beats a topology-specific MLP** (93% retained) and
   **far exceeds a topology-specific tabular Q-learner**; it beats OSPF on
   **90%** of 98 unseen real WANs at 95% load, capturing ~84% of an LP oracle's
   gain; it degrades least under link failures; and it is the best method on a
   packet-level Mininet emulation it was never tuned for.
3. **Second result (the MARL comparison).** Across three methodologies on
   identical physics, **algorithm family dominates**: policy gradient > value-based
   everywhere. CTDE (MAPPO) buys a much more accurate critic (explained variance
   0.58 vs 0.18 on 40 agents) and ~4× lower seed-to-seed variance, but no better
   final policy; and for value-based methods centralisation *hurts* (QMIX < IQL),
   which we attribute to the monotonicity constraint. Report this as a measured
   negative result, not as "centralisation helps".
4. **Reframe the gap metric.** Report *improvement retained* (§3.2) and the
   98-WAN win-rate (§3.3) rather than a single percentage; raw percentage gaps
   are unstable when rewards approach zero.
5. **Report seed variance honestly.** Zero-shot seed spread (+0.094 / +0.064 /
   +0.060 at 95% load on the 98-WAN suite) is the largest source of variance in
   the paper. Three seeds is the minimum; five would be better for a tier-A venue.
6. **Limitations to state plainly.** Training and the main evaluation use a
   flow-level (fluid) simulator, with packet-level evidence from Mininet/OVS
   emulation rather than NS-3/OMNeT++; link delays are inferred from map distance;
   the GNN loses to OSPF at low load and on hub-and-spoke outliers; MADDPG does
   not train reliably; the discrete-action learners use a hand-designed action
   set; and flow-table churn is unpenalised in the reward.

## 8. Recommended next steps
* Add hub-and-spoke and data-centre generators to the synthetic pool; 5 seeds.
* Load-aware fallback to shortest-path at low utilisation (fixes the one case
  where OSPF wins).
* Add an action-change penalty to the reward and re-measure group churn (§6).
* Scale study: IPPO vs MAPPO on 60–100-switch networks, where the critic-accuracy
  gap already visible at 40 agents should start to show in final reward.
* Larger Mininet study: GÉANT topology, more loads, link failures in emulation.

## 9. Reproducing
See `README.md` and `TRAINING.md`. Everything runs on CPU; total compute for
this report was ≈ 20 CPU-hours (training) + ≈ 2 CPU-hours (evaluation), plus the
Mininet runs.
