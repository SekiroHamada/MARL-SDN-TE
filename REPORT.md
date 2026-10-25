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

All numbers: mean over 5 traffic seeds × 200 steps (10 ms each), identical
traffic for every method. Reward uses the team weights (α, β, γ) = (1, 0.5, 2).
"Zero-shot" = trained only on 400 synthetic graphs (8–45 nodes), never on the
test topology. Two training seeds (seed 1 was stopped at update 88 by a
container restart; its best checkpoint is used).

### 3.1 Primary topologies (reward per step; higher is better)

| Topology | Load | OSPF | ECMP-hop | LP oracle | MLP spec. | GNN spec. | **GNN zero-shot s0 / s1** |
|---|---|---|---|---|---|---|---|
| Abilene (11) | 0.80 | 0.174 | 0.278 | 0.338 | 0.295 | 0.308 | **0.312 / 0.308** |
| | 0.95 | −0.037 | 0.137 | 0.215 | 0.157 | 0.175 | **0.193 / 0.177** |
| NSFNet-Zoo (13) | 0.80 | 0.222 | 0.293 | 0.326 | 0.318 | 0.326 | **0.317 / 0.296** |
| | 0.95 | 0.034 | 0.164 | 0.210 | 0.183 | 0.208 | **0.190 / 0.165** |
| GÉANT2012 (40) | 0.50 | **0.492** | 0.403 | 0.437 | 0.461 | 0.465 | 0.464 / 0.463 |
| | 0.80 | 0.409 | 0.335 | 0.428 | 0.436 | 0.442 | **0.446 / 0.431** |
| | 0.95 | 0.276 | 0.244 | 0.412 | 0.396 | 0.408 | **0.420 / 0.387** |
| | 1.10 | 0.047 | 0.125 | 0.385 | 0.316 | 0.350 | **0.372 / 0.328** |
| NSFNet team file (14) | 0.95 | −0.009 | 0.294 | 0.331 | – | – | **0.313 / 0.301** |
| Fat-Tree k=4 (36) | 0.95 | 0.385 | 0.412 | 0.440 | – | – | **0.436 / 0.436** |

Packet loss at 95% load, OSPF → GNN zero-shot (s0): Abilene 4.26% → 1.31%,
NSFNet-Zoo 3.31% → 1.44%, GÉANT 0.77% → 0.00%, NSFNet team 4.91% → 0.58%,
Fat-Tree 0.57% → 0.23%. Mean delay also falls (GÉANT 18.3 → 15.2 ms) because
queues stay short. Jain's index across flows rises (e.g. Abilene 0.990 → 0.998).

**Limitation:** at low load (0.5) on GÉANT, OSPF is better (0.492 vs 0.464):
with no congestion, any spreading only adds propagation delay. A load-aware
switch back to shortest-path routing is an easy follow-up.

### 3.2 Generalisation gap (the paper's main result)

Raw percentage gaps are unstable when rewards are near zero, so the gap is
reported as **improvement retained** = (R_zero-shot − R_OSPF) / (R_specialist − R_OSPF),
i.e. what fraction of a topology-specific model's gain over OSPF the zero-shot
model keeps without ever seeing the topology. Loads ≥ 0.8, vs the GNN specialist:

| Topology | seed 0 | seed 1 |
|---|---|---|
| Abilene | 107% | 99% |
| GÉANT2012 | 109% | 82% |
| NSFNet-Zoo | 89% | 74% |
| **All (median)** | **95%** | |

Against the per-topology **MLP**, the zero-shot GNN is equal or better in 12/18
(topology, load, seed) cases. The MLP cannot run at all on a graph of another
size (input dimension is fixed).

### 3.3 Broad zero-shot suite — 98 real Topology Zoo WANs (10–60 nodes)

| Load | Method | Mean reward | Δ vs OSPF (95% CI) | Beats OSPF on | Loss |
|---|---|---|---|---|---|
| 0.80 | OSPF | 0.324 | – | – | 1.00% |
| | ECMP-hop | 0.297 | −0.026 ± 0.021 | 42% | 1.29% |
| | **GNN zero-shot s0** | **0.364** | **+0.041 ± 0.011** | **81%** | **0.55%** |
| | GNN zero-shot s1 | 0.356 | +0.033 ± 0.011 | 73% | 0.57% |
| | LP oracle (mean TM) | 0.373 | +0.049 ± 0.009 | 91% | 0.61% |
| 0.95 | OSPF | 0.168 | – | – | 2.67% |
| | ECMP-hop | 0.172 | +0.003 ± 0.032 | 61% | 2.96% |
| | **GNN zero-shot s0** | **0.262** | **+0.094 ± 0.021** | **90%** | **1.45%** |
| | GNN zero-shot s1 | 0.248 | +0.080 ± 0.021 | 85% | 1.51% |
| | LP oracle (mean TM) | 0.280 | +0.112 ± 0.017 | 94% | 1.48% |

The GNN captures **~84% of the LP oracle's gain** over OSPF while knowing neither
the mean traffic matrix nor solving an optimisation. Failure cases: Easynet
(19 nodes, −0.48 at 0.95; cause not identified), and two hub-and-spoke networks,
Chinanet and Bellsouth, whose max-degree/mean-degree ratio (6.4, 7.0) lies
beyond the 95th percentile of the training pool (3.8) — add hub-and-spoke
generators to the synthetic pool.

### 3.4 Link-failure ablation (GÉANT2012, 80% load, 20 trials, 1–3 random links cut mid-episode)

| Method | Post-failure reward | Δ vs same-seed no-failure run | Post-failure loss |
|---|---|---|---|
| **GNN zero-shot s0** | **0.398** | −0.032 ± 0.024 | **0.16%** |
| GNN zero-shot s1 | 0.383 | −0.031 ± 0.020 | 0.15% |
| GNN specialist | 0.390 | −0.035 ± 0.026 | 0.22% |
| MLP specialist (failed inputs zeroed) | 0.365 | −0.045 ± 0.037 | 0.38% |
| ECMP-hop | 0.292 | −0.034 ± 0.029 | 0.46% |
| OSPF (instant reconvergence — optimistic) | 0.317 | −0.060 ± 0.065 | 0.92% |

The GNN re-reads the damaged graph and re-balances with no retraining; it keeps
the highest reward and lowest loss after failures.

### 3.5 Cost
Inference: ~2.0–2.5 ms per decision for the whole network on one CPU core
(PyTorch, unoptimised). Training: ~55 min wall-clock on a shared 2-core CPU (seed 0, 400 updates, 205k env steps),
best validation already at update 50 (~25k env steps).

---

## 4. Results — Methodologies 1 and 3 (trained on NSFNet, 14 nodes; 3 seeds)

Validation = 6 loads (0.5–1.05) × 64 steps; reference on the same cases:
OSPF 0.163, static softmin 0.210, ECMP-hop 0.294, LP oracle 0.344.

| | Best val (mean ± sd) | Final val (mean ± sd) | Env steps to 90% | Critic explained variance |
|---|---|---|---|---|
| **M1 IPPO** (local critics) | 0.326 ± 0.006 | 0.320 ± 0.011 | 94k | 0.27 |
| **M3 MAPPO** (central critic) | 0.324 ± 0.004 | 0.319 ± **0.003** | **85k** | **0.45** |
| M3 MADDPG (TD3-stabilised) | 0.242 ± 0.034 | 0.071 ± 0.254 | – | – |
| M3 MADDPG (vanilla) | −0.116 ± 0.131 | – | – | – |

**What non-stationarity costs here.** The central critic explains much more of
the return variance (0.45 vs 0.27), the final policies vary ~4× less across
seeds, and MAPPO reaches 90% of its best ~9% sooner — but the best policy each
reaches is the same within noise. With a shared team reward and 1-hop local
observations on a 14-node network, independent learners pay in critic accuracy
and run-to-run stability, not in final quality. Expect the gap to widen with
more agents (GÉANT, 40 switches: IPPO 0.329, MAPPO 0.324, single seed — not
enough to conclude).

**MADDPG.** Vanilla MADDPG diverged (the actor exploited Q-errors and saturated
every link cost). TD3-style fixes (twin critics, target smoothing, delayed
actor, action regularisation) make it learn, but its best checkpoints are early
and it still degrades late in training. With a 56-dimensional joint action the
central Q-function is too hard to fit from replay; MAPPO is the better CTDE
choice for this problem.

![M1 vs M3](results_unified/nonstationarity_curves_nsfnet.png)


---

## 5. All three methodologies side by side (9 target metrics)

Same simulator, topology, traffic scenario and traffic seeds for every method.
Learned methods averaged over their training seeds (M1/M3: 3 on NSFNet, 1 on
GÉANT; M2 zero-shot: 2). M1/M3 and the M2 specialists were **trained on the
topology being tested**; the M2 zero-shot GNN was **not**.

![Unified comparison](results_unified/unified_comparison.png)

### 5.1 NSFNet (14 nodes), Pareto bursty traffic, 95% load

| Method | Reward | Throughput (Mbps) | Loss | Delay (ms) | MLU | Jain | Inference / decision | Telemetry / interval |
|---|---|---|---|---|---|---|---|---|
| OSPF | −0.009 | 533.5 | 4.91% | 19.7 | 1.28 | 0.991 | – | – |
| ECMP-hop | 0.294 | 563.2 | 1.00% | 15.5 | 0.92 | 0.999 | – | – |
| LP oracle (mean TM) | 0.331 | 567.1 | 0.51% | 15.0 | 0.84 | 1.000 | (LP solve) | global TM |
| M1 IPPO | 0.296 | 564.2 | 0.85% | 15.5 | 0.94 | 0.999 | 0.26 ms (all 14 switches) | 1.8 kB, 1-hop |
| **M2 GNN zero-shot** | **0.307** | **566.5** | **0.59%** | **15.5** | 0.97 | **1.000** | 2.1 ms (central) | 1.5 kB to controller |
| M3 MAPPO | 0.273 | 561.7 | 1.16% | 15.8 | 0.98 | 0.999 | 0.26 ms | 1.8 kB, 1-hop |
| M3 MADDPG | 0.156 | 553.4 | 2.19% | 17.8 | 1.08 | 0.997 | 0.32 ms | 1.8 kB, 1-hop |

Micro-burst stress (90% load): M2 zero-shot 0.227 (1.81% loss) > ECMP-hop 0.223
> M1 IPPO 0.202 > LP 0.200 > M3 MAPPO 0.189 > MADDPG 0.087 > OSPF 0.002 (6.50% loss).

### 5.2 GÉANT2012 (40 nodes), Pareto bursty traffic, 95% load

| Method | Reward | Loss | Delay (ms) | MLU | Inference |
|---|---|---|---|---|---|
| OSPF / ECMP | 0.276 | 0.77% | 18.3 | 1.09 | – |
| LP oracle (mean TM) | 0.412 | 0.00% | 15.4 | 0.73 | – |
| **M1 IPPO** (trained on GÉANT) | **0.414** | 0.03% | 15.3 | 0.82 | 0.41 ms |
| M2 GNN specialist | 0.408 | 0.05% | 15.5 | 0.87 | 2.7 ms |
| **M2 GNN zero-shot** | 0.403 | **0.01%** | 15.6 | 0.90 | 2.8 ms |
| M2 MLP specialist | 0.396 | 0.01% | 15.8 | 0.91 | 0.37 ms |
| M3 MAPPO | 0.392 | 0.15% | 15.8 | 0.89 | 0.41 ms |
| M3 MADDPG | 0.285 | 0.77% | 18.0 | 1.08 | 0.44 ms |

Micro-burst stress: M2 zero-shot 0.340 ≈ GNN specialist 0.340 ≈ M1 IPPO 0.339
> LP 0.333 > MAPPO 0.313 > MLP 0.298 > OSPF 0.216.

### 5.3 Convergence and training cost (metric 5)

| Run | Best val | Env steps to 90% of best | Wall-clock to 90% |
|---|---|---|---|
| M1 IPPO, NSFNet (3 seeds) | 0.326 | 51k–128k | 2–5 min |
| M3 MAPPO, NSFNet (3 seeds) | 0.324 | 77k–102k | 3–4 min |
| M3 MADDPG, NSFNet (3 seeds) | 0.242 | 4k–40k (early peak, then degrades) | <3 min |
| M2 zero-shot GNN (seed 0) | 0.330 (synthetic val) | 26k | 7 min |
| M2 GNN specialist, GÉANT | 0.388 | 64k | 27 min |

Wall-clock on a shared 2-core CPU. Full table: `results_unified/training_cost.csv`.

### 5.4 Reading the comparison
* **Every learned on-policy method beats OSPF by a wide margin under bursts**
  (NSFNet 95%: loss 4.9% → 0.6–1.2%).
* **M2 zero-shot is the best non-oracle method on NSFNet in every scenario**
  (only the LP oracle, which knows the mean traffic matrix, scores higher at
  80–110% load), despite M1/M3 being trained on NSFNet. On GÉANT at ≥ 80% load
  it is within 0.017 of the best learned method. One network serves every topology.
* **M1 IPPO edges out M3 MAPPO at deployment** (NSFNet 0.296 vs 0.273, GÉANT
  0.414 vs 0.392) even though their validation rewards were equal. The central
  critic buys training stability (§4), not a better final policy, in this setting.
* **Signalling:** M1/M3 need only 1-hop telemetry (~1.8 kB per 10 ms interval
  network-wide on NSFNet, never leaving a link); M2 sends ~1.5 kB to a central
  controller and waits a round-trip. M2 per-decision inference (2–3 ms, whole
  network) is ~8× M1/M3's (0.3–0.4 ms) — still well under the control interval.
* **Fairness** (Jain's index across flows) is ≥ 0.997 for every learned
  on-policy method; it is not a differentiating metric here.

---

## 5b. Methods added after these results (train with `TRAINING.md`)

To close the remaining gaps against the project specification, the code now also
contains — **trained only briefly here; not yet in the tables above**:
* **M1 IQL-DQN** (`methodology_1_marl/independent_dqn.py`) and **M1 tabular
  Q-learning** (`methodology_1_marl/tabular_q.py`) — the spec's "Q-learning"
  independent learners, on a discrete per-switch action set (keep / avoid link k /
  prefer link k, step 0.75 on the cost residual).
* **M2 tabular specialist** — tabular agents trained per topology (Abilene,
  NSFNet-Zoo, GÉANT) join the GNN/MLP specialists in the generalisation gap.
* **M3 QMIX** (`methodology_3_ctde/qmix_mixer.py`) — monotonic mixer over the same
  per-switch Q-networks as IQL: a second controlled non-stationarity comparison
  (IQL vs QMIX) next to IPPO vs MAPPO.
* **3 training seeds everywhere** (M2 zero-shot, all M1/M3 methods on NSFNet and GÉANT).

3000-iteration sanity runs on NSFNet (one seed, half the default length): IQL-DQN
0.263, QMIX 0.204, tabular Q 0.179 best validation reward vs OSPF 0.163 and static
softmin 0.210 — all learn; QMIX's TD loss is still high at 3000 iterations, so judge
it on full-length runs;
full-length results come from `bash scripts/train_all.sh` + `bash scripts/evaluate_all.sh`.
An action step of 1.5 made greedy discrete policies oscillate (every switch
dodging the same hot link); 0.75 removed it (IQL 0.17 → 0.26).

## 6. Conclusions for the paper

1. The original review_1 numbers must be withdrawn (§1). The rebuilt
   experiments support a **stronger** claim than the original abstract:
   a GNN trained only on synthetic graphs retains a median **95%** of a
   topology-specific model's gain over OSPF on real Abilene/NSFNet/GÉANT, beats
   OSPF on **90%** of 98 unseen real WANs at 95% load (≈84% of an LP oracle's
   gain), and degrades least under link failures.
2. Suggested reframing of the generalisation-gap result: report
   *improvement retained* (§3.2) and the 98-WAN win-rate (§3.3) instead of a
   single percentage.
3. Honest limitations to state: training and the main evaluation use a
   flow-level (fluid) simulator; packet-level evidence comes from Mininet/OVS
   emulation (`mininet_validation/`), not NS-3/OMNeT++; link delays from map
   distance; GNN loses to OSPF at low load and on hub-and-spoke outliers; MADDPG
   did not train reliably; discrete-action learners (IQL, QMIX, tabular) use a
   hand-designed action set.

## 7. Recommended next steps
* Packet-level validation in Mininet: the harness is in `mininet_validation/`
  (trained policies drive OpenFlow select-group weights on OVS; real UDP traffic;
  measured loss and delay). Run it and add the results as an emulation section.
* Add hub-and-spoke and data-centre generators to the synthetic pool; 3+ seeds.
* Load-aware fallback to shortest-path at low utilisation.
* Scale study: M1 vs M3 on 40–100-switch networks with 3+ seeds, where
  non-stationarity should matter more.

## 8. Reproducing
See `README.md`. Everything runs on CPU; total compute for this report was
≈ 6 CPU-hours.


