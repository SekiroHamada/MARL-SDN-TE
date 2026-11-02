# MARL-SDN-TE: Results Report

All numbers in this report are read from the CSV files in `results_unified/`,
`results_marl/tables/`, `methodology_2_gnn/v2/results/tables/` and `results_mininet/`.
`bash scripts/evaluate_all.sh` regenerates every one of them from the included checkpoints.

## 1. Summary
* **Every learned method that trains stably beats OSPF/ECMP under congestion.** At 95%
  load on NSFNet, OSPF scores -0.009 and the best learned policies score 0.27–0.30. On
  GEANT, OSPF scores 0.276 and the best learned policies score 0.39–0.41.
* **Policy-gradient methods lead their families.** IPPO (M1) and MAPPO (M3) tie on GEANT
  (mean reward 0.399 each), and IPPO leads on NSFNet (0.305 vs 0.290). IQL, QMIX,
  tabular Q-learning and MADDPG trail them by 0.05–0.25 reward at high load.
* **Central training (M3) does not beat independent learners (M1) here.** MAPPO equals
  IPPO on GEANT and is slightly worse on NSFNet. QMIX is worse than IQL on both. Section 6
  explains why.
* **One GNN trained only on synthetic graphs (M2 zero-shot) transfers to unseen real
  networks.** It keeps 71–96% of a per-topology specialist's improvement over OSPF and
  beats OSPF on 80–90% of 98 Topology Zoo networks at 95% load (by training seed).
* **The results hold on an emulated data plane.** On Mininet (NSFNet, 95% load) M2 cut
  loss from 9.2% to 3.9%, M3 MAPPO to 5.0% and M1 IPPO to 6.2%.
* **The LP oracle stays the upper reference at the highest loads.** It is given the true
  mean traffic matrix, so learned methods matching it is the target, not beating it.

## 2. Experimental setup
**Simulator** (`environments/te_sim.py`). A flow-level SDN model with finite FIFO buffers
and fluid queues. Delay is propagation plus queueing. Losses propagate downstream. Traffic
is a gravity-model matrix plus Pareto ON/OFF heavy-tailed bursts, calibrated so that OSPF
on the mean matrix reaches a target max-link utilisation (the "load"). OSPF and ECMP are
special cases of the same split-ratio routing, so every method runs on identical physics.

**Topologies.** Team NSFNet (14 nodes) and GEANT2012 for all methods. M2 also uses
Abilene, the Topology Zoo NSFNet (13 nodes, written `Nsfnet`), a fat-tree k=4 and a suite
of 98 Topology Zoo WANs.

**Scenarios.** Pareto traffic at loads 0.5 / 0.8 / 0.95 / 1.1, plus a micro-burst stress
test at 0.9. Each cell uses 5 traffic seeds (101–105) x 200 control intervals, with common
random numbers across methods.

**Training seeds.** 3 per learned method (0, 1, 2). Per-topology specialists use 1 seed.

**Reward.** The team formula combines throughput, loss, delay and utilisation balance.
Higher is better.

**Metrics.** Reward, throughput, loss %, delay, max-link utilisation (MLU), utilisation
spread, Jain fairness, inference time per decision, signalling bytes per interval,
training cost (env steps and wall-clock to 90% of best) and seed variance.

## 3. Cross-methodology benchmark (`results_unified/`)
Reward, averaged over 3 training seeds (`unified_by_algorithm.csv`):

**NSFNet**

| Method | Pareto 0.5 | 0.8 | 0.95 | 1.1 | Micro-burst 0.9 | Mean |
|---|---|---|---|---|---|---|
| LP oracle (mean TM) | 0.449 | 0.397 | 0.331 | 0.240 | 0.200 | 0.323 |
| ECMP-hop | 0.453 | 0.371 | 0.294 | 0.203 | 0.223 | 0.309 |
| **M1 IPPO** | 0.458 | 0.374 | 0.296 | 0.196 | 0.202 | **0.305** |
| M2 GNN zero-shot | 0.461 | 0.371 | 0.289 | 0.188 | 0.204 | 0.303 |
| M3 MAPPO | 0.457 | 0.358 | 0.273 | 0.172 | 0.189 | 0.290 |
| M1 IQL-DQN | 0.426 | 0.304 | 0.203 | 0.082 | 0.147 | 0.232 |
| M3 QMIX | 0.430 | 0.295 | 0.172 | 0.033 | 0.106 | 0.207 |
| M3 MADDPG | 0.431 | 0.281 | 0.156 | 0.003 | 0.087 | 0.192 |
| M1 Tabular-Q | 0.425 | 0.266 | 0.134 | -0.023 | 0.083 | 0.177 |
| ECMP | 0.452 | 0.205 | 0.027 | -0.173 | 0.023 | 0.107 |
| OSPF | 0.446 | 0.172 | -0.009 | -0.224 | 0.002 | 0.077 |

**GEANT2012**

| Method | Pareto 0.5 | 0.8 | 0.95 | 1.1 | Micro-burst 0.9 | Mean |
|---|---|---|---|---|---|---|
| **M2 GNN specialist** | 0.465 | 0.442 | 0.408 | 0.350 | 0.340 | **0.401** |
| M3 MAPPO | 0.460 | 0.440 | 0.410 | 0.358 | 0.328 | 0.399 |
| LP oracle (mean TM) | 0.437 | 0.428 | 0.412 | 0.385 | 0.333 | 0.399 |
| M1 IPPO | 0.453 | 0.437 | 0.410 | 0.363 | 0.330 | 0.399 |
| M2 GNN zero-shot | 0.464 | 0.431 | 0.388 | 0.316 | 0.314 | 0.383 |
| M2 MLP specialist | 0.461 | 0.436 | 0.396 | 0.316 | 0.298 | 0.381 |
| M1 IQL-DQN | 0.447 | 0.408 | 0.360 | 0.281 | 0.283 | 0.356 |
| M1 Tabular-Q | 0.444 | 0.394 | 0.317 | 0.200 | 0.246 | 0.320 |
| M3 QMIX | 0.449 | 0.397 | 0.318 | 0.183 | 0.234 | 0.316 |
| M3 MADDPG | 0.455 | 0.399 | 0.302 | 0.125 | 0.231 | 0.302 |
| OSPF / ECMP | 0.492 | 0.409 | 0.276 | 0.047 | 0.216 | 0.288 |
| ECMP-hop | 0.403 | 0.335 | 0.244 | 0.125 | 0.186 | 0.259 |

**Packet loss (%) at Pareto 1.1** (the overload case):

| | OSPF | ECMP-hop | LP oracle | IPPO | IQL | Tab-Q | GNN zero-shot | MAPPO | MADDPG | QMIX |
|---|---|---|---|---|---|---|---|---|---|---|
| NSFNet | 8.09 | 1.98 | 1.16 | 1.69 | 2.35 | 3.46 | 1.52 | 2.10 | 3.91 | 2.95 |
| GEANT | 3.31 | 2.73 | 0.09 | 0.16 | 0.51 | 0.48 | 0.39 | 0.11 | 2.26 | 1.00 |

Notes:
* At light load (0.5) no method has room to improve. OSPF's shortest paths have the
  lowest delay, so it wins on GEANT, and the reward differences on NSFNet are small.
* The NSFNet specialists are absent from the NSFNet rows on purpose. The M2 specialists
  were trained on the 13-node Topology Zoo NSFNet, which is a different graph from the
  team's 14-node NSFNet.
* Per-seed rows are in `unified_summary.csv` and `unified_per_seed.csv`.

**Deployment cost** (median per decision, all switches):

| | M1 / M3 (per-switch agents) | M2 GNN (central controller) |
|---|---|---|
| Inference, NSFNet | 0.13–0.30 ms | 1.34 ms |
| Inference, GEANT | 0.33–1.00 ms | 2.1 ms |
| Signalling per interval, NSFNet / GEANT | 1848 / 5368 bytes | 1456 / 4216 bytes |

## 4. Methodology 2: generalisation (`methodology_2_gnn/v2/results/tables/`)
The zero-shot GNN is trained only on synthetic graphs and then evaluated on unseen real
topologies. It is compared with specialists trained on each target topology.

**Reward at 95% load** (zero-shot is the mean of 3 seeds, `m2_primary_summary.csv`):

| Topology | OSPF | Tabular specialist | MLP specialist | GNN specialist | **GNN zero-shot** | LP oracle |
|---|---|---|---|---|---|---|
| Abilene | -0.037 | 0.071 | 0.157 | 0.183 | **0.173** | 0.215 |
| Nsfnet (Zoo) | 0.034 | 0.112 | 0.183 | 0.208 | **0.157** | 0.210 |
| GEANT2012 | 0.276 | 0.315 | 0.396 | 0.408 | **0.388** | 0.412 |
| fat-tree k=4 | 0.385 | – | – | – | **0.430** | 0.440 |

**Share of the GNN specialist's improvement over OSPF that zero-shot keeps**
(`m2_generalization_gap.csv`, mean over seeds):

| Load | Abilene | Nsfnet (Zoo) | GEANT2012 |
|---|---|---|---|
| 0.8 | 91% | 69% | 66% |
| 0.95 | 96% | 71% | 85% |
| 1.1 | 97% | 69% | 89% |

At high load, zero-shot beats the tabular specialist on every topology and roughly
matches the MLP specialist.

**Topology Zoo suite (98 real WANs, `m2_zoo_suite_summary.csv`).** At 95% load the
best zero-shot seed beats OSPF on 89.8% of networks (+0.094 mean reward, 95% CI ±0.021).
The other seeds beat it on 79.6%. The LP oracle beats OSPF on 93.9%. At 80% load the
win rate is 69–81%.

**Link-failure resilience (GEANT2012, load 0.8, 20 trials, `m2_failure_summary.csv`).**
Links fail at interval 100. Reward drop after the failure:

| Method | Δ reward | Post-failure loss % |
|---|---|---|
| OSPF | -0.060 | 0.92 |
| Softmin (static) | -0.052 | 0.67 |
| MLP specialist | -0.045 | 0.38 |
| GNN zero-shot (3 seeds) | -0.032 to -0.042 | 0.16–0.45 |
| GNN specialist | -0.035 | 0.22 |
| ECMP-hop | -0.034 | 0.46 |

The GNN policies lose about as little reward as ECMP-hop (-0.03 to -0.04) while
keeping a much higher post-failure reward (0.35–0.40 vs 0.29). The GNN specialist and
the best zero-shot seed also have the lowest post-failure loss. OSPF loses the most.
The GNN re-routes because it reads the current graph at every step.

**Ablations (GEANT2012 specialist, best validation reward).** The full model scores
0.388 and the link-features-only variant scores 0.393. Removing node temperature drops
it to 0.293, so the node-temperature output carries most of the gain.

## 5. Methodologies 1 and 3: training behaviour (`results_unified/nonstationarity_*`)
Best validation reward and run-to-run spread over 3 seeds:

| Algorithm | Family | NSFNet best (sd) | NSFNet final | GEANT best (sd) | GEANT final | Wall-clock NSFNet / GEANT |
|---|---|---|---|---|---|---|
| IPPO | M1 | 0.326 (0.006) | 0.320 | 0.345 (0.002) | 0.332 | 318 / 969 s |
| MAPPO | M3 | 0.324 (0.004) | 0.319 | 0.340 (0.006) | 0.331 | 318 / 951 s |
| IQL-DQN | M1 | 0.269 (0.012) | 0.263 | 0.300 (0.008) | 0.297 | 103 / 422 s |
| QMIX | M3 | 0.212 (0.001) | 0.183 | 0.268 (0.018) | 0.234 | 117 / 454 s |
| Tabular-Q | M1 | 0.194 (0.004) | 0.163 | 0.276 (0.012) | 0.249 | 96 / 427 s |
| MADDPG | M3 | 0.242 (0.034) | 0.071 | 0.257 (0.004) | 0.044 | 178 / 473 s |

* MAPPO's central critic explains more of the return variance than IPPO's local critics
  (explained variance 0.45 vs 0.27 on NSFNet, 0.58 vs 0.18 on GEANT). This did not turn
  into a better policy.
* MADDPG collapses late in training: its final reward is far below its best. It is only
  usable with best-checkpoint selection. Without the TD3-style fixes it fails outright:
  the vanilla MADDPG ablation's best validation reward is -0.25 to 0.01, against
  0.22–0.28 with the fixes (`results_marl/ablation_vanilla_maddpg/`).
* Per-run training cost (steps and wall-clock to 90% of best) is in
  `results_unified/training_cost.csv`. The M2 zero-shot GNN takes about 45–55 min per
  seed on CPU.

## 6. Why CTDE did not help
In this setting each switch's best split depends mostly on the congestion of its own
outgoing links, which its local observation already contains. The extra information in
the global state therefore adds little. The central critic makes value estimates better
(higher explained variance) without making policies better. QMIX's monotonic mixing
constraint and MADDPG's deterministic off-policy updates also handle the
heavy-tailed, non-stationary traffic worse than PPO's clipped on-policy updates. Treat
this as a measured result for this environment, not a general claim. CTDE may help on
tasks where agents' actions are more tightly coupled.

## 7. Mininet validation (`results_mininet/nsfnet_runs.csv`)
NSFNet on Open vSwitch with `tc`-shaped links at 1/10 scale, real UDP traffic and
OpenFlow 1.3 select groups. Load 0.95, traffic seed 101, 60 x 1 s intervals.

| Method | Reward | Loss % | Delay (ms) | Throughput (Mbps, model units) | MLU | Jain | Decision (ms) | OpenFlow install (ms) |
|---|---|---|---|---|---|---|---|---|
| OSPF | -0.875 | 9.21 | 36.9 | 667.9 | 0.888 | 0.987 | 0.01 | 0 |
| M1 IPPO | -0.696 | 6.22 | 34.8 | 690.0 | 0.765 | 0.997 | 2.2 | 304 |
| M3 MAPPO | -0.591 | 5.02 | 33.2 | 698.8 | 0.789 | 0.998 | 2.6 | 316 |
| **M2 GNN zero-shot** | **-0.439** | **3.94** | **30.5** | **706.7** | 0.771 | 0.998 | 7.0 | 387 |

The ranking among learned methods differs from the simulator's NSFNet ranking, where
IPPO, GNN and MAPPO are within 0.025 of each other. One emulation run cannot separate
them reliably. What it does show is that all three keep a clear margin over OSPF on a
real packet data plane.

## 8. Limitations
* Training and most evaluation use a flow-level simulator, not packet-level dynamics.
* The Mininet validation is a single run (one traffic seed, one load, seed-0 models). It
  shows transfer, not statistically separated rankings. ECMP-hop and load 0.8 are in the
  runner's defaults but were not part of this run.
* M1/M3 are trained per topology (NSFNet, GEANT) and do not transfer to other graphs.
  Only M2 is topology-agnostic.
* The LP oracle sees the true mean traffic matrix. It is a reference point, not a
  deployable baseline.
* Specialists use a single training seed.
* Methodology 2 v1 (`methodology_2_gnn/`, outside `v2/`) used an environment whose
  rewards did not depend on the action. Its numbers, including the earlier 0.38%
  generalisation gap and the Sprint-0 OSPF/ECMP table, are not valid. See
  `methodology_2_gnn/V1_NOTICE.md`. All numbers in this report come from v2 and the
  shared simulator.

## 9. File index
| Result | File |
|---|---|
| Cross-methodology, per algorithm | `results_unified/unified_by_algorithm.csv` |
| Cross-methodology, per checkpoint / per seed | `results_unified/unified_summary.csv`, `unified_per_seed.csv` |
| Comparison figure | `results_unified/unified_comparison.png` |
| Training cost / non-stationarity | `results_unified/training_cost.csv`, `nonstationarity_*.csv`, `nonstationarity_curves_*.png` |
| M1 / M3 per-topology tables | `results_marl/tables/m{1,3}_{nsfnet,Geant2012}*.csv` |
| M2 primary, gap, suite, failure | `methodology_2_gnn/v2/results/tables/m2_*.csv` |
| M2 figures | `methodology_2_gnn/v2/results/figures/` |
| Week-1 OSPF/ECMP baseline table (re-run on the simulator) | `results_unified/week1_baseline_metrics.csv` |
| Mininet | `results_mininet/nsfnet_runs.csv` |
