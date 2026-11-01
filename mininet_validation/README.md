# Mininet validation (emulated SDN data plane)

Training happens in the fast simulator (`environments/te_sim.py`). This folder
**validates the trained policies on an emulated network**: Open vSwitch
switches, Linux `tc`-shaped links, real UDP packets, OpenFlow 1.3 rules.

## How it works
```
 terminal 1 (venv, PyTorch)                 terminal 2 (sudo, system python3 + Mininet)
 policy_server.py  <---- TCP JSON ---->   run_mininet.py
   loads OSPF/ECMP/LP + M1/M2/M3 models       builds topology: s1..sN (OVS), h1..hN, TCLinks
   telemetry -> same features as training     (capacity x scale, same delays, 100-pkt buffers)
   policy -> split ratios P[t, v, u]          one OpenFlow select GROUP per (switch, destination),
                                              bucket weights = split ratios
                                              traffic_agent.py in every host: bursty UDP, K flows
                                              per pair, seq + timestamp in every packet
                                              every interval: read tx counters + tc backlog/drops
                                              -> server -> install changed groups
```
* **Same traffic for every method:** the server generates the schedule with the
  simulator's gravity + Pareto + hot-spot process (fixed seed), scaled by `--scale`.
* **Measured, not modelled:** loss and one-way delay come from packet sequence
  numbers and timestamps at the receivers (Mininet hosts share the kernel clock).
* **Splitting is per flow:** OVS hashes each UDP flow onto one bucket, so every
  host pair sends `--flows` (default 8) flows for the weights to take effect.
* **Time scale:** one control interval is 1 s here (10 ms in the simulator). The
  policies only see ratios (utilisation, queue fraction, drop fraction), so they
  transfer, but bursts last seconds instead of tens of milliseconds.

## Run
```bash
# terminal 1 - repo root, venv active
python mininet_validation/policy_server.py --topology nsfnet
#   (wait for "listening on 127.0.0.1:6655")

# terminal 2 - repo root (sudo uses the system python3 that has Mininet)
sudo python3 mininet_validation/run_mininet.py --check      # build NSFNet, install OSPF, pingall
sudo python3 mininet_validation/run_mininet.py              # OSPF, ECMP-hop, M2 GNN, M1 IPPO, M3 MAPPO
                                                            # x loads 0.8, 0.95  (~11 min)
```
Results: `results_mininet/nsfnet_runs.csv`, one row per method x load x seed:
reward (team formula), loss %, one-way delay, throughput (emulated and rescaled
to model units), mean max-link-utilisation, utilisation spread, Jain's index,
policy decision time, OpenFlow install time, groups changed per interval.

| Option | Default | Meaning |
|---|---|---|
| `--methods` | OSPF, ECMP-hop, M2 GNN zero-shot (seed0), M1 IPPO (seed0), M3 MAPPO (seed0) | names from `policy_server.py --list` |
| `--loads` / `--seeds` | 0.8 0.95 / 101 | load (OSPF max-link-utilisation) and traffic seed |
| `--intervals` / `--interval-s` | 60 / 1.0 | run length |
| `--warmup` | 5 | first intervals excluded from metrics |
| `--scale` | 0.1 | link bandwidth scale (100 Mbps -> 10 Mbps); raise if your CPU keeps up |
| `--cli` | | build + OSPF, then open the Mininet CLI for manual checks |

Other topologies: start the server with `--topology Geant2012` (or any Topology Zoo
name). M1/M3 models exist for `nsfnet` and `Geant2012`; the M2 zero-shot GNN works on any.

## Troubleshooting
* `ovs-ofctl ... connection refused` or missing switches: `sudo service openvswitch-switch start`.
* Leftovers from a crashed run: `sudo mn -c`.
* High loss for **every** method, OSPF included, even at load 0.8: the CPU can't keep
  up with the senders - lower `--scale` (e.g. 0.05).
* `ModuleNotFoundError: mininet`: run with `sudo python3` (system Python), not the venv's python.
* `Connection refused` from the runner: start `policy_server.py` first and wait for "listening".

## Status
Tested without Mininet: traffic agents (exact pacing, loss/delay accounting over
loopback), policy server for all 7 methods (valid, loop-free split weights),
OpenFlow group/flow command generation, and `tc` counter parsing. The full
Mininet run needs a machine with Mininet + OVS; send the output of the first run
if anything fails.
