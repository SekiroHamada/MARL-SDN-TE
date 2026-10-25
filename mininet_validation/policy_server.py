"""
Policy server = the decision half of the SDN controller (runs in the `marl-sdn` venv).

The Mininet runner (root, system Python) connects over TCP and, every control
interval, sends link telemetry measured in the emulated network. The server
turns it into exactly the features the policies were trained on, runs the
chosen method (OSPF / ECMP / LP oracle / M1 IPPO / M2 GNN / M3 MAPPO / MADDPG),
and returns split weights per (switch, destination, next hop), which the runner
installs as OpenFlow select-group bucket weights.

    python mininet_validation/policy_server.py --topology nsfnet
    python mininet_validation/policy_server.py --topology nsfnet --list    # method names

Protocol: one JSON object per line, request -> reply.
  {"cmd": "hello"}                                   -> topology (nodes, links, directed-link order)
  {"cmd": "start", "method", "load", "seed", "intervals", "scale"}
                                                     -> traffic schedule (Mbps, already scaled) + ref delay
  {"cmd": "step", "k", "util", "queue", "drop"}      -> {"routes": [[v, t, [[u, w], ...]], ...]}
"""
import os as _os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")

import argparse, json, os, socket, sys, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from environments.topologies import load_named
from environments.flow_splitting_env import FlowSplittingEnv
from evaluation.unified_benchmark import methods_for

MIN_WEIGHT = 0.01     # buckets carrying < 1% are dropped (renormalised by OVS weights)


class Session:
    def __init__(self, G, topo):
        self.G, self.topo = G, topo
        self.methods = {m.name: m for m in methods_for(topo, G)}
        self.env = None

    def hello(self):
        e = FlowSplittingEnv(self.G)
        links = [[int(u), int(v), float(d["capacity"]), float(d["delay"]), int(d.get("queue", 100))]
                 for u, v, d in self.G.edges(data=True)]
        return dict(name=self.topo, nodes=self.G.number_of_nodes(), links=links,
                    directed=[[int(a), int(b)] for a, b in zip(e.sim.src, e.sim.dst)],
                    methods=list(self.methods))

    def start(self, method, load, seed, intervals, scale):
        if method not in self.methods:
            raise ValueError(f"unknown method {method!r}; available: {list(self.methods)}")
        self.m = self.methods[method]
        if hasattr(self.m, "b"):
            self.m.b.reset(None)
        self.env = FlowSplittingEnv(self.G, load=load, seed=seed)
        sim = self.env.sim
        # same gravity + Pareto/hot-spot process as training, calibrated to `load`
        self.sched = [sim.demand.copy()] + [sim.traffic.sample() for _ in range(intervals - 1)]
        self.lat = []
        return dict(rates_mbps=[(d * scale).round(4).tolist() for d in self.sched],
                    ref_delay_ms=float(sim.ref_delay), model_offered_mbps=float(np.mean([d.sum() for d in self.sched])))

    def step(self, k, util, queue, drop):
        sim = self.env.sim
        prev = self.sched[k - 1] if k > 0 else sim.traffic.mean_matrix
        sim.last = dict(util=np.asarray(util, float), queue=np.asarray(queue, float),
                        drop=np.asarray(drop, float), demand=np.asarray(prev, float))
        t0 = time.perf_counter()
        a = self.m.act(self.env)
        if "P" in a:
            P = a["P"]
        elif "A" in a:
            P = sim.routing_matrix(self.env.cost_from_matrix(a["A"]), "softmin")
        else:
            P = sim.routing_matrix(a["cost"], "softmin", a.get("gamma"))
        self.lat.append((time.perf_counter() - t0) * 1e3)
        routes = []
        n = sim.n
        for t in range(n):
            for v in range(n):
                if v == t:
                    continue
                row = P[t, v]
                nh = [[int(u), round(float(row[u]), 4)] for u in np.nonzero(row > MIN_WEIGHT)[0]]
                if not nh:                                # numerical corner case: keep best hop
                    nh = [[int(np.argmax(row)), 1.0]]
                routes.append([v, t, nh])
        return dict(routes=routes, decision_ms=self.lat[-1])


def serve(sess, host, port):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port)); srv.listen(1)
    print(f"policy server for '{sess.topo}' listening on {host}:{port}  methods: {list(sess.methods)}", flush=True)
    while True:
        conn, addr = srv.accept()
        print("runner connected", addr, flush=True)
        f = conn.makefile("rwb")
        for line in f:
            req = json.loads(line)
            try:
                cmd = req.pop("cmd")
                rep = getattr(sess, cmd)(**req)
                if cmd == "start":
                    print(f"  run: {req['method']} load={req['load']} seed={req['seed']}", flush=True)
            except Exception as ex:                      # report errors to the runner
                rep = dict(error=f"{type(ex).__name__}: {ex}")
            f.write((json.dumps(rep) + "\n").encode()); f.flush()
        print("runner disconnected", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--topology", default="nsfnet")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=6655)
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    import torch
    torch.set_num_threads(1)
    s = Session(load_named(a.topology), a.topology)
    if a.list:
        print("\n".join(s.methods)); sys.exit()
    serve(s, a.host, a.port)
