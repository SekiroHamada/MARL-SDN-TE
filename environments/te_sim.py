"""
Flow-level SDN traffic-engineering simulator (Methodology 2, v2).

Why this exists
---------------
The Sprint-0 `SDNInterfaceContract.step()` ignores the routing action and draws
throughput / delay / loss from `random.uniform`, so no policy can be learned or
compared there. This simulator makes every metric a deterministic function of
(topology, traffic, routing decision, queue state):

* Traffic: gravity-model base matrix (new per episode) x Poisson-like smooth
  background + Pareto ON/OFF heavy-tailed bursts per flow (self-similar
  aggregate). Calibrated so the *mean* matrix routed by OSPF gives a target
  max-link-utilisation `load` (0.5 / 0.8 / 0.95 ... like the team's evaluator).
* Routing: destination-based, hop-by-hop flow splitting. The policy outputs one
  cost per directed link; each node splits traffic for destination t among
  neighbours that are strictly closer to t under those costs ("softmin
  routing", Valadarsky et al., HotNets'17). Strict-decrease => loop-free.
  OSPF (single shortest path) and ECMP (equal split over equal-cost next hops)
  are special cases, so all methods run on identical physics.
* Links: capacity C, propagation delay, finite FIFO buffer B packets.
  Fluid queue: q' = clip(q + (L - C)dt/pkt, 0, B); overflow is dropped.
  Per-link delay = propagation + backlog/C + M/M/1 waiting term (bounded by B).
* Losses are propagated downstream (a dropped packet never reaches later hops).

All flows for all destinations are solved jointly with batched linear algebra:
for destination t with split matrix P_t, node throughput x solves
x = d_t + P_t^T x, so per step cost is a few (N x N) batched solves.
"""
from __future__ import annotations

import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

PKT_BITS = 12_000.0          # 1500-byte packets
EPS = 1e-9


class TrafficProcess:
    """Gravity base matrix + Pareto ON/OFF bursts. Independent of actions."""

    def __init__(self, n, rng, burst_share=0.5, pareto_shape=1.5, pareto_min=1.0,
                 max_period=60, noise=0.05, hotspot_gain=2.5, hotspot_duty=0.15):
        self.n, self.rng = n, rng
        self.burst_share, self.noise = burst_share, noise
        self.a, self.xm, self.cap = pareto_shape, pareto_min, max_period
        w = rng.lognormal(0.0, 0.8, n)                       # node "mass"
        base = np.outer(w, w)
        np.fill_diagonal(base, 0.0)
        self.base_shape = base / base.sum()
        self.scale = 1.0                                     # Mbps, set by calibrate
        self.on = rng.random((n, n)) < 0.5
        self.timer = self._period((n, n))
        # mean ON fraction of a symmetric ON/OFF Pareto process = 0.5
        self.p_on = 0.5
        # correlated source hotspots: a source's whole row surges together
        # (OFF periods stretched so the ON duty cycle is ~hotspot_duty)
        self.h_gain, self.h_duty = hotspot_gain, hotspot_duty
        self.h_on = rng.random(n) < hotspot_duty
        self.h_timer = self._period(n) * np.where(self.h_on, 1.0, (1 - hotspot_duty) / hotspot_duty)

    def _period(self, shape):
        p = self.xm * (1.0 + self.rng.pareto(self.a, shape))
        return np.minimum(np.ceil(p), self.cap)

    @property
    def mean_matrix(self):
        return self.base_shape * self.scale

    def sample(self):
        """Demand matrix (Mbps) for the next step."""
        self.timer -= 1
        flip = self.timer <= 0
        if flip.any():
            self.on[flip] = ~self.on[flip]
            self.timer[flip] = self._period(int(flip.sum()))
        self.h_timer -= 1
        hf = self.h_timer <= 0
        if hf.any():
            self.h_on[hf] = ~self.h_on[hf]
            k = int(hf.sum())
            stretch = np.where(self.h_on[hf], 1.0, (1 - self.h_duty) / self.h_duty)
            self.h_timer[hf] = self._period(k) * stretch
        hot = (1.0 + self.h_gain * self.h_on) / (1.0 + self.h_gain * self.h_duty)
        smooth = 1.0 - self.burst_share
        mult = smooth + self.burst_share * self.on / self.p_on
        mult = mult * np.clip(1.0 + self.noise * self.rng.standard_normal((self.n, self.n)), 0.0, None)
        d = self.mean_matrix * mult * hot[:, None]
        np.fill_diagonal(d, 0.0)
        return d


class TESimulator:
    """One network + one traffic process. Methods share identical physics."""

    def __init__(self, graph: nx.Graph, load: float = 0.8, seed: int = 0,
                 dt: float = 0.01, softmin_gamma: float = 2.0,
                 reward_weights=(1.0, 0.5, 2.0), traffic_kwargs=None):
        self.G = graph
        self.n = graph.number_of_nodes()
        self.load, self.dt, self.gamma_sm = load, dt, softmin_gamma
        self.alpha, self.beta, self.gamma_r = reward_weights
        self.seed = seed
        self.traffic_kwargs = traffic_kwargs or {}

        src, dst, cap, prop, buf, und = [], [], [], [], [], []
        for k, (u, v, d) in enumerate(graph.edges(data=True)):
            for a, b in ((u, v), (v, u)):
                src.append(a); dst.append(b); und.append(k)
                cap.append(d["capacity"] * 1e6)              # bits/s
                prop.append(d["delay"])                      # ms
                buf.append(d.get("queue", 100))
        self.src, self.dst = np.array(src), np.array(dst)
        self.undirected_id = np.array(und)
        self.cap, self.prop = np.array(cap, float), np.array(prop, float)
        self.buf = np.array(buf, float)
        self.E = len(self.src)
        self.mean_prop = float(self.prop.mean())
        self.deg = np.bincount(self.src, minlength=self.n)
        self.out_cap = np.bincount(self.src, weights=self.cap, minlength=self.n)
        self.in_cap = np.bincount(self.dst, weights=self.cap, minlength=self.n)
        self.reset()

    # ------------------------------------------------------------------ setup
    def reset(self, seed: int | None = None):
        if seed is not None:
            self.seed = seed
        self.rng = np.random.default_rng(self.seed)
        self.alive = np.ones(self.E, bool)
        self.queue = np.zeros(self.E)
        self.traffic = TrafficProcess(self.n, self.rng, **self.traffic_kwargs)
        self._calibrate()
        self.t = 0
        self.demand = self.traffic.sample()          # demand for step 0
        self.last = self._zero_telemetry()
        return self.observe()

    def _calibrate(self):
        """Scale traffic so OSPF on the mean matrix has MLU == self.load."""
        self.traffic.scale = 1.0
        P = self.routing_matrix(self.ospf_costs(), mode="sp")
        L = self._link_loads(P, self.traffic.mean_matrix * 1e6)[0]
        mlu = float((L / self.cap).max())
        self.traffic.scale = self.load / max(mlu, EPS)
        # reference delay for reward normalisation: demand-weighted OSPF prop delay
        Dsp = self._path_delay(P, self.prop)
        M = self.traffic.mean_matrix
        self.ref_delay = float((Dsp.T * M).sum() / max(M.sum(), EPS))

    def _zero_telemetry(self):
        return dict(util=np.zeros(self.E), drop=np.zeros(self.E), queue=np.zeros(self.E),
                    demand=self.traffic.mean_matrix.copy())

    # ---------------------------------------------------------------- routing
    def ospf_costs(self):
        return self.prop.copy()

    def hop_costs(self):
        return np.ones(self.E)

    def all_pairs(self, cost):
        c = np.where(self.alive, cost, 0.0)
        m = csr_matrix((c[self.alive], (self.src[self.alive], self.dst[self.alive])), shape=(self.n, self.n))
        return shortest_path(m, method="D", directed=True)         # dist[i, j]

    def routing_matrix(self, cost, mode="softmin", gamma=None):
        """Split tensor P[t, v, u]: fraction of v's traffic to t sent on v->u.

        mode = 'softmin' : p ~ exp(-gamma * excess) over strictly-closer hops
               'ecmp'    : equal split over next hops with excess == 0
               'sp'      : single next hop (lowest-excess, lowest index)
        """
        gamma = self.gamma_sm if gamma is None else gamma
        g_e = (np.asarray(gamma)[self.src] if np.ndim(gamma) else np.full(self.E, float(gamma)))[None, :]
        dist = self.all_pairs(cost)                                  # (n, n)
        n, E = self.n, self.E
        dv = dist[self.src].T                                        # (t, e) dist(src_e, t)
        du = dist[self.dst].T                                        # (t, e) dist(dst_e, t)
        ok = self.alive[None, :] & np.isfinite(du) & np.isfinite(dv) & (du < dv - 1e-9)
        excess = np.where(ok, cost[None, :] + du - dv, np.inf)
        excess = np.maximum(excess, 0.0)
        rel = excess / np.maximum(np.where(np.isfinite(dv), dv, 1.0), EPS)  # scale-free
        if mode == "softmin":
            score = np.where(ok, np.exp(-g_e * excess / self._cost_scale(cost)), 0.0)
        elif mode == "ecmp":
            score = (ok & (rel < 1e-6)).astype(float)
        elif mode == "sp":
            # single next hop per (t, src): minimum excess, ties -> lowest edge id
            key = np.where(ok, excess, np.inf)
            tt = np.arange(n)[:, None].repeat(E, 1)
            ss = self.src[None, :].repeat(n, 0)
            kmin = np.full((n, n), np.inf)
            np.minimum.at(kmin, (tt, ss), key)
            cand = ok & (key <= kmin[tt, ss] + 1e-12)
            eid = np.where(cand, np.arange(E)[None, :], E)
            emin = np.full((n, n), E)
            np.minimum.at(emin, (tt, ss), eid)
            score = (cand & (np.arange(E)[None, :] == emin[tt, ss])).astype(float)
        else:
            raise ValueError(mode)
        # normalise per (t, src)
        norm = np.zeros((n, n))
        np.add.at(norm, (np.arange(n)[:, None].repeat(E, 1), self.src[None, :].repeat(n, 0)), score)
        frac = score / np.maximum(norm[:, self.src], EPS)
        P = np.zeros((n, n, n))
        P[np.arange(n)[:, None], self.src[None, :], self.dst[None, :]] = frac
        P[np.arange(n), np.arange(n), :] = 0.0                       # destination absorbs
        self._last_frac = frac
        return P

    def _cost_scale(self, cost):
        return float(np.mean(cost[self.alive])) if self.alive.any() else 1.0

    # ------------------------------------------------------------- flow solve
    def _solve(self, P, demand_bits, survive=None):
        """x[t, v] = traffic for destination t passing through v (bits/s)."""
        n = self.n
        Pe = P if survive is None else P * self._edge_tensor(survive)
        A = np.eye(n)[None] - np.transpose(Pe, (0, 2, 1))           # (t, n, n)
        b = demand_bits.T.copy()                                     # b[t, v] = d[v, t]
        return np.linalg.solve(A, b[..., None])[..., 0], Pe

    def _edge_tensor(self, vals):
        T = np.zeros((self.n, self.n))
        T[self.src, self.dst] = vals
        return T[None]

    def _link_loads(self, P, demand_bits):
        x, _ = self._solve(P, demand_bits)
        frac = P[:, self.src, self.dst]                              # (t, e)
        return (x[:, self.src] * frac).sum(0), x

    def _path_delay(self, P, link_delay):
        """D[t, v]: expected delay (ms) from v to t under split P."""
        n = self.n
        c = (P * self._edge_tensor(link_delay)).sum(-1)              # (t, v)
        A = np.eye(n)[None] - P
        return np.linalg.solve(A, c[..., None])[..., 0]

    # ------------------------------------------------------------------- step
    def step(self, cost=None, mode="softmin", P=None, gamma=None):
        """Advance one dt. Either a split tensor P, or link costs `cost` (shape E)
        plus optional per-node softmin temperature `gamma` (shape n)."""
        if P is None:
            P = self.routing_matrix(cost, mode, gamma)
        d = self.demand
        bits = d * 1e6
        L, x = self._link_loads(P, bits)                             # offered link load
        L = np.where(self.alive, L, 0.0)
        # fluid FIFO queue with finite buffer
        arrive = L * self.dt / PKT_BITS                               # packets this step
        serve = self.cap * self.dt / PKT_BITS
        room = self.buf - self.queue
        overflow = np.maximum(0.0, arrive - serve - room)
        drop = np.where(arrive > 0, overflow / np.maximum(arrive, EPS), 0.0)
        q_new = np.clip(self.queue + arrive - serve, 0.0, self.buf)
        q_avg = 0.5 * (self.queue + q_new)
        self.queue = q_new
        # delivered traffic with losses propagated downstream
        xs, _ = self._solve(P, bits, survive=1.0 - drop)
        delivered_t = xs[np.arange(self.n), np.arange(self.n)]       # arrivals at t
        offered = bits.sum()
        delivered = float(delivered_t.sum())
        # per-flow delivery ratio via per-source solve is expensive; use path survival
        surv_log = np.log(np.clip(1.0 - drop, 1e-12, 1.0))
        S = np.exp(-self._path_delay(P, -surv_log))                  # (t, v) survival prob
        routed = P.sum(-1) > 0.5
        routed[np.arange(self.n), np.arange(self.n)] = True
        S = S * routed                                                # no route => nothing delivered
        util = L / self.cap
        rho = np.clip(util, 0.0, 0.999)
        svc_ms = PKT_BITS / self.cap * 1e3
        wait_ms = np.minimum(rho / (1.0 - rho), self.buf) * svc_ms + q_avg * svc_ms
        link_delay = self.prop + wait_ms
        D = self._path_delay(P, link_delay)                          # (t, v) ms
        flow = d.T > 0                                               # [t, v]
        w = d.T * S
        avg_delay = float((D * w)[flow].sum() / max(w[flow].sum(), EPS))
        flow_ratio = S[flow]
        jain = float(flow_ratio.sum() ** 2 / (len(flow_ratio) * (flow_ratio ** 2).sum() + EPS))
        loss = 1.0 - delivered / max(offered, EPS)

        thr_frac = delivered / max(offered, EPS)
        reward = (self.alpha * thr_frac - self.beta * avg_delay / self.ref_delay
                  - self.gamma_r * loss)
        alive_util = util[self.alive]
        info = dict(throughput_mbps=delivered / 1e6, offered_mbps=offered / 1e6,
                    loss_ratio=loss, avg_delay_ms=avg_delay,
                    mlu=float(alive_util.max()), util_mean=float(alive_util.mean()),
                    util_std=float(alive_util.std()), jain=jain, reward=reward)
        self.last = dict(util=util, drop=drop, queue=self.queue / self.buf, demand=d)
        self.t += 1
        self.demand = self.traffic.sample()
        return self.observe(), reward, info

    # ------------------------------------------------------------ observation
    def observe(self):
        """Topology-size-independent features. The agent sees *last* step's
        telemetry and demand (it cannot see the future burst)."""
        last = self.last
        d = last["demand"] * 1e6
        orig = d.sum(1) / np.maximum(self.out_cap, EPS)
        term = d.sum(0) / np.maximum(self.in_cap, EPS)
        util, q = last["util"], last["queue"]
        alive = self.alive.astype(float)
        out_util = np.bincount(self.src, weights=util * alive, minlength=self.n) / np.maximum(
            np.bincount(self.src, weights=alive, minlength=self.n), 1)
        out_q = np.bincount(self.src, weights=q * alive, minlength=self.n) / np.maximum(
            np.bincount(self.src, weights=alive, minlength=self.n), 1)
        deg = np.bincount(self.src, weights=alive, minlength=self.n)
        # what OSPF would do with the last demand: a cheap, local-ish congestion hint
        if not hasattr(self, "_P_sp") or self._P_sp_alive_key != self.alive.tobytes():
            self._P_sp = self.routing_matrix(self.ospf_costs(), "sp")
            self._P_sp_alive_key = self.alive.tobytes()
        L_sp = self._link_loads(self._P_sp, d)[0] / self.cap
        node_x = np.stack([out_q, np.clip(orig, 0, 2), out_util.clip(0, 2), np.minimum(deg / 10.0, 1.0),
                           np.clip(term, 0, 2)], 1)
        edge_x = np.stack([self.cap / 1e8, self.prop / self.mean_prop, util.clip(0, 2),
                           q, last["drop"], L_sp.clip(0, 2)], 1)
        idx = np.nonzero(self.alive)[0]
        return dict(node_x=node_x.astype(np.float32), edge_x=edge_x[idx].astype(np.float32),
                    edge_index=np.stack([self.src[idx], self.dst[idx]]).astype(np.int64),
                    edge_ids=idx)

    # ---------------------------------------------------------------- failure
    def fail_links(self, undirected_ids):
        for k in undirected_ids:
            self.alive[self.undirected_id == k] = False
        self.queue[~self.alive] = 0.0

    def connected_after(self, undirected_ids):
        H = self.G.copy()
        edges = list(self.G.edges())
        H.remove_edges_from([edges[k] for k in undirected_ids])
        return nx.is_connected(H)

    def costs_from_action(self, a_alive):
        """Map policy outputs on alive edges to positive link costs.
        cost = (prop / mean prop) * exp(a): a = 0 reproduces delay-based softmin."""
        cost = np.full(self.E, 1.0)
        a = np.clip(a_alive, -3.0, 3.0)
        cost[self.alive] = (self.prop[self.alive] / self.mean_prop) * np.exp(a)
        return cost

    def decode_action(self, a):
        """Full policy action = [one value per alive directed link, one per node].
        Link values -> link costs; node values -> that node's split temperature
        gamma_v = gamma0 * exp(b_v) (large = OSPF-like, small = spread load)."""
        a = np.asarray(a, dtype=np.float64)
        k = int(self.alive.sum())
        cost = self.costs_from_action(a[:k])
        gamma = self.gamma_sm * np.exp(np.clip(a[k:k + self.n], -3.0, 3.0)) if len(a) > k else None
        return cost, gamma
