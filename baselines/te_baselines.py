"""
Non-learning baselines, all executed inside the same TESimulator physics.

ospf      : Dijkstra on propagation delay, single next hop (the team's OSPF).
ecmp      : equal split over next hops whose delay-path cost ties the minimum
            (the team's ECMP definition). With real-valued delays ties are rare.
ecmp_hop  : ECMP on hop count (what data-centre / IGP-unit-weight ECMP does).
softmin   : delay-based softmin splitting with a = 0 (the GNN's starting point).
lp_mean   : min-MLU multi-commodity flow solved by LP on the *mean* traffic
            matrix (knows the average demand, not the bursts). Reference only:
            it needs global traffic knowledge and a central solver.
"""
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def lp_routing(sim, delay_eps=1e-3):
    """Destination-based min-MLU routing on sim's mean matrix -> split tensor P."""
    n, E = sim.n, sim.E
    alive = np.nonzero(sim.alive)[0]
    Ea = len(alive)
    D = sim.traffic.mean_matrix * 1e6                     # bits/s
    nv = n * Ea + 1                                       # f[t, e] ..., U
    U = n * Ea
    rows, cols, vals, b = [], [], [], []
    r = 0
    src, dst = sim.src[alive], sim.dst[alive]
    for t in range(n):
        for v in range(n):
            if v == t:
                continue
            out_e = np.nonzero(src == v)[0]
            in_e = np.nonzero(dst == v)[0]
            rows += [r] * (len(out_e) + len(in_e))
            cols += list(t * Ea + out_e) + list(t * Ea + in_e)
            vals += [1.0] * len(out_e) + [-1.0] * len(in_e)
            b.append(D[v, t]); r += 1
    A_eq = coo_matrix((vals, (rows, cols)), shape=(r, nv)).tocsr()
    # capacity: sum_t f[t,e] - U * c_e <= 0
    rows, cols, vals = [], [], []
    for j, e in enumerate(alive):
        rows += [j] * (n + 1)
        cols += [t * Ea + j for t in range(n)] + [U]
        vals += [1.0] * n + [-sim.cap[e]]
    A_ub = coo_matrix((vals, (rows, cols)), shape=(Ea, nv)).tocsr()
    c = np.zeros(nv)
    c[U] = 1.0
    # tiny delay term keeps flows on short, loop-free paths
    total = max(D.sum(), 1.0)
    c[:U] = np.tile(sim.prop[alive] / sim.mean_prop, n) * delay_eps / total
    res = linprog(c, A_ub=A_ub, b_ub=np.zeros(Ea), A_eq=A_eq, b_eq=np.array(b),
                  bounds=[(0, None)] * nv, method="highs")
    if res.status != 0:
        raise RuntimeError(res.message)
    f = res.x[:U].reshape(n, Ea)
    P = np.zeros((n, n, n))
    for t in range(n):
        out = np.bincount(src, weights=f[t], minlength=n)
        frac = np.where(out[src] > 1e-9, f[t] / np.maximum(out[src], 1e-12), 0.0)
        P[t, src, dst] = frac
        P[t, t, :] = 0.0
    # nodes that carry no flow to t in the LP still need a default next hop
    P_sp = sim.routing_matrix(sim.ospf_costs(), "sp")
    empty = P.sum(-1) < 0.5
    P[empty] = P_sp[empty]
    return P, float(res.x[U])


class Baseline:
    def __init__(self, name):
        self.name = name
        self._P = None
        self._key = None

    def reset(self, sim):
        self._P, self._key = None, None

    def act(self, sim, obs=None):
        key = sim.alive.tobytes()
        if self._P is None or key != self._key:          # recompute after failures
            self._P = self._routing(sim)
            self._key = key
        return dict(P=self._P)

    def _routing(self, sim):
        if self.name == "OSPF":
            return sim.routing_matrix(sim.ospf_costs(), "sp")
        if self.name == "ECMP":
            return sim.routing_matrix(sim.ospf_costs(), "ecmp")
        if self.name == "ECMP-hop":
            return sim.routing_matrix(sim.hop_costs(), "ecmp")
        if self.name == "Softmin(static)":
            return sim.routing_matrix(sim.costs_from_action(np.zeros(int(sim.alive.sum()))), "softmin")
        if self.name == "LP-oracle(mean TM)":
            return lp_routing(sim)[0]
        raise ValueError(self.name)


BASELINES = ["OSPF", "ECMP", "ECMP-hop", "Softmin(static)", "LP-oracle(mean TM)"]
