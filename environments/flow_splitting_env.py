"""
Multi-agent flow-splitting environment (Dec-POMDP) for Methodologies 1 and 3.

Agents   : one per switch (node). Agent v controls the cost of each of its
           outgoing links; the shared TESimulator turns all agents' costs into
           loop-free hop-by-hop split ratios (softmin over strictly-closer next
           hops), routes the traffic, and returns one TEAM reward.
Local obs: o_v = [own node features (5)] + for each outgoing-link slot k < D:
           [link features (6), neighbour node features (5), present-flag (1)]
           -> 5 + 12*D floats, D = max degree of the topology. Only information
           a switch can measure itself or hear from its direct neighbours.
State    : s = all node features + all link features (for CTDE critics only).
Action   : a_v in R^D (one value per slot; absent slots ignored).

PettingZoo-ParallelEnv-like API: reset() -> {agent: obs}, step({agent: act}).
Vectorised helpers (obs_matrix / act_from_matrix) are what the trainers use.
"""
import numpy as np

try:
    from environments.te_sim import TESimulator
    from environments.bursty_traffic_gen import SCENARIOS
except ImportError:
    from te_sim import TESimulator
    from bursty_traffic_gen import SCENARIOS

NODE_F, EDGE_F = 5, 6


class FlowSplittingEnv:
    def __init__(self, graph, load=0.8, seed=0, scenario="pareto", shadow_baseline=False):
        self.G, self.scenario = graph, scenario
        self.sim = TESimulator(graph, load=load, seed=seed, traffic_kwargs=dict(SCENARIOS[scenario]))
        self.n, self.E = self.sim.n, self.sim.E
        self.agents = [f"switch_{v}" for v in range(self.n)]
        # slot table: out-links of each node, ordered by neighbour id
        self.D = int(self.sim.deg.max())
        self.slot_edge = -np.ones((self.n, self.D), dtype=int)
        for v in range(self.n):
            es = np.nonzero(self.sim.src == v)[0]
            es = es[np.argsort(self.sim.dst[es])]
            self.slot_edge[v, :len(es)] = es
        self.obs_dim = NODE_F + self.D * (EDGE_F + NODE_F + 1)
        self.state_dim = self.n * NODE_F + self.E * EDGE_F
        self.act_dim = self.D
        self.shadow_baseline = shadow_baseline
        self._make_shadow(load, seed)

    def _make_shadow(self, load, seed):
        self.shadow = None
        if self.shadow_baseline:
            self.shadow = TESimulator(self.G, load=load, seed=seed, traffic_kwargs=dict(SCENARIOS[self.scenario]))
            self.P_sp = self.shadow.routing_matrix(self.shadow.ospf_costs(), "sp")

    # ------------------------------------------------------------------ API
    def reset(self, load=None, seed=None):
        if load is not None:
            self.sim.load = load
        self.sim.reset(seed)
        self._make_shadow(self.sim.load, self.sim.seed)
        return self._obs_dict()

    def step(self, actions):
        """actions: {agent: array(D)} or array (n, D)."""
        A = np.stack([actions[a] for a in self.agents]) if isinstance(actions, dict) else np.asarray(actions)
        obs, r, info = self.step_matrix(A)
        return ({a: obs[i] for i, a in enumerate(self.agents)}, {a: r for a in self.agents}, info)

    # --------------------------------------------------------- vectorised
    def step_matrix(self, A):
        cost = self.cost_from_matrix(A)
        _, r, info = self.sim.step(cost, "softmin")
        info["reward_raw"] = r
        if self.shadow is not None:
            _, r_sh, _ = self.shadow.step(P=self.P_sp)
            r = r - r_sh                      # counterfactual OSPF baseline (unbiased)
        return self.obs_matrix(), r, info

    def observe_all(self):
        """(local obs matrix (n, obs_dim), global state (state_dim,)) from one telemetry read."""
        o = self.sim.observe()
        return self.obs_matrix(o), self.state(o)

    def cost_from_matrix(self, A):
        a = np.zeros(self.E)
        m = self.slot_edge >= 0
        a[self.slot_edge[m]] = np.asarray(A, dtype=np.float64)[m]
        return self.sim.costs_from_action(a[self.sim.alive])

    def obs_matrix(self, o=None):
        o = self.sim.observe() if o is None else o
        node_x = o["node_x"]
        edge_full = np.zeros((self.E, EDGE_F), np.float32)
        edge_full[o["edge_ids"]] = o["edge_x"]
        X = np.zeros((self.n, self.obs_dim), np.float32)
        X[:, :NODE_F] = node_x
        m = self.slot_edge >= 0
        e = np.where(m, self.slot_edge, 0)
        nb = self.sim.dst[e]
        slot = np.concatenate([edge_full[e], node_x[nb], m[..., None].astype(np.float32)], -1) * m[..., None]
        X[:, NODE_F:] = slot.reshape(self.n, -1)
        return X

    def state(self, o=None):
        o = self.sim.observe() if o is None else o
        edge_full = np.zeros((self.E, EDGE_F), np.float32)
        edge_full[o["edge_ids"]] = o["edge_x"]
        return np.concatenate([o["node_x"].ravel(), edge_full.ravel()])

    def _obs_dict(self):
        X = self.obs_matrix()
        return {a: X[i] for i, a in enumerate(self.agents)}

    @property
    def slot_mask(self):
        return self.slot_edge >= 0

    def signalling_bytes(self):
        """Per control interval, float32 telemetry each decision needs.
        decentralised: each switch hears its D neighbours' node+link summaries.
        centralised  : every node+link report goes to the controller and every
                       link cost comes back."""
        dec = int(self.slot_mask.sum()) * (EDGE_F + NODE_F) * 4
        cen = (self.n * NODE_F + self.E * EDGE_F + self.E) * 4
        return dict(decentralised_bytes=dec, centralised_bytes=cen)
