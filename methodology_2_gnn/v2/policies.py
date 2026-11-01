"""
Policies for Methodology 2 (v2).

GNNPolicy  - topology-agnostic: GINE message passing with edge-state updates.
             Outputs one Gaussian mean per *directed link* (a residual on the
             delay-based link cost). One set of weights for every graph size.
MLPPolicy  - fixed-structure baseline: flattens all node+edge features of ONE
             topology; output size = that topology's link count. Cannot be
             applied to a graph of different size (the generalisation problem).

Both expose:  forward(obs_list) -> mu (sum_E,), edge_graph (sum_E,), value (B,)
"""
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GINEConv, global_mean_pool, global_max_pool

NODE_F, EDGE_F = 5, 6


def obs_to_data(obs):
    return Data(x=torch.from_numpy(obs["node_x"]),
                edge_index=torch.from_numpy(obs["edge_index"]),
                edge_attr=torch.from_numpy(obs["edge_x"]))


def mlp(i, h, o, layers=2):
    # ELU, not ReLU: with ReLU the actor head's hidden units died early in
    # training (0/64 alive in one run) and the policy froze, because a uniform
    # output bias has no effect on scale-invariant softmin routing.
    mods, d = [], i
    for _ in range(layers - 1):
        mods += [nn.Linear(d, h), nn.ELU()]
        d = h
    mods.append(nn.Linear(d, o))
    return nn.Sequential(*mods)


class GraphEncoder(nn.Module):
    """GINE message passing with residual node AND edge-state updates."""

    def __init__(self, hidden, layers):
        super().__init__()
        self.node_in = mlp(NODE_F, hidden, hidden)
        self.edge_in = mlp(EDGE_F, hidden, hidden)
        self.convs = nn.ModuleList(GINEConv(mlp(hidden, hidden, hidden), edge_dim=hidden) for _ in range(layers))
        self.edge_upd = nn.ModuleList(mlp(3 * hidden, hidden, hidden) for _ in range(layers))
        self.n_norm = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(layers))
        self.e_norm = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(layers))

    def forward(self, b):
        h, e = self.node_in(b.x), self.edge_in(b.edge_attr)
        s, d = b.edge_index
        for conv, upd, nn_, en in zip(self.convs, self.edge_upd, self.n_norm, self.e_norm):
            h = nn_(h + torch.relu(conv(h, b.edge_index, e)))
            e = en(e + torch.relu(upd(torch.cat([h[s], h[d], e], -1))))
        pooled = torch.cat([global_mean_pool(h, b.batch), global_max_pool(h, b.batch)], -1)
        return h, e, pooled


class GNNPolicy(nn.Module):
    """Actor and critic have SEPARATE encoders: with a shared trunk the value
    gradient was ~80x the policy gradient and starved the actor under global
    grad-norm clipping (measured; see README)."""

    def __init__(self, hidden=64, layers=4, global_context=True, init_log_std=-1.0,
                 critic_hidden=48, critic_layers=3, node_actions=False, node_init_log_std=-2.0):
        super().__init__()
        self.enc = GraphEncoder(hidden, layers)
        self.global_context = global_context
        g = 2 * hidden if global_context else 0
        self.actor = mlp(3 * hidden + g, hidden, 1)
        nn.init.uniform_(self.actor[-1].weight, -1e-3, 1e-3)    # start at a ~ 0
        nn.init.zeros_(self.actor[-1].bias)
        self.node_actor = mlp(hidden + g, hidden, 1)          # per-node split temperature
        nn.init.uniform_(self.node_actor[-1].weight, -1e-3, 1e-3)
        nn.init.zeros_(self.node_actor[-1].bias)
        self.log_std = nn.Parameter(torch.tensor(float(init_log_std)))
        # node temperatures rescale *every* split at a node, so they get their
        # own (smaller) exploration noise
        self.node_actions = node_actions
        self.node_log_std = nn.Parameter(torch.tensor(float(node_init_log_std)))
        self.critic_enc = GraphEncoder(critic_hidden, critic_layers)
        self.critic = mlp(3 * critic_hidden, critic_hidden, 1)

    def actor_parameters(self):
        return [p for n, p in self.named_parameters() if not n.startswith("critic")]

    def critic_parameters(self):
        return [p for n, p in self.named_parameters() if n.startswith("critic")]

    def forward(self, obs_list, with_value=True):
        b = Batch.from_data_list([obs_to_data(o) for o in obs_list])
        h, e, pooled = self.enc(b)
        s = b.edge_index[0]
        edge_graph = b.batch[s]
        z = [h[s], h[b.edge_index[1]], e]
        if self.global_context:
            z.append(pooled[edge_graph])
        mu_e = self.actor(torch.cat(z, -1)).squeeze(-1)
        zn = [h] + ([pooled[b.batch]] if self.global_context else [])
        mu_n = self.node_actor(torch.cat(zn, -1)).squeeze(-1)
        if not self.node_actions:
            mu_n = mu_n[:0]
        self.last_log_std = torch.cat([self.log_std.expand(len(mu_e)), self.node_log_std.expand(len(mu_n))])
        value = None
        if with_value:
            hc, ec, pc = self.critic_enc(b)
            value = self.critic(torch.cat([pc, global_mean_pool(ec, edge_graph, size=b.num_graphs)], -1)).squeeze(-1)
        # agents = links then nodes; masking by graph id keeps [links_i, nodes_i] order
        node_graph = b.batch if self.node_actions else b.batch[:0]
        return torch.cat([mu_e, mu_n]), torch.cat([edge_graph, node_graph]), value


class MLPPolicy(nn.Module):
    """Per-topology MLP. Uses the full (fixed-size) feature vector of one graph;
    failed links are zero-filled and their outputs ignored."""

    def __init__(self, n_nodes, n_edges, hidden=256, init_log_std=-1.0, node_actions=False):
        super().__init__()
        self.n, self.E = n_nodes, n_edges
        self.n_act = n_nodes if node_actions else 0
        d = n_nodes * NODE_F + n_edges * EDGE_F
        self.body = nn.Sequential(nn.Linear(d, hidden), nn.ELU(), nn.Linear(hidden, hidden), nn.ELU())
        self.actor = nn.Linear(hidden, n_edges + self.n_act)
        nn.init.uniform_(self.actor.weight, -1e-3, 1e-3); nn.init.zeros_(self.actor.bias)
        self.critic = nn.Sequential(nn.Linear(d, hidden), nn.ELU(), nn.Linear(hidden, 1))
        self.node_log_std = nn.Parameter(torch.tensor(-2.0))
        self.log_std = nn.Parameter(torch.tensor(float(init_log_std)))

    @staticmethod
    def flat(obs, n_edges):
        ex = np.zeros((n_edges, EDGE_F), np.float32)
        ex[obs["edge_ids"]] = obs["edge_x"]
        return np.concatenate([obs["node_x"].ravel(), ex.ravel()])

    def actor_parameters(self):
        return [p for n, p in self.named_parameters() if not n.startswith("critic")]

    def critic_parameters(self):
        return [p for n, p in self.named_parameters() if n.startswith("critic")]

    def forward(self, obs_list, with_value=True):
        if obs_list[0]["node_x"].shape[0] != self.n:
            raise ValueError(f"MLP built for {self.n} nodes cannot run on a "
                             f"{obs_list[0]['node_x'].shape[0]}-node graph")
        x = torch.from_numpy(np.stack([self.flat(o, self.E) for o in obs_list]))
        z = self.body(x)
        mu_full = self.actor(z)                                     # (B, E + n)
        mus, gid = [], []
        for i, o in enumerate(obs_list):
            ids = torch.from_numpy(o["edge_ids"])
            mus += [mu_full[i, ids], mu_full[i, self.E:]]
            gid.append(torch.full((len(ids) + self.n_act,), i))
        self.last_log_std = torch.cat([torch.cat([self.log_std.expand(len(o["edge_ids"])), self.node_log_std.expand(self.n_act)])
                                       for o in obs_list])
        return torch.cat(mus), torch.cat(gid), (self.critic(x).squeeze(-1) if with_value else None)


def logprob_per_graph(mu, log_std, actions, edge_graph, num_graphs):
    dist = torch.distributions.Normal(mu, log_std.exp())
    lp = dist.log_prob(actions)
    out = torch.zeros(num_graphs).index_add_(0, edge_graph, lp)
    return out, dist


def count_params(m):
    return sum(p.numel() for p in m.parameters())
