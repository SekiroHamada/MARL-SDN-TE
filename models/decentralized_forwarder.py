"""
Decentralised forwarding policies (actors) for Methodologies 1 and 3.

Every switch has its OWN parameters (no sharing): this is what makes the
learners "independent" and exposes the non-stationarity problem. For speed the
n per-agent MLPs are evaluated in one batched einsum (GroupedLinear).

At execution time a switch needs only its local observation o_v and its own
weights - no controller round-trip.
"""
import math
import torch
import torch.nn as nn


class GroupedLinear(nn.Module):
    """n independent Linear(in, out) layers applied in parallel: x (B, n, in) -> (B, n, out)."""

    def __init__(self, n, i, o):
        super().__init__()
        self.w = nn.Parameter(torch.empty(n, i, o))
        self.b = nn.Parameter(torch.zeros(n, o))
        bound = 1 / math.sqrt(i)
        nn.init.uniform_(self.w, -bound, bound)

    def forward(self, x):
        return torch.einsum("bni,nio->bno", x, self.w) + self.b


class GroupedMLP(nn.Module):
    def __init__(self, n, i, h, o, small_last=False):
        super().__init__()
        self.l1, self.l2, self.l3 = GroupedLinear(n, i, h), GroupedLinear(n, h, h), GroupedLinear(n, h, o)
        if small_last:
            nn.init.uniform_(self.l3.w, -1e-3, 1e-3)

    def forward(self, x):
        return self.l3(nn.functional.elu(self.l2(nn.functional.elu(self.l1(x)))))   # ELU: no dead units


class DecentralizedForwarder(nn.Module):
    """Gaussian per-switch policy over its D outgoing-link cost residuals.
    `deterministic=True` variant (MADDPG) bounds outputs to [-3, 3] with tanh."""

    def __init__(self, n_agents, obs_dim, act_dim, hidden=64, init_log_std=-1.0, deterministic=False):
        super().__init__()
        self.net = GroupedMLP(n_agents, obs_dim, hidden, act_dim, small_last=True)
        self.log_std = nn.Parameter(torch.full((n_agents, act_dim), float(init_log_std)))
        self.deterministic = deterministic

    def forward(self, obs):                       # obs (B, n, obs_dim)
        mu = self.net(obs)
        return 3.0 * torch.tanh(mu / 3.0) if self.deterministic else mu

    def dist(self, obs):
        return torch.distributions.Normal(self(obs), self.log_std.exp())


def count_params(m):
    return sum(p.numel() for p in m.parameters())
