"""
Methodology 3: QMIX monotonic mixing network (Rashid et al., ICML 2018).

Q_tot(s, a) = f_s(Q_1(o_1, a_1), ..., Q_n(o_n, a_n)) with dQ_tot/dQ_i >= 0, so each
switch's greedy action on its own Q_i is also greedy for Q_tot: training is
centralised (the mixer sees the global state s), execution is decentralised
(each switch argmaxes its own Q_i; the mixer is discarded).
Hypernetworks produce state-dependent, non-negative mixing weights.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class QMixer(nn.Module):
    def __init__(self, n_agents, state_dim, embed=32, hyper_hidden=64):
        super().__init__()
        self.n, self.e = n_agents, embed
        self.hyper_w1 = nn.Sequential(nn.Linear(state_dim, hyper_hidden), nn.ELU(), nn.Linear(hyper_hidden, n_agents * embed))
        self.hyper_b1 = nn.Linear(state_dim, embed)
        self.hyper_w2 = nn.Sequential(nn.Linear(state_dim, hyper_hidden), nn.ELU(), nn.Linear(hyper_hidden, embed))
        self.hyper_b2 = nn.Sequential(nn.Linear(state_dim, embed), nn.ELU(), nn.Linear(embed, 1))

    def forward(self, q, s):                      # q (B, n), s (B, S) -> (B,)
        w1 = torch.abs(self.hyper_w1(s)).view(-1, self.n, self.e)
        h = F.elu(torch.bmm(q.unsqueeze(1), w1).squeeze(1) + self.hyper_b1(s))
        w2 = torch.abs(self.hyper_w2(s))
        return (h * w2).sum(-1) + self.hyper_b2(s).squeeze(-1)
