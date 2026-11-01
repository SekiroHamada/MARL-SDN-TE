"""
Methodology 1 critic: one value network PER SWITCH on its LOCAL observation.

V_v(o_v) estimates the team return from what switch v alone can see. Other
switches' changing policies are invisible to it, so its regression target
drifts as they learn - the non-stationarity Methodology 3's central critic
is designed to remove.
"""
import torch.nn as nn

from models.decentralized_forwarder import GroupedMLP


class LocalCritics(nn.Module):
    def __init__(self, n_agents, obs_dim, hidden=64):
        super().__init__()
        self.net = GroupedMLP(n_agents, obs_dim, hidden, 1)   # independent weights per switch

    def forward(self, obs, state=None):           # (B, n, obs) -> (B, n); global state unused
        return self.net(obs).squeeze(-1)
