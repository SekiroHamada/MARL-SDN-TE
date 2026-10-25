"""
Methodology 3 critics (used ONLY during training; discarded at execution).
The M1 counterpart, LocalCritics, is in methodology_1_marl/local_critic.py.

CentralValueCritic (M3, MAPPO)  : V(s) on the global state, used only in
                                  training (discarded at execution).
CentralQCritic     (M3, MADDPG) : Q(s, a_1..a_n) on global state + joint action.
"""
import torch
import torch.nn as nn



class CentralValueCritic(nn.Module):
    def __init__(self, n_agents, state_dim, hidden=256):
        super().__init__()
        self.n = n_agents
        self.net = nn.Sequential(nn.Linear(state_dim, hidden), nn.ELU(),
                                 nn.Linear(hidden, hidden), nn.ELU(), nn.Linear(hidden, 1))

    def forward(self, obs, state):                # -> (B, n): same team value for every agent
        return self.net(state).expand(-1, self.n)


class CentralQCritic(nn.Module):
    def __init__(self, state_dim, joint_act_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim + joint_act_dim, hidden), nn.ELU(),
                                 nn.Linear(hidden, hidden), nn.ELU(), nn.Linear(hidden, 1))

    def forward(self, state, joint_act):          # (B, S), (B, n*D) -> (B,)
        return self.net(torch.cat([state, joint_act], -1)).squeeze(-1)
