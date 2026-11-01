"""
Methodology 1 (value-based variant): independent DQN per switch (IQL).

Each switch has its own Q-network Q_v(o_v, a_v) over the discrete action set in
models/marl_value.py and regresses the TEAM return on its LOCAL observation,
treating every other switch as part of the environment. Its QMIX counterpart
(Methodology 3) uses the same agent networks but trains them through a
monotonic mixer that sees the global state.
"""
from models.marl_value import train_value, QAgents, DiscreteActions, greedy_fn  # noqa: F401


def train_iql(G, out_dir, tag="iql", **over):
    return train_value(G, out_dir, tag, make_mixer=None, **over)
