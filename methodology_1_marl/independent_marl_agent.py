"""
Methodology 1: Decentralised Independent MARL (IPPO) with multi-path flow splitting.

Each switch v is an independent learner:
  * its own Gaussian actor pi_v(a_v | o_v)          (models/decentralized_forwarder.py)
  * its own critic V_v(o_v) on LOCAL observation     (methodology_1_marl/local_critic.py)
  * no parameter sharing, no communication beyond 1-hop telemetry.
All agents receive the team reward R = alpha*Thr - beta*Delay - gamma*Loss
(team weights from configs/shared_env_config.py), differenced against a
counterfactual OSPF network on the same traffic for variance reduction.

Because every agent treats the others as part of its environment, the
transition and reward each one sees drift as the others learn - the
non-stationarity that Methodology 3's centralised critic is meant to remove.
"""
from models.marl_onpolicy import train as _train, evaluate, DEFAULTS  # noqa: F401
from methodology_1_marl.local_critic import LocalCritics


def make_local_critic(env, cfg):
    return LocalCritics(env.n, env.obs_dim, cfg["hidden"])


def train_ippo(G, out_dir, tag="ippo", **over):
    return _train(G, make_local_critic, out_dir, tag, **over)
