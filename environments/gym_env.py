"""
Gymnasium wrapper around the shared traffic-engineering simulator.

The project's own training loops talk to `FlowSplittingEnv` / `TESimulator`
directly (they need the multi-agent observation matrix and the global state for
the CTDE critics). This module exposes the SAME simulator through the standard
Gymnasium API, so that:

  * the environment can be inspected with the usual tooling
    (`gymnasium.utils.env_checker.check_env`),
  * a trained policy from any of the three methodologies can be stepped through
    `reset()` / `step()` like any other Gymnasium environment,
  * a third-party agent (Stable-Baselines3, CleanRL, ...) can be pointed at the
    same physics without touching the project's code.

Two views of the same environment:

    view="switch"  (Methodologies 1 and 3)
        observation : (n_switches, obs_dim)  local 1-hop observations
        action      : (n_switches, max_degree) cost residual per outgoing link
    view="link"    (Methodology 2)
        observation : flat vector of all node and link features
        action      : (n_links,) cost residual per directed link

Register and use:

    import gymnasium as gym
    import environments.gym_env                      # registers the ids
    env = gym.make("MARL-SDN-TE-Switch-v0", topology="nsfnet", load=0.95)

or directly:

    from environments.gym_env import TEGymEnv
    env = TEGymEnv(topology="nsfnet", view="link", load=0.95)
"""
import os as _os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")

import os
import sys

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as e:                                    # pragma: no cover
    raise ImportError(
        "gymnasium is not installed in this environment.\n"
        "    pip install gymnasium\n"
        "(the project's own training does not need it; this wrapper does)"
    ) from e

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from environments.flow_splitting_env import FlowSplittingEnv     # noqa: E402
from environments.topologies import load_named                   # noqa: E402

ACT_BOUND = 3.0          # the simulator clips cost residuals to [-3, 3]
METRICS = ("reward", "throughput_mbps", "offered_mbps", "loss_ratio",
           "avg_delay_ms", "mlu", "util_std", "jain")


class TEGymEnv(gym.Env):
    """Gymnasium view of the flow-splitting traffic-engineering problem."""

    metadata = {"render_modes": ["ansi"]}

    def __init__(self, topology="nsfnet", view="switch", load=0.95,
                 scenario="pareto", episode_len=200, seed=101, render_mode=None):
        super().__init__()
        if view not in ("switch", "link"):
            raise ValueError("view must be 'switch' or 'link'")
        self.topology, self.view, self.scenario = topology, view, scenario
        self.load, self.episode_len, self.render_mode = load, episode_len, render_mode
        self._seed = seed
        self.G = load_named(topology)
        self.env = FlowSplittingEnv(self.G, load=load, seed=seed, scenario=scenario)
        self.sim = self.env.sim
        self.n, self.E, self.D = self.env.n, self.env.E, self.env.D
        self.t = 0

        if view == "switch":
            self.observation_space = spaces.Box(-np.inf, np.inf,
                                                shape=(self.n, self.env.obs_dim), dtype=np.float32)
            self.action_space = spaces.Box(-ACT_BOUND, ACT_BOUND,
                                           shape=(self.n, self.D), dtype=np.float32)
        else:
            self.observation_space = spaces.Box(-np.inf, np.inf,
                                                shape=(self.env.state_dim,), dtype=np.float32)
            self.action_space = spaces.Box(-ACT_BOUND, ACT_BOUND,
                                           shape=(self.E,), dtype=np.float32)

    # ------------------------------------------------------------------ API
    def _obs(self):
        if self.view == "switch":
            return self.env.obs_matrix().astype(np.float32)
        _, state = self.env.observe_all()
        return state.astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        load = options.get("load", self.load)
        s = seed if seed is not None else self._seed
        self.env.reset(load=load, seed=int(s))
        self.sim = self.env.sim
        self.t = 0
        return self._obs(), {}

    def step(self, action):
        a = np.asarray(action, dtype=np.float64)
        if self.view == "switch":
            _, reward, info = self.env.step_matrix(a.reshape(self.n, self.D))
        else:
            cost = self.sim.costs_from_action(np.clip(a.reshape(self.E), -ACT_BOUND, ACT_BOUND))
            _, reward, info = self.sim.step(cost, "softmin")
        self.t += 1
        truncated = self.t >= self.episode_len
        info = {k: float(info[k]) for k in METRICS if k in info}
        return self._obs(), float(reward), False, truncated, info

    def render(self):
        if self.render_mode != "ansi":
            return None
        i = self.sim.last_info if hasattr(self.sim, "last_info") else {}
        return (f"t={self.t:4d}  mlu={i.get('mlu', float('nan')):.2f}  "
                f"loss={100 * i.get('loss_ratio', 0):.2f}%")

    def close(self):
        pass


# ---------------------------------------------------------------- registration
for _vid, _view in (("MARL-SDN-TE-Switch-v0", "switch"), ("MARL-SDN-TE-Link-v0", "link")):
    try:
        gym.register(id=_vid, entry_point="environments.gym_env:TEGymEnv",
                     kwargs={"view": _view}, max_episode_steps=None)
    except Exception:                                      # already registered
        pass


# ---------------------------------------------------------------- policies
def trained_policy(name, gym_env):
    """Return act(obs) -> action for a trained checkpoint, driven through the
    Gymnasium observation where the architecture allows it.

    `name` is any method listed by evaluation.unified_benchmark.methods_for,
    e.g. 'M1 IPPO (seed0)', 'M2 GNN zero-shot (seed0)', 'M3 MAPPO (seed0)'.
    """
    import torch
    import evaluation.unified_benchmark as UB
    methods = {m.name: m for m in UB.methods_for(gym_env.topology, gym_env.G)}
    if name not in methods:
        raise KeyError(f"{name!r} not found. Available:\n  " + "\n  ".join(sorted(methods)))
    m = methods[name]

    sim, env = gym_env.sim, gym_env.env
    n, E = gym_env.n, gym_env.E

    # --- M1 / M3 continuous actors: observation -> action, nothing else needed
    if gym_env.view == "switch" and hasattr(m, "actor"):
        def act(obs):
            with torch.no_grad():
                return m.actor(torch.from_numpy(np.asarray(obs, np.float32))[None])[0].numpy()
        act.pure, act.method = True, m
        return act

    # --- M2 graph policy: the flat link-view observation is exactly
    #     [node_x (n x 5) | edge_x (E x 6)], so the graph input is rebuilt from
    #     it plus the (static) edge index. The policy still sees only the obs.
    if gym_env.view == "link" and hasattr(m, "model"):
        def act(obs):
            o = np.asarray(obs, np.float32)
            node_x = o[:n * 5].reshape(n, 5)
            edge_full = o[n * 5:n * 5 + E * 6].reshape(E, 6)
            idx = np.nonzero(sim.alive)[0]
            graph_obs = dict(node_x=node_x, edge_x=edge_full[idx],
                             edge_index=np.stack([sim.src[idx], sim.dst[idx]]).astype(np.int64),
                             edge_ids=idx)
            with torch.no_grad():
                mu, _, _ = m.model([graph_obs], with_value=False)
            mu = mu.numpy().ravel()
            a = np.zeros(E, np.float32)
            a[idx] = mu[:len(idx)]            # link residuals; node temperatures unused here
            return a
        act.pure, act.method = True, m
        return act

    # --- baselines and discrete greedy-Q policies (IQL / QMIX / tabular) act on
    #     the simulator's own structures; they are driven through the wrapped env
    #     and their effect is applied directly, so the returned action is the
    #     equivalent residual vector where one exists.
    def act(obs):
        a = m.act(env)
        if "A" in a and gym_env.view == "switch":
            return np.asarray(a["A"], np.float32)
        if "cost" in a and gym_env.view == "link":
            c = np.asarray(a["cost"], np.float64)
            base = sim.prop / sim.mean_prop
            with np.errstate(divide="ignore", invalid="ignore"):
                res = np.log(np.where(base > 0, c / np.maximum(base, 1e-12), 1.0))
            return np.clip(np.nan_to_num(res), -ACT_BOUND, ACT_BOUND).astype(np.float32)
        raise RuntimeError(f"{name!r} cannot be expressed in view={gym_env.view!r}; "
                           f"use view='switch' for M1/M3 and view='link' for M2/baselines")
    act.pure, act.method = False, m
    return act


def available_methods(topology="nsfnet"):
    import evaluation.unified_benchmark as UB
    G = load_named(topology)
    return sorted(m.name for m in UB.methods_for(topology, G))
