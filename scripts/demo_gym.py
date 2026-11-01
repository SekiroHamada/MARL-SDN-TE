"""
Run the trained policies of all three methodologies through the standard
Gymnasium API (environments/gym_env.py).

    python scripts/demo_gym.py                 # all three + baselines, ~40 s
    python scripts/demo_gym.py --check         # also run gymnasium's env_checker
    python scripts/demo_gym.py --verify        # prove the wrapper == the native env
    python scripts/demo_gym.py --topology Geant2012 --load 1.1 --steps 300

Nothing is trained here. The point is to show that the same simulator the paper
uses is a conforming Gymnasium environment, and that a trained policy from any
methodology can be stepped through reset() / step() like any other agent.
"""
import os as _os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")

import argparse
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.append(os.path.join(ROOT, "methodology_2_gnn", "v2"))

W = 76


def rule(ch="-"):
    print(ch * W)


def title(t):
    print()
    rule("=")
    print(f"  {t}")
    rule("=")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topology", default="nsfnet")
    ap.add_argument("--load", type=float, default=0.95)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--check", action="store_true", help="run gymnasium's env_checker")
    ap.add_argument("--verify", action="store_true", help="compare wrapper vs native env")
    a = ap.parse_args()

    try:
        import gymnasium as gym
    except ImportError:
        print("gymnasium is not installed.  pip install gymnasium")
        return
    import torch
    torch.set_num_threads(1)
    import environments.gym_env as G

    title("A STANDARD GYMNASIUM ENVIRONMENT FOR SDN TRAFFIC ENGINEERING")
    print(f"  gymnasium {gym.__version__}   topology '{a.topology}'   load {a.load}\n")
    print("  Two registered ids, two views of the SAME simulator:\n")
    for vid, view, who in (("MARL-SDN-TE-Switch-v0", "switch", "Methodologies 1 and 3"),
                           ("MARL-SDN-TE-Link-v0", "link", "Methodology 2")):
        e = gym.make(vid, topology=a.topology, load=a.load, episode_len=a.steps).unwrapped
        print(f"    {vid:26s} view='{view}'   {who}")
        print(f"      observation_space = {e.observation_space}")
        print(f"      action_space      = {e.action_space}")
    print()

    if a.check:
        from gymnasium.utils.env_checker import check_env
        for vid in ("MARL-SDN-TE-Switch-v0", "MARL-SDN-TE-Link-v0"):
            e = gym.make(vid, topology=a.topology, load=a.load, episode_len=20).unwrapped
            check_env(e, skip_render_check=True)
            print(f"  gymnasium env_checker: {vid} PASSED")
        print()

    if a.verify:
        verify(a)
        return

    # ---------------------------------------------------------------- rollouts
    title("TRAINED POLICIES, STEPPED THROUGH reset() / step()")
    plan = [
        ("M1 IPPO (seed0)", "switch", "M1  independent learners"),
        ("M1 IQL-DQN (seed0)", "switch", "M1  independent learners"),
        ("M2 GNN zero-shot (seed0)", "link", "M2  graph policy, zero-shot"),
        ("M3 MAPPO (seed0)", "switch", "M3  centralised training"),
        ("M3 QMIX (seed0)", "switch", "M3  centralised training"),
        ("M3 MADDPG (seed0)", "switch", "M3  centralised training"),
    ]
    have = set(G.available_methods(a.topology))
    rows = []
    print(f"  {'method':30s}{'family':28s}{'policy input':14s}")
    rule()
    for name, view, fam in plan:
        if name not in have:
            print(f"  {name:30s}(checkpoint not found - skipped)")
            continue
        env = gym.make("MARL-SDN-TE-Switch-v0" if view == "switch" else "MARL-SDN-TE-Link-v0",
                       topology=a.topology, load=a.load, episode_len=a.steps).unwrapped
        pol = G.trained_policy(name, env)
        obs, _ = env.reset(seed=a.seed)
        acc = []
        while True:
            obs, r, term, trunc, info = env.step(pol(obs))
            acc.append(info)
            if term or trunc:
                break
        rows.append((name, fam, {k: float(np.mean([m[k] for m in acc])) for k in G.METRICS}))
        print(f"  {name:30s}{fam:28s}{'obs only' if pol.pure else 'env-driven':14s}")
    # OSPF / ECMP emit a split tensor under a DIFFERENT routing mode, so they are
    # not expressible as a cost residual: they are run natively, for reference.
    for name in ("OSPF", "ECMP-hop"):
        rows.insert(0, (name, "Baseline (native)", native_baseline(name, a)))
        print(f"  {name:30s}{'Baseline':28s}{'native*':14s}")
    rule()
    print("  * OSPF and ECMP select a different routing mode (single shortest path /")
    print("    equal-cost) rather than a cost residual, so they are outside this")
    print("    action space by construction and are run on the simulator directly.")

    title(f"RESULT  ({a.steps} steps, load {a.load}, identical traffic seed {a.seed})")
    print(f"  {'method':30s}{'reward':>9s}{'loss %':>9s}{'delay ms':>10s}{'MLU':>8s}")
    rule()
    best = max(rows, key=lambda r: r[2]["reward"])[0] if rows else None
    for name, fam, m in rows:
        print(f"  {name:30s}{m['reward']:+9.3f}{100 * max(0, m['loss_ratio']):9.2f}"
              f"{m['avg_delay_ms']:10.1f}{m['mlu']:8.2f}"
              + ("   <-- best" if name == best else ""))
    rule()
    print("  Every LEARNED row above was produced through the Gymnasium API alone:")
    print("    obs, _ = env.reset(seed=...)   then   obs, r, term, trunc, info = env.step(action)")
    print("  'obs only' means the policy saw nothing but the observation vector.")
    print()
    print("  Any third-party agent can therefore be trained on this environment, e.g.")
    print("      from stable_baselines3 import PPO")
    print("      model = PPO('MlpPolicy', gym.make('MARL-SDN-TE-Link-v0')).learn(200_000)")
    print("  (the paper's own results use the project's multi-agent loops, not SB3).")
    print()


def native_baseline(name, a):
    """OSPF / ECMP on the same simulator, same traffic seed (see note above)."""
    import environments.gym_env as G
    from environments.flow_splitting_env import FlowSplittingEnv
    from environments.topologies import load_named
    from baselines.te_baselines import Baseline
    env = FlowSplittingEnv(load_named(a.topology), load=a.load, seed=a.seed, scenario="pareto")
    b = Baseline(name)
    b.reset(env.sim)
    acc = []
    for _ in range(a.steps):
        _, _, info = env.sim.step(P=b.act(env.sim)["P"])
        acc.append(info)
    return {k: float(np.mean([m[k] for m in acc])) for k in G.METRICS}


def verify(a):
    """Step the wrapper and the native environment on the same seed and action
    stream, and show that they produce identical physics."""
    import gymnasium as gym
    import environments.gym_env as G
    from environments.flow_splitting_env import FlowSplittingEnv
    from environments.topologies import load_named

    title("VERIFICATION: the Gymnasium wrapper IS the environment used in the paper")
    Gr = load_named(a.topology)
    rng = np.random.default_rng(0)
    acts = [rng.uniform(-1, 1, size=(Gr.number_of_nodes(), 0)) for _ in range(0)]

    env = gym.make("MARL-SDN-TE-Switch-v0", topology=a.topology, load=a.load,
                   episode_len=60).unwrapped
    native = FlowSplittingEnv(Gr, load=a.load, seed=a.seed, scenario="pareto")
    env.reset(seed=a.seed)
    native.reset(load=a.load, seed=a.seed)
    rng = np.random.default_rng(7)
    gym_r, nat_r = [], []
    for _ in range(60):
        A = rng.uniform(-1.5, 1.5, size=(env.n, env.D))
        _, r1, _, _, i1 = env.step(A)
        _, r2, i2 = native.step_matrix(A)
        gym_r.append(i1["loss_ratio"])
        nat_r.append(i2["loss_ratio"])
    d = float(np.max(np.abs(np.array(gym_r) - np.array(nat_r))))
    print(f"  60 identical random actions, same seed:")
    print(f"    mean loss, Gymnasium wrapper : {100 * np.mean(gym_r):.4f} %")
    print(f"    mean loss, native env        : {100 * np.mean(nat_r):.4f} %")
    print(f"    max per-step difference      : {d:.2e}")
    print()
    print("  identical" if d < 1e-12 else "  DIFFERENT - investigate")
    print()


if __name__ == "__main__":
    main()
