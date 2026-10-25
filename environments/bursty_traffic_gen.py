"""
Bursty traffic generation (Methodology 1 deliverable, shared by all methodologies).

The traffic process itself lives in environments/te_sim.py (TrafficProcess) so
that every methodology sees byte-identical demand for the same seed. This module
exposes it with the API planned in Sprint 0 plus named scenario presets.

Model
-----
demand[s,t](k) = mean[s,t] * ((1-b) + b * ON[s,t](k)/p_on) * HOT[s](k) * noise
  * mean        : gravity matrix (lognormal node masses), rescaled per episode so
                  OSPF on the mean matrix has max-link-utilisation == `load`
  * (1-b)       : smooth Poisson-like background
  * ON[s,t]     : per-flow Pareto ON/OFF (shape 1.5 => infinite variance,
                  self-similar aggregate)
  * HOT[s]      : correlated source hot-spots (Pareto ON/OFF, low duty cycle):
                  a source's whole row surges together - this is what actually
                  overloads links; independent per-flow bursts average out.
"""
import numpy as np

try:
    from environments.te_sim import TrafficProcess
except ImportError:                       # executed from inside environments/
    from te_sim import TrafficProcess

SCENARIOS = {
    # name: TrafficProcess kwargs
    "poisson":    dict(burst_share=0.0, hotspot_gain=0.0, noise=0.10),
    "pareto":     dict(burst_share=0.5, hotspot_gain=2.5, hotspot_duty=0.15),   # default
    "microburst": dict(burst_share=0.7, hotspot_gain=5.0, hotspot_duty=0.08,
                       pareto_shape=1.2, max_period=20),                         # stress test
}


class BurstyTrafficGenerator:
    """Stand-alone generator: yields demand matrices (Mbps) step by step."""

    def __init__(self, n_nodes, scenario="pareto", seed=0, scale_mbps=1.0, **overrides):
        kw = dict(SCENARIOS[scenario]); kw.update(overrides)
        self.rng = np.random.default_rng(seed)
        self.proc = TrafficProcess(n_nodes, self.rng, **kw)
        self.proc.scale = scale_mbps

    def next(self):
        return self.proc.sample()

    def generate_poisson_burst(self, steps):
        """Smooth background only (no ON/OFF, no hot-spots)."""
        g = BurstyTrafficGenerator(self.proc.n, "poisson", int(self.rng.integers(1 << 31)), self.proc.scale)
        g.proc.base_shape = self.proc.base_shape
        return np.stack([g.next() for _ in range(steps)])

    def generate_pareto_burst(self, steps):
        """Heavy-tailed bursty sequence (this generator's scenario)."""
        return np.stack([self.next() for _ in range(steps)])


def burstiness_report(seq):
    """Peak-to-mean and index of dispersion of aggregate demand."""
    agg = seq.sum(axis=(1, 2))
    return dict(peak_to_mean=float(agg.max() / agg.mean()), cv=float(agg.std() / agg.mean()),
                idc=float(agg.var() / agg.mean()))
