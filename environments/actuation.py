"""
Forwarding-rule churn, measured the way the Mininet installer
(mininet_validation/run_mininet.py) decides to re-send an OpenFlow select group:
a (switch, destination) group is re-installed when its set of next hops changes
or any bucket weight moves by more than 2%.
"""
import numpy as np


def groups_changed(P_old, P_new, tol=0.02):
    """Number of (switch, destination) groups that differ between split tensors
    P[t, v, u] (destination t, switch v, next hop u). 0 for the first install."""
    if P_old is None:
        return 0
    n = P_new.shape[0]
    on_o, on_n = P_old > 1e-6, P_new > 1e-6
    diff = (on_o != on_n).any(-1) | (np.abs(P_old - P_new).max(-1) > tol)
    diff[np.arange(n), np.arange(n)] = False              # switch t keeps no group for t
    return int(diff.sum())


def churn_fraction(P_old, P_new, tol=0.02):
    """Share of all n(n-1) groups that would be re-installed (0..1)."""
    n = P_new.shape[0]
    return groups_changed(P_old, P_new, tol) / max(n * (n - 1), 1)


def deadband(P_inst, P_new, delta):
    """Installer-side hysteresis. A destination's forwarding rules are replaced only
    if some group for it would gain/lose a next hop or move a weight by more than
    `delta`; otherwise the installed rules for that destination are kept. Updating
    whole destinations keeps every per-destination forwarding graph loop-free."""
    if P_inst is None or delta <= 0:
        return P_new
    on_o, on_n = P_inst > 1e-6, P_new > 1e-6
    big = ((on_o != on_n).any(-1) | (np.abs(P_inst - P_new).max(-1) > delta)).any(-1)   # per destination
    return np.where(big[:, None, None], P_new, P_inst)
