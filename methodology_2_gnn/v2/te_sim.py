"""Moved to environments/te_sim.py (shared by all three methodologies). Kept as a shim."""
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
if _ROOT not in _sys.path:
    _sys.path.append(_ROOT)
from environments.te_sim import *  # noqa: F401,F403
from environments.te_sim import TESimulator, TrafficProcess, PKT_BITS, EPS  # noqa: F401
