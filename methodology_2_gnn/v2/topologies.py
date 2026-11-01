"""Moved to environments/topologies.py (shared by all three methodologies). Kept as a shim."""
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
if _ROOT not in _sys.path:
    _sys.path.append(_ROOT)
from environments.topologies import *  # noqa: F401,F403
from environments.topologies import ZOO_DIR, REPO_ROOT, PRIMARY_TEST, load_named, load_any, load_topology_zoo, synthetic_pool, synthetic_graph, exists_exact  # noqa: F401
