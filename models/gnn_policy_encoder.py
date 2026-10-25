"""Methodology 2 GNN encoder (GINE message passing with edge-state updates).
Implementation lives in methodology_2_gnn/v2/policies.py; re-exported here to
match the Sprint-0 file layout."""
import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "methodology_2_gnn", "v2"))
from policies import GraphEncoder, GNNPolicy  # noqa: E402,F401
