"""Moved to baselines/te_baselines.py (shared by all three methodologies). Kept as a shim.
Loaded by file path because this module's name shadows the top-level `baselines` package."""
import importlib.util as _u, os as _os
_p = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))), "baselines", "te_baselines.py")
_spec = _u.spec_from_file_location("te_baselines_shared", _p)
_m = _u.module_from_spec(_spec); _spec.loader.exec_module(_m)
Baseline, BASELINES, lp_routing = _m.Baseline, _m.BASELINES, _m.lp_routing
