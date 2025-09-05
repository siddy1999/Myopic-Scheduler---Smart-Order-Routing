# sor/__init__.py
"""Top‑level API for the Smart‑Order‑Routing package.

After `pip install -e .` (or adding the *sor* folder to PYTHONPATH) users can:

    from sor import execute_optimization, generate_lookup_table, fetch_live_market_data

without drilling into sub‑modules.  Only *stable* high‑level functions are
re‑exported here; the rest of the internal surface may change.
"""
from importlib import import_module
from types import ModuleType
from typing import Any, TYPE_CHECKING

# Helper to lazily import heavy sub‑modules (prevents Kafka import cost when
# users only need CSV optimisation).


def _lazy(name: str) -> ModuleType:
    return import_module(f"sor.{name}")

# Core façade ---------------------------------------------------------------
_facade = _lazy("facade")
execute_optimization = _facade.execute_optimization  # type: ignore[attr-defined]
run_optimization_facade = _facade.run_optimization_facade  # type: ignore[attr-defined]

# Lookup helpers ------------------------------------------------------------
_lookup = _lazy("lookup")
generate_lookup_table = _lookup.generate_lookup_table  # type: ignore[attr-defined]
load_ratio_table = _lookup.load_ratio_table  # type: ignore[attr-defined]

# Live‑data fetch -----------------------------------------------------------
_live = _lazy("live_data")
fetch_live_market_data = _live.fetch_live_market_data  # type: ignore[attr-defined]

# Explicit re‑export list ----------------------------------------------------
__all__: list[str] = [
    "execute_optimization",
    "run_optimization_facade",
    "generate_lookup_table",
    "load_ratio_table",
    "fetch_live_market_data",
]

# Optional type‑checking stubs ---------------------------------------------
if TYPE_CHECKING:
    from numpy import ndarray  # noqa: F401
