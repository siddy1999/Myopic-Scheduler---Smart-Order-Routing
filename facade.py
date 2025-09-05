#!/usr/bin/env python3
"""
High-level orchestration layer for SOR optimization:
- `run_optimization_facade` prepares data and calls the optimizer.
- `execute_optimization` adds lookup-table mode.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

from data_io import get_spread_at_time
from market_metrics import (
    calculate_queue_depth,
    calculate_outflow_fills_all_intervals,
    estimate_adverse_selection,
)
from optimisation import (
    StochasticOrderOptimizer,
    InsufficientDataError,
)

__all__: list[str] = ["run_optimization_facade", "execute_optimization"]


def run_optimization_facade(
    dfs: Dict[str, pd.DataFrame],
    target_time,  # str | pd.Timestamp
    *,
    S: float,
    T: int,
    f: float,
    r: List[float],
    lambda_u: float,
    lambda_o: float,
    N: int = 1000,
    method: str = "exp",
    side="B",
) -> Tuple[np.ndarray, float]:
    """Prepare inputs then call `StochasticOrderOptimizer.optimize`."""
    if not dfs:
        raise ValueError("dfs cannot be empty")

    target_time = pd.Timestamp(target_time)

    # clip each DF to history <= target_time
    valid = {v: df[df["timestamp"] <= target_time] for v, df in dfs.items() if not df.empty}
    if not valid:
        return None, None

    # adjust target_time to latest available across venues
    target_time = max(df["timestamp"].max() for df in valid.values())

    # queue depth vector Q
    Q_map = calculate_queue_depth(valid, target_time, side)
    Q = np.array([Q_map.get(v, 0.0) for v in valid])

    # half-spread estimate h
    spreads = [get_spread_at_time(df, target_time) for df in valid.values()]
    h = float(np.mean(spreads) / 2 if spreads else 0.0)

    # outflow history
    outflows = calculate_outflow_fills_all_intervals(valid, side)
    for v in valid:
        if v not in outflows or outflows[v].empty:
            outflows[v] = pd.DataFrame({"outflow_fill": [0.0]})

    # adverse selection
    adv_sel = estimate_adverse_selection(valid, 10, target_time, side)
    rebates = np.asarray(r, dtype=float) + np.array([adv_sel.get(v, 0.0) for v in valid])

    params = {
        "S": S,
        "T": T,
        "h": h,
        "f": f,
        "r": rebates,
        "theta": f * 0.1,
        "lambda_u": lambda_u,
        "lambda_o": lambda_o,
        "Q": Q,
        "outflows": outflows,
        "s": +1 if side.upper() == "B" else -1,
    }

    opt = StochasticOrderOptimizer(params, len(valid))
    return opt.optimize(N=N, method=method)


def execute_optimization(
    dfs: Dict[str, pd.DataFrame],
    target_time,  # str | pd.Timestamp
    *,
    S: float,
    T: int,
    f: float,
    r: List[float],
    lambda_u: float,
    lambda_o: float,
    N: int = 1000,
    method: str = "exp",  # or "kde" or "lookup"
    stock: str | None = None,
    side="B",
):
    """Main API exposed to end-users / entry-point scripts."""
    if method != "lookup":
        return run_optimization_facade(
            dfs,
            target_time,
            S=S,
            T=T,
            f=f,
            r=r,
            lambda_u=lambda_u,
            lambda_o=lambda_o,
            N=N,
            method=method,
            side=side,
        )

    if stock is None:
        raise ValueError("stock symbol required in lookup mode")
    fn = f"ratio_table_{stock}.pkl"
    if not os.path.exists(fn):
        raise FileNotFoundError(fn)

    from lookup import load_ratio_table, get_queue_volume_for_minute, get_bucket

    table = load_ratio_table(fn)
    target_time = pd.Timestamp(target_time)
    minute_before = target_time.floor("min")

    # build composite key
    key_parts, sample_th = [], None
    for entry in table.values():
        if entry.get("thresholds"):
            sample_th = entry["thresholds"]
            break
    if sample_th is None:
        raise ValueError("No thresholds in ratio table")

    for venue, df in dfs.items():
        qv = get_queue_volume_for_minute(df, minute_before)
        if qv is None:
            raise ValueError(f"No data for {venue} at {minute_before}")
        th = sample_th[venue]
        qb = get_bucket(
            qv["queue"],
            (th["queue_thresholds"]["low_to_medium"], th["queue_thresholds"]["medium_to_high"]),
        )
        vb = get_bucket(
            qv["volume"],
            (th["volume_thresholds"]["low_to_medium"], th["volume_thresholds"]["medium_to_high"]),
        )
        key_parts.append(f"{venue}:{qb.lower()}_{vb.lower()}")
    comp_key = "|".join(key_parts)

    if comp_key not in table or table[comp_key].get("avg_ratio") is None:
        raise ValueError(f"Composite key {comp_key} not found in ratio table")

    ratio = table[comp_key]["avg_ratio"]
    return ratio * S, None
