# lookup.py
"""Generate and load per‑minute queue/volume ratio lookup tables.

This module relies on *market_metrics* for queue/volume features and performs
multiple optimisation runs (via `run_optimization_facade`) to learn an average
allocation ratio for each composite market state.

It is imported only by the high‑level facade; **no other sub‑module depends on
it**, so there is no circular‑import risk.  To avoid one, the optimisation
facade is imported lazily *inside* `generate_lookup_table`.
"""
from __future__ import annotations

import os
import pickle
from typing import Dict, List

import numpy as np
import pandas as pd

from .market_metrics import (
    compute_queue_and_volume_size,
    get_queue_volume_for_minute,
    get_bucket,
)

__all__: list[str] = ["generate_lookup_table", "load_ratio_table"]

# ---------------------------------------------------------------------------
# Generate lookup table
# ---------------------------------------------------------------------------

def generate_lookup_table(
    dfs: Dict[str, pd.DataFrame],
    target_time: pd.Timestamp,
    *,
    S: float,
    T: int,
    f: float,
    r: List[float],
    lambda_u: float,
    lambda_o: float,
    N: int,
    method: str,
    stock: str,
):
    """Build and persist a lookup table mapping *composite bucket keys* → avg ratio.

    This is essentially Cell 3 of your notebook made into a function.
    """
    # (1) Per‑venue threshold calc ------------------------------------------------
    thresholds: Dict[str, Dict[str, Dict[str, float]]] = {}
    for venue, df in dfs.items():
        qv = compute_queue_and_volume_size({venue: df})[venue]
        thresholds[venue] = {
            "queue_thresholds": {
                "low_to_medium":  qv["queue" ].quantile(0.33),
                "medium_to_high": qv["queue" ].quantile(0.66),
            },
            "volume_thresholds": {
                "low_to_medium":  qv["volume"].quantile(0.33),
                "medium_to_high": qv["volume"].quantile(0.66),
            },
        }

    # (2) Union of minutes -------------------------------------------------------
    all_minutes = sorted({m for df in dfs.values() for m in df["minute"].unique()})

    def composite_key(minute):
        parts = []
        for v, df in dfs.items():
            qv = get_queue_volume_for_minute(df, minute)
            if qv is None:
                parts.append(f"{v}:NA")
                continue
            th = thresholds[v]
            q_bucket = get_bucket(qv["queue"],  (th["queue_thresholds"]["low_to_medium"],  th["queue_thresholds"]["medium_to_high"]))
            v_bucket = get_bucket(qv["volume"], (th["volume_thresholds"]["low_to_medium"], th["volume_thresholds"]["medium_to_high"]))
            parts.append(f"{v}:{q_bucket.lower()}_{v_bucket.lower()}")
        return "|".join(parts)

    keys = [composite_key(m) for m in all_minutes]

    # (3) Group consecutive minutes with same key -------------------------------
    segments: list[tuple[str, list[pd.Timestamp]]] = []
    current_key, seg = None, []
    for m, k in zip(all_minutes, keys):
        if current_key is None or k == current_key and (not seg or m - seg[-1] <= pd.Timedelta("1min")):
            current_key = k
            seg.append(m)
        else:
            segments.append((current_key, seg))
            current_key, seg = k, [m]
    if seg:
        segments.append((current_key, seg))

    # (4) Optimise each segment --------------------------------------------------
    from .facade import run_optimization_facade  # lazy import to avoid cycles

    seg_results: dict[str, list[np.ndarray]] = {}
    for key, minutes in segments:
        dseg = {v: df[df["minute"].isin(minutes)] for v, df in dfs.items()}
        if all(d.empty for d in dseg.values()):
            continue
        sol, _ = run_optimization_facade(
            dseg,
            target_time,
            S=S,
            T=T,
            f=f,
            r=r,
            lambda_u=lambda_u,
            lambda_o=lambda_o,
            N=N,
            method=method,
        )
        if sol is None:
            continue
        seg_results.setdefault(key, []).append(sol / sol.sum() if sol.sum() else sol)

    # (5) Average ratios per key --------------------------------------------------
    table = {
        k: {
            "avg_ratio": np.mean(np.vstack(v), axis=0) if v else None,
            "thresholds": thresholds,
        }
        for k, v in seg_results.items()
    }

    # (6) Persist to pickle -------------------------------------------------------
    fn = f"ratio_table_{stock}.pkl"
    with open(fn, "wb") as fh:
        pickle.dump(table, fh)
    print(f"Saved lookup table → {fn}")
    return table

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_ratio_table(path: str):
    with open(path, "rb") as fh:
        table = pickle.load(fh)
    return table
