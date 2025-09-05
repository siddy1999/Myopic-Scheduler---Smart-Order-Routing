# data_io.py
"""Utilities for loading raw MBP‑10 data and lightweight price‑time helpers.
This module contains **no trading logic**; it's purely I/O + simple spreads.
It is imported by downstream modules (market_metrics, optimisation, facade).
"""
from __future__ import annotations
import os
from typing import Dict
import numpy as np
import pandas as pd
__all__: list[str] = [
    "preprocess_data",
    "get_timestamp_values",
    "get_spread_at_time",
]
# ─────────────────────────────────────────────────────────────────────────────
# CSV PRE‑PROCESSOR
# ─────────────────────────────────────────────────────────────────────────────
def preprocess_data(file_path: str | os.PathLike) -> pd.DataFrame:
    """Read a *single‑venue* MBP‑10 CSV dump and return a clean DataFrame.
    The function validates required columns, converts timestamps to UTC,
    renames bid/ask columns to *best_bid* / *best_ask*, and adds helper
    columns (*timestamp*, *minute*).  The resulting DataFrame is indexed by
    **ts_event** (tz‑aware) and sorted chronologically.
    """
    # ── basic file‑type sanity check ────────────────────────────────────────
    if isinstance(file_path, str):
        _, ext = os.path.splitext(file_path)
        if ext.lower() != ".csv":
            raise AttributeError("Unsupported file format; expected .csv")
    # ── attempt UTF‑8 → UTF‑16 fallback read ───────────────────────────────
    try:
        try:
            df = pd.read_csv(file_path, on_bad_lines="error", encoding="utf‑8")
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, on_bad_lines="error", encoding="utf‑16")
    except pd.errors.ParserError as e:
        raise pd.errors.ParserError(f"Error parsing CSV: {e}") from None
    # ── required columns present? ─────────────────────────────────────────-
    required = {"ts_event", "bid_px_00", "ask_px_00"}
    if missing := required - set(df.columns):
        raise KeyError(f"Missing required columns: {missing}")
    # ── convert & validate types ───────────────────────────────────────────
    df["ts_event"] = pd.to_datetime(df["ts_event"], utc=True)
    df["bid_px_00"] = pd.to_numeric(df["bid_px_00"])
    df["ask_px_00"] = pd.to_numeric(df["ask_px_00"])
    # ── sort and add helpers ───────────────────────────────────────────────
    df = df.sort_values("ts_event").reset_index(drop=True)
    df["timestamp"] = df["ts_event"]
    df["minute"] = df["timestamp"].dt.floor("1min")
    # df = df.rename(columns={"bid_px_00": "best_bid", "ask_px_00": "best_ask"})
    df = df.rename(columns={
        "bid_px_00": "best_bid",
        "ask_px_00": "best_ask",
        "bid_sz_00": "bid_sz_00",      # keep as-is if present
        "ask_sz_00": "ask_sz_00",      # optional
    })
    df = df.set_index("ts_event")
    return df
# ─────────────────────────────────────────────────────────────────────────────
# PRICE / TIME HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _as_timestamp(t) -> pd.Timestamp:
    """Coerce *t* to a timezone‑aware pandas Timestamp (UTC if naive)."""
    ts = pd.Timestamp(t)
    return ts if ts.tzinfo else ts.tz_localize("UTC")
# def get_timestamp_values(df: pd.DataFrame, target_time) -> Dict[str, float]:
#     """Return best bid/ask (and derived spread) at the row closest to *target_time*."""
#     if df.empty:
#         return {"best_bid": 0.0, "best_ask": 0.0, "spread": 0.0, "half_spread": 0.0}
#     if not isinstance(df.index, pd.DatetimeIndex):
#         raise TypeError("DataFrame index must be DatetimeIndex")
#     target_time = _as_timestamp(target_time).tz_convert(df.index.tz)
#     #idx_closest = np.abs((df.index - target_time)).idxmin()
#     diff = np.abs(df.index - target_time)
#     pos = diff.argmin()
#     idx_closest = df.index[pos]
#     row = df.loc[idx_closest]
#     if isinstance(row, pd.DataFrame):
#         row = row.iloc[0]
#     print(row)
#     bid, ask = row.get("best_bid", np.nan), row.get("best_ask", np.nan)
#     spread = ask - bid if np.isfinite([bid, ask]).all() else np.nan
#     return {
#         "closest_timestamp": idx_closest,
#         "best_bid": bid,
#         "best_ask": ask,
#         "spread": spread,
#         "half_spread": spread / 2 if np.isfinite(spread) else np.nan,
#     }



def get_timestamp_values(df: pd.DataFrame, target_time) -> Dict[str, float]:
    """Return best bid/ask (and derived spread) at the row closest to *target_time*."""
    print("========== DEBUGGING GET_TIMESTAMP_VALUES ==========")
    print(f"Target time: {target_time}")
    
    if df.empty:
        print("DataFrame is empty, returning default values")
        return {"best_bid": 0.0, "best_ask": 0.0, "spread": 0.0, "half_spread": 0.0}
    
    print(f"DataFrame shape: {df.shape}")
    print(f"DataFrame columns: {df.columns.tolist()}")
    print(f"DataFrame index type: {type(df.index)}")
    
    if not isinstance(df.index, pd.DatetimeIndex):
        print("ERROR: DataFrame index is not DatetimeIndex")
        raise TypeError("DataFrame index must be DatetimeIndex")
    
    try:
        target_time = _as_timestamp(target_time).tz_convert(df.index.tz)
        print(f"Converted target time: {target_time}")
    except Exception as e:
        print(f"Error converting target time: {e}")
        # Try without timezone conversion
        target_time = _as_timestamp(target_time)
        print(f"Fallback target time: {target_time}")
    
    diff = np.abs(df.index - target_time)
    pos = diff.argmin()
    idx_closest = df.index[pos]
    print(f"Closest index: {idx_closest}")
    
    row = df.loc[idx_closest]
    print(f"Row type: {type(row)}")
    

    if isinstance(row, pd.DataFrame):
        print("Row is a DataFrame, taking first row")
        row = row.iloc[0]
    
    print("Row data:")
    print(row)
    
    # Debug specific price fields
    print("\nPrice debugging:")
    bid = row.get("best_bid", np.nan)
    ask = row.get("best_ask", np.nan)
    print(f"Raw best_bid: {bid}, type: {type(bid)}")
    print(f"Raw best_ask: {ask}, type: {type(ask)}")
    
    # Check if the prices seem reasonable or need scaling
    if np.isfinite(bid) and bid > 1000:
        print(f"NOTE: Bid price is unusually high: {bid}")
    if np.isfinite(ask) and ask > 1000:
        print(f"NOTE: Ask price is unusually high: {ask}")
    
    spread = ask - bid if np.isfinite([bid, ask]).all() else np.nan
    half_spread = spread / 2 if np.isfinite(spread) else np.nan
    
    print(f"Calculated spread: {spread}")
    print(f"Calculated half spread: {half_spread}")
    print("========== END DEBUGGING ==========")
    
    return {
        "closest_timestamp": idx_closest,
        "best_bid": bid,
        "best_ask": ask,
        "spread": spread,
        "half_spread": half_spread,
    }



def get_spread_at_time(df: pd.DataFrame, target_time) -> float:
    """Convenience wrapper used by optimisation layer."""
    if df.empty:
        raise ValueError("Cannot compute spread from empty DataFrame")
    diff = np.abs(df.index - _as_timestamp(target_time))
    pos = diff.argmin()
    idx = df.index[pos]
    row = df.loc[idx]
    bid, ask = row["best_bid"], row["best_ask"]
    if not np.isfinite([bid, ask]).all():
        raise ValueError("Non‑finite best_bid / best_ask encountered")
    return float(ask - bid)