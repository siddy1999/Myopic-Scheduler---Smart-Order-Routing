# market_metrics.py
"""Functions for minute‑wise queue/outflow analytics and distribution fitting.
These utilities sit *between* raw data I/O (`data_io`) and the optimiser.

Exports (see `__all__`):
    process_minute_wise_data
    calculate_queue_depth
    calculate_outflow_fills_all_intervals
    fit_outflows_to_distribution
    generate_xi
    estimate_adverse_selection
    compute_queue_and_volume_size
    get_queue_volume_for_minute
    get_bucket
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import expon, gaussian_kde
from datetime import timedelta
from typing import Dict, List

from .data_io import get_timestamp_values  # local import – keeps layering clean

def _sel(bid_col: str, ask_col: str, side: str) -> str:
    """Return the side-specific column name."""
    return bid_col if side.upper() == "B" else ask_col


__all__: list[str] = [
    "process_minute_wise_data",
    "calculate_queue_depth",
    "calculate_outflow_fills_all_intervals",
    "fit_outflows_to_distribution",
    "generate_xi",
    "estimate_adverse_selection",
    "compute_queue_and_volume_size",
    "get_queue_volume_for_minute",
    "get_bucket",
]

# ─────────────────────────────────────────────────────────────────────────────
# 1. Minute‑wise snapshot helper
# ─────────────────────────────────────────────────────────────────────────────

def process_minute_wise_data(dfs: Dict[str, pd.DataFrame]):
    """Return a dict {venue: minute‑resampled DF (+ max_best_bid column)}."""
    if not dfs:
        raise ValueError("dfs dictionary is empty")
    for v, df in dfs.items():
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Value for {v} must be a DataFrame")

    dfs_minute, best_bid_series = {}, []

    for venue, df in dfs.items():
        for col in ("best_bid", "bid_sz_00", "best_ask"):
            if col not in df.columns:
                raise KeyError(f"{venue} missing column {col}")

        res_bid      = df["best_bid"  ].resample("1min").last().shift(1)
        res_bid_size = df["bid_sz_00"].resample("1min").last().shift(1)
        res_ask      = df["best_ask"  ].resample("1min").last().shift(1)

        df_m = pd.DataFrame({
            "minute": res_bid.index,
            "best_bid": res_bid.values,
            "best_bid_size": res_bid_size.values,
            "best_ask": res_ask.values,
        }).dropna(subset=["minute"]).reset_index(drop=True)

        dfs_minute[venue] = df_m
        best_bid_series.append(res_bid.rename(venue))

    combined = pd.concat(best_bid_series, axis=1)
    max_bid = pd.DataFrame({"minute": combined.index, "max_best_bid": combined.max(axis=1)})

    for venue in dfs_minute:
        dfs_minute[venue] = dfs_minute[venue].merge(max_bid, on="minute", how="left")

    return dfs_minute

# ─────────────────────────────────────────────────────────────────────────────
# 2. Queue depth (Q) at target timestamp
# ─────────────────────────────────────────────────────────────────────────────

def _find_nearest_timestamp(df: pd.DataFrame, target_time: pd.Timestamp):
    ts = df.loc[df["timestamp"] <= target_time, "timestamp"].max()
    return None if pd.isna(ts) else ts


def calculate_queue_depth(dfs: Dict[str, pd.DataFrame], target_time: pd.Timestamp, side: str = "B",) -> Dict[str, float]:
    target_time = pd.Timestamp(target_time)
    bid_px, qdepth = {}, {}

    for venue, df in dfs.items():
        ts = _find_nearest_timestamp(df, target_time)
        if ts is None:
            continue
        row = df.loc[ts]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        # bid_px[venue] = row["best_bid"]
        # qdepth[venue] = row["bid_sz_00"]
        best_col  = _sel("best_bid", "best_ask", side)
        depth_col = _sel("bid_sz_00", "ask_sz_00", side)
        bid_px[venue] = row[best_col]
        qdepth[venue] = row[depth_col]

    if not bid_px:
        return {}
    max_bid = max(bid_px.values())
    for v in qdepth:
        if bid_px[v] < max_bid:
            qdepth[v] = 0
    return qdepth

# ─────────────────────────────────────────────────────────────────────────────
# 3. Outflow fills per‑interval
# ─────────────────────────────────────────────────────────────────────────────

def _outflow_for_interval(df: pd.DataFrame, best_px: float, side: str) -> float:
    """
    How many shares left the front of the book during this interval:
      • cancellations at price == best_px
      • trades at price == best_px *and* hitting our side of the book
    """
    if df.empty or pd.isna(best_px):
        return 0.0

    # 1) cancels at that price level
    cancels = df.loc[
        (df["price"] == best_px) & (df["action"] == "C"),
        "size"
    ].sum()

    # 2) trades at that price *and* on the correct side:
    #    if we're buying (side=="B"), sellers hit our bids → df.side=="B"
    #    if we're selling (side=="S"), buyers hit our asks → df.side=="A"
    trade_side = "B" if side.upper() == "B" else "A"
    trades = df.loc[
        (df["price"] == best_px) & (df["action"] == "T") & (df["side"] == trade_side),
        "size"
    ].sum()

    return cancels + trades


def calculate_outflow_fills_all_intervals(
    dfs: Dict[str, pd.DataFrame],
    side: str = "B",
) -> Dict[str, pd.DataFrame]:
    """
    For each venue DF, group by the floor-minute of its own timestamps,
    then call _outflow_for_interval on that minute’s rows and the
    corresponding best_bid or best_ask.
    """
    results: Dict[str, pd.DataFrame] = {}

    for venue, df in dfs.items():
        # ensure there is a 'minute' column
        if "minute" not in df.columns:
            df = df.copy()
            df["minute"] = df.index.floor("1min")

        records = []
        for minute_val, group in df.groupby("minute"):
            best_px = (
                group["best_bid"].iloc[0] if side.upper() == "B"
                else group["best_ask"].iloc[0]
            )
            xi = _outflow_for_interval(group, best_px, side)
            records.append({"minute": minute_val, "outflow_fill": xi})

        results[venue] = pd.DataFrame(records)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 4. Fit distribution & sample ξ
# ─────────────────────────────────────────────────────────────────────────────

def fit_outflows_to_distribution(outflows: Dict[str, pd.DataFrame], method: str = "exp"):
    params = {}
    for venue, df in outflows.items():
        data = df["outflow_fill"].values
        if len(data) == 0 or np.all(data == 0):
            continue
        if method == "exp":
            loc, scale = expon.fit(data)
            if scale > 0:  # sanity
                params[venue] = {"loc": loc, "scale": scale}
        else:  # KDE
            bw = "scott"
            params[venue] = {"kde": gaussian_kde(data, bw_method=bw)}
    return params


def generate_xi(outflows: Dict[str, pd.DataFrame], T: int, method: str = "exp") -> np.ndarray:
    if T <= 0:
        return np.zeros(len(outflows))
    params = fit_outflows_to_distribution(outflows, method)
    xi_vec = []
    for venue in outflows:
        if venue not in params:
            xi_vec.append(0.0)
            continue
        p = params[venue]
        if method == "exp":
            xi_vec.append(float(np.sum(expon.rvs(loc=p["loc"], scale=p["scale"], size=T))))
        else:
            xi_vec.append(float(np.sum(p["kde"].resample(T))))
    return np.array(xi_vec)

# ─────────────────────────────────────────────────────────────────────────────
# 5. Adverse selection estimator (Δprice after 10 s)
# ─────────────────────────────────────────────────────────────────────────────

def estimate_adverse_selection(
    dfs: Dict[str, pd.DataFrame],
    sec_delta: int,
    target_time,
    side: str = "B",
) -> Dict[str, float]:
    """
    For each venue, pick every trade that hit our side of the book
    in a ±1-minute window around target_time, then measure
    the average signed move in mid-price after sec_delta seconds.
    """
    delta = timedelta(seconds=sec_delta)
    out: Dict[str, float] = {}

    for venue, df in dfs.items():
        # 1) filter only the trades that hit our side
        #    if buying → we care about df.side=="B" trades
        #    if selling → we care about df.side=="A" trades
        trade_side = "B" if side.upper() == "B" else "A"      # ←  restore this
        trades = df.loc[
        (df["action"] == "T") & (df["side"] == trade_side),
        ["price"]
    ]
        trades = df.loc[
            (df["action"] == "T") & (df["side"] == trade_side),
            ["price"]
        ].copy()
        trades = trades[~trades.index.duplicated(keep="first")]

        # 2) restrict to ±1 minute around target_time
        base = pd.Timestamp(target_time).floor("1min")
        mask = (
            (trades.index >= base - pd.Timedelta("1min")) &
            (trades.index <= base + pd.Timedelta("1min"))
        )
        trades = trades.loc[mask]

        # 3) for each trade, grab future mid‐price and sign it
        deltas: list[float] = []
        for ts, row in trades.iterrows():
            future = ts + delta
            snap = df[df["minute"] == future.floor("1min")]
            px = get_timestamp_values(snap, future)
            if px["best_bid"] == 0 and px["best_ask"] == 0:
                continue
            mpx = (px["best_bid"] + px["best_ask"]) / 2
            sign = +1 if side.upper() == "B" else -1
            deltas.append(sign * (mpx - row["price"]))

        out[venue] = float(np.nanmean(deltas)) if deltas else np.nan

    return out



# ─────────────────────────────────────────────────────────────────────────────
# 6. Queue / volume buckets (used by lookup table)
# ─────────────────────────────────────────────────────────────────────────────

def compute_queue_and_volume_size(dfs: Dict[str, pd.DataFrame]):
    minute_data = process_minute_wise_data(dfs)
    out = {}
    for venue, df_min in minute_data.items():
        venue_rows = []
        vdf = dfs[venue]
        for _, r in df_min.iterrows():
            minute = r["minute"]
            chunk = vdf[vdf["minute"] == minute]
            queue = chunk["size"].sum()
            vol   = chunk[chunk["action"] == "T"]["size"].sum()
            venue_rows.append({"minute": minute, "queue": queue, "volume": vol})
        out[venue] = pd.DataFrame(venue_rows)
    return out


def get_queue_volume_for_minute(df: pd.DataFrame, minute_val):
    chunk = df[df["minute"] == minute_val]
    if chunk.empty:
        return None
    return {
        "queue":  chunk["size"].sum(),
        "volume": chunk[chunk["action"] == "T"]["size"].sum(),
    }


def get_bucket(value: float, thresholds: tuple[float, float]):
    low, high = thresholds
    return "Low" if value < low else "High" if value > high else "Medium"
