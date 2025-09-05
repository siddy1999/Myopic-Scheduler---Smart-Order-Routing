# live_adapter.py  – put next to your notebook or just run the cell
import pandas as pd

def mbp_row_to_sor(row: pd.Series) -> pd.Series:
    """
    Convert one flattened MBP‑1 row (already produced by flatten_row)
    into the column names the SOR code expects.

    Required incoming columns (from flatten_row):
        bid_price, bid_size, ask_price, ask_size,
        price, size, action, side, timestamp
    """
    return pd.Series({
        # --- columns used by queue‑depth logic -----------------------------
        'timestamp' : row['timestamp'],
        'minute'    : row['timestamp'].floor('1min'),
        'best_bid'  : row['bid_price'],
        'best_ask'  : row['ask_price'],
        'bid_sz_00' : row['bid_size'],     # size at best bid
        'ask_sz_00' : row['ask_size'],
        # --- columns used by outflow/adverse‑selection logic ---------------
        'price'     : row['price'],
        'size'      : row['size'],
        'action'    : row['action'],       # 'T', 'C', …
        'side'      : row['side']          # 'A' (ask aggressor) / 'B' etc.
    })
