import pandas as pd

def load_mbp10(path: str) -> pd.DataFrame:
    df = pd.read_csv(path,
                     parse_dates=["ts_recv", "ts_event"],
                     low_memory=False)
    # rename top-of-book columns to match the framework’s expectations
    df = df.rename(columns={
        "bid_px_00": "bid_price_0",
        "ask_px_00": "ask_price_0",
        "bid_sz_00": "bid_size_0",
        "ask_sz_00": "ask_size_0",
    })
    # compute mid price
    df["mid_price"] = (df.bid_price_0 + df.ask_price_0) / 2
    # index by receive timestamp
    return df.set_index("ts_recv").sort_index()
