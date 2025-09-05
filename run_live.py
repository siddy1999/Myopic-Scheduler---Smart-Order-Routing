import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sor.live_data import fetch_live_market_data
from sor.data_io import get_timestamp_values
from sor import execute_optimization

def notional_to_size(df: pd.DataFrame, notional: float, side: str) -> int:
    print(f"notional{notional} side{side}")
    ask = df["best_ask"].iloc[-1]
    bid = df["best_bid"].iloc[-1]
    px = ask if side.upper() == "B" else bid if side.upper() == "S" else (ask + bid) / 2
    px = px/1e9
    print(f"px{px} ")
    if px == 0:
        raise ValueError("Price is zero, cannot calculate size")
    return int(np.floor(notional / px))

def normalise_live_df(df: pd.DataFrame) -> pd.DataFrame:
    df = (
        df.rename(columns={"bid_price": "best_bid", "ask_price": "best_ask"})
          .assign(timestamp=lambda d: pd.to_datetime(d["timestamp"], utc=True))
          .sort_values("timestamp")
          .set_index("timestamp")
    )
    df["minute"] = df.index.floor("1min")
    df["best_bid"] = df["best_bid"].astype(float) / 1e9
    df["best_ask"] = df["best_ask"].astype(float) / 1e9
    return df


def main():
    p = argparse.ArgumentParser(description="Run live SOR allocation")
    p.add_argument("--symbol", default="AAPL")
    p.add_argument("--buffer", type=int, default=None,
                   help="Seconds to buffer live data (mutually exclusive with --n-ticks)")
    p.add_argument("--n-ticks", type=int, default=None,
                   help="Number of latest ticks to fetch (mutually exclusive with --buffer)")
    p.add_argument("--size", type=float, default=0)
    p.add_argument("--notional", type=float, default=0)
    p.add_argument("--side", default="B", choices=["B","S"])  
    p.add_argument("--flatten", action="store_true")
    args = p.parse_args()

    if args.buffer is not None and args.n_ticks is not None:
        raise SystemExit("--buffer and --n-ticks are mutually exclusive")
    if args.buffer is None and args.n_ticks is None:
        raise SystemExit("Specify either --buffer or --n-ticks")

    dfs = fetch_live_market_data(
        n_ticks=args.n_ticks,
        buffer_time_sec=args.buffer or 0,
        flatten=True
    )
    # print({k: len(v) for k,v in dfs.items()})

    df_live = normalise_live_df(dfs[args.symbol])
    dfs_live = {"v1": df_live}

    S = args.size
    if S == 0:
        if args.notional <= 0:
            raise SystemExit("Either --size or --notional must be > 0")
        S = notional_to_size(df_live, args.notional, args.side)
    print(f"Effective share size S = {S}")

    mode = "lookup" if Path(f"ratio_table_{args.symbol}.pkl").exists() else "exp"
    target_time = df_live.index[len(df_live)//2]

    allocation, _ = execute_optimization(
        dfs_live,
        target_time,
        S=S,
        T=1,
        f=0.003,
        r=[0.002],
        lambda_u=0.05,
        lambda_o=0.05,
        method=mode,
        stock=args.symbol,
        side=args.side

    )

    prices = get_timestamp_values(df_live, target_time)

    # result = {
    #     "number_of_market_orders": int(round(allocation[0])),
    #     "market_order_price": round(prices["best_ask"], 5),
    #     "number_of_limit_orders": int(round(allocation[1])),
    #     "limit_order_price": round(prices["best_bid"], 5),
    # }

    # 6. Reverse market/limit order prices if selling ----------------------
    if args.side.upper() == "B":
        m_px, l_px = prices["best_ask"], prices["best_bid"]
    else:                    # selling reverses the book
        m_px, l_px = prices["best_bid"], prices["best_ask"]

    result = {
        "side": args.side.upper(),
        "number_of_market_orders": int(round(allocation[0])),
        "market_order_price": round(m_px, 5),
        "number_of_limit_orders": int(round(allocation[1])),
        "limit_order_price": round(l_px, 5),
    }

    out_path = Path("allocation_output.json")
    out_path.write_text(json.dumps(result, indent=4))

    print(json.dumps(result, indent=4))

if __name__ == "__main__":
    main()
