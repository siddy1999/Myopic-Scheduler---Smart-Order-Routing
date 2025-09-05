#!/usr/bin/env python3
"""
main_myopic_integration.py

Enhanced execution script integrating VWAP, TWAP, and Myopic scheduling
on top of a Smart Order Router (SOR) using MBP‑10 data.
"""
import os
import logging
from datetime import datetime
import pandas as pd
import pandas as dd  # alias dd → pandas


from benchmark_implementation import BenchmarkConfig, BenchmarkRunner
from mbp10_loader import load_mbp10

# -------------------------------
# 1. Configuration
# -------------------------------
TICKER       = "AAPL"
DAYS         = ["2025-04-01", "2025-04-02"]      # Expand as needed
DATA_ROOT    = "data/mbp10/"
OUTPUT_ROOT  = "results/"
LOG_DIR      = "logs/"
ALGORITHMS   = ["VWAP", "TWAP", "Myopic"]

# ensure output directories exist
os.makedirs(os.path.join(OUTPUT_ROOT, "schedules"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_ROOT, "metrics"), exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# -------------------------------
# 2. Logging
# -------------------------------
logfile = os.path.join(LOG_DIR, f"backtest_{TICKER}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(logfile),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# -------------------------------
# 3. Main
# -------------------------------

def main():
    # Build a per-date run config
    summary = []
    for day in DAYS:
        date_str = day.replace("-", "")
        logger.info(f"Starting backtest for {TICKER} on {day}")

        # Construct data paths
        data_paths = {
            "VWAP": os.path.join(DATA_ROOT, f"{TICKER}/xnas-itch-{date_str}.mbp-10.csv"),
            "TWAP": os.path.join(DATA_ROOT, f"{TICKER}/xnas-itch-{date_str}.mbp-10.csv"),
            "Myopic": os.path.join(DATA_ROOT, f"{TICKER}/xnas-itch-{date_str}.mbp-10.csv")
        }

        # Instantiate config & runner
        config = BenchmarkConfig(
            ticker=TICKER,
            trading_dates=[day],
            data_paths=data_paths,
            algorithms=ALGORITHMS,
            output_dir=OUTPUT_ROOT,
            log_dir=LOG_DIR
        )
        runner = BenchmarkRunner(config)

        # Monkey-patch the data loader for Myopic
        original_loader = runner._load_data
        def _load_data_override(algo, date):
            path = data_paths[algo]
            if algo == "Myopic":
                df = load_mbp10(path)
            else:
                df = pd.read_csv(path, parse_dates=["ts_recv", "ts_event"], low_memory=False)
                # if needed compute mid_price for VWAP/TWAP as well
                df["mid_price"] = (df.bid_px_00 + df.ask_px_00) / 2.0
            return df
        runner._load_data = _load_data_override

        # Run backtest for this day
        daily_results = runner.run_all()
        summary.append(daily_results)

    # Save/aggregate summary
    runner.save_summary(summary)
    logger.info("All backtests complete.")
    return summary


if __name__ == "__main__":
    main()
