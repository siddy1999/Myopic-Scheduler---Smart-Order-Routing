import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime, timedelta,timezone
import ast
import logging
import subprocess

from Main_1 import backtest #backtest_type
from metrics import metrics
from concurrent.futures import ThreadPoolExecutor, as_completed


def read_data(day,market_data_path,stock):
    
    date_str = day.replace("-", "")
    local_path = f"{market_data_path}{stock}/xnas-itch-{date_str}.mbp-10.csv"
    df = pd.read_csv(local_path)
    return df

def run_job(stock, day, strategy_params, data_path, order_freq, start_time, end_time, lookup_duration, market_data_path):
    df = read_data(day, data_path, stock)
    cost = backtest(stock, day, strategy_params, data_path, order_freq, start_time, end_time, lookup_duration, df.copy())
    metrics([stock], [day], market_data_path, start_time, end_time, df)
    return stock, cost

def main():
    
    #backtest_type("ofi") # ofi or look_up

    mode = "ofi"
    if mode == "look_up":
        from gen_lookup_table import preprocess_data,find_nearest_timestamp,calculate_queue_depth,execute_optimization,generate_lookup_table
    elif mode == "ofi":
        from gen_lookup_table_ofi import preprocess_data,find_nearest_timestamp,calculate_queue_depth,execute_optimization,generate_lookup_table
    strategy_params = { #define the parameters
            'S': 100,
            'T': 5,
            'f': 0.003,
            'r': [0.003],
            'theta': 0.0005,
            'lambda_u': 0.05,
            'lambda_o': 0.05,
            'N': 1000
        }

    order_freq = 120 # unit sec
    start_time = ("09","30") #At least from 9:45 each day (hour,minute) string
    end_time = ("16","00") #End before 16:00 (hour,minute) string
    stocks = ["AAPL"]
    market_data_path = f"s3://blockhouse-databento-mbp10/lookup-table/"
    data_path = f"s3://blockhouse-databento-mbp10/lookup-table/"
    lookup_duration = (0,15) #(hour,minute) int
    days = ["2025-04-02"]
    average_cost = {}

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for stock in stocks:
            for day in days:
                futures.append(executor.submit(
                    run_job, stock, day, strategy_params,
                    data_path, order_freq, start_time,
                    end_time, lookup_duration, market_data_path
                ))

        for f in as_completed(futures):
            stock, cost = f.result()
            average_cost[stock] = cost
    
    print("Done:", average_cost)

            
if __name__ == "__main__":
    main()
