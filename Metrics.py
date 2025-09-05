import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timezone
import ast
import logging
import numpy as np
import databento as db
import subprocess

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_data(ticker, date,market_data_path,initia_time,ending_time,df):
    """Download and load databento data for a specific ticker and date."""
    # date_str = date.replace("-", "")
    # local_path = f"{market_data_path}{ticker}/xnas-itch-{date_str}.mbp-10.csv"
    try:
    
        # df = pd.read_csv(local_path)
        logging.info("Data loaded successfully!")
        
        # Process timestamps and time zones
        df['ts_event'] = pd.to_datetime(df['ts_event'], utc=True)
        start_dt = datetime.strptime(f"{date} {initia_time[0]}:{initia_time[1]}", "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
        end_dt = datetime.strptime(f"{date} {ending_time[0]}:{ending_time[1]}", "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
        

        # Filter within time range
        filtered_df = df[(df['ts_event'] >= start_dt) & (df['ts_event'] <= end_dt) & (df['action'] == 'T')] 
        
        return filtered_df
    except FileNotFoundError:
        logging.error("Error: AWS CLI not found. Please install and configure it.")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        return None

def slice_market_data(market_data, table_df):
    """
    Slice market data according to the times specified in table_df.
    
    Parameters:
    - market_data: DataFrame containing market data from databento
    - table_df: DataFrame with trade information including times and horizons
    
    Returns:
    - List of DataFrames, each containing a slice of market data
    """
    all_slices = []

    for idx, row in table_df.iterrows():
        # Get the time string from the row
        start_time_str = row['time'] if ':' in str(row['time']) else '00:00:00'
        
        # Extract the date from the market data
        date_str = market_data['ts_event'].dt.date.iloc[0].strftime('%Y-%m-%d')
        
        # Construct a timestamp with timezone
        start_time = pd.Timestamp(f"{date_str} {start_time_str}", tz='America/New_York')
        
        # Add time horizon in minutes
        horizon = int(row['Time Horizon'])
        end_time = start_time + pd.Timedelta(minutes=horizon)
        
        # Slice data between start_time and end_time
        data_slice = market_data[
            (market_data['ts_event'] >= start_time) &
            (market_data['ts_event'] <= end_time)
        ]
        
        # Filter trades if needed
        trades_slice = data_slice[data_slice['action'] == 'T'] if 'action' in data_slice.columns else data_slice
        
        all_slices.append(data_slice)  # Keep the full slice for book data
        
        logging.info(f"Row {idx}: From {start_time} to {end_time}")
        logging.info(f"Total entries: {len(data_slice)}, Trade entries: {len(trades_slice)}")
        
    return all_slices

def calculate_vwap_price(df):
    """Calculate Volume-Weighted Average Price per share from databento data."""
    if len(df) == 0:
        return 0
    
    # Filter for trades only
    trades_df = df[df['action'] == 'T'] if 'action' in df.columns else df
    
    if len(trades_df) == 0 or 'price' not in trades_df.columns or 'size' not in trades_df.columns:
        return 0
    
    total_volume = trades_df["size"].sum()
    if total_volume <= 0:
        return 0
    
    # Sum of (price * size) / sum of size
    weighted_price_sum = (trades_df["price"] * trades_df["size"]).sum()
    vwap_price = weighted_price_sum / total_volume
    
    return vwap_price


def calculate_twap_price(df):
    """
    Calculate true Time-Weighted Average Price (TWAP) per share from databento data.
    TWAP is calculated as the sum of (price * time interval) divided by the total time.
    """
    if len(df) == 0 or 'ts_event' not in df.columns:
        return 0

    # Use mid prices if available, else trade prices
    if 'best_bid' in df.columns and 'best_ask' in df.columns:
        valid_quotes = df[(df['best_bid'] > 0) & (df['best_ask'] > 0)].copy()
        if len(valid_quotes) > 1:
            valid_quotes['mid'] = (valid_quotes['best_bid'] + valid_quotes['best_ask']) / 2
            valid_quotes = valid_quotes.sort_values('ts_event')
            # Calculate time deltas in seconds
            time_deltas = valid_quotes['ts_event'].diff().dt.total_seconds().shift(-1)
            # Drop the last row (no next interval)
            valid_quotes = valid_quotes.iloc[:-1]
            time_deltas = time_deltas.iloc[:-1]
            weighted_sum = (valid_quotes['mid'] * time_deltas).sum()
            total_time = time_deltas.sum()
            return weighted_sum / total_time if total_time > 0 else 0

    # Fallback to trade prices if available
    if 'price' in df.columns:
        trades_df = df[df['action'] == 'T'].copy() if 'action' in df.columns else df.copy()
        if len(trades_df) > 1 and 'ts_event' in trades_df.columns:
            trades_df = trades_df.sort_values('ts_event')
            time_deltas = trades_df['ts_event'].diff().dt.total_seconds().shift(-1)
            trades_df = trades_df.iloc[:-1]
            time_deltas = time_deltas.iloc[:-1]
            weighted_sum = (trades_df['price'] * time_deltas).sum()
            total_time = time_deltas.sum()
            return weighted_sum / total_time if total_time > 0 else 0

    return 0

def calculate_rolling_twap(market_data_slices, result):
    """Calculate TWAP for each market data slice and add to result DataFrame."""
    result["TWAP"] = np.nan
    for i in range(len(market_data_slices)):
        twap_temp = calculate_twap_price(market_data_slices[i])
        result.at[i, "TWAP"] = twap_temp
        logging.info(f"Calculated TWAP for slice {i}: {twap_temp}")

    return result
def rolling_vwap(market_data_slices, result_df):
    """Calculate VWAP for each market data slice and add to result DataFrame."""
    result_df["VWAP"] = np.nan
    
    for i in range(len(market_data_slices)):
        # Get trade data from the slice
        trades_df = market_data_slices[i][market_data_slices[i]['action'] == 'T'] if 'action' in market_data_slices[i].columns else market_data_slices[i]
        
        vwap_temp = calculate_vwap_price(trades_df)
        result_df.at[i,"VWAP"] = vwap_temp
        logging.info(f"Calculated VWAP for slice {i}: {vwap_temp}")
        
    return result_df

def parse_mid_prices(mid_price_str):
    """Parse mid price string into price and volume tuples."""
    try:
        mid_list = ast.literal_eval(mid_price_str)
        return mid_list  # List of [price, volume] pairs
    except (ValueError, SyntaxError) as e:
        logging.warning(f"Error parsing mid price: {e}")
        return []

def calculate_mid_price_avg(mid_prices):
    """Calculate volume-weighted average mid price."""
    if not mid_prices:
        return 0
    
    total_value = sum(price * volume for price, volume in mid_prices)
    total_volume = sum(volume for _, volume in mid_prices)
    
    if total_volume == 0:
        return 0
    
    return total_value / total_volume

def safe_division(numerator, denominator, default=0):
    """Safely divide values to avoid division by zero errors."""
    return numerator / denominator if denominator != 0 else default

def calculate_bps_slippage(benchmark_price, execution_price):
    """Calculate slippage in basis points (bps).
    Positive value means execution was better than benchmark.
    """
    
    if benchmark_price <= 0:
        return 0
    
    # (Benchmark - Execution) / Benchmark * 10000 = BPS difference
    # If execution price is lower than benchmark, this is positive (good)
    return (benchmark_price - execution_price) / benchmark_price
    
def metrics(tickers, days,market_data_path,initia_time,ending_time,df):
    """Calculate trading metrics for given tickers and days using databento data."""
    
    all_results = pd.DataFrame()  # To store all results for summary statistics
    
    for ticker in tickers:
        for day in days:
            # Reset summary for each day to avoid appending to previous days
            summary = []
            
            logging.info(f"Processing {ticker} for {day}")
            sor_path = f"{market_data_path}{ticker}/{day}_result.csv"
            
            
            try:
                sor_result = pd.read_csv(sor_path)
                logging.info(f"Loaded SOR result from {sor_path} with {len(sor_result)} rows")
            except Exception as e:
                logging.error(f"Failed to load SOR result: {e}")
                continue
            
            # Load market data from databento
            market_data = get_data(ticker, day, market_data_path,initia_time,ending_time,df)
            if market_data is None:
                logging.error(f"Failed to load databento data for {ticker} on {day}, skipping")
                continue
            
            # Slice market data according to SOR result
            market_data_slices = slice_market_data(market_data, sor_result)
            
            # Add VWAP and TWAP to result
            calculate_rolling_twap(market_data_slices, sor_result)
            rolling_vwap(market_data_slices, sor_result)
            
            # Process timestamps in SOR result
            sor_result['time'] = pd.to_datetime(sor_result['time'], errors='coerce')
            sor_result = sor_result.dropna(subset=['time'])
            if sor_result['time'].dt.tz is None:
                sor_result['time'] = sor_result['time'].dt.tz_localize('UTC')
            else:
                sor_result['time'] = sor_result['time'].dt.tz_convert('UTC')
                
            for i in range(len(sor_result)):
                try:
                    # Calculate order and execution metrics
                    order_size = (sor_result["market_v"].iloc[i] + sor_result["limit_v"].iloc[i])
                    executed_qty = (sor_result["market_v"].iloc[i] + sor_result["limit_fill"].iloc[i])
                    
                    # Skip if no order was placed
                    if order_size <= 0:
                        continue
                    
                    # Calculate execution value and average price
                    market_value = sor_result["market_v"].iloc[i] * sor_result["market_p"].iloc[i]
                    limit_value = sor_result["limit_fill"].iloc[i] * sor_result["limit_p"].iloc[i]
                    executed_value = market_value + limit_value
                    
                    avg_execution_price = safe_division(executed_value, executed_qty)
                    
                    # Parse mid prices
                    mid_prices = parse_mid_prices(sor_result["mid_price"].iloc[i])
                    mid_price_avg = calculate_mid_price_avg(mid_prices)
                    
                    # Calculate notional values for mid price
                    mid_value = sum(price * volume for price, volume in mid_prices)
                    mid_volume = sum(volume for _, volume in mid_prices)
                    
                    # Skip further calculations if nothing was executed
                    if executed_qty <= 0:
                        continue
                    
                    # Calculate slippage in basis points
                    twap_price = sor_result["TWAP"].iloc[i]
                    vwap_price = sor_result["VWAP"].iloc[i]
                    sor_twap = calculate_bps_slippage(twap_price, avg_execution_price)
                    sor_vwap = calculate_bps_slippage(vwap_price, avg_execution_price)
                    sor_mid = calculate_bps_slippage(mid_price_avg, avg_execution_price)
                    
                    # Calculate fill ratio
                    fill_ratio = safe_division(executed_qty, order_size)
                    
                    # Calculate latency
                    list_time = sor_result["time"].iloc[i]
                    if pd.isna(list_time):
                        latency = None
                    else:
                        # Parse fill_time safely
                        fill_time_raw = sor_result["fill_time"].iloc[i]
                        try:
                            fill_time_parsed = ast.literal_eval(fill_time_raw)
                            
                            # Handle both single timestamp and list
                            if isinstance(fill_time_parsed, list):
                                fill_times = [pd.to_datetime(ft) for ft in fill_time_parsed]
                                fill_times = [ft.tz_localize('UTC') if ft.tzinfo is None else ft for ft in fill_times]
                                latency = [(ft - list_time).total_seconds() for ft in fill_times]
                                latency = latency[0] if latency else None
                            else:
                                fill_time = pd.to_datetime(fill_time_parsed)
                                fill_time = fill_time.tz_localize('UTC') if fill_time.tzinfo is None else fill_time
                                latency = (fill_time - list_time).total_seconds()
                        except Exception as e:
                            logging.warning(f"Error processing fill time: {e}")
                            latency = None
                    
                    summary.append({
                        "ticker": ticker,
                        "date": day,  # Add date to track which day this is for
                        "time": sor_result["time"].iloc[i],
                        "order_size": order_size,
                        "executed_qty": executed_qty,
                        "fill_ratio": fill_ratio,
                        "avg_execution_price": avg_execution_price,
                        "twap_price": twap_price,
                        "vwap_price": vwap_price,
                        "mid_price": mid_price_avg,
                        "sor_twap": sor_twap,
                        "sor_vwap": sor_vwap,
                        "sor_midprice": sor_mid,
                        "mid_value": mid_value,
                        "latency": latency,
                        "filled": sor_result["filled"].iloc[i]
                    })
                    
                except Exception as e:
                    logging.error(f"Error processing row {i}: {e}")
                    logging.error(f"Row data: {sor_result.iloc[i]}")
            
            # Create DataFrame for this day's results
            day_df = pd.DataFrame(summary)
            
            if not day_df.empty:
                # Save results for this specific day
                output_file = f"{market_data_path}{ticker}/{day}_output.csv"
                logging.info(f"Saving results for {ticker} on {day} to {output_file}")
                day_df.to_csv(output_file, index=False)
                
                # Append to overall results for summary statistics
                all_results = pd.concat([all_results, day_df], ignore_index=True)
                
                # # Analyze last 15 minutes for this day
                # logging.info(f"\n=== ANALYZING LAST 15 MINUTES OF TRADING FOR {ticker} ON {day} ===")
                # last_15_min_data = analyze_last_15_min(ticker, day)
                # if last_15_min_data is not None and not last_15_min_data.empty:
                #     logging.info(f"Last 15 minutes analysis complete with {len(last_15_min_data)} trades")
                    
                #     # Save last 15 minutes analysis to a separate file
                #     last_15_min_output_path = f"s3://blockhouse-databento-mbp10/lookup-table/Backtest-SOR/Dixit/Strat-Dev-05052025-1/{ticker}/{day}_LAST_15MIN.csv"
                #     last_15_min_data.to_csv(last_15_min_output_path, index=False)
                #     logging.info(f"Saved last 15 minutes analysis to {last_15_min_output_path}")
            else:
                logging.warning(f"No results to save for {ticker} on {day}")
    
    return all_results

# def analyze_last_15_min(ticker, date,market_data_path):
#     """Analyze trading data for the last 15 minutes of regular trading."""
#     try:
#         # Get databento data
#         df = get_data(ticker, date,market_data_path)
        
#         if df is None:
#             logging.error("Failed to load databento data")
#             return None
            
#         # Define the market close time range
#         start_time = pd.Timestamp(f'{date} 15:45:00', tz='America/New_York')
#         end_time = pd.Timestamp(f'{date} 16:00:00', tz='America/New_York')
        
#         # Filter rows within the last 15 minutes of regular trading
#         last_15_min_df = df[(df['ts_event_nyc'] >= start_time) & (df['ts_event_nyc'] < end_time)]
#         logging.info(f"Found {len(last_15_min_df)} entries in the last 15 minutes")
        
#         # Filter for trades within the time range
#         last_15_min_trades_df = last_15_min_df[last_15_min_df['action'] == 'T']
#         logging.info(f"Found {len(last_15_min_trades_df)} trades in the last 15 minutes")
        
#         # Basic stats for trades
#         if len(last_15_min_trades_df) > 0:
#             logging.info(f"Average trade price: {last_15_min_trades_df['price'].mean():.4f}")
#             logging.info(f"Total traded volume: {last_15_min_trades_df['size'].sum()}")
            
#         return last_15_min_trades_df
            
#     except Exception as e:
#         logging.error(f"Error analyzing last 15 minutes: {e}")
#         return None






















#############--------------MAIN--------------#############

# def main():
#     """Main function to run the metrics calculation."""
#     # Configure logging for better debug output
#     logging.basicConfig(
#         level=logging.INFO,
#         format='%(asctime)s - %(levelname)s - %(message)s',
#         handlers=[
#             logging.FileHandler("trading_metrics.log"),
#             logging.StreamHandler()
#         ]
#     )
    
    
#     # Define tickers and dates to process
#     tickers = ["AAPL"]  # Add more tickers as needed
#     days = ["2025-04-01"]  # Format: YYYY-MM-DD
#     market_data_path = f"s3://blockhouse-databento-mbp10/lookup-table/"
    
#     # Calculate metrics
#     start_time = ("15","45") #At least from 9:45 each day (hour,minute) string
#     end_time = ("16","00") #End before 16:00 (hour,minute) string
    
        
#     results = metrics(tickers, days,market_data_path,start_time,end_time)
        
        
            
            
# if __name__ == "__main__":
#     main()