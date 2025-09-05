# 1. Lib and Data Imports

import pandas as pd
import numpy as np
from scipy.stats import expon, gaussian_kde
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Dict, Tuple, List
import os
import pickle
from ofi import calculate_ofi_signal, estimate_xi_from_ofi, calculate_causal_ofi_signal


def preprocess_data(file_path):
    """
    Preprocess the MBP-10 data.
    Correctly index, timestamp, and rename.
    
    Required columns:
    - ts_event
    - bid_px_00
    - ask_px_00
    
    Raises:
        KeyError: If required columns are missing
        ValueError: If data processing fails
        UnicodeDecodeError: If file encoding is invalid
        pd.errors.ParserError: If CSV parsing fails
        AttributeError: If file format is not supported
    """
    try:
        # Check if file_path is a string (actual file path)
        if isinstance(file_path, str):
            # Check file extension
            _, extension = os.path.splitext(file_path)
            if extension.lower() != '.csv':
                raise AttributeError(f"Unsupported file format: {extension}. Expected .csv")

        # Try reading with UTF-8 encoding first
        try:
            df = pd.read_csv(file_path, on_bad_lines='error', encoding='utf-8')
        except UnicodeDecodeError as e:
            # If UTF-8 fails, try with UTF-16
            try:
                df = pd.read_csv(file_path, on_bad_lines='error', encoding='utf-16')
            except UnicodeDecodeError as e:
                raise
        
        # Check for required columns
        required_columns = {'ts_event', 'bid_px_00', 'ask_px_00'}
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            raise KeyError(f"Missing required columns: {missing_columns}")

        # Verify all required columns have valid data
        if df['ts_event'].isna().any().any():
            raise ValueError("Missing values in 'ts_event' columns")

        # Verify price columns contain valid numeric data
        try:
            df['bid_px_00'] = pd.to_numeric(df['bid_px_00'])
            df['ask_px_00'] = pd.to_numeric(df['ask_px_00'])
        except ValueError as e:
            raise ValueError(f"Invalid price values: {str(e)}")

        df['ts_event'] = pd.to_datetime(df['ts_event'], utc=True)
        df = df.sort_values('ts_event').reset_index(drop=True)
        df['timestamp'] = pd.to_datetime(df['ts_event'])
        df['Date'] = df['timestamp'].dt.date
        df['minute'] = df['timestamp'].dt.floor('1min')
        df.rename(columns={'bid_px_00': 'best_bid', 'ask_px_00': 'best_ask'}, inplace=True)
        df.set_index('ts_event', inplace=True)

        return df
    except pd.errors.ParserError as e:
        raise pd.errors.ParserError(f"Error parsing CSV file: {str(e)}")
    except (ValueError, UnicodeDecodeError) as e:
        # Only catch ValueError and UnicodeDecodeError, let KeyError pass through
        raise ValueError(f"Error processing data: {str(e)}")


def get_timestamp_values(df: pd.DataFrame, target_time: pd.Timestamp) -> dict:
    '''
    For an inputted timestamp, return the timestamp from the data that is closest and its relevant info
    
    Input: 
        df: DataFrame with DatetimeIndex
        target_time: pd.Timestamp or string that can be converted to timestamp
    Output: 
        dict of closest timestamp, best bid, best ask, spread, half_spread
    
    Raises:
        TypeError: If df index is not DatetimeIndex or target_time can't be converted to Timestamp
    '''
    if df.empty:
        return {
            "best_bid": 0.0,
            "best_ask": 0.0,
            "spread": 0.0,
            "half_spread": 0.0
        }

    # Validate DataFrame structure
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame index must be DatetimeIndex")

    # Convert target_time to Timestamp if it isn't already
    if not isinstance(target_time, pd.Timestamp):
        try:
            target_time = pd.Timestamp(target_time)
        except (ValueError, TypeError):
            raise TypeError("target_time must be convertible to Timestamp")

    # Convert target timezone to match DataFrame if needed
    if target_time.tz != df.index.tz:
        if target_time.tz is None:
            target_time = target_time.tz_localize(df.index.tz)
        else:
            target_time = target_time.tz_convert(df.index.tz)

    # Calculate differences as a plain TimedeltaIndex
    timedeltas = df.index - target_time
    
    # Make a Series whose index is the same as df.index, but the data is the Timedelta
    timedeltas_s = pd.Series(timedeltas, index=df.index)
    
    # Take absolute value of each Timedelta
    abs_diff = timedeltas_s.abs()
    
    # Find closest timestamp
    idx_closest = abs_diff.idxmin()
    row = df.loc[idx_closest]

    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]

    # Extract values with safe fallbacks for missing or invalid data
    best_bid = row.get("best_bid", 0.0)
    best_ask = row.get("best_ask", 0.0)
    
    # Handle NaN/Inf values
    if pd.isna(best_bid) or pd.isna(best_ask):
        best_bid = float('nan')
        best_ask = float('nan')
        spread_value = float('nan')
        half_spread_value = float('nan')
    elif np.isinf(best_bid) or np.isinf(best_ask):
        best_bid = float('inf') if np.isinf(best_bid) else best_bid
        best_ask = float('inf') if np.isinf(best_ask) else best_ask
        spread_value = float('inf')
        half_spread_value = float('inf')
    else:
        spread_value = best_ask - best_bid
        half_spread_value = spread_value / 2.0

    return {
        "closest_timestamp": idx_closest,
        "best_bid": best_bid,
        "best_ask": best_ask,
        "spread": spread_value,
        "half_spread": half_spread_value
    }

def get_spread_at_time(df: pd.DataFrame, target_time: pd.Timestamp) -> float:
    """
    Finds the row in `df` whose timestamp is closest to target_time
    and returns the spread = best_ask - best_bid from that row.
    
    Args:
        df: DataFrame with DatetimeIndex and best_bid, best_ask columns
        target_time: Target timestamp to find spread for
        
    Returns:
        float: Spread value (best_ask - best_bid)
        
    Raises:
        ValueError: If DataFrame is empty or missing required columns
        TypeError: If price data is not numeric
        KeyError: If required columns are missing
    """
    if df.empty:
        raise ValueError("Cannot calculate spread from empty DataFrame")
    
    if not isinstance(target_time, pd.Timestamp):
        target_time = pd.Timestamp(target_time)

    # Verify required columns exist
    required_columns = {'best_bid', 'best_ask'}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise KeyError(f"Missing required columns for spread calculation: {missing_columns}")

    # Calculate absolute differences using np.abs
    timedeltas = df.index - target_time
    abs_diff = pd.Series(np.abs(timedeltas), index=df.index)

    # Find the index of the closest row(s)
    idx_closest = abs_diff.idxmin()
    rows = df.loc[[idx_closest]]  # Get as DataFrame with single index
    row = rows.iloc[0]  # Take first row if multiple exist

    try:
        best_bid = row['best_bid']
        best_ask = row['best_ask']
        
        # Handle NaN/None values first
        if pd.isna(best_bid) or pd.isna(best_ask) or best_bid is None or best_ask is None:
            raise ValueError(f"Invalid price data: bid={best_bid}, ask={best_ask}")
        
        # Then check numeric types
        if not (np.issubdtype(type(best_bid), np.number) and np.issubdtype(type(best_ask), np.number)):
            raise TypeError(f"Price data must be numeric. Got bid: {type(best_bid)}, ask: {type(best_ask)}")
            
        return float(best_ask - best_bid)
    except (AttributeError, TypeError) as e:
        if "issubdtype" in str(e):
            # If the error is from our type checking, raise the original error
            raise
        raise TypeError(f"Error accessing price data: {str(e)}")

# spread = get_spread_at_time(df1, pd.Timestamp("2024-10-21 12:47:03", tz="UTC"))
# print(f"Spread: {spread}")

def process_minute_wise_data(dfs):
    """
    Calculate the max best bid price across all venues and aggregated minute-level data.
    
    Args:
        dfs: Dictionary mapping venue names to DataFrames
        
    Raises:
        ValueError: If dfs is empty
        TypeError: If any value in dfs is not a DataFrame
    """
    if not dfs:
        raise ValueError("Empty dictionary provided")
        
    # Validate input types
    for venue, df in dfs.items():
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Value for venue {venue} must be a DataFrame, got {type(df)}")

    # Dictionaries to store minute-wise DataFrames
    dfs_minute = {}
    best_bid_series = []

    for venue, df in dfs.items():
        # Validate required columns
        required_columns = {'best_bid', 'bid_sz_00', 'best_ask'}
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            raise KeyError(f"Missing required columns for venue {venue}: {missing_columns}")

        # 'Cluster' the data into 1-min intervals, and shift to get the last value from the previous minute
        resampled_bid = df['best_bid'].resample('1min').last().shift(1)
        resampled_bid_size = df['bid_sz_00'].resample('1min').last().shift(1)
        resampled_ask = df['best_ask'].resample('1min').last().shift(1)

        # Combine resampled data into a single DataFrame for the venue
        df_minute = pd.DataFrame({
            'minute': resampled_bid.index,
            'best_bid': resampled_bid.values,
            'best_bid_size': resampled_bid_size.values,
            'best_ask': resampled_ask.values,
        })

        df_minute.dropna(subset=['minute'], inplace=True)

        # Reset index to ensure 'minute' is a column
        df_minute.reset_index(drop=True, inplace=True)

        # Store the per-venue DataFrame in the dictionary
        dfs_minute[venue] = df_minute

        # Collect Series for aggregated metrics
        best_bid_series.append(resampled_bid.rename(venue))

    # Combine all best bid Series to calculate the maximum best bid across venues
    combined_best_bid = pd.concat(best_bid_series, axis=1)
    max_best_bid_df = pd.DataFrame({
        'minute': combined_best_bid.index,
        'max_best_bid': combined_best_bid.max(axis=1)
    }).reset_index(drop=True)

    # Merge max best bid column back to each venues dataframe
    for venue in dfs_minute:
        dfs_minute[venue] = dfs_minute[venue].merge(max_best_bid_df, how='left', on='minute')

    return dfs_minute

# 2. Calculate parameters

def find_nearest_timestamp(df: pd.DataFrame, target_time: pd.Timestamp) -> pd.Timestamp:
    '''
    Helper function: The nearest timestamp before the target time
    Prevent future data leakage
    '''
    if df.empty:
        return None

    if not isinstance(target_time, pd.Timestamp):
        target_time = pd.Timestamp(target_time)
    nearest_timestamp = df.loc[df['timestamp'] <= target_time, 'timestamp'].max()

    if pd.isna(nearest_timestamp):
        return None

    return nearest_timestamp

def calculate_queue_depth(dfs, target_time) -> dict:
    """
    Calculate the queue depth for the best bid at a specific timestamp.
    """
    if not isinstance(target_time, pd.Timestamp):
        target_time = pd.Timestamp(target_time)

    bid_price = {}
    queue_depth = {}

    for venue, df in dfs.items():
        nearest_timestamp = find_nearest_timestamp(df, target_time)
        if nearest_timestamp is None:
            continue

        row = df.loc[nearest_timestamp]
        # If multiple rows are returned, select the first one.
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]

        bid_price[venue] = row['best_bid']
        queue_depth[venue] = row['bid_sz_00']

    # Determine the maximum best bid among venues.
    max_best_bid = max(list(bid_price.values()))

    # Adjust queue depth: if a venue's bid is less than the max, set its depth to zero.
    for venue in queue_depth:
        # Ensure that bid_price[venue] is a scalar.
        if bid_price[venue] < max_best_bid:
            queue_depth[venue] = 0

    return queue_depth

def calculate_outflow_fills_for_interval(df, best_bid, max_best_bid) -> dict:
    """
    Calculate queue size (Q) and outflow fills (xi) for a single time interval.
    """
    # if data is missing
    if len(df) == 0 or pd.isna(best_bid):
        return {'outflow_fill': 0}

    # Orders ahead of us can leave through:
    # 1. Cancellations at our price OR BETTER
    cancellations = df[(df['price'] == best_bid) & 
                    (df['action'] == 'C')]['size'].sum()
    # 2. Executed orders at our price OR BETTER
    execution = df[#(df['price'] >= best_bid) & 
                (df['action'] == 'T')]['size'].sum()
    outflow_fill = cancellations + execution

    return {'outflow_fill': outflow_fill}


def calculate_outflow_fills_all_intervals(dfs):
    results = {}
    dfs_minute = process_minute_wise_data(dfs)
    
    for venue, df_minute in dfs_minute.items():
        venue_results = []
        venue_df = dfs[venue]
        
        for _, row in df_minute.iterrows():
            interval = row['minute']
            best_bid = row['best_bid']
            max_best_bid = row['max_best_bid']
            
            # Calculate the outflows for just one venue at a time
            interval_data = venue_df[venue_df['minute'] == interval]
            
            result = calculate_outflow_fills_for_interval(
                interval_data, best_bid, max_best_bid
            )
            result['venue'] = venue
            result['minute'] = interval
            venue_results.append(result)
            
        results[venue] = pd.DataFrame(venue_results)
    
    return results


import numpy as np
from scipy.stats import gaussian_kde, expon

def fit_outflows_to_distribution(outflow_fills, method="exp"):
    """
    Fit outflow fills data to a distribution based on the specified method.
    
    Parameters:
        outflow_fills (dict): Dictionary where each key is a venue and each value is a DataFrame 
                              containing an 'outflow_fill' column.
        method (str): The fitting method to use: "kde" for kernel density estimation or "exp" for 
                      an exponential distribution fit.
    
    Returns:
        dict: A dictionary with venues as keys and the fitted distribution parameters as values.
              For method "kde", the value is a dict with keys 'kde', 'bw', and 'R2'.
              For method "exp", the value is a dict with keys 'loc', 'scale', and 'R2'.
    """
    results = {}
    
    for venue, df in outflow_fills.items():
        # Check if the venue data exists (this check is redundant when iterating over keys)
        if venue not in outflow_fills:
            continue

        xi_data = df['outflow_fill']
        
        # Skip empty, all-zero, or invalid data
        if (len(xi_data) == 0 or 
            np.all(xi_data == 0) or 
            np.any(np.isnan(xi_data)) or 
            np.any(np.isinf(xi_data))):
            continue

        try:
            if method == "kde":
                # Create histogram of observed densities
                hist, bin_edges = np.histogram(xi_data, bins=10, density=True)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # Compute bin centers
                
                # Precompute total sum of squares (TSS) from the histogram (independent of bw)
                tss = np.sum((hist - np.mean(hist)) ** 2)
                
                # Define a function to compute R² for a given bandwidth
                def r2_for_bw(bw):
                    kde_candidate = gaussian_kde(xi_data, bw_method=bw)
                    kde_vals = kde_candidate(bin_centers)
                    rss = np.sum((hist - kde_vals) ** 2)
                    return 1 - (rss / tss)
                
                # Ternary search for the optimal bandwidth in [0, 1]
                low, high = 0.0, 1.0
                tol = 1e-3  # stopping tolerance
                
                while high - low > tol:
                    left = low + (high - low) / 3
                    right = high - (high - low) / 3
                    f_left = r2_for_bw(left)
                    f_right = r2_for_bw(right)
                    
                    # Discard the third with the lower R² value
                    if f_left < f_right:
                        low = left
                    else:
                        high = right
                
                optimal_bw = (low + high) / 2
                optimal_r2 = r2_for_bw(optimal_bw)
                kde_optimal = gaussian_kde(xi_data, bw_method=optimal_bw)
                
                results[venue] = {'kde': kde_optimal, 'bw': optimal_bw, 'R2': optimal_r2}
            
            elif method == "exp":
                # Fit exponential distribution
                exp_params = expon.fit(xi_data)  # Returns (loc, scale)
                exp_loc, exp_scale = exp_params
                
                # Skip if parameters are invalid
                if (np.isnan(exp_loc) or np.isnan(exp_scale) or 
                    np.isinf(exp_loc) or np.isinf(exp_scale) or 
                    exp_scale <= 0):
                    continue
                
                # Calculate R² for goodness of fit
                hist, bin_edges = np.histogram(xi_data, bins=50, density=True)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                exp_pdf = expon.pdf(bin_centers, loc=exp_loc, scale=exp_scale)
                
                rss_exp = np.nansum((hist - exp_pdf) ** 2)
                tss = np.nansum((hist - np.nanmean(hist)) ** 2)
                r2_exp = 1 - (rss_exp / tss) if tss != 0 else np.nan
                
                results[venue] = {'loc': exp_loc, 'scale': exp_scale, 'R2': r2_exp}
            
            else:
                raise ValueError("Invalid method specified. Use 'kde' or 'exp'.")
                
        except (ValueError, RuntimeWarning) as e:
            # Log error and skip this venue
            print(f"Warning: Failed to fit distribution for venue {venue}: {str(e)}")
            continue
    
    return results

def generate_xi(outflows, T, method="exp") -> np.ndarray:
    """
    Randomly sample outflow values for T minutes across exchanges.
    
    Depending on the chosen method, it dynamically computes distribution parameters using
    the combined function `fit_outflows_to_distribution`:
      - method="kde": uses kernel density estimation.
      - method="exp": fits an exponential distribution.
    
    Args:
        outflows (dict): Dictionary of DataFrames containing an 'outflow_fill' column for each venue.
        T (int): Number of minutes to sample for.
        method (str): Sampling method to use, either "kde" or "exp".
    
    Returns:
        np.ndarray: Array of sampled outflow values (one per valid venue).
        
    Raises:
        ValueError: If T is negative, non-integer, or if an invalid method is specified.
    """
    # Validate T
    try:
        T = int(T)
        if T < 0:
            raise ValueError("T must be non-negative")
    except (TypeError, ValueError):
        raise ValueError("T must be a non-negative integer")
    
    if method not in ["kde", "exp"]:
        raise ValueError("Invalid method specified. Use 'kde' or 'exp'.")
    
    # Compute distri bution parameters using the combined function
    # This function should be defined elsewhere and accept a `method` parameter.
    distribution_params = fit_outflows_to_distribution(outflows, method=method)
    
    # Special case: if T==0, return zeros for each venue with valid parameters.
    if T == 0:
        return np.zeros(len(distribution_params))
    
    result = []
    # Generate samples for each venue.
    for venue in outflows.keys():
        if venue in distribution_params:
            params = distribution_params[venue]
            
            # For exponential method, ensure parameters are valid.
            if method == "exp":
                if any(np.isnan([params['loc'], params['scale']])) or \
                   any(np.isinf([params['loc'], params['scale']])) or \
                   params['scale'] <= 0:
                    continue
            
            try:
                # Process samples in batches to avoid memory issues.
                batch_size = min(10000, T)
                n_batches = T // batch_size
                remainder = T % batch_size
                sampled_xi = 0.0
                
                if method == "kde":
                    # Use the pre-fitted KDE to generate samples.
                    for _ in range(n_batches):
                        sampled_xi += np.sum(params['kde'].resample(batch_size).flatten())
                    if remainder > 0:
                        sampled_xi += np.sum(params['kde'].resample(remainder).flatten())
                
                elif method == "exp":
                    # Use the exponential parameters to generate samples.
                    for _ in range(n_batches):
                        sampled_xi += np.sum(expon.rvs(
                            loc=params['loc'],
                            scale=params['scale'],
                            size=batch_size
                        ))
                    if remainder > 0:
                        sampled_xi += np.sum(expon.rvs(
                            loc=params['loc'],
                            scale=params['scale'],
                            size=remainder
                        ))
                
                result.append(sampled_xi)
            except (ValueError, MemoryError) as e:
                print(f"Warning: Failed to generate samples for venue {venue}: {str(e)}")
                continue
    
    return np.array(result)

def estimate_adverse_selection(dfs, sec_delta, target_time):
    """Estimate adverse selection impact from price changes after trades."""
    print("\nDEBUG: Starting adverse selection calculation")
    print(f"Target time: {target_time}")
    results = {}
    delta = timedelta(seconds=sec_delta)

    for venue, df in dfs.items():
        print(f"\nProcessing venue: {venue}")
        # Find the execution time of buy limit orders.
        bid_limit_execution = df[(df['action'] == 'T') & (df['side'] == 'A')][['timestamp', 'price']]
        print(f"Found {len(bid_limit_execution)} total trades")
        
        # Remove duplicate timestamps if any.
        bid_limit_execution = bid_limit_execution[~bid_limit_execution.index.duplicated(keep='first')]
        print(f"After removing duplicates: {len(bid_limit_execution)} trades")

        venue_result = []  # To store price differences for this venue
        
        # Get trades in a window around the target minute
        target_minute = pd.Timestamp(target_time).floor('1min')
        window_start = target_minute - pd.Timedelta(minutes=1)
        window_end = target_minute + pd.Timedelta(minutes=1)
        print(f"Looking for trades between {window_start} and {window_end}")
        
        minute_trades = bid_limit_execution[
            (bid_limit_execution.index >= window_start) & 
            (bid_limit_execution.index <= window_end)
        ]
        print(f"Found {len(minute_trades)} trades in window")

        for time in minute_trades.index:
            print(f"\nProcessing trade at {time}")
            target_time = time + delta
            print(f"Looking for future prices at {target_time}")

            # Get the data for the minute corresponding to the target time.
            search_time_1 = df[df['minute'] == target_time.floor('1min')]
            print(f"Found {len(search_time_1)} rows in target minute")


            # Find the row with the closest timestamp to the target.
            row = get_timestamp_values(search_time_1, target_time)
            print(f"Got prices: bid={row['best_bid']}, ask={row['best_ask']}")
            
            # Skip if we don't have valid future prices
            if row['best_bid'] == 0.0 and row['best_ask'] == 0.0:
                print("Skipping due to invalid prices")
                continue

            # Check if the row is actually in the future
            if row['closest_timestamp'] <= time:
                print(f"No future data found: closest timestamp {row['closest_timestamp']} is not after trade time {time}")
                results[venue] = np.nan
                break  # Exit the loop since we won't have future data for other trades either

            price_at_execution = bid_limit_execution.loc[time]['price']
            price_after_execution = (row['best_bid'] + row['best_ask']) / 2
            print(f"Price difference: {price_after_execution - price_at_execution}")
            venue_result.append(price_after_execution - price_at_execution)

        if venue_result:
            results[venue] = np.mean(venue_result)
            print(f"\nVenue {venue} result: {results[venue]}")
        else:
            results[venue] = np.nan
            print(f"\nVenue {venue} result: NaN (no valid price differences)")

    return results

def calculate_filled_amounts(Q: np.ndarray, L: np.ndarray, xi: np.ndarray) -> np.ndarray:
    """
    Calculate filled amounts for limit orders based on queue outflows.
    
    Args:
        Q: Queue sizes at each exchange
        L: Limit order sizes at each exchange
        xi: Order outflows at each exchange
        
    Returns:
        np.ndarray: Amount filled at each exchange
        
    Raises:
        ValueError: If arrays have different shapes or contain invalid values
    """
    # Convert inputs to numpy arrays if they aren't already
    Q = np.asarray(Q, dtype=np.float64)
    L = np.asarray(L, dtype=np.float64)
    xi = np.asarray(xi, dtype=np.float64)
    
    # Check dimensions
    if not (Q.ndim == L.ndim == xi.ndim == 1):
        raise ValueError("All input arrays must be 1-dimensional")
    
    # Check shapes match
    if not (Q.shape == L.shape == xi.shape):
        raise ValueError(f"All input arrays must have the same shape. Got shapes: Q: {Q.shape}, L: {L.shape}, xi: {xi.shape}")
    
    # Check for NaN/Inf in all values
    if np.any(np.isnan([Q, L, xi])) or np.any(np.isinf([Q, L, xi])):
        raise ValueError("Input arrays cannot contain NaN or Inf values")

    return np.maximum(xi - Q, 0) - np.maximum(xi - Q - L, 0)


def cost_func(X, xi, params) -> float:
    """
    Calculate the cost v(X,ξ) as defined in equation (5) of the paper.
    Parameters:
    X: Order allocation vector (M, L1,...,LK)
        M: Market order size
        L1...LK: Limit order sizes for K exchanges
    xi: Order outflow vector (ξ1,...,ξK)
    params: Dictionary containing market parameters
    Returns:
    float: Total cost
    
    Raises:
        ValueError: If input arrays contain NaN or Inf values
    """
    M = X[0]  # Market order size
    L = X[1:]  # Limit order sizes
    
    if np.any(np.isnan(M)) or np.any(np.isinf(M)) or np.any(np.isnan(L)) or np.any(np.isinf(L)):
        raise ValueError("Input arrays cannot contain NaN or Inf values")
    
    # Handle empty xi array
    if xi.size == 0:
        xi = np.zeros_like(L)
    
    # Calculate total filled amounts for limit orders
    filled_amounts = calculate_filled_amounts(params['Q'], L, xi)
    total_filled = M + np.sum(filled_amounts)  # Function A of the Paper

    S = params['S']
    # Calculate explicit trading costs
    market_cost = (params['h'] + params['f']) * M
    limit_cost = -np.sum((params['h'] + params['r']) * filled_amounts)

    # Calculate market impact
    total_orders = M + np.sum(L)
    catch_up = max(S - total_filled, 0)  # Additional market orders needed
    impact = params['theta'] * (total_orders + catch_up)

    # Calculate shortfall penalties
    underfill = params['lambda_u'] * max(S - total_filled, 0)
    overfill = params['lambda_o'] * max(total_filled - S, 0)
    total_cost = market_cost + limit_cost + impact + underfill + overfill
    return total_cost

# 3. Optimize
class StochasticOrderOptimizer:
    def __init__(self, params: Dict, n_exchanges: int):
        """
        Initialize optimizer with market parameters.
        Args:
            params: Dictionary containing market parameters (h, f, r, theta, etc.)
            n_exchanges: Number of trading venues
        """
        self.params = params
        self.n_exchanges = n_exchanges
        self.dimension = n_exchanges + 1  # M + L1,...,LK
        self.xi_estimates = None  # Will store regression-based xi estimates

    def update_xi_estimates(self, dfs: Dict[str, pd.DataFrame], target_time: pd.Timestamp):
        """
        Update xi estimates using regression on OFI.
        """
        self.xi_estimates = estimate_xi_from_ofi(
            dfs, target_time,
            window_minutes=5,
            bin_size=10
        )

    def calculate_gradient(self, X: np.ndarray, xi: np.ndarray) -> np.ndarray:
        """
        Calculate gradient g(X,ξ) = ∇v(X,ξ) as described in the paper.
        Uses regression-based xi estimates if available.
        """
        # Handle empty xi array
        if xi.size == 0:
            # Use zeros for outflows when no valid xi data
            xi = np.zeros(self.n_exchanges)
        
        M = float(X[0])
        L = X[1:].astype(float)
        Q = self.params['Q'].astype(float)

        # Calculate filled amounts and total execution
        filled_amounts = calculate_filled_amounts(Q, L, xi)
        total_filled = M + np.sum(filled_amounts)
        
        indicator_shortfall = float(total_filled < self.params['S'])
        indicator_surplus = float(total_filled > self.params['S'])
        
        gradient = np.zeros(self.dimension, dtype=float)
        
        # Gradient component for market orders (M)
        gradient[0] = (self.params['h'] + self.params['f'] + self.params['theta']
                      - (self.params['lambda_u'] + self.params['theta']) * indicator_shortfall
                      + self.params['lambda_o'] * indicator_surplus)

        # Gradient components for limit orders (L1,...,LK)
        for k in range(self.n_exchanges):
            # Use regression-based xi estimate if available
            if self.xi_estimates is not None:
                venue = list(self.params['outflows'].keys())[k]
                xi_estimate, reg_coef = self.xi_estimates.get(venue, (0.0, 0.0))
                # Adjust xi based on regression coefficient
                adjusted_xi = xi[k] * (1 + reg_coef)
            else:
                adjusted_xi = xi[k]
                
            indicator_isfilled = 1. if (adjusted_xi > self.params['Q'][k] + L[k]) else 0.
            gradient[k + 1] = (self.params['theta'] + indicator_isfilled
                             * ((-self.params['h'] - self.params['r'][k]) - (
                                self.params['lambda_u'] + self.params['theta'])
                                * indicator_shortfall + self.params['lambda_o'] * indicator_surplus))
        return gradient

    def compute_step_size(self, N: int) -> float:
        """
        Compute step size γ as described in Section 4.1 of the paper.
        Args:
            N: Total number of optimization steps
        Return:
            gamma: step size
        """
        K = self.n_exchanges
        h = self.params['h']
        f = self.params['f']
        theta = self.params['theta']
        lambda_u = self.params['lambda_u']
        lambda_o = self.params['lambda_o']
        r = self.params['r']  # multi-dimensional
        S = self.params['S']

        first_term = N * (h + f + theta + lambda_u + lambda_o) ** 2
        second_term = 0
        for rk in r:
            second_term += (N * (h + rk + theta + lambda_u + lambda_o) ** 2)

        gamma = (K ** 0.5 * S) / np.sqrt(
            first_term + second_term
        )
        return gamma

    def project_to_feasible(self, X: np.ndarray) -> np.ndarray:
        """
        Project X onto the feasible set C defined in Proposition 1.
        C = {X ∈ R^(K+1)₊ | 0 ≤ M ≤ S, 0 ≤ Lₖ ≤ S - M, k = 1,...,K, M + ΣLₖ ≥ S}
        Args:
            X: numpy array version of our vector X
        """
        M = X[0]
        L = X[1:]
        S = self.params['S']
        K = len(L)

        # Enforce non-negativity
        M = max(0, M)
        L = np.maximum(L, 0)

        # Constraint 1: 0 <= M <= S
        M = min(M, S)

        # Constraint 2: 0 <= Lₖ <= S - M
        L = np.minimum(L, S - M)

        # Constraint 3: M + ΣLₖ >= S
        total = M + np.sum(L)
        if total < S:
            if total == 0:
                # Special case: if total is zero, distribute S evenly
                if K > 0:
                    # Allocate evenly between limit orders
                    L = np.array([S / K] * K)
                    M = 0
                else:
                    # If no limit orders possible, put everything in market order
                    M = S
            else:
                # Scale up proportionally to meet S
                scale = S / total
                M = min(M * scale, S)
                remaining = S - M
                if np.sum(L) > 0:  # Avoid divide by zero
                    L = L * (remaining / np.sum(L))  # Scale limit orders to fill remaining amount
                else:
                    M = S  # If no limit orders, put everything in market order

        return np.concatenate(([M], L))

    def optimize(self, N: int = 1000, method="exp") -> Tuple[np.ndarray, float]:
        """
        Implement the stochastic approximation algorithm from Section 4 of the paper.
        Args:
            N: Number of iterations
        Returns:
            Tuple of (optimal allocation X*, estimated optimal value V(X*))
        """
        X = np.ones(self.dimension) * self.params['S'] / self.dimension
        X_sum = np.zeros_like(X)
        for n in range(1, N + 1):
            # Generate random order flow
            xi = generate_xi(self.params['outflows'], self.params['T'], method=method)
            # Compute gradient
            gradient = self.calculate_gradient(X, xi)
            # Compute step size
            gamma = self.compute_step_size(N)
            X = X - gamma * gradient
            X = self.project_to_feasible(X)
            X_sum += X

        # Return averaged solution
        X_star = X_sum / N
        # Estimate optimal value using last 100 iterations
        V_star = 0
        for _ in range(100):
            xi = generate_xi(self.params['outflows'], self.params['T'])
            V_star += cost_func(X_star, xi, self.params)
        V_star /= 100
        return X_star, V_star
    
class InsufficientDataError(ValueError):
    """Raised when there is insufficient data to perform a calculation"""
    pass

def compute_queue_and_volume_size(dfs):
    results = {}
    dfs_minute = process_minute_wise_data(dfs)
    for venue, df_minute in dfs_minute.items():
        venue_results = []
        venue_df = dfs[venue]
        for _, row in df_minute.iterrows():
            interval = row['minute']
            # Calculate the queue size for just one venue at a time
            interval_data = venue_df[venue_df['minute'] == interval]
            result = compute_queue_and_volume_size_interval(interval_data)
            result['venue'] = venue
            result['minute'] = interval
            venue_results.append(result)
            
        results[venue] = pd.DataFrame(venue_results)
    
    return results
    
def compute_queue_and_volume_size_interval(df) -> dict:
    # if data is missing
    if len(df) == 0:
        return {'queue_size': 0}

    queue = df['size'].sum()
    volume = df[(df['action'] == 'T')]['size'].sum()

    return {'queue': queue, 'volume': volume}

def get_queue_volume_for_minute(df: pd.DataFrame, minute_val: pd.Timestamp) -> dict:
    """
    Computes the aggregated queue and trading volume for the specified minute.
    It filters df for rows where the 'minute' column equals minute_val,
    then sums up 'size' (for queue) and the 'size' of rows where action=='T' (for volume).
    Returns a dict or None if no data is found.
    """
    df_min = df[df['minute'] == minute_val]
    if df_min.empty:
        return None
    total_queue = df_min['size'].sum()
    total_volume = df_min[df_min['action'] == 'T']['size'].sum()
    return {'queue': total_queue, 'volume': total_volume}

def get_bucket(value, thresholds):
    low, high = thresholds
    if value < low:
        return "Low"
    elif value > high:
        return "High"
    else:
        return "Medium"

def generate_lookup_table(dfs, target_time, S, T, f, r, lambda_u, lambda_o, N, method, stock):
    """
    Builds a lookup table for future use.
    
    Steps:
      1. For each venue, compute its per-minute queue/volume data (via compute_queue_and_volume_size)
         and then compute its critical thresholds (33% and 66% quantiles).
      2. For the union of minutes across venues, compute for each minute the bucket of each venue
         (using that venue's own thresholds). For one venue "v1", a bucket might be "v1:low_low"
         (meaning both queue and volume are low).
      3. Group consecutive minutes with the same composite key.
      4. For each group (segment), filter the full dfs to those minutes and run run_optimization_facade
         to obtain a solution vector; compute its ratio (solution divided by sum(solution)).
      5. Average the ratio vectors for segments with the same composite key.
      6. Also store the per-venue thresholds (so that later, in lookup mode, we can determine the key).
      7. Save the lookup table (a dict mapping composite keys to a dict with keys "avg_ratio" and "thresholds")
         to a pickle file.
    """
    # (1) Compute thresholds per venue.
    thresholds_dict = {}
    for venue, df in dfs.items():
        qv_data = compute_queue_and_volume_size({venue: df})[venue]
        # Compute critical thresholds from the historical aggregated data.
        q_thresholds = qv_data['queue'].quantile([0.33, 0.66]).values
        v_thresholds = qv_data['volume'].quantile([0.33, 0.66]).values
        thresholds_dict[venue] = {
            "queue_thresholds": {"low_to_medium": q_thresholds[0], "medium_to_high": q_thresholds[1]},
            "volume_thresholds": {"low_to_medium": v_thresholds[0], "medium_to_high": v_thresholds[1]}
        }

    # (2) Compute the union of all minutes.
    all_minutes = set()
    for df in dfs.values():
        all_minutes.update(df['minute'].unique())
    all_minutes = sorted(list(all_minutes))
    
    # For each minute, form a composite key (for each venue, get its bucket based on that minute's queue & volume).
    # Here the bucket string for a venue is: "venue:queueBucket_volumeBucket", e.g., "v1:low_low".
    composite_keys = {}  # mapping minute -> composite key string
    for m in all_minutes:
        key_parts = []
        for venue, df in dfs.items():
            qv = get_queue_volume_for_minute(df, m)
            # If no data for this minute for a venue, skip this minute entirely.
            if qv is None:
                key_parts.append(f"{venue}:NA")
                continue
            th = thresholds_dict[venue]
            q_bucket = get_bucket(qv['queue'], (th["queue_thresholds"]["low_to_medium"], th["queue_thresholds"]["medium_to_high"]))
            v_bucket = get_bucket(qv['volume'], (th["volume_thresholds"]["low_to_medium"], th["volume_thresholds"]["medium_to_high"]))
            key_parts.append(f"{venue}:{q_bucket.lower()}_{v_bucket.lower()}")
        composite_keys[m] = "|".join(key_parts)
    
    # (3) Group consecutive minutes that share the same composite key.
    segments = []  # list of tuples: (composite_key, list of minutes)
    current_key = None
    current_segment = []
    prev_minute = None
    for m in all_minutes:
        key = composite_keys[m]
        if current_key is None:
            current_key = key
            current_segment = [m]
            prev_minute = m
        else:
            # If the key is the same and m is continuous (<=1 minute gap), extend the segment.
            if key == current_key and (m - prev_minute) <= pd.Timedelta(minutes=1):
                current_segment.append(m)
            else:
                segments.append((current_key, current_segment))
                current_key = key
                current_segment = [m]
            prev_minute = m
    if current_segment:
        segments.append((current_key, current_segment))
    
    # (4) For each segment, filter full data (dfs) to rows where 'minute' is in the segment,
    # then run run_optimization_facade to obtain a solution vector, then compute its ratio.
    segment_results = {}  # mapping composite key -> list of ratio vectors
    for comp_key, minutes in segments:
        # Filter each venue's DataFrame to rows with minute in this segment.
        filtered_dfs = {}
        for venue, df in dfs.items():
            filtered_dfs[venue] = df[df['minute'].isin(minutes)]
        if all(df_seg.empty for df_seg in filtered_dfs.values()):
            continue
        try:
            sol, _ = run_optimization_facade(
                filtered_dfs,
                target_time,
                S=S,
                T=T,
                f=f,
                r=r,
                lambda_u=lambda_u,
                lambda_o=lambda_o,
                N=N,
                method=method
            )
        except InsufficientDataError as e:
            print(f"[WARNING] Skipping minute: {str(e)}")
            continue
        if sol is None:
            continue
        total = np.sum(sol)
        if total != 0:
            ratio = sol / total
        else:
            ratio = np.zeros_like(sol)
        segment_results.setdefault(comp_key, []).append(ratio)
    
    # (5) Average ratio vectors for each composite key.
    lookup_table = {}
    for comp_key, ratios in segment_results.items():
        avg_ratio = np.mean(np.vstack(ratios), axis=0) if ratios else None
        lookup_table[comp_key] = {"avg_ratio": avg_ratio, "thresholds": thresholds_dict}
    
    # (6) Save the lookup table to file.
    filename = f"ratio_table_{stock}.pkl"
    with open(filename, "wb") as f_out:
        pickle.dump(lookup_table, f_out)
    print(f"Saved lookup table to {filename}")
    # (Optionally display the lookup table)
    for key, entry in lookup_table.items():
        print(f"Key {key}: {entry}")
    return lookup_table

def load_ratio_table(file_path):
    """
    Load the ratio table from a pickle file and print its contents.
    """
    with open(file_path, "rb") as f_in:
        ratio_table = pickle.load(f_in)
    print("Loaded Ratio Table:")
    for key, value in ratio_table.items():
        print(f"{key}: {value}")
    return ratio_table

def run_optimization_facade(
    dfs: Dict[str, pd.DataFrame],
    target_time: str,
    *,  # Force keyword arguments after this point
    S: float,
    T: int,
    f: float,
    r: List[float],
    lambda_u: float,
    lambda_o: float,
    N: int = 1000,
    method: str = "exp",
    use_ofi_signal: bool = False
) -> Tuple[np.ndarray, float]:
    """
    Facade for running the stochastic optimization.
    
    Args:
        dfs: Dictionary of venue DataFrames
        target_time: Target timestamp for optimization
        S: Target order size
        T: Time horizon
        f: Fixed fee
        r: List of rebates per venue
        lambda_u: Underfill penalty
        lambda_o: Overfill penalty
        N: Number of iterations (default=1000)
        method: Optimization method (default="exp")
        use_ofi_signal: Whether to use OFI signal in optimization
        
    Returns:
        Tuple of (optimal allocation X*, optimal value V*)
    """
    # Validate input parameters
    if not isinstance(dfs, dict):
        raise TypeError("dfs must be a dictionary of DataFrames")
    
    if not dfs:
        raise ValueError("dfs dictionary cannot be empty")
        
    for venue, df in dfs.items():
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Value for venue {venue} must be a DataFrame")
        if df.empty:
            raise ValueError(f"DataFrame for venue {venue} cannot be empty")
        required_columns = {'timestamp', 'best_bid', 'best_ask', 'bid_sz_00', 'price', 'action', 'side', 'size'}
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            raise ValueError(f"DataFrame for venue {venue} is missing required columns: {missing_columns}")

    if not isinstance(S, (int, float)) or S <= 0:
        raise ValueError("Target size (S) must be a positive number")

    if not isinstance(T, int) or T <= 0:
        raise ValueError("Time horizon (T) must be a positive integer")

    if not isinstance(f, (int, float)) or f < 0:
        raise ValueError("Market order fee (f) must be a non-negative number")

    if not isinstance(r, (list, np.ndarray)):
        raise TypeError("Rebates (r) must be a list or numpy array")
    r = np.asarray(r)
    if len(r) != len(dfs):
        raise ValueError(f"Length of rebates array ({len(r)}) must match number of venues ({len(dfs)})")
    if np.any(r < 0):
        raise ValueError("All rebates must be non-negative")

    if not isinstance(lambda_u, (int, float)) or lambda_u < 0:
        raise ValueError("Underfill penalty (lambda_u) must be a non-negative number")

    if not isinstance(lambda_o, (int, float)) or lambda_o < 0:
        raise ValueError("Overfill penalty (lambda_o) must be a non-negative number")

    if not isinstance(N, int) or N <= 0:
        raise ValueError("Number of iterations (N) must be a positive integer")
    
    # Convert target_time to Timestamp.
    try:
        if not isinstance(target_time, pd.Timestamp):
            target_time = pd.Timestamp(target_time)
    except (ValueError, TypeError):
        raise ValueError("Invalid timestamp format")
    # Filter each venue's DataFrame to only include rows at or before target_time.
    valid_dfs = {}
    for venue, df in dfs.items():
        print(target_time)
        df_valid = df[df['timestamp'] <= target_time]
        if df_valid.empty:
            print(f"Venue {venue}: no data available before target time. Skipping optimization for this venue.")
            continue
        valid_dfs[venue] = df_valid

    if not valid_dfs:
        print("No valid timestamp found before target time for any venue. Stopping optimization.")
        return (None, None)
    
    # Find the nearest timestamp across the filtered data.
    nearest_timestamps = []
    for df in valid_dfs.values():
        nearest_ts = find_nearest_timestamp(df, target_time)
        if nearest_ts is not None:
            nearest_timestamps.append(nearest_ts)
    
    if not nearest_timestamps:
        print("No valid timestamp found before target time after filtering. Stopping optimization.")
        return (None, None)
    # Use the latest valid timestamp among venues.
    target_time = max(nearest_timestamps)
    # Step 2: Compute queue sizes (Q) at the given timestamp
    queue_depth = calculate_queue_depth(valid_dfs, target_time)
    Q = np.array([queue_depth[venue] if venue in queue_depth else 0 for venue in valid_dfs.keys()])
    
    # Step 3: Compute half-spread (h) - average across venues
    spreads = []
    for df in valid_dfs.values():
        spread = get_spread_at_time(df, target_time)
        if spread > 0:
            spreads.append(spread)
    h = np.mean(spreads) / 2 if spreads else 0.0
    
    # Step 4: Compute outflows based on historical data
    outflows = calculate_outflow_fills_all_intervals(valid_dfs)
    
    # Validate outflows data and ensure each venue has at least one value.
    for venue in valid_dfs.keys():
        if venue not in outflows or outflows[venue].empty:
            outflows[venue] = pd.DataFrame({'outflow_fill': [0.0]})
        elif 'outflow_fill' not in outflows[venue].columns:
            outflows[venue]['outflow_fill'] = 0.0
        if outflows[venue]['outflow_fill'].isnull().all():
            outflows[venue]['outflow_fill'] = outflows[venue]['outflow_fill'].fillna(0.0)
    
    # Step 5: Handle adverse selection (this example assumes it uses valid_dfs)
    r = np.asarray(r, dtype=float)
    adverse_selection = estimate_adverse_selection(valid_dfs, sec_delta=10, target_time=target_time)
    if any(np.isnan(val) for val in adverse_selection.values()):
        raise InsufficientDataError("Unable to calculate adverse selection - insufficient future price data")
    adverse_selection_array = np.array([adverse_selection.get(venue, 0.0) for venue in valid_dfs.keys()], dtype=float)
    r = r + adverse_selection_array
    
    # Step 6: Define market parameters.
    params = {
        'S': S,
        'T': T,
        'h': h,
        'f': f,
        'r': r,
        'theta': f * 0.1,
        'lambda_u': lambda_u,
        'lambda_o': lambda_o,
        'Q': Q,
        'outflows': outflows
    }
    
    # Step 7: Initialize optimizer and run optimization.
    optimizer = StochasticOrderOptimizer(params, n_exchanges=len(valid_dfs))
    
    # Update xi estimates using regression on OFI if use_ofi_signal is True
    if use_ofi_signal:
        optimizer.update_xi_estimates(valid_dfs, target_time)
    
    X_star, V_star = optimizer.optimize(N=N, method=method)
    print(X_star)
    return (X_star, V_star)

def execute_optimization(
    dfs: Dict[str, pd.DataFrame],
    target_time: str,
    *,
    S: float,
    T: int,
    f: float,
    r: List[float],
    lambda_u: float,
    lambda_o: float,
    N: int = 1000,
    method: str = "exp",
    stock: str = None,  # required for lookup mode
    alpha: float = 0.1,  # OFI adjustment parameter
    use_ofi_signal: bool = False  # whether to use OFI as signal
) -> tuple:
    """
    If method=="lookup", loads the lookup table from file (using the stock symbol), then:
      - For each venue, computes the queue and volume for the minute immediately before target_time 
        (using get_queue_volume_for_minute).
      - Determines each venue's bucket (using that venue's stored thresholds from the lookup table).
      - Forms a composite key by concatenating each venue's bucket (e.g. for one venue, "v1:low_low").
      - Looks up that composite key in the table and returns its associated avg_ratio vector.
      - If use_ofi_signal is True, adjusts the allocation based on OFI signal.
    Otherwise, calls run_optimization_facade.
    """
    target_time = pd.Timestamp(target_time)
    if method == "lookup":
        if stock is None:
            raise ValueError("Stock symbol must be provided when using the lookup method.")
        filename = f"ratio_table_{stock}.pkl"
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Lookup table {filename} does not exist.")
        with open(filename, "rb") as f_in:
            lookup_table = pickle.load(f_in)
        print(f"Loaded lookup table from {filename}")
        # Compute the minute immediately before target_time.
        minute_before = (target_time - pd.Timedelta(minutes=1)).floor('T')
        key_parts = []
        # For each venue, compute queue & volume for minute_before and determine bucket.
        # Here we use the thresholds stored in the lookup table (they are the same for all keys per venue).
        # We pick one arbitrary entry (any composite key) to get the thresholds.
        sample_entry = None
        for key, entry in lookup_table.items():
            if entry is not None and "thresholds" in entry:
                sample_entry = entry["thresholds"]
                break
        if sample_entry is None:
            raise ValueError("No threshold information found in the lookup table.")
        
        for venue, df in dfs.items():
            qv = get_queue_volume_for_minute(df, minute_before)
            if qv is None:
                raise ValueError(f"No queue/volume data available for venue {venue} at minute {minute_before}")
            th = sample_entry[venue]  # get this venue's thresholds
            q_bucket = get_bucket(qv['queue'], (th["queue_thresholds"]["low_to_medium"], th["queue_thresholds"]["medium_to_high"]))
            v_bucket = get_bucket(qv['volume'], (th["volume_thresholds"]["low_to_medium"], th["volume_thresholds"]["medium_to_high"]))
            key_parts.append(f"{venue}:{q_bucket.lower()}_{v_bucket.lower()}")
        composite_key = "|".join(key_parts)
        if composite_key not in lookup_table or lookup_table[composite_key]["avg_ratio"] is None:
            raise ValueError(f"Lookup table entry {composite_key} not found or is None.")
        base_ratio = lookup_table[composite_key]["avg_ratio"]
        
        if use_ofi_signal:
            # Calculate OFI signal for the target time window
            try:
                ofi_signal = calculate_causal_ofi_signal(
                    dfs, target_time,
                    window_minutes=5,
                    bin_size=10,
                    alpha=alpha
                )
                
                # Adjust the base ratio based on OFI signal
                if ofi_signal:
                    # Calculate adjustment factors based on OFI
                    adjustment_factors = {}
                    for venue in dfs.keys():
                        venue_ofi = ofi_signal.get(venue, 0)
                        # Adjust ratio based on OFI (positive OFI -> increase allocation)
                        adjustment_factors[venue] = 1 + alpha * venue_ofi
                    
                    # Apply adjustments while maintaining sum to 1
                    adjusted_ratio = base_ratio * np.array([adjustment_factors.get(venue, 1) for venue in dfs.keys()])
                    adjusted_ratio = adjusted_ratio / np.sum(adjusted_ratio)  # Renormalize
                    
                    print(f"Base ratio: {base_ratio}")
                    print(f"OFI signal: {ofi_signal}")
                    print(f"Adjusted ratio: {adjusted_ratio}")
                    return (adjusted_ratio * S, None)
            except Exception as e:
                print(f"Warning: Failed to calculate OFI signal: {str(e)}")
                print("Using base ratio without OFI adjustment")
        
        return (base_ratio * S, None)
    else:
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
            use_ofi_signal=use_ofi_signal
        )