# Helper Functions for OFI calculation
import pandas as pd
import numpy as np
from scipy.stats import expon, gaussian_kde, pearsonr
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Dict, Tuple, List
import os

def calculate_bid_order_flow(df, level):
    """Calculate bid order flow"""
    if level == 1:
        price_col = 'best_bid'
        qty_col = 'bid_sz_00'
    else:
        price_col = f'bid_px_{level-1:02d}'
        qty_col = f'bid_sz_{level-1:02d}'
    
    # Calculate price and quantity deltas
    price_delta = df[price_col].diff()
    qty_delta = df[qty_col].diff()
    
    # Create order flow series with same logic as PySpark
    order_flow = pd.Series(index=df.index, dtype=float)
    order_flow = np.where(price_delta > 0, df[qty_col],
                  np.where(price_delta < 0, -df[qty_col].shift(),
                  qty_delta))
    
    # Set first row to 0
    df['bid_of'] = order_flow
    df.loc[df.index[0], 'bid_of'] = 0
    
    return df

def calculate_ask_order_flow(df, level):
    """Calculate ask order flow"""
    if level == 1:
        price_col = 'best_ask'
        qty_col = 'ask_sz_00'
    else:
        price_col = f'ask_px_{level-1:02d}'
        qty_col = f'ask_sz_{level-1:02d}'
    
    # Calculate price and quantity deltas
    price_delta = df[price_col].diff()
    qty_delta = df[qty_col].diff()
    
    # Create order flow series with same logic as PySpark
    order_flow = pd.Series(index=df.index, dtype=float)
    order_flow = np.where(price_delta > 0, -df[qty_col],
                  np.where(price_delta < 0, df[qty_col].shift(),
                  -qty_delta))
    
    # Set first row to 0
    df['ask_of'] = order_flow
    df.loc[df.index[0], 'ask_of'] = 0
    
    return df

def calculate_level_OFI(df, level):
    """Calculate Order Flow Imbalance (OFI) for a specific level"""
    # Get bid and ask order flows
    df = calculate_bid_order_flow(df, level)
    df = calculate_ask_order_flow(df, level)
    
    # Calculate OFI as bid OF - ask OF
    df[f"OFI_L{level}"] = df['bid_of'] - df['ask_of']
    
    # Drop intermediate columns
    df = df.drop(columns=['bid_of', 'ask_of'])
    
    return df


def calculate_ofi_signal(dfs, target_time, window_minutes=5, levels=1):
    """
    Calculate OFI signal from order book data.
    
    Args:
        dfs: Dictionary of venue DataFrames
        target_time: Target timestamp for optimization
        window_minutes: Window size for OFI calculation in minutes
        levels: Number of order book levels to consider for OFI
        
    Returns:
        Dictionary of normalized OFI values per venue
    """
    ofi_values = {}
    
    for venue, df in dfs.items():
        # print(venue, df)
        # Filter data for the recent window
        window_start = target_time - pd.Timedelta(minutes=window_minutes)
        window_df = df[(df['timestamp'] >= window_start) & (df['timestamp'] <= target_time)].copy()
        
        if window_df.empty:
            print(f"Warning: No data available for OFI calculation at venue {venue}")
            ofi_values[venue] = 0
            continue
            
        # Calculate OFI for each level and sum them
        venue_ofi = 0
        for level in range(1, levels + 1):
            # Calculate OFI for this level without renaming columns
            if level == 1:
                level_bid_price = 'best_bid'
                level_ask_price = 'best_ask'
            else:
                level_bid_price = f'bid_px_{level-1:02d}'
                level_ask_price = f'ask_px_{level-1:02d}'
            
            # Calculate OFI for this level
            window_df = calculate_level_OFI(window_df, level)
            
            # Sum the OFI values
            level_ofi = window_df[f"OFI_L{level}"].sum()
            venue_ofi += level_ofi
        
        # Store OFI value for this venue
        ofi_values[venue] = venue_ofi
    
    # Normalize OFI values
    if ofi_values:
        max_abs_ofi = max(abs(ofi) for ofi in ofi_values.values())
        if max_abs_ofi > 0:
            normalized_ofi = {venue: ofi/max_abs_ofi for venue, ofi in ofi_values.items()}
        else:
            normalized_ofi = {venue: 0 for venue in ofi_values}
    else:
        normalized_ofi = {}
        
    return normalized_ofi

def calculate_metrics_with_bin(df: pd.DataFrame, bin_size: int) -> pd.DataFrame:
    """
    Calculate price variation and order flow metrics based on bin size.
    
    Args:
        df: DataFrame with price and order flow data
        bin_size: Size of time bin in seconds
        
    Returns:
        DataFrame with calculated metrics
    """
    # Ensure timestamp is datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Calculate price variation
    df['price_variation'] = df['best_ask'] - df['best_bid']
    
    # Calculate order flow
    df['order_flow'] = df['bid_sz_00'] - df['ask_sz_00']
    
    # Resample to bins
    metrics = df.resample(f'{bin_size}S', on='timestamp').agg({
        'price_variation': ['mean', 'std'],
        'order_flow': ['mean', 'std']
    })
    
    # Flatten column names
    metrics.columns = ['_'.join(col).strip() for col in metrics.columns.values]
    
    return metrics

def build_correlation_matrix(metrics: Dict[str, pd.DataFrame], column: str) -> pd.DataFrame:
    """
    Build correlation matrix across venues for a specific metric.
    
    Args:
        metrics: Dictionary of venue DataFrames with calculated metrics
        column: Column name to calculate correlations for
        
    Returns:
        DataFrame containing correlation matrix
    """
    venues = list(metrics.keys())
    n_venues = len(venues)
    corr_matrix = np.zeros((n_venues, n_venues))
    
    for i in range(n_venues):
        for j in range(i+1):
            if i == j:
                corr_matrix[i,j] = 1.0
            else:
                venue1, venue2 = venues[i], venues[j]
                corr, _ = pearsonr(
                    metrics[venue1][column].dropna(),
                    metrics[venue2][column].dropna()
                )
                corr_matrix[i,j] = corr_matrix[j,i] = corr
                
    return pd.DataFrame(corr_matrix, index=venues, columns=venues)

def estimate_xi_from_ofi(
    dfs: Dict[str, pd.DataFrame],
    target_time: pd.Timestamp,
    window_minutes: int = 5,
    bin_size: int = 10
) -> Dict[str, Tuple[float, float]]:
    """
    Estimate xi (outflow fills) from OFI using regression.
    
    Args:
        dfs: Dictionary of venue DataFrames
        target_time: Target timestamp for estimation
        window_minutes: Window size in minutes
        bin_size: Size of time bin in seconds
        
    Returns:
        Dictionary mapping venues to (xi_estimate, regression_coef) tuples
    """
    # Filter data for recent window
    window_start = target_time - pd.Timedelta(minutes=window_minutes)
    filtered_dfs = {
        venue: df[
            (df['timestamp'] >= window_start) & 
            (df['timestamp'] <= target_time)
        ].copy() for venue, df in dfs.items()
    }
    
    # Calculate metrics for each venue
    metrics = {
        venue: calculate_metrics_with_bin(df, bin_size)
        for venue, df in filtered_dfs.items()
    }
    
    xi_estimates = {}
    
    for venue in dfs.keys():
        try:
            # Prepare data for regression
            X = metrics[venue]['order_flow_mean'].values.reshape(-1, 1)
            y = filtered_dfs[venue]['bid_sz_00'].values
            
            # Fit linear regression
            from sklearn.linear_model import LinearRegression
            reg = LinearRegression()
            reg.fit(X, y)
            
            # Get current OFI
            current_ofi = metrics[venue]['order_flow_mean'].iloc[-1]
            
            # Predict xi using current OFI
            current_xi = reg.predict([[current_ofi]])[0]
            xi_estimates[venue] = (max(0, current_xi), reg.coef_[0])
        except Exception as e:
            print(f"Warning: Failed to estimate xi for venue {venue}: {str(e)}")
            xi_estimates[venue] = (0.0, 0.0)
    
    return xi_estimates

def calculate_causal_ofi_signal(
    dfs: Dict[str, pd.DataFrame],
    target_time: pd.Timestamp,
    window_minutes: int = 5,
    bin_size: int = 10,
    alpha: float = 0.1
) -> Dict[str, float]:
    """
    Calculate OFI signal using causal inference approach.
    
    Args:
        dfs: Dictionary of venue DataFrames
        target_time: Target timestamp for OFI calculation
        window_minutes: Window size in minutes
        bin_size: Size of time bin in seconds
        alpha: OFI impact factor
        
    Returns:
        Dictionary of OFI signals per venue
    """
    # Filter data for recent window
    window_start = target_time - pd.Timedelta(minutes=window_minutes)
    filtered_dfs = {
        venue: df[
            (df['timestamp'] >= window_start) & 
            (df['timestamp'] <= target_time)
        ].copy() for venue, df in dfs.items()
    }
    
    # Calculate metrics for each venue
    metrics = {
        venue: calculate_metrics_with_bin(df, bin_size)
        for venue, df in filtered_dfs.items()
    }
    
    # Build correlation matrices
    price_corr = build_correlation_matrix(metrics, 'price_variation_mean')
    flow_corr = build_correlation_matrix(metrics, 'order_flow_mean')
    
    # Calculate causal OFI signals
    ofi_signals = {}
    
    for venue in dfs.keys():
        # Calculate direct OFI
        venue_metrics = metrics[venue]
        direct_ofi = venue_metrics['order_flow_mean'].mean()
        
        # Calculate indirect OFI through price correlations
        indirect_ofi = 0
        for other_venue in dfs.keys():
            if other_venue != venue:
                corr = price_corr.loc[venue, other_venue]
                other_flow = metrics[other_venue]['order_flow_mean'].mean()
                indirect_ofi += corr * other_flow
        
        # Combine direct and indirect OFI
        total_ofi = direct_ofi + alpha * indirect_ofi
        
        # Normalize based on statistical significance
        if venue_metrics['order_flow_std'].mean() > 0:
            z_score = total_ofi / venue_metrics['order_flow_std'].mean()
            ofi_signals[venue] = np.tanh(z_score)  # Sigmoid normalization
        else:
            ofi_signals[venue] = 0.0
    
    return ofi_signals