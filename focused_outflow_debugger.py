# focused_outflow_debugger.py
"""
Focused debugger to understand why outflow is always 0
"""

import pandas as pd
import numpy as np
from pathlib import Path

def debug_outflow_systematically():
    """Debug outflow calculation step by step"""
    
    print("🔍 SYSTEMATIC OUTFLOW DEBUGGING")
    print("=" * 50)
    
    # Load the same file that showed 0 outflow
    data_path = Path("data/mbp10")
    file_path = data_path / "xnas-itch-20250528.mbp-10.csv"
    
    print(f"📄 Loading: {file_path.name}")
    
    # Load first 5000 rows for analysis
    df = pd.read_csv(file_path, nrows=5000)
    df['ts_event'] = pd.to_datetime(df['ts_event'], utc=True)
    
    print(f"✅ Loaded {len(df)} records")
    
    # Add required columns
    if 'best_bid' not in df.columns and 'bid_px_00' in df.columns:
        df['best_bid'] = df['bid_px_00']
    if 'best_ask' not in df.columns and 'ask_px_00' in df.columns:
        df['best_ask'] = df['ask_px_00']
    
    print(f"\n📊 STEP 1: COLUMN VERIFICATION")
    required_cols = ['action', 'side', 'price', 'size', 'best_bid']
    for col in required_cols:
        if col in df.columns:
            print(f"  ✅ {col}: {df[col].dtype}")
        else:
            print(f"  ❌ {col}: MISSING")
    
    if not all(col in df.columns for col in required_cols):
        print("\n❌ MISSING REQUIRED COLUMNS - This is why outflow = 0!")
        print("📋 Available columns:")
        for col in df.columns:
            print(f"    • {col}")
        return
    
    print(f"\n📊 STEP 2: DATA VALUE ANALYSIS")
    print(f"  Action values: {sorted(df['action'].unique())}")
    print(f"  Side values: {sorted(df['side'].unique())}")
    print(f"  Price range: ${df['price'].min():.4f} - ${df['price'].max():.4f}")
    print(f"  Best bid range: ${df['best_bid'].min():.4f} - ${df['best_bid'].max():.4f}")
    print(f"  Size range: {df['size'].min()} - {df['size'].max()}")
    
    # Check for the specific patterns we need
    print(f"\n📊 STEP 3: ORDER FLOW PATTERN ANALYSIS")
    
    # Count bid cancellations
    bid_cancels = df[(df['action'] == 'C') & (df['side'] == 'B')]
    print(f"  Total bid cancellations (C + B): {len(bid_cancels)}")
    if len(bid_cancels) > 0:
        print(f"    Price range: ${bid_cancels['price'].min():.4f} - ${bid_cancels['price'].max():.4f}")
        print(f"    Size range: {bid_cancels['size'].min()} - {bid_cancels['size'].max()}")
        print(f"    Sample prices: {bid_cancels['price'].head(5).tolist()}")
    
    # Count seller trades
    seller_trades = df[(df['action'] == 'T') & (df['side'] == 'A')]
    print(f"  Total seller trades (T + A): {len(seller_trades)}")
    if len(seller_trades) > 0:
        print(f"    Price range: ${seller_trades['price'].min():.4f} - ${seller_trades['price'].max():.4f}")
        print(f"    Size range: {seller_trades['size'].min()} - {seller_trades['size'].max()}")
        print(f"    Sample prices: {seller_trades['price'].head(5).tolist()}")
    
    print(f"\n📊 STEP 4: EXACT PRICE MATCHING TEST")
    
    # Test the exact logic on actual data
    sample_best_bids = df['best_bid'].dropna().unique()[:10]
    print(f"  Testing {len(sample_best_bids)} unique best_bid values...")
    
    total_outflow_found = 0
    
    for i, best_bid in enumerate(sample_best_bids):
        print(f"\n  Test {i+1}: best_bid = ${best_bid:.6f}")
        
        # YOUR EXACT logic
        top = df[df["price"] == best_bid]
        print(f"    Records at EXACT price: {len(top)}")
        
        if len(top) > 0:
            # Show what we found
            print(f"    Actions in exact matches: {top['action'].unique()}")
            print(f"    Sides in exact matches: {top['side'].unique()}")
            
            # Apply YOUR exact outflow logic
            cancels = top.loc[(top["action"] == "C") & (top["side"] == "B"), "size"].sum()
            trades = top.loc[(top["action"] == "T") & (top["side"] == "A"), "size"].sum()
            outflow = cancels + trades
            
            print(f"    Bid cancellations: {cancels}")
            print(f"    Seller trades: {trades}")
            print(f"    OUTFLOW: {outflow}")
            
            if outflow > 0:
                total_outflow_found += outflow
                print(f"    🎯 FOUND OUTFLOW!")
        
        else:
            # Check nearby prices
            tolerance = 0.01
            nearby = df[
                (df['price'] >= best_bid - tolerance) & 
                (df['price'] <= best_bid + tolerance)
            ]
            print(f"    Records within ±${tolerance}: {len(nearby)}")
            if len(nearby) > 0:
                unique_nearby = sorted(nearby['price'].unique())
                print(f"    Nearby prices: {unique_nearby[:5]}...")  # Show first 5
    
    print(f"\n📊 STEP 5: SUMMARY")
    print(f"  Total outflow found across all tests: {total_outflow_found}")
    
    if total_outflow_found == 0:
        print(f"\n❌ PROBLEM IDENTIFIED:")
        print(f"  🔍 No exact price matches between 'price' and 'best_bid'")
        print(f"  💡 This means:")
        print(f"     • Order actions happen at prices ≠ best_bid")
        print(f"     • Price precision mismatch")
        print(f"     • Different data sources for prices")
        
        print(f"\n🔬 DETAILED ANALYSIS:")
        
        # Check price precision
        price_decimals = df['price'].apply(lambda x: len(str(x).split('.')[-1]) if '.' in str(x) else 0)
        bid_decimals = df['best_bid'].apply(lambda x: len(str(x).split('.')[-1]) if '.' in str(x) else 0)
        
        print(f"  Price decimal places (mode): {price_decimals.mode().iloc[0] if not price_decimals.empty else 'N/A'}")
        print(f"  Best bid decimal places (mode): {bid_decimals.mode().iloc[0] if not bid_decimals.empty else 'N/A'}")
        
        # Check if prices are ever close
        min_diff = float('inf')
        for best_bid in sample_best_bids[:5]:
            diffs = abs(df['price'] - best_bid)
            min_diff_this = diffs.min()
            if min_diff_this < min_diff:
                min_diff = min_diff_this
        
        print(f"  Smallest price difference found: ${min_diff:.6f}")
        
        if min_diff < 0.01:
            print(f"  💡 SOLUTION: Use tolerance-based matching")
        else:
            print(f"  💡 SOLUTION: Check if 'price' and 'best_bid' are from same data source")
    
    else:
        print(f"\n✅ OUTFLOW DETECTED!")
        print(f"  🎯 YOUR exact logic should work in some intervals")
    
    # Final recommendation
    print(f"\n💡 RECOMMENDATIONS:")
    print(f"  1. 🔍 Check if 'price' column represents trade/cancel prices")
    print(f"  2. 🔍 Check if 'best_bid' represents quote prices")
    print(f"  3. 🔄 Consider using tolerance: abs(price - best_bid) < 0.005")
    print(f"  4. 📊 Verify data sources are synchronized")

def inspect_raw_data_sample():
    """Look at raw data to understand the structure"""
    
    print(f"\n" + "="*60)
    print("🔍 RAW DATA INSPECTION")
    print("="*60)
    
    data_path = Path("data/mbp10")
    file_path = data_path / "xnas-itch-20250528.mbp-10.csv"
    
    # Load just 20 rows to inspect
    df_raw = pd.read_csv(file_path, nrows=20)
    
    print(f"📋 Raw data sample (first 10 rows):")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    print(df_raw[['ts_event', 'action', 'side', 'price', 'size', 'bid_px_00', 'ask_px_00']].head(10))
    
    print(f"\n📊 Data types:")
    relevant_cols = ['action', 'side', 'price', 'size', 'bid_px_00', 'ask_px_00']
    for col in relevant_cols:
        if col in df_raw.columns:
            print(f"  {col}: {df_raw[col].dtype}")
    
    # Look for patterns
    print(f"\n🔍 Pattern analysis:")
    if 'action' in df_raw.columns:
        print(f"  Actions: {df_raw['action'].value_counts().to_dict()}")
    if 'side' in df_raw.columns:
        print(f"  Sides: {df_raw['side'].value_counts().to_dict()}")

if __name__ == "__main__":
    debug_outflow_systematically()
    inspect_raw_data_sample()
