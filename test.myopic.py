import pandas as pd
import numpy as np
from datetime import datetime
import os

# Import enhanced backtest
from Main_1_enhanced import enhanced_backtest

def test_myopic_integration():
    """Test myopic integration with existing backtesting engine"""
    
    print("🧪 MYOPIC INTEGRATION TEST")
    print("=" * 50)
    
    # Test parameters (same as your existing)
    strategy_params = {
        'S': 100,
        'T': 5,
        'f': 0.003,
        'r': [0.002],
        'theta': 0.0005,
        'lambda_u': 0.05,
        'lambda_o': 0.05,
        'N': 1000
    }
    
    # Test configuration
    stock = "AAPL"
    days = ["2025-04-11"]  # Single day test first
    data_path = "Data/"  # Adjust path as needed
    order_freq = 120
    start_time = ("15", "45")
    end_time = ("16", "00")
    lookup_duration = (0, 15)
    
    print(f"Testing with: {stock} on {days[0]}")
    print(f"Window: {start_time[0]}:{start_time[1]} - {end_time[0]}:{end_time[1]}")
    
    # Create results directory if not exists
    os.makedirs(f"Result/{stock}", exist_ok=True)
    
    try:
        # Test 1: Traditional backtest (baseline)
        print("\n📊 Test 1: Traditional Backtest (Baseline)")
        traditional_results = enhanced_backtest(
            stock=stock,
            days=days,
            strategy_params=strategy_params,
            data_path=data_path,
            frequency=order_freq,
            start_time=start_time,
            end_time=end_time,
            lookup_duration=lookup_duration,
            use_myopic=False  # Traditional approach
        )
        
        print(f"✅ Traditional backtest completed: {len(traditional_results)} orders")
        
    except Exception as e:
        print(f"❌ Traditional backtest failed: {e}")
        traditional_results = []
    
    try:
        # Test 2: Myopic enhanced backtest
        print("\n🚀 Test 2: Myopic Enhanced Backtest")
        myopic_results = enhanced_backtest(
            stock=stock,
            days=days,
            strategy_params=strategy_params,
            data_path=data_path,
            frequency=order_freq,
            start_time=start_time,
            end_time=end_time,
            lookup_duration=lookup_duration,
            use_myopic=True  # Myopic approach
        )
        
        print(f"✅ Myopic backtest completed: {len(myopic_results)} orders")
        
        # Analysis
        if myopic_results:
            analyze_myopic_results(myopic_results)
            
    except Exception as e:
        print(f"❌ Myopic backtest failed: {e}")
        print("This is normal for first integration - let's debug!")
        return False
    
    # Comparison
    if traditional_results and myopic_results:
        compare_results(traditional_results, myopic_results)
    
    return True

def analyze_myopic_results(results):
    """Analyze myopic-specific results"""
    print("\n📈 MYOPIC ANALYSIS")
    print("-" * 30)
    
    df = pd.DataFrame(results)
    
    if 'myopic_quantity' in df.columns:
        print(f"Myopic quantities used: {df['myopic_quantity'].describe()}")
    
    if 'lambda_used' in df.columns:
        print(f"Lambda parameter used: {df['lambda_used'].iloc[0]:.2f}")
    
    if 'myopic_alpha' in df.columns:
        alpha_values = df['myopic_alpha'].dropna()
        if len(alpha_values) > 0:
            print(f"Alpha signals: mean={alpha_values.mean():.6f}, std={alpha_values.std():.6f}")
    
    # Execution quality
    filled_count = df['filled'].sum()
    fill_rate = filled_count / len(df) * 100
    print(f"Fill rate: {fill_rate:.1f}% ({filled_count}/{len(df)})")

def compare_results(traditional, myopic):
    """Compare traditional vs myopic results"""
    print("\n⚖️  COMPARISON ANALYSIS")
    print("-" * 30)
    
    trad_df = pd.DataFrame(traditional)
    myop_df = pd.DataFrame(myopic)
    
    print(f"Decision count - Traditional: {len(trad_df)}, Myopic: {len(myop_df)}")
    
    if len(trad_df) > 0 and len(myop_df) > 0:
        trad_fill_rate = trad_df['filled'].sum() / len(trad_df) * 100
        myop_fill_rate = myop_df['filled'].sum() / len(myop_df) * 100
        
        print(f"Fill rate - Traditional: {trad_fill_rate:.1f}%, Myopic: {myop_fill_rate:.1f}%")
        print(f"Fill rate improvement: {myop_fill_rate - trad_fill_rate:.1f} percentage points")

if __name__ == "__main__":
    success = test_myopic_integration()
    if success:
        print("\n🎉 Integration test completed successfully!")
    else:
        print("\n🔧 Integration needs debugging - this is expected for first run")