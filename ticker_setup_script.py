# quick_ticker_setup.py
"""
Quick setup script to run Enhanced Myopic SOR on your new ticker
"""

import os
import glob
from pathlib import Path

def find_available_tickers():
    """Find all available ticker data files"""
    
    data_path = Path("data/mbp10")
    
    if not data_path.exists():
        print("❌ data/mbp10 directory not found!")
        print("📁 Please create the directory and add your CSV files")
        return []
    
    csv_files = list(data_path.glob("*.csv"))
    
    if not csv_files:
        print("❌ No CSV files found in data/mbp10/")
        return []
    
    print(f"📁 Found {len(csv_files)} data files:")
    for i, file in enumerate(csv_files):
        print(f"  {i+1}. {file.name}")
    
    return csv_files

def run_enhanced_myopic_sor():
    """Run the enhanced myopic SOR backtester"""
    
    print("🎯 ENHANCED MYOPIC SOR QUICK SETUP")
    print("=" * 50)
    
    # Find available files
    files = find_available_tickers()
    
    if not files:
        return
    
    # Let user choose file or specify pattern
    print(f"\n📊 Choose your approach:")
    print(f"1. Select from available files")
    print(f"2. Enter custom ticker pattern")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        if len(files) == 1:
            selected_file = files[0].name
            print(f"📄 Using: {selected_file}")
        else:
            print(f"\nSelect file number (1-{len(files)}):")
            try:
                file_num = int(input("File number: ")) - 1
                if 0 <= file_num < len(files):
                    selected_file = files[file_num].name
                    print(f"📄 Selected: {selected_file}")
                else:
                    print("❌ Invalid selection")
                    return
            except ValueError:
                print("❌ Invalid input")
                return
    
    elif choice == "2":
        selected_file = input("Enter ticker pattern (e.g., 'AAPL', '20250527', or full filename): ").strip()
        print(f"📄 Using pattern: {selected_file}")
    
    else:
        print("❌ Invalid choice")
        return
    
    # Get order parameters
    print(f"\n⚙️ Configure backtest parameters:")
    
    try:
        order_size = int(input("Order size (default 1000): ") or "1000")
        time_horizon = int(input("Time horizon in minutes (default 30): ") or "30")
    except ValueError:
        print("❌ Invalid parameters, using defaults")
        order_size = 1000
        time_horizon = 30
    
    print(f"\n🚀 Running Enhanced Myopic SOR Backtest...")
    print(f"📊 File/Pattern: {selected_file}")
    print(f"📦 Order Size: {order_size:,} shares")
    print(f"⏱️ Time Horizon: {time_horizon} minutes")
    print("🔬 All parameters calculated dynamically from market data")
    print("🎯 Using YOUR exact outflow logic: restrict to top-of-book events")
    print("=" * 70)
    
    # Import and run the backtester
    try:
        from enhanced_myopic_sor_backtester_new_ticker import EnhancedMyopicSORBacktester
        
        backtester = EnhancedMyopicSORBacktester()
        result = backtester.run_backtest(
            ticker_pattern=selected_file,
            order_size=order_size,
            time_horizon=time_horizon
        )
        
        if result:
            print(f"\n🎉 SUCCESS! Enhanced Myopic SOR completed!")
            
            # Summary
            print(f"\n📋 QUICK SUMMARY:")
            print(f"  💰 Price improvement: {result['price_improvement_bps']:.1f} bps")
            print(f"  🏦 SOR savings: ${result['sor_savings']:.2f}")
            print(f"  🌊 Outflow benefits: ${result['outflow_benefits']:.2f}")
            print(f"  💵 Total benefits: ${result['total_benefits']:.2f}")
            print(f"  ✅ Fill rate: {result['fill_rate']:.1%}")
            print(f"  🔄 Outflow detected: {result['total_outflow_detected']:.0f}")
            
            # Show dynamic parameters
            if hasattr(backtester, 'params') and backtester.params:
                print(f"\n🔬 DYNAMIC PARAMETERS CALCULATED:")
                print(f"  • λ (Market Impact): {backtester.params.get('lambda_value', 0):.0f}")
                print(f"  • β (Decay Rate): {backtester.params.get('beta', 0):.4f}")
                print(f"  • σ (Volatility): {backtester.params.get('volatility', 0):.6f}")
                print(f"  • γ (Risk Aversion): {backtester.params.get('risk_aversion', 0):.2f}")
                print(f"  • α (Alpha Strength): {backtester.params.get('alpha_strength', 0):.4f}")
            
            # Create visualization
            create_viz = input("\n📊 Create visualization? (y/n): ").lower().strip()
            if create_viz == 'y':
                backtester.create_visualization(result)
            
            return result
        else:
            print("❌ Backtest failed")
            return None
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure enhanced_myopic_sor_backtester_new_ticker.py is in the current directory")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def quick_data_check():
    """Quick check of your data format"""
    
    print("🔍 QUICK DATA FORMAT CHECK")
    print("=" * 30)
    
    files = find_available_tickers()
    if not files:
        return
    
    # Check first file
    import pandas as pd
    
    try:
        sample_file = files[0]
        print(f"📄 Checking: {sample_file.name}")
        
        # Load just the header
        df_header = pd.read_csv(sample_file, nrows=5)
        
        print(f"📊 Columns found:")
        for col in df_header.columns:
            print(f"  • {col}")
        
        # Check for required columns
        required_basic = ['ts_event', 'bid_px_00', 'ask_px_00']
        required_order_flow = ['action', 'side', 'price', 'size']
        
        print(f"\n✅ Basic columns:")
        for col in required_basic:
            status = "✓" if col in df_header.columns else "✗"
            print(f"  {status} {col}")
        
        print(f"\n🔄 Order flow columns (for actual outflow):")
        for col in required_order_flow:
            status = "✓" if col in df_header.columns else "✗"
            print(f"  {status} {col}")
        
        # Show sample data
        print(f"\n📋 Sample data:")
        print(df_header[['ts_event', 'bid_px_00', 'ask_px_00']].head(3))
        
        if all(col in df_header.columns for col in required_order_flow):
            print(f"\n🎯 Perfect! You have order flow data for YOUR exact outflow calculation")
            print(f"    ✓ Will use: top = df[df['price'] == best_bid]")
            print(f"    ✓ Cancellations: action='C' & side='B'")
            print(f"    ✓ Seller trades: action='T' & side='A'")
        else:
            print(f"\n🔄 Note: Will use synthetic outflow (missing order flow columns)")
            print(f"    • Consider getting order book data with action/side/price columns")
        
    except Exception as e:
        print(f"❌ Error checking data: {e}")

def main():
    """Main setup function"""
    
    print("🎯 ENHANCED MYOPIC SOR - QUICK SETUP")
    print("=" * 45)
    print("This script helps you run your Enhanced Myopic SOR algorithm")
    print("on new ticker data with minimal setup.")
    print()
    
    while True:
        print("🚀 What would you like to do?")
        print("1. 🔍 Check data format")
        print("2. 🎯 Run Enhanced Myopic SOR backtest")
        print("3. 📁 Show data directory info")
        print("4. ❌ Exit")
        
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == "1":
            quick_data_check()
        elif choice == "2":
            run_enhanced_myopic_sor()
        elif choice == "3":
            find_available_tickers()
        elif choice == "4":
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice")
        
        print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    main()