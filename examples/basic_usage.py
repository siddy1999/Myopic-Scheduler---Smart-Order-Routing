#!/usr/bin/env python3
"""
Basic Usage Example for Myopic Scheduler

This example demonstrates how to use the Myopic Scheduler for basic
algorithmic trading scenarios.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add the parent directory to the path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from myopic_sor_scheduler import MyopicScheduler, MyopicParameters
from myopic_analysis_utils import MyopicAnalyzer

def create_sample_market_data():
    """Create sample market data for demonstration."""
    np.random.seed(42)  # For reproducible results
    
    # Generate 1 hour of minute-level data
    start_time = datetime.now().replace(hour=9, minute=30, second=0, microsecond=0)
    timestamps = [start_time + timedelta(minutes=i) for i in range(60)]
    
    # Generate realistic price data
    base_price = 150.0
    price_changes = np.random.normal(0, 0.001, 60)  # 0.1% volatility
    prices = [base_price]
    for change in price_changes[1:]:
        prices.append(prices[-1] * (1 + change))
    
    # Generate order book data
    data = []
    for i, (ts, price) in enumerate(zip(timestamps, prices)):
        spread = np.random.uniform(0.01, 0.05)  # 1-5 cent spread
        bid = price - spread/2
        ask = price + spread/2
        
        # Generate volume data
        volume = np.random.poisson(1000)  # Average 1000 shares per minute
        signed_volume = np.random.normal(0, volume/2)  # Some buy/sell imbalance
        
        data.append({
            'ts_event': ts,
            'best_bid': bid,
            'best_ask': ask,
            'bid_fill': max(0, signed_volume),
            'ask_fill': max(0, -signed_volume),
            'signed_volume': signed_volume,
            'Volatility': 0.01,  # 1% daily volatility
            'ADV': 1000000,  # 1M average daily volume
            'size': volume
        })
    
    return pd.DataFrame(data)

def basic_myopic_example():
    """Basic example of using the Myopic Scheduler."""
    print("🚀 Myopic Scheduler - Basic Usage Example")
    print("=" * 50)
    
    # 1. Create sample market data
    print("📊 Creating sample market data...")
    market_data = create_sample_market_data()
    print(f"   Generated {len(market_data)} data points")
    
    # 2. Configure myopic parameters
    print("\n⚙️  Configuring myopic parameters...")
    myopic_params = MyopicParameters(
        lambda_value=25000.0,  # Market impact coefficient
        beta=0.693,           # Impact decay (1-hour half-life)
        volatility=0.01,      # 1% volatility
        adv=1000000.0,       # 1M average daily volume
        T=6.5,               # Trading session length
        Q_0=0.01             # Position normalization
    )
    print(f"   Lambda: {myopic_params.lambda_value}")
    print(f"   Beta: {myopic_params.beta}")
    print(f"   Volatility: {myopic_params.volatility}")
    
    # 3. Create scheduler
    print("\n🧠 Creating myopic scheduler...")
    scheduler = MyopicScheduler(myopic_params)
    
    # 4. Estimate lambda from data
    print("\n📈 Estimating lambda parameter from data...")
    try:
        lambda_values = scheduler.estimate_lambda(market_data)
        if lambda_values:
            best_lambda = lambda_values.get('60s', lambda_values.get('30s', 25000.0))
            scheduler.params.lambda_value = best_lambda
            print(f"   ✅ Estimated lambda: {best_lambda:.2f}")
        else:
            print("   ⚠️  Lambda estimation failed, using default")
    except Exception as e:
        print(f"   ⚠️  Lambda estimation error: {e}")
    
    # 5. Generate trading schedule
    print("\n🎯 Generating trading schedule...")
    total_quantity = 1000  # Trade 1000 shares
    time_horizon = 30      # Over 30 minutes
    
    try:
        schedule = scheduler.generate_trading_schedule(
            df=market_data,
            total_quantity=total_quantity,
            time_horizon=time_horizon
        )
        print(f"   ✅ Generated {len(schedule)} trading decisions")
        
        # Display first few decisions
        print("\n📋 Sample trading decisions:")
        for i, decision in enumerate(schedule[:3]):
            print(f"   Decision {i+1}:")
            print(f"     Time: {decision['timestamp']}")
            print(f"     Quantity: {decision['optimal_quantity']:.2f} shares")
            print(f"     Price Impact: {decision['price_impact']:.6f}")
            print(f"     Alpha Signal: {decision['alpha']:.6f}")
            print()
            
    except Exception as e:
        print(f"   ❌ Schedule generation failed: {e}")
        return
    
    # 6. Analyze results
    print("📊 Analyzing results...")
    analyzer = MyopicAnalyzer()
    
    # Create mock results for analysis
    mock_results = {
        'avg_cost_per_share': 150.25,
        'num_decisions': len(schedule),
        'total_size': sum(abs(d['optimal_quantity']) for d in schedule),
        'lambda_used': scheduler.params.lambda_value,
        'results': [
            {
                'time': d['timestamp'],
                'market_v': abs(d['optimal_quantity']) // 2,
                'limit_v': abs(d['optimal_quantity']) - abs(d['optimal_quantity']) // 2,
                'market_p': d['mid_price'],
                'limit_p': d['mid_price'] - 0.01,
                'myopic_signal': d['alpha'],
                'myopic_impact': d['price_impact']
            }
            for d in schedule
        ]
    }
    
    # Create visualization
    print("📈 Creating visualization...")
    try:
        analyzer.create_schedule_visualization(
            mock_results, 
            save_path="examples/myopic_schedule_example.png"
        )
        print("   ✅ Visualization saved to examples/myopic_schedule_example.png")
    except Exception as e:
        print(f"   ⚠️  Visualization failed: {e}")
    
    # 7. Summary
    print("\n📋 Summary:")
    print(f"   Total quantity to trade: {total_quantity} shares")
    print(f"   Time horizon: {time_horizon} minutes")
    print(f"   Trading decisions generated: {len(schedule)}")
    print(f"   Lambda parameter used: {scheduler.params.lambda_value:.2f}")
    print(f"   Total quantity in schedule: {sum(abs(d['optimal_quantity']) for d in schedule):.2f}")
    
    print("\n✅ Basic example completed successfully!")
    return schedule

def parameter_sensitivity_example():
    """Example showing parameter sensitivity analysis."""
    print("\n🔬 Parameter Sensitivity Analysis Example")
    print("=" * 50)
    
    # Create sample data
    market_data = create_sample_market_data()
    
    # Test different lambda values
    lambda_values = [10000, 15000, 20000, 25000, 30000]
    results = {}
    
    print("Testing different lambda values...")
    for lambda_val in lambda_values:
        print(f"   Testing lambda = {lambda_val}")
        
        # Create scheduler with this lambda
        params = MyopicParameters(
            lambda_value=lambda_val,
            beta=0.693,
            volatility=0.01,
            adv=1000000.0
        )
        scheduler = MyopicScheduler(params)
        
        # Generate schedule
        try:
            schedule = scheduler.generate_trading_schedule(
                df=market_data,
                total_quantity=1000,
                time_horizon=30
            )
            
            # Calculate metrics
            total_quantity = sum(abs(d['optimal_quantity']) for d in schedule)
            avg_impact = np.mean([d['price_impact'] for d in schedule])
            
            results[lambda_val] = {
                'decisions': len(schedule),
                'total_quantity': total_quantity,
                'avg_impact': avg_impact
            }
            
        except Exception as e:
            print(f"     ❌ Failed: {e}")
            results[lambda_val] = None
    
    # Display results
    print("\n📊 Sensitivity Analysis Results:")
    print("Lambda | Decisions | Total Qty | Avg Impact")
    print("-" * 45)
    for lambda_val, result in results.items():
        if result:
            print(f"{lambda_val:6d} | {result['decisions']:9d} | {result['total_quantity']:8.1f} | {result['avg_impact']:9.6f}")
        else:
            print(f"{lambda_val:6d} | {'ERROR':9s} | {'N/A':8s} | {'N/A':9s}")
    
    return results

def main():
    """Main function to run all examples."""
    print("🎯 Myopic Scheduler Examples")
    print("=" * 60)
    
    try:
        # Run basic example
        schedule = basic_myopic_example()
        
        # Run sensitivity analysis
        sensitivity_results = parameter_sensitivity_example()
        
        print("\n🎉 All examples completed successfully!")
        print("\nNext steps:")
        print("1. Check the generated visualization: examples/myopic_schedule_example.png")
        print("2. Modify parameters in the examples to see different behaviors")
        print("3. Try with your own market data")
        print("4. Explore the advanced examples in the examples/ directory")
        
    except Exception as e:
        print(f"\n❌ Example failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
