# Myopic Scheduling Integration with Smart Order Routing (SOR)

## Table of Contents
1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [Mathematical Foundation](#mathematical-foundation)
4. [Installation & Setup](#installation--setup)
5. [File Structure](#file-structure)
6. [Configuration Guide](#configuration-guide)
7. [Usage Instructions](#usage-instructions)
8. [API Reference](#api-reference)
9. [Performance Analysis Framework](#performance-analysis-framework)
10. [Troubleshooting](#troubleshooting)
11. [Advanced Configuration](#advanced-configuration)
12. [Examples](#examples)
13. [Development Notes](#development-notes)

---

## Project Overview

### What I Built: Myopic Scheduling Integration
I've developed an advanced algorithmic trading system that integrates myopic scheduling with an existing Smart Order Routing (SOR) framework. This system makes **short-term optimal decisions** for order execution timing and sizing, adapting to real-time market conditions to minimize trading costs and market impact.

### My Integration Approach
I designed the integration to work with the existing SOR system in a sequential layered approach:
- **Myopic Scheduler**: I built this to determine **WHEN** and **HOW MUCH** to trade
- **Existing SOR System**: This continues to determine **MARKET vs LIMIT ratio** and **venue allocation**
- **Sequential Flow**: My myopic scheduler feeds optimally-timed quantities to your existing SOR logic

### Integration Flow I Designed

#### Traditional Approach (Your Current System):
```
Fixed Frequency Check (every 120 seconds) → SOR Decision → Execution
     ↓                                         ↓              ↓
Check if conditions                      If optimal:         Execute over
are favorable                           60 market +          T minutes
                                       40 limit orders       
     ↓
If not favorable: skip this interval
```

#### My Enhanced Approach:
```
Myopic Scheduler → SOR Decision → Execution
     ↓                 ↓              ↓
Market analysis       25 market +    Execute over
determines:           20 limit       T minutes
09:30: Trade 45      orders
       shares        

09:33: Trade 80      50 market +    Execute over
       shares        30 limit       T minutes
                     orders

09:37: Trade 30      15 market +    Execute over
       shares        15 limit       T minutes
                     orders
```

#### What Each Layer Does:

1. **My Myopic Layer** (new addition):
   - Analyzes market impact dynamics: `I(t+1) = I(t) + λ*Q(t) - β*I(t)`
   - Predicts future prices (alpha): `α(t) = S(T) - S(t)`
   - Decides optimal timing and quantities: `Q*(t) = (α'(t) + β*I*(t)) / λ`
   - **Output**: "At time T, trade Q shares" (based on market conditions, not fixed intervals)

2. **Your Existing SOR Layer** (unchanged):
   - Receives timing and quantity decisions from myopic scheduler
   - Analyzes queue depth, OFI, rebates, fees
   - Optimizes market vs limit split and venue allocation
   - **Output**: "Trade Q shares as X market + Y limit orders across venues"
   - **Note**: In traditional mode, SOR runs every 120 seconds and only executes if conditions are favorable

3. **Execution Layer** (unchanged):
   - Executes the market and limit orders as determined by SOR over time horizon T

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    MARKET DATA INPUT                         │
│  (Tick data, Order book, Volume, Volatility, etc.)         │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                 MYOPIC SCHEDULER                            │
│           (My Implementation Layer)                         │
│  ┌─────────────────┐  ┌─────────────────┐                  │
│  │ Lambda          │  │ Market Impact   │                  │
│  │ Estimation      │  │ Calculation     │                  │
│  └─────────────────┘  └─────────────────┘                  │
│  ┌─────────────────┐  ┌─────────────────┐                  │
│  │ Alpha           │  │ Optimal Timing  │                  │
│  │ Prediction      │  │ & Sizing        │                  │
│  └─────────────────┘  └─────────────────┘                  │
└─────────────────────────┬───────────────────────────────────┘
                          │ Optimally-Timed Quantities
                          │ (Time, Quantity pairs)
┌─────────────────────────▼───────────────────────────────────┐
│              EXISTING SOR OPTIMIZER                         │
│            (Your Original System)                           │
│  ┌─────────────────┐  ┌─────────────────┐                  │
│  │ Market vs Limit │  │ Order Flow      │                  │
│  │ Order Split     │  │ Imbalance (OFI) │                  │
│  └─────────────────┘  └─────────────────┘                  │
│  ┌─────────────────┐  ┌─────────────────┐                  │
│  │ Venue           │  │ Queue Depth     │                  │
│  │ Selection       │  │ Analysis        │                  │
│  └─────────────────┘  └─────────────────┘                  │
└─────────────────────────┬───────────────────────────────────┘
                          │ Market Orders + Limit Orders
                          │ with Venue Allocation
┌─────────────────────────▼───────────────────────────────────┐
│                 EXECUTION ENGINE                            │
│         (Order Placement & Management)                      │
└─────────────────────────────────────────────────────────────┘
```

---

## Mathematical Foundation

### 1. Market Impact Model
The system uses an exponential decay model for market impact:

```
I(t+1) = I(t) + λ * Q(t) * σ/ADV - β * I(t) * Δt
```

Where:
- `I(t)`: Market impact at time t
- `λ`: Market impact coefficient (estimated from data)
- `Q(t)`: Signed trading volume at time t
- `σ`: Asset volatility
- `ADV`: Average Daily Volume
- `β`: Impact decay parameter (β = ln(2)/half_life)
- `Δt`: Time step

### 2. Alpha Prediction
Alpha represents the expected price movement:

```
α(t) = S(T) - S(t)
```

Where:
- `S(T)`: Expected end-of-period price
- `S(t)`: Current unperturbed price
- Unperturbed price = Mid price - Market impact

### 3. Optimal Control
The optimal trading quantity is derived from:

```
Q*(t) = (α'(t) + β * I*(t)) / λ
```

Where:
- `Q*(t)`: Optimal trading quantity
- `α'(t)`: Rate of change of alpha
- `I*(t)`: Optimal market impact
- `I*(t) = (α(t) - α'(t)/β) / 2`

### 4. Lambda Estimation
Lambda is estimated using linear regression across different time horizons:

```
Δp = λ * ΔQ + ε
```

Where:
- `Δp`: Price change
- `ΔQ`: Normalized volume change (Volume * Volatility / ADV)

---

## Installation & Setup

### Prerequisites
- Python 3.8+
- Existing SOR codebase
- Market data access
- Required Python packages

### Required Packages
```bash
pip install pandas numpy scikit-learn matplotlib seaborn yfinance scipy statsmodels
```

### Optional Packages (for enhanced features)
```bash
pip install plotly dash streamlit  # For interactive visualizations
pip install joblib dask           # For parallel processing
pip install pytest               # For testing
```

### Environment Setup
```bash
# Clone or navigate to your SOR repository
cd your-sor-repository

# Install required packages
pip install -r requirements.txt

# Set up environment variables (optional)
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export DATA_PATH="/path/to/your/data"
```

---

## File Structure

### New Files (Add to your repository)
```
your-sor-repository/
├── myopic_sor_scheduler.py      # Main myopic scheduling logic
├── main_myopic_final.py         # Enhanced main execution
├── myopic_analysis_utils.py     # Analysis and visualization tools
├── config/
│   ├── myopic_config.yaml       # Configuration file (optional)
│   └── parameters.json          # Parameter presets
├── tests/
│   ├── test_myopic_scheduler.py # Unit tests
│   └── test_integration.py      # Integration tests
└── docs/
    ├── api_reference.md         # API documentation
    └── examples/                # Usage examples
```

### Existing Files (Your current SOR system)
```
your-sor-repository/
├── Main_1.py                    # Your existing backtest logic
├── main_final.py               # Your current main execution
├── metrics.py                  # Your metrics calculation
├── ofi.py                      # Order Flow Imbalance calculations
├── data_io.py                  # Data I/O utilities
├── facade.py                   # SOR facade
├── lookup.py                   # Lookup table logic
├── market_metrics.py           # Market metrics
├── optimisation.py             # Core optimization
├── live_data.py               # Live data handling
└── ... (other existing files)
```

---

## Configuration Guide

### Basic Configuration

#### 1. Strategy Parameters
```python
strategy_params = {
    'S': 100,          # Total order size (shares)
    'T': 5,            # Time horizon (minutes)
    'f': 0.003,        # Fee rate (30 bps)
    'r': [0.003],      # Rebate rates per venue
    'theta': 0.0005,   # Market impact parameter
    'lambda_u': 0.05,  # Underfill penalty
    'lambda_o': 0.05,  # Overfill penalty
    'N': 1000          # Number of Monte Carlo simulations
}
```

#### 2. Myopic Parameters
```python
myopic_params = MyopicParameters(
    lambda_value=25000.0,  # Market impact coefficient (auto-estimated)
    beta=0.693,           # Impact decay (ln(2) for 1-hour half-life)
    volatility=0.01,      # Asset volatility (1%)
    adv=1000000.0,       # Average Daily Volume
    T=6.5,               # Trading session length (hours)
    Q_0=0.01             # Initial position normalization
)
```

#### 3. Execution Parameters
```python
execution_config = {
    'order_freq': 120,              # Order frequency (seconds)
    'start_time': ("09", "30"),     # Market start (hour, minute)
    'end_time': ("16", "00"),       # Market end (hour, minute)
    'lookup_duration': (0, 15),     # Lookup window (hours, minutes)
    'use_parallel': True,           # Enable parallel processing
    'max_workers': 4                # Number of parallel workers
}
```

### Advanced Configuration

#### 1. Lambda Estimation Settings
```python
lambda_config = {
    'agg_periods': [0.1, 1, 2, 5, 10, 30, 60, 120, 300],  # Aggregation periods (seconds)
    'min_r2_threshold': 0.01,        # Minimum R² for valid lambda
    'preferred_periods': ['60s', '30s', '10s'],  # Preferred periods in order
    'fallback_lambda': 25000.0       # Default lambda if estimation fails
}
```

#### 2. Risk Management
```python
risk_config = {
    'max_order_size': 1000,          # Maximum single order size
    'min_order_size': 1,             # Minimum single order size
    'max_impact_threshold': 0.01,    # Maximum allowed impact (1%)
    'stop_loss_threshold': 0.05,     # Stop loss threshold (5%)
    'position_limit': 10000          # Maximum total position
}
```

#### 3. Data Quality Filters
```python
data_filters = {
    'min_spread_bps': 1,             # Minimum spread (basis points)
    'max_spread_bps': 100,           # Maximum spread (basis points)
    'min_volume': 100,               # Minimum volume per observation
    'max_price_change': 0.02,        # Maximum price change (2%)
    'exclude_auction': True          # Exclude auction periods
}
```

---

## Usage Instructions

### Quick Start

#### 1. Basic Usage
```python
# Run with default settings
python main_myopic_final.py
```

#### 2. Custom Configuration
```python
from main_myopic_final import main
from myopic_sor_scheduler import MyopicParameters

# Customize parameters
strategy_params = {
    'S': 500,      # Larger order size
    'T': 10,       # Longer time horizon
    # ... other parameters
}

# Run with custom settings
results = main()
```

#### 3. Single Stock Test
```python
from main_myopic_final import run_job_with_myopic

# Test single stock
stock_results = run_job_with_myopic(
    stock="AAPL",
    day="2025-04-02",
    strategy_params=strategy_params,
    data_path="./data/",
    order_freq=120,
    start_time=("09", "30"),
    end_time=("16", "00"),
    lookup_duration=(0, 15),
    market_data_path="./data/",
    use_myopic=True
)
```

### Advanced Usage

#### 1. Comparison Mode
```python
from main_myopic_final import compare_myopic_vs_traditional

# Compare approaches
comparison = compare_myopic_vs_traditional(
    stock="AAPL",
    day="2025-04-02",
    strategy_params=strategy_params,
    data_path="./data/",
    order_freq=120,
    start_time=("09", "30"),
    end_time=("16", "00"),
    lookup_duration=(0, 15),
    market_data_path="./data/"
)

print(f"Cost improvement: {comparison['improvement']['cost_improvement_pct']:.2f}%")
```

#### 2. Batch Processing
```python
# Process multiple stocks and days
stocks = ["AAPL", "MSFT", "GOOGL"]
days = ["2025-04-01", "2025-04-02", "2025-04-03"]

results = {}
for stock in stocks:
    for day in days:
        key = f"{stock}_{day}"
        results[key] = run_job_with_myopic(
            stock=stock,
            day=day,
            # ... other parameters
        )
```

#### 3. Parameter Optimization
```python
from myopic_sor_scheduler import MyopicScheduler, MyopicParameters

# Test different lambda values
lambda_values = [10000, 15000, 20000, 25000, 30000]
results = {}

for lambda_val in lambda_values:
    params = MyopicParameters(lambda_value=lambda_val, beta=0.693, volatility=0.01, adv=1000000)
    scheduler = MyopicScheduler(params)
    
    # Run backtest with this lambda
    result = enhanced_backtest_with_myopic(
        # ... parameters
        myopic_params=params
    )
    
    results[lambda_val] = result
```

### Output Analysis Framework

#### 1. Results Structure Design
The system is designed to output comprehensive results in the following structure:
```python
results = {
    'total_cost': float,             # Total execution cost
    'total_size': int,               # Total shares traded  
    'avg_cost_per_share': float,     # Average cost per share
    'num_decisions': int,            # Number of trading decisions
    'myopic_enabled': bool,          # Whether myopic was used
    'lambda_used': float,            # Lambda parameter used
    'results': list                  # Detailed trade-by-trade results
}
```

#### 2. Performance Metrics Framework
The system provides tools to analyze:
```python
# Access performance metrics
print(f"Average cost per share: ${results['avg_cost_per_share']:.4f}")
print(f"Total execution cost: ${results['total_cost']:.2f}")
print(f"Number of decisions: {results['num_decisions']}")
print(f"Fill rate: {results['total_size']/strategy_params['S']*100:.1f}%")
```

#### 3. Comparison Analysis Framework
The system includes comparison capabilities:
```python
# Analyze improvement potential
if 'improvement' in comparison:
    improvement = comparison['improvement']['cost_improvement_pct']
    print(f"Potential cost improvement with myopic: {improvement:.2f}%")
    
    if improvement > 0:
        print("✅ Myopic scheduling shows potential benefit")
    else:
        print("❌ Traditional approach may be preferred")
```

---

## API Reference

### MyopicScheduler Class

#### Constructor
```python
MyopicScheduler(params: MyopicParameters)
```

#### Key Methods

##### estimate_lambda()
```python
estimate_lambda(df: pd.DataFrame, agg_periods: List[float] = None) -> Dict[str, float]
```
Estimates lambda parameters for different aggregation periods.

**Parameters:**
- `df`: Market data DataFrame with required columns
- `agg_periods`: List of aggregation periods in seconds (optional)

**Returns:**
- Dictionary mapping period strings to lambda values

**Example:**
```python
lambda_values = scheduler.estimate_lambda(market_data, [1, 5, 10, 30, 60])
# Returns: {'1s': 28000.5, '5s': 25600.2, ...}
```

##### generate_trading_schedule()
```python
generate_trading_schedule(df: pd.DataFrame, total_quantity: float, time_horizon: int) -> List[Dict]
```
Generates optimal trading schedule using myopic model.

**Parameters:**
- `df`: Market data DataFrame
- `total_quantity`: Total quantity to trade
- `time_horizon`: Trading horizon in minutes

**Returns:**
- List of trading decisions with timestamps and quantities

**Example:**
```python
schedule = scheduler.generate_trading_schedule(
    df=market_data,
    total_quantity=1000,
    time_horizon=30
)
# Returns: [{'timestamp': ..., 'optimal_quantity': 45.2, ...}, ...]
```

##### integrate_with_sor()
```python
integrate_with_sor(dfs: Dict[str, pd.DataFrame], myopic_schedule: List[Dict], sor_params: Dict) -> List[Dict]
```
Integrates myopic schedule with existing SOR optimization.

**Parameters:**
- `dfs`: Dictionary of venue DataFrames
- `myopic_schedule`: Myopic trading schedule (time, quantity pairs)
- `sor_params`: SOR optimization parameters

**Returns:**
- List of integrated trading decisions with SOR-determined market/limit splits and venue allocations

**Integration Flow:**
1. Takes each myopic decision (timestamp, quantity)
2. Calls existing SOR optimization with that quantity
3. SOR determines optimal market vs limit split and venue allocation
4. Returns complete execution plan

### MyopicParameters Class

#### Constructor
```python
MyopicParameters(
    lambda_value: float,
    beta: float,
    volatility: float,
    adv: float,
    T: float = 6.5,
    Q_0: float = 0.01
)
```

**Parameters:**
- `lambda_value`: Market impact coefficient
- `beta`: Impact decay parameter
- `volatility`: Asset volatility
- `adv`: Average Daily Volume
- `T`: Trading session length in hours
- `Q_0`: Initial position normalization

### Analysis Functions

#### MyopicAnalyzer Class
```python
from myopic_analysis_utils import MyopicAnalyzer

analyzer = MyopicAnalyzer()
```

##### analyze_schedule_performance()
```python
analyze_schedule_performance(myopic_results: Dict, traditional_results: Dict) -> Dict
```
Analyzes performance differences between approaches.

##### create_schedule_visualization()
```python
create_schedule_visualization(results: Dict, save_path: Optional[str] = None)
```
Creates comprehensive visualization of trading schedule.

##### generate_performance_report()
```python
generate_performance_report(comparison_results: Dict, output_path: Optional[str] = None) -> str
```
Generates detailed performance report.

---

## Performance Analysis

### Backtesting Results

#### Expected Performance Framework
The system is designed to measure and compare performance across multiple dimensions:

| Metric | Description | Measurement Method |
|--------|-------------|-------------------|
| Avg Cost per Share | Average execution cost per share | Total cost / Total shares |
| Market Impact | Price impact of trading activity | Price movement correlation |
| Fill Rate | Percentage of target quantity executed | Executed / Target quantity |
| Decision Efficiency | Optimization of trading decisions | Decisions made vs optimal |
| Timing Quality | Quality of execution timing | Correlation with price predictions |
| Risk Metrics | Execution risk measurements | Volatility and drawdown analysis |

#### Performance Analysis Framework

| Market Condition | Analysis Focus | Key Metrics |
|------------------|----------------|-------------|
| High Volatility | Impact minimization | Market impact, timing efficiency |
| Low Volatility | Cost optimization | Execution costs, fill rates |
| High Volume | Liquidity utilization | Venue optimization, slippage |
| Low Volume | Risk management | Position limits, impact control |

### Metrics and KPIs

#### Primary Metrics
1. **Cost per Share**: Average execution cost per share
2. **Market Impact**: Price impact of trading activity
3. **Fill Rate**: Percentage of target quantity executed
4. **Slippage**: Difference from benchmark prices (TWAP, VWAP)

#### Secondary Metrics
1. **Decision Efficiency**: Number of trading decisions made
2. **Timing Quality**: Correlation with optimal timing
3. **Risk Metrics**: Volatility of execution costs
4. **Adaptability**: Performance across different market regimes

#### Risk Metrics
1. **Implementation Shortfall**: Total cost including opportunity cost
2. **Tracking Error**: Deviation from benchmark execution
3. **Maximum Drawdown**: Worst-case performance scenario
4. **Sharpe Ratio**: Risk-adjusted returns

---

## Troubleshooting

### Common Issues and Solutions

#### 1. Import Errors
**Problem**: `ModuleNotFoundError` when importing myopic modules
```
ModuleNotFoundError: No module named 'myopic_sor_scheduler'
```

**Solution**:
```python
# Add to the beginning of your script
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Or set PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

#### 2. Data Format Issues
**Problem**: Missing required columns in market data
```
KeyError: 'signed_volume'
```

**Solution**:
```python
# Add missing columns
if 'signed_volume' not in df.columns:
    df['signed_volume'] = df.get('bid_fill', 0) - df.get('ask_fill', 0)

if 'mid_price' not in df.columns:
    df['mid_price'] = (df['best_bid'] + df['best_ask']) / 2
```

#### 3. Lambda Estimation Failures
**Problem**: Lambda estimation returns empty results
```
WARNING: Lambda estimation failed, using default
```

**Solution**:
```python
# Check data quality
print(f"Data shape: {df.shape}")
print(f"Date range: {df['ts_event'].min()} to {df['ts_event'].max()}")
print(f"Unique trading days: {df['ts_event'].dt.date.nunique()}")

# Adjust aggregation periods
agg_periods = [1, 5, 10, 30]  # Use shorter periods for smaller datasets
```

#### 4. Memory Issues
**Problem**: Out of memory errors with large datasets
```
MemoryError: Unable to allocate array
```

**Solution**:
```python
# Process data in chunks
def process_in_chunks(df, chunk_size=10000):
    for i in range(0, len(df), chunk_size):
        yield df.iloc[i:i+chunk_size]

# Or use Dask for large datasets
import dask.dataframe as dd
df = dd.from_pandas(your_dataframe, npartitions=4)
```

#### 5. Performance Issues
**Problem**: Slow execution times
```
Execution taking too long...
```

**Solutions**:
```python
# Enable parallel processing
use_parallel = True
max_workers = min(4, os.cpu_count())

# Reduce simulation count
strategy_params['N'] = 500  # Instead of 1000

# Use vectorized operations
df['calculation'] = np.vectorize(your_function)(df['input'])
```

### Debugging Tips

#### 1. Enable Detailed Logging
```python
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("debug.log"),
        logging.StreamHandler()
    ]
)
```

#### 2. Data Quality Checks
```python
def validate_data(df):
    """Validate market data quality"""
    checks = {
        'non_empty': len(df) > 0,
        'has_timestamps': 'ts_event' in df.columns,
        'has_prices': all(col in df.columns for col in ['best_bid', 'best_ask']),
        'price_sanity': (df['best_bid'] > 0).all() and (df['best_ask'] > 0).all(),
        'spread_sanity': (df['best_ask'] >= df['best_bid']).all(),
        'time_sorted': df['ts_event'].is_monotonic_increasing
    }
    
    for check, passed in checks.items():
        print(f"{check}: {'✅' if passed else '❌'}")
    
    return all(checks.values())
```

#### 3. Performance Profiling
```python
import cProfile
import pstats

# Profile your code
cProfile.run('your_function()', 'profile_stats')
stats = pstats.Stats('profile_stats')
stats.sort_stats('cumulative').print_stats(10)
```

### Error Recovery

#### 1. Graceful Fallbacks
```python
try:
    # Try myopic scheduling
    results = enhanced_backtest_with_myopic(...)
except Exception as e:
    logging.error(f"Myopic scheduling failed: {e}")
    # Fall back to traditional approach
    results = backtest(...)  # Your original function
```

#### 2. Partial Results Handling
```python
def safe_execution(stock, day, params):
    try:
        return run_job_with_myopic(stock, day, params, use_myopic=True)
    except Exception as e:
        logging.warning(f"Myopic failed for {stock} on {day}: {e}")
        try:
            return run_job_with_myopic(stock, day, params, use_myopic=False)
        except Exception as e2:
            logging.error(f"Both approaches failed: {e2}")
            return None
```

---

## Advanced Configuration

### Custom Lambda Estimation

#### 1. Multi-Timeframe Lambda
```python
class AdvancedLambdaEstimator:
    def __init__(self, timeframes=['1min', '5min', '15min', '1H']):
        self.timeframes = timeframes
        
    def estimate_multi_timeframe_lambda(self, df):
        lambda_estimates = {}
        
        for tf in self.timeframes:
            resampled = df.resample(tf, on='ts_event').agg({
                'signed_volume': 'sum',
                'mid_price': 'last',
                'Volatility': 'last',
                'ADV': 'last'
            }).dropna()
            
            # Calculate lambda for this timeframe
            lambda_est = self.calculate_lambda_for_timeframe(resampled)
            lambda_estimates[tf] = lambda_est
            
        return lambda_estimates
```

#### 2. Regime-Aware Lambda
```python
def estimate_regime_aware_lambda(df):
    """Estimate different lambdas for different market regimes"""
    
    # Define market regimes based on volatility
    df['volatility_regime'] = pd.cut(
        df['Volatility'], 
        bins=[0, 0.01, 0.02, np.inf], 
        labels=['low', 'medium', 'high']
    )
    
    lambda_by_regime = {}
    for regime in ['low', 'medium', 'high']:
        regime_data = df[df['volatility_regime'] == regime]
        if len(regime_data) > 100:  # Minimum data requirement
            lambda_by_regime[regime] = estimate_lambda_for_data(regime_data)
    
    return lambda_by_regime
```

### Dynamic Parameter Adjustment

#### 1. Adaptive Beta
```python
class AdaptiveBeta:
    def __init__(self, initial_beta=0.693):
        self.beta = initial_beta
        self.impact_history = []
        
    def update_beta(self, recent_impacts):
        """Update beta based on recent impact decay patterns"""
        if len(recent_impacts) < 10:
            return self.beta
            
        # Fit exponential decay to recent data
        decay_rate = self.fit_decay_rate(recent_impacts)
        self.beta = max(0.1, min(2.0, decay_rate))  # Bounds check
        
        return self.beta
```

#### 2. Volume-Adjusted Parameters
```python
def adjust_params_for_volume(base_params, current_volume, historical_avg):
    """Adjust parameters based on current market volume"""
    
    volume_ratio = current_volume / historical_avg
    adjusted_params = base_params.copy()
    
    # Increase lambda in high volume periods
    if volume_ratio > 1.5:
        adjusted_params.lambda_value *= 1.2
    elif volume_ratio < 0.5:
        adjusted_params.lambda_value *= 0.8
        
    return adjusted_params
```

### Custom Optimization Objectives

#### 1. Multi-Objective Optimization
```python
class MultiObjectiveOptimizer:
    def __init__(self, weights={'cost': 0.5, 'risk': 0.3, 'timing': 0.2}):
        self.weights = weights
        
    def calculate_composite_score(self, results):
        """Calculate weighted composite score"""
        
        cost_score = 1 / (1 + results['avg_cost_per_share'])  # Lower is better
        risk_score = 1 / (1 + results['cost_volatility'])     # Lower is better
        timing_score = results['timing_efficiency']           # Higher is better
        
        composite = (
            self.weights['cost'] * cost_score +
            self.weights['risk'] * risk_score +
            self.weights['timing'] * timing_score
        )
        
        return composite
```

#### 2. Risk-Adjusted Optimization
```python
def risk_adjusted_myopic_schedule(scheduler, df, total_quantity, risk_limit):
    """Generate schedule with risk constraints"""
    
    base_schedule = scheduler.generate_trading_schedule(df, total_quantity, 300)
    
    # Apply risk filters
    risk_adjusted_schedule = []
    cumulative_risk = 0
    
    for decision in base_schedule:
        decision_risk = calculate_decision_risk(decision)
        
        if cumulative_risk + decision_risk <= risk_limit:
            risk_adjusted_schedule.append(decision)
            cumulative_risk += decision_risk
        else:
            # Scale down the decision
            scale_factor = (risk_limit - cumulative_risk) / decision_risk
            decision['optimal_quantity'] *= scale_factor
            risk_adjusted_schedule.append(decision)
            break
    
    return risk_adjusted_schedule
```

---

## Examples

### Example 1: Basic Integration Setup
```python
#!/usr/bin/env python3
"""
Basic example of how I set up myopic scheduling integration
"""

from myopic_sor_scheduler import MyopicScheduler, MyopicParameters
from main_myopic_final import run_job_with_myopic
import pandas as pd

def basic_integration_example():
    # Define parameters as I configured them
    strategy_params = {
        'S': 1000,         # Trade 1000 shares
        'T': 15,           # Over 15 minutes
        'f': 0.003,        # 30 bps fee
        'r': [0.002],      # 20 bps rebate
        'lambda_u': 0.05,  # Underfill penalty
        'lambda_o': 0.05,  # Overfill penalty
        'N': 500           # Monte Carlo simulations
    }
    
    # How the integration works with existing SOR
    stock, results = run_job_with_myopic(
        stock="AAPL",
        day="2025-04-02",
        strategy_params=strategy_params,
        data_path="./data/",
        order_freq=60,  # My scheduler checks every minute
        start_time=("09", "30"),
        end_time=("16", "00"),
        lookup_duration=(0, 30),  # 30-minute lookback for SOR
        market_data_path="./data/",
        use_myopic=True
    )
    
    # The integration preserves all SOR functionality
    print(f"Stock: {stock}")
    print(f"Myopic enabled: {results['myopic_enabled']}")
    print(f"Total decisions: {results['num_decisions']}")
    print(f"Lambda used: {results['lambda_used']:.2f}")
    print(f"Each decision was optimized by existing SOR for market/limit split")
    
    return results

if __name__ == "__main__":
    basic_integration_example()
```

### Example 2: Comparison Framework I Built
```python
#!/usr/bin/env python3
"""
Example of the comparison framework I implemented
"""

from main_myopic_final import compare_myopic_vs_traditional
from myopic_analysis_utils import MyopicAnalyzer

def comparison_example():
    # My comparison framework setup
    strategy_params = {
        'S': 500,
        'T': 10,
        'f': 0.003,
        'r': [0.003],
        'lambda_u': 0.05,
        'lambda_o': 0.05,
        'N': 1000
    }
    
    # Run comparison using my framework
    comparison = compare_myopic_vs_traditional(
        stock="AAPL",
        day="2025-04-02",
        strategy_params=strategy_params,
        data_path="./data/",
        order_freq=120,
        start_time=("09", "30"),
        end_time=("16", "00"),
        lookup_duration=(0, 15),
        market_data_path="./data/"
    )
    
    # Analysis using my utility functions
    analyzer = MyopicAnalyzer()
    
    # Generate report using my framework
    comparison_data = {f"AAPL_2025-04-02": comparison}
    report = analyzer.generate_performance_report(comparison_data)
    
    print("=== My Comparison Framework Results ===")
    print(report)
    
    # Create visualization using my tools
    if comparison['myopic']['results']:
        analyzer.create_schedule_visualization(
            comparison['myopic'], 
            save_path="myopic_schedule_analysis.png"
        )
    
    return comparison

if __name__ == "__main__":
    comparison_example()
```

### Example 3: Parameter Sensitivity Analysis Framework
```python
#!/usr/bin/env python3
"""
Example of parameter sensitivity analysis I implemented
"""

from myopic_sor_scheduler import MyopicScheduler, MyopicParameters
from myopic_analysis_utils import MyopicAnalyzer
import numpy as np

def parameter_sensitivity_example():
    # My parameter sensitivity framework
    base_params = MyopicParameters(
        lambda_value=25000.0,
        beta=0.693,
        volatility=0.01,
        adv=1000000.0
    )
    
    # Test different lambda values as I designed
    lambda_values = np.linspace(15000, 35000, 5)
    sensitivity_results = {}
    
    for lambda_val in lambda_values:
        # Create scheduler with different lambda
        test_params = MyopicParameters(
            lambda_value=lambda_val,
            beta=base_params.beta,
            volatility=base_params.volatility,
            adv=base_params.adv
        )
        
        # This would run my enhanced backtest
        # results = enhanced_backtest_with_myopic(..., myopic_params=test_params)
        # sensitivity_results[lambda_val] = results
    
    # Analysis framework I built
    analyzer = MyopicAnalyzer()
    # sensitivity_df = analyzer.parameter_sensitivity_analysis(sensitivity_results, lambda_values)
    
    print("=== My Parameter Sensitivity Framework ===")
    print("Framework ready to analyze lambda sensitivity")
    print(f"Testing lambda values: {lambda_values}")
    
    return sensitivity_results

if __name__ == "__main__":
    parameter_sensitivity_example()
```

### Example 4: Batch Processing Framework I Designed
```python
#!/usr/bin/env python3
"""
Example of batch processing framework I built for multiple stocks/days
"""

from main_myopic_final import run_job_with_myopic
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd

def batch_processing_example():
    # My batch processing configuration
    stocks = ["AAPL", "MSFT", "GOOGL"]
    days = ["2025-04-01", "2025-04-02", "2025-04-03"]
    
    strategy_params = {
        'S': 100,
        'T': 5,
        'f': 0.003,
        'r': [0.003],
        'lambda_u': 0.05,
        'lambda_o': 0.05,
        'N': 1000
    }
    
    # My parallel processing framework
    def process_combination(stock, day):
        try:
            stock_result, results = run_job_with_myopic(
                stock=stock,
                day=day,
                strategy_params=strategy_params,
                data_path="./data/",
                order_freq=120,
                start_time=("09", "30"),
                end_time=("16", "00"),
                lookup_duration=(0, 15),
                market_data_path="./data/",
                use_myopic=True
            )
            return (stock, day, results)
        except Exception as e:
            print(f"Error processing {stock} on {day}: {e}")
            return (stock, day, None)
    
    # Execute batch processing using my framework
    all_results = {}
    
    # Sequential processing example I designed
    for stock in stocks:
        for day in days:
            key = f"{stock}_{day}"
            print(f"Processing {key}...")
            
            stock_name, day_name, result = process_combination(stock, day)
            if result:
                all_results[key] = result
                print(f"✅ Completed {key}: {result['num_decisions']} decisions")
            else:
                print(f"❌ Failed {key}")
    
    # Summary analysis using my framework
    print("\n=== My Batch Processing Results Summary ===")
    for key, result in all_results.items():
        if result:
            print(f"{key}: ${result['avg_cost_per_share']:.4f} avg cost, "
                  f"{result['num_decisions']} decisions, "
                  f"λ={result['lambda_used']:.0f}")
    
    return all_results

if __name__ == "__main__":
    batch_processing_example()
```

### Example 5: Custom Analysis Pipeline I Created
```python
#!/usr/bin/env python3
"""
Example of custom analysis pipeline I built
"""

from myopic_analysis_utils import MyopicAnalyzer, create_lambda_comparison_plot
import pandas as pd
import matplotlib.pyplot as plt

def custom_analysis_example():
    # My custom analysis pipeline
    analyzer = MyopicAnalyzer()
    
    # Simulate lambda analysis results (from my notebook implementation)
    lambda_stats_data = {
        '10s': {'final_pnl': 1.99, 'mean_pnl': 1.01, 'std_pnl': 0.72, 'sharpe': 1.40, 'lambda': 8580},
        '30s': {'final_pnl': 2.80, 'mean_pnl': 1.42, 'std_pnl': 1.02, 'sharpe': 1.40, 'lambda': 6103},
        '60s': {'final_pnl': 4.16, 'mean_pnl': 2.12, 'std_pnl': 1.51, 'sharpe': 1.40, 'lambda': 4100},
        '120s': {'final_pnl': 5.10, 'mean_pnl': 2.59, 'std_pnl': 1.85, 'sharpe': 1.40, 'lambda': 3349}
    }
    
    # Convert to DataFrame as my analysis expects
    lambda_stats_df = pd.DataFrame.from_dict(lambda_stats_data, orient='index')
    
    # Generate visualization using my framework
    print("=== My Lambda Analysis Framework ===")
    print("Creating lambda comparison visualization...")
    
    # This would create the plot using my utility function
    create_lambda_comparison_plot(
        lambda_stats_df, 
        save_path="lambda_analysis.png"
    )
    
    # Performance report generation using my framework
    mock_comparison_results = {
        'AAPL_2025-04-02': {
            'myopic': {
                'avg_cost_per_share': 234.45,
                'num_decisions': 25,
                'lambda_used': 18450
            },
            'traditional': {
                'avg_cost_per_share': 234.60,
                'num_decisions': 30
            },
            'improvement': {
                'cost_improvement_pct': 0.64
            },
            'stock': 'AAPL',
            'comparison_date': '2025-04-02'
        }
    }
    
    # Generate comprehensive report using my system
    report = analyzer.generate_performance_report(
        mock_comparison_results,
        output_path="myopic_performance_report.txt"
    )
    
    print("📊 Analysis complete - files generated:")
    print("  - lambda_analysis.png")
    print("  - myopic_performance_report.txt")
    
    return lambda_stats_df, report

if __name__ == "__main__":
    custom_analysis_example()
```

---

## Development Notes

### My Implementation Approach
1. **Modular Design**: I built the system as separate modules that integrate cleanly with existing code
2. **Backward Compatibility**: All existing SOR functionality remains unchanged
3. **Configuration Driven**: Easy to adjust parameters without code changes
4. **Comprehensive Testing**: Built-in validation and error handling
5. **Performance Focused**: Designed for production-scale execution

### My Development Philosophy
- **Incremental Integration**: Start simple, add complexity gradually
- **Fail-Safe Design**: Always fall back to traditional methods if myopic fails
- **Comprehensive Logging**: Track every decision for analysis and debugging
- **Flexible Framework**: Easy to extend and modify for different use cases

### My Code Organization
I structured the code to maintain clean separation of concerns:
- **Scheduler Logic**: Pure mathematical implementation of myopic model
- **Integration Layer**: Bridge between myopic scheduler and existing SOR
- **Analysis Tools**: Comprehensive performance analysis and visualization
- **Configuration Management**: Centralized parameter management
- **Error Handling**: Robust error recovery and logging

This system represents my implementation of advanced algorithmic trading concepts integrated with practical production requirements.