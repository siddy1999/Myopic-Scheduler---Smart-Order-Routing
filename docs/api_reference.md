# API Reference

This document provides detailed API documentation for the Myopic Scheduler project.

## Table of Contents

- [MyopicScheduler](#myopicscheduler)
- [MyopicParameters](#myopicparameters)
- [MyopicAnalyzer](#myopicanalyzer)
- [Utility Functions](#utility-functions)

## MyopicScheduler

The main class for implementing myopic scheduling algorithms.

### Constructor

```python
MyopicScheduler(params: MyopicParameters)
```

**Parameters:**
- `params` (MyopicParameters): Configuration parameters for the scheduler

**Example:**
```python
from myopic_sor_scheduler import MyopicScheduler, MyopicParameters

params = MyopicParameters(
    lambda_value=25000.0,
    beta=0.693,
    volatility=0.01,
    adv=1000000.0
)
scheduler = MyopicScheduler(params)
```

### Methods

#### estimate_lambda

```python
estimate_lambda(df: pd.DataFrame, agg_periods: List[float] = None) -> Dict[str, float]
```

Estimates lambda parameters for different aggregation periods using linear regression.

**Parameters:**
- `df` (pd.DataFrame): Market data with required columns
- `agg_periods` (List[float], optional): List of aggregation periods in seconds

**Returns:**
- `Dict[str, float]`: Dictionary mapping period strings to lambda values

**Required DataFrame Columns:**
- `ts_event`: Timestamp
- `bid_fill`: Bid-side fill volume
- `ask_fill`: Ask-side fill volume
- `signed_volume`: Signed volume (bid_fill - ask_fill)
- `best_bid`: Best bid price
- `best_ask`: Best ask price
- `Volatility`: Asset volatility
- `ADV`: Average Daily Volume

**Example:**
```python
lambda_values = scheduler.estimate_lambda(market_data, [1, 5, 10, 30, 60])
# Returns: {'1s': 28000.5, '5s': 25600.2, '10s': 24000.1, ...}
```

#### calculate_price_impact

```python
calculate_price_impact(df: pd.DataFrame, lambda_val: float = None) -> pd.DataFrame
```

Calculates cumulative price impact using the myopic model.

**Parameters:**
- `df` (pd.DataFrame): Market data DataFrame
- `lambda_val` (float, optional): Override lambda value

**Returns:**
- `pd.DataFrame`: DataFrame with added `price_impact` column

**Formula:**
```
I(t+1) = I(t) + λ * Q(t) * σ/ADV - β * I(t) * Δt
```

#### calculate_alpha_and_derivatives

```python
calculate_alpha_and_derivatives(df: pd.DataFrame) -> pd.DataFrame
```

Calculates alpha (price prediction) and its derivatives.

**Parameters:**
- `df` (pd.DataFrame): DataFrame with price impact calculations

**Returns:**
- `pd.DataFrame`: DataFrame with alpha calculations

**Added Columns:**
- `unperturbed_price`: Mid price minus market impact
- `determined_alpha`: Expected price movement
- `determined_alpha_prime`: Rate of change of alpha

#### calculate_optimal_impact

```python
calculate_optimal_impact(df: pd.DataFrame) -> pd.DataFrame
```

Calculates optimal impact I* using the myopic model.

**Parameters:**
- `df` (pd.DataFrame): DataFrame with alpha calculations

**Returns:**
- `pd.DataFrame`: DataFrame with optimal impact calculations

**Added Columns:**
- `I_star_t`: Optimal market impact
- `I_star_prime`: Derivative of optimal impact
- `delta_Q_star`: Optimal trading quantity

#### generate_trading_schedule

```python
generate_trading_schedule(df: pd.DataFrame, total_quantity: float, time_horizon: int) -> List[Dict]
```

Generates a myopic trading schedule.

**Parameters:**
- `df` (pd.DataFrame): Market data DataFrame
- `total_quantity` (float): Total quantity to trade
- `time_horizon` (int): Trading horizon in minutes

**Returns:**
- `List[Dict]`: List of trading decisions

**Decision Dictionary Keys:**
- `timestamp`: Decision timestamp
- `optimal_quantity`: Optimal trading quantity
- `price_impact`: Current price impact
- `optimal_impact`: Optimal market impact
- `alpha`: Alpha signal value
- `mid_price`: Mid price at decision time

#### integrate_with_sor

```python
integrate_with_sor(dfs: Dict[str, pd.DataFrame], myopic_schedule: List[Dict], sor_params: Dict) -> List[Dict]
```

Integrates myopic schedule with SOR optimization.

**Parameters:**
- `dfs` (Dict[str, pd.DataFrame]): Dictionary of venue DataFrames
- `myopic_schedule` (List[Dict]): Myopic trading schedule
- `sor_params` (Dict): SOR optimization parameters

**Returns:**
- `List[Dict]`: List of integrated trading decisions

## MyopicParameters

Configuration parameters for the myopic scheduling model.

### Constructor

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
- `lambda_value` (float): Market impact coefficient
- `beta` (float): Impact decay parameter
- `volatility` (float): Asset volatility
- `adv` (float): Average Daily Volume
- `T` (float, optional): Trading session length in hours (default: 6.5)
- `Q_0` (float, optional): Initial position normalization (default: 0.01)

### Attributes

- `lambda_value`: Market impact coefficient
- `beta`: Impact decay parameter (β = ln(2)/half_life)
- `volatility`: Asset volatility
- `adv`: Average Daily Volume
- `T`: Trading session length in hours
- `Q_0`: Initial position normalization

## MyopicAnalyzer

Utility class for analyzing myopic scheduling results.

### Constructor

```python
MyopicAnalyzer()
```

### Methods

#### analyze_schedule_performance

```python
analyze_schedule_performance(myopic_results: Dict, traditional_results: Dict) -> Dict
```

Analyzes performance differences between myopic and traditional approaches.

**Parameters:**
- `myopic_results` (Dict): Results from myopic scheduling
- `traditional_results` (Dict): Results from traditional SOR

**Returns:**
- `Dict`: Performance analysis results

**Analysis Keys:**
- `cost_improvement_pct`: Cost improvement percentage
- `myopic_decisions`: Number of myopic decisions
- `traditional_decisions`: Number of traditional decisions
- `size_efficiency`: Size efficiency ratio
- `lambda_used`: Lambda parameter used

#### create_schedule_visualization

```python
create_schedule_visualization(results: Dict, save_path: Optional[str] = None)
```

Creates visualization of the trading schedule and impacts.

**Parameters:**
- `results` (Dict): Myopic scheduling results
- `save_path` (str, optional): Path to save the plot

**Generated Plots:**
- Order quantities over time
- Price impact evolution
- Market vs Limit price levels
- Myopic alpha signals

#### parameter_sensitivity_analysis

```python
parameter_sensitivity_analysis(base_results: Dict, lambda_values: List[float]) -> pd.DataFrame
```

Analyzes sensitivity to lambda parameter values.

**Parameters:**
- `base_results` (Dict): Base case results
- `lambda_values` (List[float]): List of lambda values to test

**Returns:**
- `pd.DataFrame`: Sensitivity analysis results

#### generate_performance_report

```python
generate_performance_report(comparison_results: Dict, output_path: Optional[str] = None) -> str
```

Generates a comprehensive performance report.

**Parameters:**
- `comparison_results` (Dict): Dictionary of comparison results
- `output_path` (str, optional): Path to save the report

**Returns:**
- `str`: Performance report text

## Utility Functions

### create_lambda_comparison_plot

```python
create_lambda_comparison_plot(lambda_stats_df: pd.DataFrame, save_path: Optional[str] = None)
```

Creates visualization comparing different lambda values and their performance.

**Parameters:**
- `lambda_stats_df` (pd.DataFrame): DataFrame with lambda statistics
- `save_path` (str, optional): Path to save the plot

### integrate_myopic_with_existing_metrics

```python
integrate_myopic_with_existing_metrics(myopic_results: Dict, traditional_metrics: pd.DataFrame) -> pd.DataFrame
```

Integrates myopic results with existing metrics framework.

**Parameters:**
- `myopic_results` (Dict): Results from myopic scheduling
- `traditional_metrics` (pd.DataFrame): Traditional metrics DataFrame

**Returns:**
- `pd.DataFrame`: Combined metrics DataFrame

## Data Requirements

### Market Data Format

The system expects market data in the following format:

```python
{
    'ts_event': pd.Timestamp,      # Timestamp
    'best_bid': float,             # Best bid price
    'best_ask': float,             # Best ask price
    'bid_fill': float,             # Bid-side fill volume
    'ask_fill': float,             # Ask-side fill volume
    'signed_volume': float,        # Signed volume (bid_fill - ask_fill)
    'Volatility': float,           # Asset volatility
    'ADV': float,                  # Average Daily Volume
    'size': float                  # Trade size (optional)
}
```

### SOR Parameters

```python
sor_params = {
    'T': int,                      # Time horizon
    'f': float,                    # Fee rate
    'r': List[float],              # Rebate rates per venue
    'lambda_u': float,             # Underfill penalty
    'lambda_o': float,             # Overfill penalty
    'N': int,                      # Monte Carlo simulations
    'method': str,                 # Optimization method
    'stock': str                   # Stock symbol
}
```

## Error Handling

### Common Exceptions

- `ValueError`: Invalid parameter values
- `KeyError`: Missing required DataFrame columns
- `ImportError`: Missing SOR modules
- `RuntimeError`: Optimization failures

### Error Recovery

The system includes built-in error recovery mechanisms:

1. **Lambda Estimation Fallback**: Uses default lambda if estimation fails
2. **SOR Integration Fallback**: Skips problematic decisions
3. **Data Validation**: Checks data quality before processing
4. **Graceful Degradation**: Falls back to traditional methods when needed

## Performance Considerations

### Memory Usage

- Process data in chunks for large datasets
- Use `dask` for distributed processing
- Cache intermediate results when possible

### Computational Complexity

- Lambda estimation: O(n log n) where n is data points
- Schedule generation: O(n) where n is time horizon
- SOR integration: O(m) where m is number of decisions

### Optimization Tips

1. Use vectorized operations where possible
2. Enable parallel processing for multiple stocks
3. Cache frequently used calculations
4. Use appropriate data types (float32 vs float64)
5. Profile code to identify bottlenecks

## Examples

See the `examples/` directory for comprehensive usage examples:

- `basic_usage.py`: Basic usage example
- `advanced_configuration.py`: Advanced configuration
- `batch_processing.py`: Batch processing example
- `parameter_sensitivity.py`: Parameter sensitivity analysis
