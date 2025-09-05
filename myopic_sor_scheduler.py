# myopic_sor_scheduler.py
"""
Myopic scheduling algorithm integrated with SOR (Smart Order Routing).

This module integrates the myopic market impact model with the existing SOR
optimization framework to provide optimal timing and sizing decisions.
"""

import pandas as pd
import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression
import yfinance as yf

# Import existing SOR modules
from ofi import calculate_ofi_signal, calculate_causal_ofi_signal
from facade import execute_optimization
from data_io import preprocess_data


@dataclass
class MyopicParameters:
    """Parameters for the myopic scheduling model"""
    lambda_value: float  # Market impact parameter
    beta: float         # Impact decay parameter
    volatility: float   # Asset volatility
    adv: float         # Average daily volume
    T: float = 6.5     # Trading session length (hours)
    Q_0: float = 0.01  # Initial position normalization


class MyopicScheduler:
    """
    Myopic scheduler that integrates with SOR optimization.
    
    This class implements the myopic market impact model to determine
    optimal trading quantities and timing, then uses SOR to determine
    venue allocation.
    """
    
    def __init__(self, params: MyopicParameters):
        self.params = params
        self.lambda_values = {}  # Store lambda values for different time horizons
        self.price_impact_history = []
        self.logger = logging.getLogger(__name__)
        
    def estimate_lambda(self, df: pd.DataFrame, agg_periods: List[float] = None) -> Dict[str, float]:
        """
        Estimate lambda parameter for different aggregation periods.
        
        Args:
            df: Market data DataFrame with required columns
            agg_periods: List of aggregation periods in seconds
            
        Returns:
            Dictionary mapping period strings to lambda values
        """
        if agg_periods is None:
            agg_periods = [0.1, 1, 2, 5, 10, 30, 60, 120, 300]
            
        # Ensure required columns exist
        required_cols = ['ts_event', 'bid_fill', 'ask_fill', 'signed_volume', 
                        'best_bid', 'best_ask', 'Volatility', 'ADV']
        
        # Set market hours (adjust as needed)
        market_open = pd.to_datetime('13:30:00').time()
        market_close = pd.to_datetime('20:00:00').time()
        
        # Filter for market hours
        df['ts_event'] = pd.to_datetime(df['ts_event'])
        df.set_index('ts_event', inplace=True)
        df_market = df.between_time(market_open, market_close)
        
        lambda_values = {}
        
        for period in agg_periods:
            try:
                # Resample data
                df_resampled = df_market.resample(f'{period}s').agg({
                    'bid_fill': 'sum',
                    'ask_fill': 'sum',
                    'signed_volume': 'sum',
                    'best_bid': 'last',
                    'best_ask': 'last',
                    'Volatility': 'last',
                    'ADV': 'last',
                }).dropna()
                
                if len(df_resampled) < 10:  # Need minimum data
                    continue
                    
                # Calculate mid price and deltas
                df_resampled['mid_price'] = (df_resampled['best_bid'] + df_resampled['best_ask']) / 2
                df_resampled['delta_p'] = df_resampled['mid_price'].diff()
                df_resampled['delta_Q'] = (df_resampled['signed_volume'] * 
                                         df_resampled['Volatility'] / df_resampled['ADV'])
                
                # Linear regression
                X = df_resampled[['delta_Q']].values[1:]
                y = df_resampled['delta_p'].values[1:]
                
                if len(X) > 0 and np.std(X) > 0:
                    model = LinearRegression()
                    model.fit(X, y)
                    lambda_values[f'{period}s'] = float(model.coef_[0])
                    
            except Exception as e:
                self.logger.warning(f"Failed to estimate lambda for period {period}s: {e}")
                continue
                
        self.lambda_values = lambda_values
        return lambda_values
    
    def calculate_price_impact(self, df: pd.DataFrame, lambda_val: float = None) -> pd.DataFrame:
        """
        Calculate cumulative price impact using the myopic model.
        
        Args:
            df: Market data DataFrame
            lambda_val: Override lambda value (uses default if None)
            
        Returns:
            DataFrame with price impact calculations
        """
        if lambda_val is None:
            lambda_val = self.params.lambda_value
            
        df = df.copy()
        df['price_impact'] = 0.0
        
        # Calculate impact iteratively
        for i in range(1, len(df)):
            if df.iloc[i]['ts_event'].hour == 13:  # Start of day reset
                continue
                
            I_t = df.iloc[i-1]['price_impact']
            delta_Q = (df.iloc[i]['signed_volume'] * 
                      df.iloc[i]['Volatility'] / df.iloc[i]['ADV'])
            
            delta_I = lambda_val * delta_Q - self.params.beta * I_t
            df.iloc[i, df.columns.get_loc('price_impact')] = I_t + delta_I
            
        return df
    
    def calculate_alpha_and_derivatives(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate alpha (price prediction) and its derivatives.
        
        Args:
            df: DataFrame with price impact calculations
            
        Returns:
            DataFrame with alpha calculations
        """
        df = df.copy()
        df['date'] = df['ts_event'].dt.date
        df['time'] = df['ts_event'].dt.time
        
        # Calculate unperturbed price
        df['unperturbed_price'] = df['mid_price'] - df['price_impact']
        
        # Calculate alpha (end-of-day price minus current price)
        end_time = pd.to_datetime('20:00:00').time()
        
        def get_closing_price(group):
            end_prices = group[group['time'] == end_time]['unperturbed_price']
            if len(end_prices) > 0:
                return end_prices.iloc[0]
            # Fallback to last available price
            return group['unperturbed_price'].iloc[-1]
        
        closing_prices = df.groupby('date').apply(get_closing_price)
        df['closing_price'] = df['date'].map(closing_prices)
        df['determined_alpha'] = df['closing_price'] - df['unperturbed_price']
        
        # Calculate alpha prime (derivative)
        df['determined_alpha_prime'] = df.groupby('date')['determined_alpha'].diff().fillna(0)
        
        return df.drop(['date', 'time', 'closing_price'], axis=1)
    
    def calculate_optimal_impact(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate optimal impact I* using the myopic model.
        
        Args:
            df: DataFrame with alpha calculations
            
        Returns:
            DataFrame with optimal impact calculations
        """
        df = df.copy()
        
        # Calculate optimal impact I*
        df['I_star_t'] = ((df['determined_alpha'] - 
                          (df['determined_alpha_prime'] / self.params.beta)) / 2).fillna(0)
        
        # Calculate optimal trade quantity
        df['I_star_prime'] = df['I_star_t'].diff().fillna(0)
        df['delta_Q_star'] = ((df['I_star_prime'] + self.params.beta * df['I_star_t']) / 
                             self.params.lambda_value).fillna(0)
        
        return df
    
    def generate_trading_schedule(self, df: pd.DataFrame, total_quantity: float, 
                                time_horizon: int) -> List[Dict]:
        """
        Generate a myopic trading schedule.
        
        Args:
            df: Market data DataFrame
            total_quantity: Total quantity to trade
            time_horizon: Trading horizon in minutes
            
        Returns:
            List of trading decisions with timestamps and quantities
        """
        # Calculate myopic model components
        df_with_impact = self.calculate_price_impact(df)
        df_with_alpha = self.calculate_alpha_and_derivatives(df_with_impact)
        df_optimal = self.calculate_optimal_impact(df_with_alpha)
        
        schedule = []
        current_time = df_optimal['ts_event'].iloc[0]
        end_time = current_time + timedelta(minutes=time_horizon)
        
        # Filter data for trading window
        trading_window = df_optimal[
            (df_optimal['ts_event'] >= current_time) & 
            (df_optimal['ts_event'] <= end_time)
        ]
        
        for _, row in trading_window.iterrows():
            if abs(row['delta_Q_star']) > 1e-6:  # Only include meaningful trades
                # Scale the optimal quantity by total quantity and time remaining
                scaled_quantity = row['delta_Q_star'] * total_quantity
                
                schedule.append({
                    'timestamp': row['ts_event'],
                    'optimal_quantity': scaled_quantity,
                    'price_impact': row['price_impact'],
                    'optimal_impact': row['I_star_t'],
                    'alpha': row['determined_alpha'],
                    'mid_price': row['mid_price']
                })
                
        return schedule
    
    def integrate_with_sor(self, dfs: Dict[str, pd.DataFrame], myopic_schedule: List[Dict], 
                          sor_params: Dict) -> List[Dict]:
        """
        Integrate myopic schedule with SOR optimization.
        
        Args:
            dfs: Dictionary of venue DataFrames
            myopic_schedule: Myopic trading schedule
            sor_params: SOR optimization parameters
            
        Returns:
            List of integrated trading decisions with venue allocations
        """
        integrated_schedule = []
        
        for trade_decision in myopic_schedule:
            timestamp = trade_decision['timestamp']
            quantity = abs(trade_decision['optimal_quantity'])
            
            if quantity < 1:  # Skip very small trades
                continue
                
            try:
                # Use SOR to determine venue allocation
                sor_result = execute_optimization(
                    dfs=dfs,
                    target_time=timestamp,
                    S=quantity,
                    **sor_params
                )
                
                if sor_result is not None and len(sor_result) > 0:
                    allocation = sor_result[0]  # Get allocation vector
                    
                    # Create venue-specific orders
                    venue_orders = {}
                    for i, venue in enumerate(dfs.keys()):
                        if i < len(allocation) and allocation[i] > 0.5:  # Minimum order size
                            venue_orders[venue] = {
                                'market_orders': int(allocation[i]),
                                'limit_orders': int(allocation[i+len(dfs)] if i+len(dfs) < len(allocation) else 0)
                            }
                    
                    integrated_decision = {
                        'timestamp': timestamp,
                        'total_quantity': quantity,
                        'myopic_signal': trade_decision['alpha'],
                        'price_impact': trade_decision['price_impact'],
                        'venue_allocation': venue_orders,
                        'sor_cost': sor_result[1] if len(sor_result) > 1 else None
                    }
                    
                    integrated_schedule.append(integrated_decision)
                    
            except Exception as e:
                self.logger.warning(f"SOR optimization failed for timestamp {timestamp}: {e}")
                continue
                
        return integrated_schedule


class MyopicSORBacktester:
    """
    Backtesting framework for the integrated myopic-SOR system.
    """
    
    def __init__(self, scheduler: MyopicScheduler):
        self.scheduler = scheduler
        self.results = []
        
    def backtest(self, dfs: Dict[str, pd.DataFrame], sor_params: Dict, 
                total_quantity: float, time_horizon: int) -> Dict:
        """
        Run backtest of the integrated system.
        
        Args:
            dfs: Dictionary of venue market data
            sor_params: SOR parameters
            total_quantity: Total quantity to trade
            time_horizon: Trading horizon in minutes
            
        Returns:
            Backtest results dictionary
        """
        # Use the first venue's data for myopic calculations
        primary_venue = list(dfs.keys())[0]
        df_primary = dfs[primary_venue].copy()
        
        # Generate myopic schedule
        myopic_schedule = self.scheduler.generate_trading_schedule(
            df_primary, total_quantity, time_horizon
        )
        
        # Integrate with SOR
        integrated_schedule = self.scheduler.integrate_with_sor(
            dfs, myopic_schedule, sor_params
        )
        
        # Calculate performance metrics
        total_traded = sum(decision['total_quantity'] for decision in integrated_schedule)
        avg_impact = np.mean([decision['price_impact'] for decision in integrated_schedule])
        
        results = {
            'total_decisions': len(integrated_schedule),
            'total_quantity_traded': total_traded,
            'fill_rate': total_traded / total_quantity if total_quantity > 0 else 0,
            'average_price_impact': avg_impact,
            'myopic_schedule': myopic_schedule,
            'integrated_schedule': integrated_schedule,
            'sor_params': sor_params
        }
        
        return results


# Integration with existing main execution flow
def run_myopic_sor_backtest(stock: str, day: str, strategy_params: Dict, 
                           data_path: str, order_freq: int, start_time: Tuple, 
                           end_time: Tuple, lookup_duration: Tuple, df: pd.DataFrame):
    """
    Modified version of the backtest function that includes myopic scheduling.
    
    This function integrates with your existing Main_1.py backtest workflow.
    """
    
    # Initialize myopic parameters (you may want to make these configurable)
    myopic_params = MyopicParameters(
        lambda_value=25000.0,  # Default lambda, will be estimated from data
        beta=math.log(2) / 1.0,  # 1-hour half-life
        volatility=0.01,  # Will be updated from data
        adv=1000000.0     # Will be updated from data
    )
    
    # Create scheduler
    scheduler = MyopicScheduler(myopic_params)
    
    # Prepare data for myopic model
    df_myopic = df.copy()
    
    # Add required columns if missing
    if 'signed_volume' not in df_myopic.columns:
        df_myopic['signed_volume'] = df_myopic.get('bid_fill', 0) - df_myopic.get('ask_fill', 0)
    
    if 'mid_price' not in df_myopic.columns:
        df_myopic['mid_price'] = (df_myopic['best_bid'] + df_myopic['best_ask']) / 2
    
    # Estimate lambda from historical data
    try:
        lambda_values = scheduler.estimate_lambda(df_myopic)
        if lambda_values:
            # Use the lambda value for 60-second aggregation if available
            best_lambda = lambda_values.get('60s', lambda_values.get('30s', 25000.0))
            scheduler.params.lambda_value = best_lambda
            print(f"Using estimated lambda: {best_lambda}")
    except Exception as e:
        print(f"Lambda estimation failed, using default: {e}")
    
    # Prepare SOR parameters
    sor_params = {
        'T': strategy_params['T'],
        'f': strategy_params['f'],
        'r': strategy_params['r'],
        'lambda_u': strategy_params['lambda_u'],
        'lambda_o': strategy_params['lambda_o'],
        'N': strategy_params['N'],
        'method': 'lookup',
        'stock': stock
    }
    
    # Create venue data dictionary (modify based on your actual venue structure)
    dfs = {'v1': preprocess_data(df)}
    
    # Run integrated backtest
    backtester = MyopicSORBacktester(scheduler)
    results = backtester.backtest(
        dfs=dfs,
        sor_params=sor_params,
        total_quantity=strategy_params['S'],
        time_horizon=strategy_params['T']
    )
    
    print(f"Myopic-SOR Backtest Results for {stock} on {day}:")
    print(f"Total decisions: {results['total_decisions']}")
    print(f"Fill rate: {results['fill_rate']:.2%}")
    print(f"Average price impact: {results['average_price_impact']:.6f}")
    
    return results


# Example usage function
def example_usage():
    """
    Example of how to use the myopic scheduler with your existing SOR system.
    """
    
    # This would integrate with your main_final.py
    strategy_params = {
        'S': 100,
        'T': 5,
        'f': 0.003,
        'r': [0.003],
        'theta': 0.0005,
        'lambda_u': 0.05,
        'lambda_o': 0.05,
        'N': 1000
    }
    
    # Example data loading and processing
    # df = pd.read_csv('your_market_data.csv')
    # results = run_myopic_sor_backtest(
    #     stock='AAPL',
    #     day='2025-04-02',
    #     strategy_params=strategy_params,
    #     data_path='your_data_path/',
    #     order_freq=120,
    #     start_time=('09', '30'),
    #     end_time=('16', '00'),
    #     lookup_duration=(0, 15),
    #     df=df
    # )
    
    print("Example usage function - integrate with your main execution flow")


if __name__ == "__main__":
    example_usage()
