# benchmark_framework.py
"""
End-to-End Scheduler Benchmark & Backtest Integration
Implements VWAP, TWAP, and Myopic schedulers with unified interface
"""

import pandas as pd
import numpy as np
import json
import time
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import argparse
import os
from pathlib import Path

# Import existing modules
from Main_1_enhanced import enhanced_backtest
from myopic_sor_scheduler import MyopicScheduler, MyopicParameters
from metrics import metrics


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs"""
    tickers: List[str]
    dates: List[str]
    venues: List[str]
    algorithms: List[str]
    
    # Strategy parameters
    order_size: int = 100
    time_horizon: int = 5
    fee_rate: float = 0.003
    rebate_rates: List[float] = None
    lambda_u: float = 0.05
    lambda_o: float = 0.05
    n_simulations: int = 1000
    
    # Execution parameters
    order_freq: int = 120
    start_time: Tuple[str, str] = ("09", "30")
    end_time: Tuple[str, str] = ("16", "00")
    lookup_duration: Tuple[int, int] = (0, 15)
    
    # Scheduler-specific parameters
    vwap_slice_size: int = 60  # seconds
    twap_interval: int = 120   # seconds
    myopic_lookback: int = 60  # minutes
    myopic_half_life: float = 1.0  # hours
    
    # Data paths
    data_path: str = "./data/"
    output_path: str = "./benchmark_results/"
    
    def __post_init__(self):
        if self.rebate_rates is None:
            self.rebate_rates = [0.002] * len(self.venues)


class BaseScheduler(ABC):
    """Abstract base class for all scheduling algorithms"""
    
    def __init__(self, name: str, config: BenchmarkConfig):
        self.name = name
        self.config = config
        self.logger = logging.getLogger(f"scheduler.{name}")
    
    @abstractmethod
    def generate_schedule(self, df: pd.DataFrame, total_quantity: float, 
                         time_horizon: int) -> List[Dict]:
        """Generate trading schedule for given parameters"""
        pass
    
    def get_strategy_params(self) -> Dict:
        """Get strategy parameters for SOR integration"""
        return {
            'S': self.config.order_size,
            'T': self.config.time_horizon,
            'f': self.config.fee_rate,
            'r': self.config.rebate_rates,
            'lambda_u': self.config.lambda_u,
            'lambda_o': self.config.lambda_o,
            'N': self.config.n_simulations
        }


class VWAPScheduler(BaseScheduler):
    """VWAP-based scheduling algorithm"""
    
    def __init__(self, config: BenchmarkConfig):
        super().__init__("VWAP", config)
        self.slice_size = config.vwap_slice_size
    
    def generate_schedule(self, df: pd.DataFrame, total_quantity: float, 
                         time_horizon: int) -> List[Dict]:
        """Generate VWAP schedule based on historical volume patterns"""
        
        # Calculate historical volume profile
        df['minute'] = df['ts_event'].dt.floor('1min')
        volume_profile = df.groupby('minute')['size'].sum()
        
        # Normalize to create percentage allocation
        total_volume = volume_profile.sum()
        if total_volume == 0:
            # Fallback to uniform distribution
            volume_weights = pd.Series(1.0 / len(volume_profile), 
                                     index=volume_profile.index)
        else:
            volume_weights = volume_profile / total_volume
        
        # Generate schedule
        schedule = []
        start_time = df['ts_event'].min()
        
        for minute, weight in volume_weights.items():
            if minute < start_time:
                continue
                
            quantity = total_quantity * weight
            if quantity >= 1:  # Minimum trade size
                schedule.append({
                    'timestamp': minute,
                    'optimal_quantity': quantity,
                    'algorithm': 'VWAP',
                    'weight': weight,
                    'scheduler_info': {
                        'slice_size': self.slice_size,
                        'volume_weight': weight
                    }
                })
        
        return schedule


class TWAPScheduler(BaseScheduler):
    """TWAP-based scheduling algorithm"""
    
    def __init__(self, config: BenchmarkConfig):
        super().__init__("TWAP", config)
        self.interval = config.twap_interval
    
    def generate_schedule(self, df: pd.DataFrame, total_quantity: float, 
                         time_horizon: int) -> List[Dict]:
        """Generate TWAP schedule with uniform time distribution"""
        
        start_time = df['ts_event'].min()
        end_time = start_time + pd.Timedelta(minutes=time_horizon)
        
        # Calculate number of slices
        total_seconds = (end_time - start_time).total_seconds()
        num_slices = max(1, int(total_seconds / self.interval))
        quantity_per_slice = total_quantity / num_slices
        
        # Generate uniform schedule
        schedule = []
        current_time = start_time
        
        for i in range(num_slices):
            schedule.append({
                'timestamp': current_time,
                'optimal_quantity': quantity_per_slice,
                'algorithm': 'TWAP',
                'slice_number': i + 1,
                'scheduler_info': {
                    'interval': self.interval,
                    'total_slices': num_slices,
                    'quantity_per_slice': quantity_per_slice
                }
            })
            current_time += pd.Timedelta(seconds=self.interval)
        
        return schedule


class MyopicSchedulerWrapper(BaseScheduler):
    """Wrapper for existing MyopicScheduler to match interface"""
    
    def __init__(self, config: BenchmarkConfig):
        super().__init__("Myopic", config)
        
        # Initialize myopic parameters
        myopic_params = MyopicParameters(
            lambda_value=25000.0,  # Will be estimated from data
            beta=np.log(2) / config.myopic_half_life,
            volatility=0.01,
            adv=1000000.0
        )
        
        self.scheduler = MyopicScheduler(myopic_params)
    
    def generate_schedule(self, df: pd.DataFrame, total_quantity: float, 
                         time_horizon: int) -> List[Dict]:
        """Generate myopic schedule using existing implementation"""
        
        # Prepare data for myopic model (same as your existing code)
        df_analysis = df.copy()
        
        # Add required columns if missing
        if 'signed_volume' not in df_analysis.columns:
            df_analysis['signed_volume'] = (df_analysis.get('bid_fill', 0) - 
                                           df_analysis.get('ask_fill', 0))
        
        if 'mid_price' not in df_analysis.columns:
            df_analysis['mid_price'] = (df_analysis['best_bid'] + df_analysis['best_ask']) / 2
        
        # Add volatility and ADV if not present
        if 'Volatility' not in df_analysis.columns:
            df_analysis['Volatility'] = df_analysis['mid_price'].pct_change().rolling(20).std().fillna(0.01)
        if 'ADV' not in df_analysis.columns:
            df_analysis['ADV'] = df_analysis.get('size', 0).rolling(100).mean().fillna(1000000.0)
        
        # Estimate lambda parameter
        try:
            lambda_values = self.scheduler.estimate_lambda(df_analysis)
            if lambda_values:
                best_period = '60s' if '60s' in lambda_values else list(lambda_values.keys())[0]
                self.scheduler.params.lambda_value = lambda_values[best_period]
                self.logger.info(f"Using estimated lambda: {self.scheduler.params.lambda_value}")
        except Exception as e:
            self.logger.warning(f"Lambda estimation failed, using default: {e}")
        
        # Generate myopic schedule
        try:
            schedule = self.scheduler.generate_trading_schedule(
                df=df_analysis,
                total_quantity=total_quantity,
                time_horizon=time_horizon
            )
            
            # Add algorithm identifier
            for decision in schedule:
                decision['algorithm'] = 'Myopic'
                decision['scheduler_info'] = {
                    'lambda_used': self.scheduler.params.lambda_value,
                    'beta': self.scheduler.params.beta,
                    'lookback_minutes': self.config.myopic_lookback
                }
            
            return schedule
            
        except Exception as e:
            self.logger.error(f"Myopic schedule generation failed: {e}")
            return []


@dataclass
class BenchmarkResult:
    """Container for benchmark results"""
    ticker: str
    date: str
    algorithm: str
    
    # Performance metrics
    avg_cost_per_share: float = 0.0
    total_cost: float = 0.0
    total_size: int = 0
    fill_rate: float = 0.0
    num_decisions: int = 0
    
    # Slippage metrics
    slippage_vs_twap: float = 0.0
    slippage_vs_vwap: float = 0.0
    slippage_vs_midprice: float = 0.0
    
    # Execution metrics
    execution_time: float = 0.0
    compute_overhead: float = 0.0
    
    # Algorithm-specific info
    scheduler_info: Dict = None
    
    # Success flags
    completed_successfully: bool = False
    error_message: str = ""


class BenchmarkRunner:
    """Main benchmark execution engine"""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.schedulers = self._initialize_schedulers()
        self.results = []
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f"{config.output_path}/benchmark.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("benchmark")
        
        # Create output directory
        Path(config.output_path).mkdir(parents=True, exist_ok=True)
    
    def _initialize_schedulers(self) -> Dict[str, BaseScheduler]:
        """Initialize all schedulers"""
        schedulers = {}
        
        if "VWAP" in self.config.algorithms:
            schedulers["VWAP"] = VWAPScheduler(self.config)
        if "TWAP" in self.config.algorithms:
            schedulers["TWAP"] = TWAPScheduler(self.config)
        if "Myopic" in self.config.algorithms:
            schedulers["Myopic"] = MyopicSchedulerWrapper(self.config)
        
        return schedulers
    
    def run_single_benchmark(self, ticker: str, date: str, algorithm: str) -> BenchmarkResult:
        """Run benchmark for single ticker/date/algorithm combination"""
        
        result = BenchmarkResult(ticker=ticker, date=date, algorithm=algorithm)
        start_time = time.time()
        
        try:
            self.logger.info(f"Running {algorithm} for {ticker} on {date}")
            
            # Load market data
            df = self._load_market_data(ticker, date)
            if df is None or df.empty:
                raise ValueError(f"No data available for {ticker} on {date}")
            
            # Get scheduler
            scheduler = self.schedulers[algorithm]
            
            # Measure compute overhead
            compute_start = time.time()
            
            # Generate schedule
            schedule = scheduler.generate_schedule(
                df, self.config.order_size, self.config.time_horizon
            )
            
            compute_overhead = time.time() - compute_start
            
            if not schedule:
                raise ValueError(f"Empty schedule generated by {algorithm}")
            
            # Run backtest using your existing infrastructure
            backtest_result = self._run_backtest_with_schedule(
                ticker, date, schedule, scheduler.get_strategy_params()
            )
            
            # Calculate metrics
            result = self._calculate_metrics(result, backtest_result, schedule)
            result.compute_overhead = compute_overhead
            result.execution_time = time.time() - start_time
            result.completed_successfully = True
            
            # Store scheduler-specific info
            if schedule:
                result.scheduler_info = schedule[0].get('scheduler_info', {})
            
            self.logger.info(f"✅ {algorithm} completed for {ticker} on {date}")
            
        except Exception as e:
            self.logger.error(f"❌ {algorithm} failed for {ticker} on {date}: {e}")
            result.error_message = str(e)
            result.execution_time = time.time() - start_time
        
        return result
    
    def _load_market_data(self, ticker: str, date: str) -> pd.DataFrame:
        """Load market data for given ticker and date"""
        # Use your existing data loading logic
        date_str = date.replace("-", "")
        file_path = f"{self.config.data_path}{ticker}/xnas-itch-{date_str}.mbp-10.csv"
        
        try:
            df = pd.read_csv(file_path)
            df['ts_event'] = pd.to_datetime(df['ts_event'], utc=True)
            
            # Filter for trading window
            start = f"{date} {self.config.start_time[0]}:{self.config.start_time[1]}"
            end = f"{date} {self.config.end_time[0]}:{self.config.end_time[1]}"
            start_dt = pd.to_datetime(start, utc=True)
            end_dt = pd.to_datetime(end, utc=True)
            
            filtered_df = df[(df['ts_event'] >= start_dt) & (df['ts_event'] <= end_dt)]
            return filtered_df
            
        except Exception as e:
            self.logger.error(f"Failed to load data for {ticker} on {date}: {e}")
            return None
    
    def _run_backtest_with_schedule(self, ticker: str, date: str, 
                                   schedule: List[Dict], strategy_params: Dict) -> Dict:
        """Run backtest using provided schedule"""
        
        # This integrates with your existing enhanced_backtest
        # You'll need to modify enhanced_backtest to accept pre-generated schedules
        
        try:
            if schedule[0]['algorithm'] == 'Myopic':
                # Use your existing myopic backtest
                results = enhanced_backtest(
                    stock=ticker,
                    days=[date],
                    strategy_params=strategy_params,
                    data_path=self.config.data_path,
                    frequency=self.config.order_freq,
                    start_time=self.config.start_time,
                    end_time=self.config.end_time,
                    lookup_duration=self.config.lookup_duration,
                    use_myopic=True
                )
            else:
                # Use traditional backtest with custom schedule
                results = self._run_custom_schedule_backtest(
                    ticker, date, schedule, strategy_params
                )
            
            return results
            
        except Exception as e:
            self.logger.error(f"Backtest failed: {e}")
            return {}
    
    def _run_custom_schedule_backtest(self, ticker: str, date: str, 
                                     schedule: List[Dict], strategy_params: Dict) -> Dict:
        """Run backtest with custom (VWAP/TWAP) schedule"""
        # Implement custom backtest logic for VWAP/TWAP schedules
        # This should follow similar structure to your enhanced_backtest
        # but use the provided schedule instead of generating one
        
        # Placeholder implementation
        return {
            'total_cost': 0,
            'total_size': 0,
            'avg_cost_per_share': 0,
            'num_decisions': len(schedule),
            'results': []
        }
    
    def _calculate_metrics(self, result: BenchmarkResult, 
                          backtest_result: Dict, schedule: List[Dict]) -> BenchmarkResult:
        """Calculate performance metrics from backtest results"""
        
        # Basic metrics
        result.total_cost = backtest_result.get('total_cost', 0)
        result.total_size = backtest_result.get('total_size', 0)
        result.avg_cost_per_share = backtest_result.get('avg_cost_per_share', 0)
        result.num_decisions = backtest_result.get('num_decisions', len(schedule))
        
        # Calculate fill rate
        if self.config.order_size > 0:
            result.fill_rate = result.total_size / self.config.order_size
        
        # TODO: Calculate slippage metrics using your existing metrics framework
        # This would integrate with your metrics.py
        
        return result
    
    def run_full_benchmark(self) -> List[BenchmarkResult]:
        """Run complete benchmark across all configurations"""
        
        self.logger.info(f"Starting full benchmark")
        self.logger.info(f"Tickers: {self.config.tickers}")
        self.logger.info(f"Dates: {self.config.dates}")
        self.logger.info(f"Algorithms: {self.config.algorithms}")
        
        total_runs = len(self.config.tickers) * len(self.config.dates) * len(self.config.algorithms)
        completed_runs = 0
        
        for ticker in self.config.tickers:
            for date in self.config.dates:
                for algorithm in self.config.algorithms:
                    result = self.run_single_benchmark(ticker, date, algorithm)
                    self.results.append(result)
                    
                    completed_runs += 1
                    progress = (completed_runs / total_runs) * 100
                    self.logger.info(f"Progress: {progress:.1f}% ({completed_runs}/{total_runs})")
        
        # Save results
        self._save_results()
        
        return self.results
    
    def _save_results(self):
        """Save benchmark results to files"""
        
        # Save as JSON
        results_dict = [asdict(result) for result in self.results]
        json_path = f"{self.config.output_path}/benchmark_results.json"
        with open(json_path, 'w') as f:
            json.dump(results_dict, f, indent=2, default=str)
        
        # Save as CSV
        df = pd.DataFrame(results_dict)
        csv_path = f"{self.config.output_path}/benchmark_results.csv"
        df.to_csv(csv_path, index=False)
        
        # Save summary statistics
        self._generate_summary_report()
        
        self.logger.info(f"Results saved to {json_path} and {csv_path}")
    
    def _generate_summary_report(self):
        """Generate summary statistics and report"""
        
        df = pd.DataFrame([asdict(result) for result in self.results])
        
        # Filter successful runs
        successful_df = df[df['completed_successfully'] == True]
        
        if successful_df.empty:
            self.logger.warning("No successful runs to analyze")
            return
        
        # Calculate summary statistics by algorithm
        summary = successful_df.groupby('algorithm').agg({
            'avg_cost_per_share': ['mean', 'std', 'min', 'max'],
            'fill_rate': ['mean', 'std'],
            'execution_time': ['mean', 'std'],
            'compute_overhead': ['mean', 'std']
        }).round(6)
        
        # Save summary
        summary_path = f"{self.config.output_path}/benchmark_summary.csv"
        summary.to_csv(summary_path)
        
        # Generate text report
        report_lines = [
            "BENCHMARK SUMMARY REPORT",
            "=" * 50,
            f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            f"Total runs: {len(self.results)}",
            f"Successful runs: {len(successful_df)}",
            f"Success rate: {len(successful_df)/len(self.results)*100:.1f}%",
            "",
            "PERFORMANCE BY ALGORITHM:",
            "-" * 30
        ]
        
        for algorithm in self.config.algorithms:
            algo_results = successful_df[successful_df['algorithm'] == algorithm]
            if not algo_results.empty:
                avg_cost = algo_results['avg_cost_per_share'].mean()
                avg_fill_rate = algo_results['fill_rate'].mean()
                avg_time = algo_results['execution_time'].mean()
                
                report_lines.extend([
                    f"{algorithm}:",
                    f"  Average cost per share: ${avg_cost:.6f}",
                    f"  Average fill rate: {avg_fill_rate:.2%}",
                    f"  Average execution time: {avg_time:.2f}s",
                    ""
                ])
        
        report_path = f"{self.config.output_path}/benchmark_report.txt"
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        self.logger.info(f"Summary report saved to {report_path}")


def main():
    """Main entry point for benchmark execution"""
    
    parser = argparse.ArgumentParser(description='Run scheduler benchmark')
    parser.add_argument('--config', type=str, help='Path to config JSON file')
    parser.add_argument('--tickers', nargs='+', default=['AAPL'], help='Tickers to test')
    parser.add_argument('--dates', nargs='+', default=['2025-04-02'], help='Dates to test')
    parser.add_argument('--algorithms', nargs='+', default=['Myopic', 'VWAP', 'TWAP'], 
                       help='Algorithms to test')
    parser.add_argument('--output-path', type=str, default='./benchmark_results/', 
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
        config = BenchmarkConfig(**config_dict)
    else:
        config = BenchmarkConfig(
            tickers=args.tickers,
            dates=args.dates,
            venues=['v1'],  # Default single venue
            algorithms=args.algorithms,
            output_path=args.output_path
        )
    
    # Run benchmark
    runner = BenchmarkRunner(config)
    results = runner.run_full_benchmark()
    
    print(f"\nBenchmark completed! Results saved to {config.output_path}")
    print(f"Total runs: {len(results)}")
    print(f"Successful runs: {sum(1 for r in results if r.completed_successfully)}")


if __name__ == "__main__":
    main()
