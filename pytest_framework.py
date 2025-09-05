# tests/conftest.py
"""
Pytest configuration and fixtures for scheduler testing
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import shutil
from pathlib import Path

from benchmark_framework import BenchmarkConfig, VWAPScheduler, TWAPScheduler, MyopicSchedulerWrapper


@pytest.fixture
def sample_market_data():
    """Generate sample market data for testing"""
    
    start_time = pd.Timestamp('2025-04-02 09:30:00', tz='UTC')
    end_time = pd.Timestamp('2025-04-02 16:00:00', tz='UTC')
    
    # Generate timestamps every 100ms
    timestamps = pd.date_range(start_time, end_time, freq='100ms')
    n_rows = len(timestamps)
    
    # Generate realistic market data
    np.random.seed(42)  # For reproducible tests
    
    # Price data with random walk
    initial_price = 150.0
    price_changes = np.random.normal(0, 0.01, n_rows).cumsum()
    mid_prices = initial_price + price_changes
    
    # Bid-ask spread
    spreads = np.random.gamma(2, 0.02, n_rows)  # Realistic spread distribution
    best_bids = mid_prices - spreads/2
    best_asks = mid_prices + spreads/2
    
    # Volume data
    sizes = np.random.lognormal(3, 1, n_rows).astype(int)
    
    # Order flow data
    actions = np.random.choice(['T', 'C', 'A'], n_rows, p=[0.7, 0.2, 0.1])
    sides = np.random.choice(['A', 'B'], n_rows, p=[0.5, 0.5])
    
    df = pd.DataFrame({
        'ts_event': timestamps,
        'best_bid': best_bids,
        'best_ask': best_asks,
        'bid_sz_00': np.random.poisson(100, n_rows),
        'ask_sz_00': np.random.poisson(100, n_rows),
        'price': mid_prices,
        'size': sizes,
        'action': actions,
        'side': sides,
        'depth': np.random.randint(0, 5, n_rows),
        'bid_fill': np.random.poisson(10, n_rows),
        'ask_fill': np.random.poisson(10, n_rows)
    })
    
    return df


@pytest.fixture
def test_config():
    """Generate test configuration"""
    return BenchmarkConfig(
        tickers=['AAPL'],
        dates=['2025-04-02'],
        venues=['v1'],
        algorithms=['Myopic', 'VWAP', 'TWAP'],
        order_size=100,
        time_horizon=5,
        data_path='./test_data/',
        output_path='./test_results/'
    )


@pytest.fixture
def temp_directory():
    """Create temporary directory for test files"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_backtest_result():
    """Mock backtest result for testing"""
    return {
        'total_cost': 15000.0,
        'total_size': 100,
        'avg_cost_per_share': 150.0,
        'num_decisions': 5,
        'results': [
            {
                'id': 0,
                'time': '2025-04-02 09:30:00',
                'market_v': 20,
                'limit_v': 0,
                'filled': True,
                'avg_cost': 150.05
            }
        ]
    }


# tests/unit/test_schedulers.py
"""
Unit tests for individual scheduler components
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from benchmark_framework import VWAPScheduler, TWAPScheduler, MyopicSchedulerWrapper


class TestVWAPScheduler:
    """Test VWAP scheduler functionality"""
    
    def test_initialization(self, test_config):
        scheduler = VWAPScheduler(test_config)
        assert scheduler.name == "VWAP"
        assert scheduler.slice_size == test_config.vwap_slice_size
    
    def test_generate_schedule_basic(self, test_config, sample_market_data):
        scheduler = VWAPScheduler(test_config)
        schedule = scheduler.generate_schedule(sample_market_data, 1000, 30)
        
        # Basic validation
        assert isinstance(schedule, list)
        assert len(schedule) > 0
        
        # Check schedule structure
        for decision in schedule:
            assert 'timestamp' in decision
            assert 'optimal_quantity' in decision
            assert 'algorithm' in decision
            assert decision['algorithm'] == 'VWAP'
            assert decision['optimal_quantity'] >= 0
    
    def test_volume_based_allocation(self, test_config, sample_market_data):
        scheduler = VWAPScheduler(test_config)
        schedule = scheduler.generate_schedule(sample_market_data, 1000, 30)
        
        # Total quantity should approximately equal target
        total_quantity = sum(d['optimal_quantity'] for d in schedule)
        assert abs(total_quantity - 1000) < 10  # Allow for rounding
    
    def test_empty_data_handling(self, test_config):
        scheduler = VWAPScheduler(test_config)
        empty_df = pd.DataFrame(columns=['ts_event', 'size'])
        
        # Should handle empty data gracefully
        schedule = scheduler.generate_schedule(empty_df, 1000, 30)
        assert isinstance(schedule, list)
    
    def test_zero_volume_handling(self, test_config, sample_market_data):
        # Create data with zero volume
        zero_volume_data = sample_market_data.copy()
        zero_volume_data['size'] = 0
        
        scheduler = VWAPScheduler(test_config)
        schedule = scheduler.generate_schedule(zero_volume_data, 1000, 30)
        
        # Should fallback to uniform distribution
        if schedule:
            quantities = [d['optimal_quantity'] for d in schedule]
            # Check if quantities are roughly uniform
            assert np.std(quantities) / np.mean(quantities) < 0.5  # Low coefficient of variation


class TestTWAPScheduler:
    """Test TWAP scheduler functionality"""
    
    def test_initialization(self, test_config):
        scheduler = TWAPScheduler(test_config)
        assert scheduler.name == "TWAP"
        assert scheduler.interval == test_config.twap_interval
    
    def test_generate_schedule_uniform(self, test_config, sample_market_data):
        scheduler = TWAPScheduler(test_config)
        schedule = scheduler.generate_schedule(sample_market_data, 1000, 30)
        
        # Basic validation
        assert isinstance(schedule, list)
        assert len(schedule) > 0
        
        # Check uniform distribution
        quantities = [d['optimal_quantity'] for d in schedule]
        expected_quantity = 1000 / len(schedule)
        
        for quantity in quantities:
            assert abs(quantity - expected_quantity) < 1  # Allow for rounding
    
    def test_time_spacing(self, test_config, sample_market_data):
        scheduler = TWAPScheduler(test_config)
        schedule = scheduler.generate_schedule(sample_market_data, 1000, 600)  # 10 minutes
        
        # Check time spacing
        if len(schedule) > 1:
            time_diffs = []
            for i in range(1, len(schedule)):
                diff = (schedule[i]['timestamp'] - schedule[i-1]['timestamp']).total_seconds()
                time_diffs.append(diff)
            
            # All intervals should be approximately equal
            expected_interval = test_config.twap_interval
            for diff in time_diffs:
                assert abs(diff - expected_interval) < 1  # 1 second tolerance


class TestMyopicScheduler:
    """Test Myopic scheduler functionality"""
    
    def test_initialization(self, test_config):
        scheduler = MyopicSchedulerWrapper(test_config)
        assert scheduler.name == "Myopic"
        assert hasattr(scheduler, 'scheduler')
    
    def test_generate_schedule_basic(self, test_config, sample_market_data):
        scheduler = MyopicSchedulerWrapper(test_config)
        
        # Add required columns for myopic model
        sample_market_data['signed_volume'] = (sample_market_data['bid_fill'] - 
                                              sample_market_data['ask_fill'])
        sample_market_data['mid_price'] = ((sample_market_data['best_bid'] + 
                                           sample_market_data['best_ask']) / 2)
        
        schedule = scheduler.generate_schedule(sample_market_data, 1000, 30)
        
        # Basic validation
        assert isinstance(schedule, list)
        
        # If schedule is generated, validate structure
        if schedule:
            for decision in schedule:
                assert 'timestamp' in decision
                assert 'optimal_quantity' in decision
                assert 'algorithm' in decision
                assert decision['algorithm'] == 'Myopic'
    
    def test_lambda_estimation_fallback(self, test_config, sample_market_data):
        scheduler = MyopicSchedulerWrapper(test_config)
        
        # Test with minimal data that might cause estimation to fail
        minimal_data = sample_market_data.head(10).copy()
        minimal_data['signed_volume'] = 0
        minimal_data['mid_price'] = 150.0
        
        # Should not crash even with poor data
        try:
            schedule = scheduler.generate_schedule(minimal_data, 100, 5)
            assert isinstance(schedule, list)
        except Exception as e:
            # If it fails, it should fail gracefully
            assert "estimation" in str(e).lower() or "data" in str(e).lower()


# tests/unit/test_benchmark_runner.py
"""
Unit tests for benchmark runner components
"""

import pytest
import json
from unittest.mock import Mock, patch

from benchmark_framework import BenchmarkRunner, BenchmarkResult


class TestBenchmarkRunner:
    """Test benchmark runner functionality"""
    
    def test_initialization(self, test_config):
        runner = BenchmarkRunner(test_config)
        assert len(runner.schedulers) == len(test_config.algorithms)
        assert 'VWAP' in runner.schedulers
        assert 'TWAP' in runner.schedulers
        assert 'Myopic' in runner.schedulers
    
    def test_result_creation(self, test_config):
        result = BenchmarkResult(
            ticker='AAPL',
            date='2025-04-02',
            algorithm='VWAP'
        )
        
        assert result.ticker == 'AAPL'
        assert result.date == '2025-04-02'
        assert result.algorithm == 'VWAP'
        assert result.completed_successfully == False  # Default
        assert result.avg_cost_per_share == 0.0  # Default
    
    @patch('benchmark_framework.BenchmarkRunner._load_market_data')
    @patch('benchmark_framework.BenchmarkRunner._run_backtest_with_schedule')
    def test_single_benchmark_success(self, mock_backtest, mock_load_data, 
                                    test_config, sample_market_data, mock_backtest_result):
        """Test successful single benchmark run"""
        
        # Setup mocks
        mock_load_data.return_value = sample_market_data
        mock_backtest.return_value = mock_backtest_result
        
        runner = BenchmarkRunner(test_config)
        result = runner.run_single_benchmark('AAPL', '2025-04-02', 'VWAP')
        
        assert result.completed_successfully == True
        assert result.ticker == 'AAPL'
        assert result.algorithm == 'VWAP'
        assert result.execution_time > 0
        assert result.total_cost == mock_backtest_result['total_cost']
    
    @patch('benchmark_framework.BenchmarkRunner._load_market_data')
    def test_single_benchmark_data_failure(self, mock_load_data, test_config):
        """Test benchmark run with data loading failure"""
        
        # Setup mock to return None (data loading failure)
        mock_load_data.return_value = None
        
        runner = BenchmarkRunner(test_config)
        result = runner.run_single_benchmark('AAPL', '2025-04-02', 'VWAP')
        
        assert result.completed_successfully == False
        assert "No data available" in result.error_message
    
    def test_scheduler_initialization_filtering(self, test_config):
        """Test that only requested algorithms are initialized"""
        
        # Test with only VWAP
        config_vwap_only = test_config
        config_vwap_only.algorithms = ['VWAP']
        
        runner = BenchmarkRunner(config_vwap_only)
        assert len(runner.schedulers) == 1
        assert 'VWAP' in runner.schedulers
        assert 'TWAP' not in runner.schedulers
        assert 'Myopic' not in runner.schedulers


# tests/integration/test_end_to_end.py
"""
Integration tests for end-to-end benchmark execution
"""

import pytest
import json
import pandas as pd
from pathlib import Path

from benchmark_framework import BenchmarkRunner, BenchmarkConfig


class TestEndToEndBenchmark:
    """Test complete benchmark execution"""
    
    def test_mini_benchmark_execution(self, sample_market_data, temp_directory):
        """Test a minimal end-to-end benchmark"""
        
        # Setup test data
        data_dir = Path(temp_directory) / "data" / "AAPL"
        data_dir.mkdir(parents=True)
        
        # Save sample data
        data_file = data_dir / "xnas-itch-20250402.mbp-10.csv"
        sample_market_data.to_csv(data_file, index=False)
        
        # Create test config
        config = BenchmarkConfig(
            tickers=['AAPL'],
            dates=['2025-04-02'],
            venues=['v1'],
            algorithms=['VWAP', 'TWAP'],  # Skip Myopic for faster test
            order_size=100,
            time_horizon=5,
            data_path=str(Path(temp_directory) / "data") + "/",
            output_path=str(Path(temp_directory) / "results") + "/"
        )
        
        # Run benchmark
        runner = BenchmarkRunner(config)
        
        # Mock the backtest execution to avoid complex dependencies
        def mock_backtest(ticker, date, schedule, params):
            return {
                'total_cost': len(schedule) * 150.0,
                'total_size': 100,
                'avg_cost_per_share': 150.0 + len(schedule) * 0.01,
                'num_decisions': len(schedule),
                'results': []
            }
        
        runner._run_backtest_with_schedule = mock_backtest
        runner._run_custom_schedule_backtest = mock_backtest
        
        # Execute
        results = runner.run_full_benchmark()
        
        # Validate results
        assert len(results) == 2  # AAPL * 1 date * 2 algorithms
        assert all(r.ticker == 'AAPL' for r in results)
        assert all(r.date == '2025-04-02' for r in results)
        
        # Check that files were created
        results_dir = Path(temp_directory) / "results"
        assert (results_dir / "benchmark_results.json").exists()
        assert (results_dir / "benchmark_results.csv").exists()
        assert (results_dir / "benchmark_summary.csv").exists()
    
    def test_results_file_format(self, sample_market_data, temp_directory):
        """Test that results files have correct format"""
        
        # Setup minimal test
        data_dir = Path(temp_directory) / "data" / "AAPL"
        data_dir.mkdir(parents=True)
        data_file = data_dir / "xnas-itch-20250402.mbp-10.csv"
        sample_market_data.to_csv(data_file, index=False)
        
        config = BenchmarkConfig(
            tickers=['AAPL'],
            dates=['2025-04-02'],
            venues=['v1'],
            algorithms=['TWAP'],  # Single algorithm for simplicity
            data_path=str(Path(temp_directory) / "data") + "/",
            output_path=str(Path(temp_directory) / "results") + "/"
        )
        
        runner = BenchmarkRunner(config)
        
        # Mock backtest
        runner._run_custom_schedule_backtest = lambda *args: {
            'total_cost': 15000.0,
            'total_size': 100,
            'avg_cost_per_share': 150.0,
            'num_decisions': 5,
            'results': []
        }
        
        results = runner.run_full_benchmark()
        
        # Check JSON format
        json_file = Path(temp_directory) / "results" / "benchmark_results.json"
        with open(json_file, 'r') as f:
            json_data = json.load(f)
        
        assert isinstance(json_data, list)
        assert len(json_data) == 1
        
        result = json_data[0]
        required_fields = [
            'ticker', 'date', 'algorithm', 'avg_cost_per_share',
            'total_cost', 'total_size', 'fill_rate', 'completed_successfully'
        ]
        
        for field in required_fields:
            assert field in result
        
        # Check CSV format
        csv_file = Path(temp_directory) / "results" / "benchmark_results.csv"
        df = pd.read_csv(csv_file)
        
        assert len(df) == 1
        assert all(field in df.columns for field in required_fields)


# tests/integration/test_performance_bounds.py
"""
Performance and bounds testing for schedulers
"""

import pytest
import time
import numpy as np

from benchmark_framework import VWAPScheduler, TWAPScheduler, MyopicSchedulerWrapper


class TestPerformanceBounds:
    """Test performance characteristics and bounds"""
    
    def test_vwap_performance_large_dataset(self, test_config):
        """Test VWAP performance with large dataset"""
        
        # Generate large dataset
        n_rows = 100000
        large_df = pd.DataFrame({
            'ts_event': pd.date_range('2025-04-02 09:30', periods=n_rows, freq='100ms'),
            'size': np.random.lognormal(3, 1, n_rows),
            'best_bid': np.random.normal(150, 1, n_rows),
            'best_ask': np.random.normal(150.05, 1, n_rows)
        })
        
        scheduler = VWAPScheduler(test_config)
        
        start_time = time.time()
        schedule = scheduler.generate_schedule(large_df, 10000, 60)
        execution_time = time.time() - start_time
        
        # Performance bounds
        assert execution_time < 5.0  # Should complete within 5 seconds
        assert len(schedule) > 0
        
        # Memory usage check (basic)
        total_quantity = sum(d['optimal_quantity'] for d in schedule)
        assert abs(total_quantity - 10000) < 100  # Reasonable accuracy
    
    def test_twap_schedule_bounds(self, test_config, sample_market_data):
        """Test TWAP schedule bounds and constraints"""
        
        scheduler = TWAPScheduler(test_config)
        
        # Test various order sizes
        for order_size in [10, 100, 1000, 10000]:
            schedule = scheduler.generate_schedule(sample_market_data, order_size, 30)
            
            # Bounds checking
            assert all(d['optimal_quantity'] >= 0 for d in schedule)
            assert all(d['optimal_quantity'] <= order_size for d in schedule)
            
            total_quantity = sum(d['optimal_quantity'] for d in schedule)
            assert abs(total_quantity - order_size) < 1  # Very tight bound for TWAP
    
    def test_scheduler_memory_usage(self, test_config, sample_market_data):
        """Test that schedulers don't have memory leaks"""
        
        import gc
        import sys
        
        schedulers = [
            VWAPScheduler(test_config),
            TWAPScheduler(test_config)
        ]
        
        initial_objects = len(gc.get_objects())
        
        # Run multiple iterations
        for _ in range(10):
            for scheduler in schedulers:
                schedule = scheduler.generate_schedule(sample_market_data, 1000, 30)
                del schedule  # Explicit cleanup
        
        gc.collect()  # Force garbage collection
        final_objects = len(gc.get_objects())
        
        # Object count shouldn't grow significantly
        object_growth = final_objects - initial_objects
        assert object_growth < 1000  # Allow some growth but not excessive
    
    def test_extreme_parameters(self, test_config, sample_market_data):
        """Test schedulers with extreme parameter values"""
        
        # Test with very small order size
        vwap = VWAPScheduler(test_config)
        schedule = vwap.generate_schedule(sample_market_data, 1, 5)  # 1 share
        
        if schedule:  # May be empty due to minimum trade size
            assert all(d['optimal_quantity'] >= 0 for d in schedule)
            assert sum(d['optimal_quantity'] for d in schedule) <= 1.1  # Allow rounding
        
        # Test with very large order size
        schedule = vwap.generate_schedule(sample_market_data, 1000000, 30)  # 1M shares
        assert len(schedule) > 0
        assert sum(d['optimal_quantity'] for d in schedule) > 999000  # Should allocate most
        
        # Test with very short time horizon
        twap = TWAPScheduler(test_config)
        schedule = twap.generate_schedule(sample_market_data, 1000, 1)  # 1 minute
        assert len(schedule) >= 1  # Should create at least one slice


# tests/unit/test_regression_detection.py
"""
Test regression detection functionality
"""

import pytest
import json
from pathlib import Path

# Import the regression detector from scripts
import sys
sys.path.append('scripts')
from regression_detector import RegressionDetector


class TestRegressionDetector:
    """Test regression detection logic"""
    
    def test_no_regression_detected(self, temp_directory):
        """Test case where no regression should be detected"""
        
        current_results = [
            {'algorithm': 'VWAP', 'avg_cost_per_share': 150.00, 'fill_rate': 0.95, 'completed_successfully': True},
            {'algorithm': 'VWAP', 'avg_cost_per_share': 150.01, 'fill_rate': 0.94, 'completed_successfully': True},
            {'algorithm': 'VWAP', 'avg_cost_per_share': 149.99, 'fill_rate': 0.96, 'completed_successfully': True},
        ]
        
        baseline_results = [
            {'algorithm': 'VWAP', 'avg_cost_per_share': 150.02, 'fill_rate': 0.94, 'completed_successfully': True},
            {'algorithm': 'VWAP', 'avg_cost_per_share': 150.01, 'fill_rate': 0.95, 'completed_successfully': True},
            {'algorithm': 'VWAP', 'avg_cost_per_share': 150.00, 'fill_rate': 0.94, 'completed_successfully': True},
        ]
        
        detector = RegressionDetector(threshold=0.05)
        regressions = detector.detect_regressions(current_results, baseline_results)
        
        assert len(regressions) == 0
    
    def test_cost_regression_detected(self, temp_directory):
        """Test case where cost regression should be detected"""
        
        # Current results are significantly worse
        current_results = [
            {'algorithm': 'VWAP', 'avg_cost_per_share': 160.00, 'fill_rate': 0.95, 'completed_successfully': True},
            {'algorithm': 'VWAP', 'avg_cost_per_share': 161.00, 'fill_rate': 0.94, 'completed_successfully': True},
            {'algorithm': 'VWAP', 'avg_cost_per_share': 159.50, 'fill_rate': 0.96, 'completed_successfully': True},
        ]
        
        baseline_results = [
            {'algorithm': 'VWAP', 'avg_cost_per_share': 150.00, 'fill_rate': 0.94, 'completed_successfully': True},
            {'algorithm': 'VWAP', 'avg_cost_per_share': 150.50, 'fill_rate': 0.95, 'completed_successfully': True},
            {'algorithm': 'VWAP', 'avg_cost_per_share': 149.80, 'fill_rate': 0.94, 'completed_successfully': True},
        ]
        
        detector = RegressionDetector(threshold=0.05)
        regressions = detector.detect_regressions(current_results, baseline_results)
        
        assert len(regressions) == 1
        assert regressions[0]['algorithm'] == 'VWAP'
        assert regressions[0]['regression_type'] == 'cost'
        assert regressions[0]['percentage_change'] > 0.05  # More than 5% worse
    
    def test_fill_rate_regression_detected(self, temp_directory):
        """Test case where fill rate regression should be detected"""
        
        current_results = [
            {'algorithm': 'TWAP', 'avg_cost_per_share': 150.00, 'fill_rate': 0.80, 'completed_successfully': True},
            {'algorithm': 'TWAP', 'avg_cost_per_share': 150.01, 'fill_rate': 0.79, 'completed_successfully': True},
            {'algorithm': 'TWAP', 'avg_cost_per_share': 149.99, 'fill_rate': 0.81, 'completed_successfully': True},
        ]
        
        baseline_results = [
            {'algorithm': 'TWAP', 'avg_cost_per_share': 150.02, 'fill_rate': 0.95, 'completed_successfully': True},
            {'algorithm': 'TWAP', 'avg_cost_per_share': 150.01, 'fill_rate': 0.94, 'completed_successfully': True},
            {'algorithm': 'TWAP', 'avg_cost_per_share': 150.00, 'fill_rate': 0.96, 'completed_successfully': True},
        ]
        
        detector = RegressionDetector(threshold=0.05)
        regressions = detector.detect_regressions(current_results, baseline_results)
        
        assert len(regressions) == 1
        assert regressions[0]['algorithm'] == 'TWAP'
        assert regressions[0]['regression_type'] == 'fill_rate'
        assert regressions[0]['percentage_change'] < -0.05  # More than 5% worse
    
    def test_insufficient_data_handling(self, temp_directory):
        """Test handling of insufficient data for statistical tests"""
        
        # Only 2 data points (insufficient for t-test)
        current_results = [
            {'algorithm': 'VWAP', 'avg_cost_per_share': 160.00, 'fill_rate': 0.95, 'completed_successfully': True},
            {'algorithm': 'VWAP', 'avg_cost_per_share': 161.00, 'fill_rate': 0.94, 'completed_successfully': True},
        ]
        
        baseline_results = [
            {'algorithm': 'VWAP', 'avg_cost_per_share': 150.00, 'fill_rate': 0.94, 'completed_successfully': True},
            {'algorithm': 'VWAP', 'avg_cost_per_share': 150.50, 'fill_rate': 0.95, 'completed_successfully': True},
        ]
        
        detector = RegressionDetector(threshold=0.05)
        regressions = detector.detect_regressions(current_results, baseline_results)
        
        # Should not detect regression due to insufficient data
        assert len(regressions) == 0


# tests/smoke/test_smoke.py
"""
Smoke tests - quick validation that basic functionality works
"""

import pytest
import pandas as pd
import numpy as np

from benchmark_framework import BenchmarkConfig, VWAPScheduler, TWAPScheduler


class TestSmoke:
    """Basic smoke tests for critical functionality"""
    
    def test_schedulers_can_be_imported(self):
        """Test that all schedulers can be imported and instantiated"""
        
        config = BenchmarkConfig(
            tickers=['AAPL'],
            dates=['2025-04-02'],
            venues=['v1'],
            algorithms=['VWAP', 'TWAP']
        )
        
        # Should not raise exceptions
        vwap = VWAPScheduler(config)
        twap = TWAPScheduler(config)
        
        assert vwap.name == "VWAP"
        assert twap.name == "TWAP"
    
    def test_basic_schedule_generation(self):
        """Test that schedulers can generate basic schedules"""
        
        # Minimal test data
        df = pd.DataFrame({
            'ts_event': pd.date_range('2025-04-02 09:30', periods=100, freq='1min'),
            'size': np.random.randint(10, 1000, 100),
            'best_bid': np.random.normal(150, 0.5, 100),
            'best_ask': np.random.normal(150.05, 0.5, 100)
        })
        
        config = BenchmarkConfig(
            tickers=['AAPL'],
            dates=['2025-04-02'],
            venues=['v1'],
            algorithms=['VWAP', 'TWAP']
        )
        
        vwap = VWAPScheduler(config)
        twap = TWAPScheduler(config)
        
        # Should generate schedules without errors
        vwap_schedule = vwap.generate_schedule(df, 1000, 30)
        twap_schedule = twap.generate_schedule(df, 1000, 30)
        
        assert isinstance(vwap_schedule, list)
        assert isinstance(twap_schedule, list)
    
    def test_config_validation(self):
        """Test basic configuration validation"""
        
        # Valid config should work
        config = BenchmarkConfig(
            tickers=['AAPL'],
            dates=['2025-04-02'],
            venues=['v1'],
            algorithms=['VWAP']
        )
        
        assert config.tickers == ['AAPL']
        assert config.order_size == 100  # Default value
        assert config.rebate_rates == [0.002]  # Default calculated in __post_init__


# conftest.py additions for performance testing
@pytest.fixture
def performance_test_data():
    """Generate larger dataset for performance testing"""
    
    n_rows = 50000  # Larger dataset
    start_time = pd.Timestamp('2025-04-02 09:30:00', tz='UTC')
    
    timestamps = pd.date_range(start_time, periods=n_rows, freq='50ms')
    
    np.random.seed(42)
    
    df = pd.DataFrame({
        'ts_event': timestamps,
        'best_bid': np.random.normal(150, 0.1, n_rows),
        'best_ask': np.random.normal(150.05, 0.1, n_rows),
        'size': np.random.lognormal(3, 1, n_rows).astype(int),
        'bid_sz_00': np.random.poisson(100, n_rows),
        'ask_sz_00': np.random.poisson(100, n_rows),
        'action': np.random.choice(['T', 'C', 'A'], n_rows),
        'side': np.random.choice(['A', 'B'], n_rows)
    })
    
    return df


# Performance test marks
pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

# Custom markers for different test types
def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "performance: marks tests as performance tests")
    config.addinivalue_line("markers", "smoke: marks tests as smoke tests")


# Example pytest.ini configuration
"""
[tool:pytest]
minversion = 6.0
addopts = 
    -ra 
    -q 
    --strict-markers 
    --strict-config 
    --cov=benchmark_framework 
    --cov-report=term-missing 
    --cov-report=html
testpaths = 
    tests
markers =
    slow: marks tests as slow
    integration: marks tests as integration tests  
    performance: marks tests as performance tests
    smoke: marks tests as smoke tests
filterwarnings =
    ignore::DeprecationWarning
    ignore::PendingDeprecationWarning
"""