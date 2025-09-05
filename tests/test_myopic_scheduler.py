#!/usr/bin/env python3
"""
Unit tests for MyopicScheduler class.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from myopic_sor_scheduler import MyopicScheduler, MyopicParameters

class TestMyopicParameters:
    """Test cases for MyopicParameters dataclass."""
    
    def test_initialization(self):
        """Test parameter initialization."""
        params = MyopicParameters(
            lambda_value=25000.0,
            beta=0.693,
            volatility=0.01,
            adv=1000000.0
        )
        
        assert params.lambda_value == 25000.0
        assert params.beta == 0.693
        assert params.volatility == 0.01
        assert params.adv == 1000000.0
        assert params.T == 6.5  # Default value
        assert params.Q_0 == 0.01  # Default value
    
    def test_default_values(self):
        """Test default parameter values."""
        params = MyopicParameters(
            lambda_value=1000.0,
            beta=0.5,
            volatility=0.02,
            adv=500000.0
        )
        
        assert params.T == 6.5
        assert params.Q_0 == 0.01

class TestMyopicScheduler:
    """Test cases for MyopicScheduler class."""
    
    @pytest.fixture
    def sample_params(self):
        """Create sample parameters for testing."""
        return MyopicParameters(
            lambda_value=25000.0,
            beta=0.693,
            volatility=0.01,
            adv=1000000.0
        )
    
    @pytest.fixture
    def sample_data(self):
        """Create sample market data for testing."""
        np.random.seed(42)
        timestamps = [datetime.now() + timedelta(minutes=i) for i in range(10)]
        
        data = []
        for i, ts in enumerate(timestamps):
            price = 150.0 + i * 0.1
            data.append({
                'ts_event': ts,
                'best_bid': price - 0.01,
                'best_ask': price + 0.01,
                'bid_fill': np.random.poisson(100),
                'ask_fill': np.random.poisson(100),
                'signed_volume': np.random.normal(0, 50),
                'Volatility': 0.01,
                'ADV': 1000000.0
            })
        
        return pd.DataFrame(data)
    
    def test_initialization(self, sample_params):
        """Test scheduler initialization."""
        scheduler = MyopicScheduler(sample_params)
        
        assert scheduler.params == sample_params
        assert scheduler.lambda_values == {}
        assert scheduler.price_impact_history == []
        assert scheduler.logger is not None
    
    def test_lambda_estimation(self, sample_params, sample_data):
        """Test lambda parameter estimation."""
        scheduler = MyopicScheduler(sample_params)
        
        # Test with valid data
        lambda_values = scheduler.estimate_lambda(sample_data)
        
        assert isinstance(lambda_values, dict)
        # Should have some lambda values for different periods
        assert len(lambda_values) > 0
    
    def test_lambda_estimation_empty_data(self, sample_params):
        """Test lambda estimation with empty data."""
        scheduler = MyopicScheduler(sample_params)
        empty_df = pd.DataFrame()
        
        lambda_values = scheduler.estimate_lambda(empty_df)
        
        assert isinstance(lambda_values, dict)
        assert len(lambda_values) == 0
    
    def test_calculate_price_impact(self, sample_params, sample_data):
        """Test price impact calculation."""
        scheduler = MyopicScheduler(sample_params)
        
        result_df = scheduler.calculate_price_impact(sample_data)
        
        assert 'price_impact' in result_df.columns
        assert len(result_df) == len(sample_data)
        assert not result_df['price_impact'].isna().all()
    
    def test_calculate_alpha_and_derivatives(self, sample_params, sample_data):
        """Test alpha calculation."""
        scheduler = MyopicScheduler(sample_params)
        
        # First calculate price impact
        df_with_impact = scheduler.calculate_price_impact(sample_data)
        result_df = scheduler.calculate_alpha_and_derivatives(df_with_impact)
        
        assert 'determined_alpha' in result_df.columns
        assert 'determined_alpha_prime' in result_df.columns
        assert 'unperturbed_price' in result_df.columns
    
    def test_calculate_optimal_impact(self, sample_params, sample_data):
        """Test optimal impact calculation."""
        scheduler = MyopicScheduler(sample_params)
        
        # Calculate all required components
        df_with_impact = scheduler.calculate_price_impact(sample_data)
        df_with_alpha = scheduler.calculate_alpha_and_derivatives(df_with_impact)
        result_df = scheduler.calculate_optimal_impact(df_with_alpha)
        
        assert 'I_star_t' in result_df.columns
        assert 'I_star_prime' in result_df.columns
        assert 'delta_Q_star' in result_df.columns
    
    def test_generate_trading_schedule(self, sample_params, sample_data):
        """Test trading schedule generation."""
        scheduler = MyopicScheduler(sample_params)
        
        schedule = scheduler.generate_trading_schedule(
            df=sample_data,
            total_quantity=1000,
            time_horizon=30
        )
        
        assert isinstance(schedule, list)
        
        if len(schedule) > 0:
            decision = schedule[0]
            required_keys = ['timestamp', 'optimal_quantity', 'price_impact', 
                           'optimal_impact', 'alpha', 'mid_price']
            for key in required_keys:
                assert key in decision
    
    def test_generate_trading_schedule_empty_data(self, sample_params):
        """Test trading schedule with empty data."""
        scheduler = MyopicScheduler(sample_params)
        empty_df = pd.DataFrame()
        
        schedule = scheduler.generate_trading_schedule(
            df=empty_df,
            total_quantity=1000,
            time_horizon=30
        )
        
        assert isinstance(schedule, list)
        assert len(schedule) == 0
    
    def test_integrate_with_sor(self, sample_params, sample_data):
        """Test SOR integration."""
        scheduler = MyopicScheduler(sample_params)
        
        # Create mock SOR data
        dfs = {'venue1': sample_data}
        myopic_schedule = [
            {
                'timestamp': sample_data['ts_event'].iloc[0],
                'optimal_quantity': 100,
                'alpha': 0.001,
                'price_impact': 0.0001,
                'mid_price': 150.0
            }
        ]
        sor_params = {
            'T': 5,
            'f': 0.003,
            'r': [0.002],
            'lambda_u': 0.05,
            'lambda_o': 0.05,
            'N': 1000,
            'method': 'lookup',
            'stock': 'TEST'
        }
        
        # This might fail due to missing SOR modules, but should not crash
        try:
            integrated_schedule = scheduler.integrate_with_sor(
                dfs, myopic_schedule, sor_params
            )
            assert isinstance(integrated_schedule, list)
        except ImportError:
            # Expected if SOR modules are not available
            pytest.skip("SOR modules not available for integration test")

class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_negative_parameters(self):
        """Test handling of negative parameters."""
        with pytest.raises((ValueError, TypeError)):
            MyopicParameters(
                lambda_value=-1000.0,  # Negative value
                beta=0.693,
                volatility=0.01,
                adv=1000000.0
            )
    
    def test_invalid_data_types(self):
        """Test handling of invalid data types."""
        with pytest.raises(TypeError):
            MyopicParameters(
                lambda_value="invalid",  # String instead of float
                beta=0.693,
                volatility=0.01,
                adv=1000000.0
            )
    
    def test_missing_columns(self):
        """Test handling of missing required columns."""
        params = MyopicParameters(
            lambda_value=25000.0,
            beta=0.693,
            volatility=0.01,
            adv=1000000.0
        )
        scheduler = MyopicScheduler(params)
        
        # Create data with missing columns
        incomplete_data = pd.DataFrame({
            'ts_event': [datetime.now()],
            'best_bid': [150.0],
            'best_ask': [150.01]
            # Missing required columns
        })
        
        # Should handle missing columns gracefully
        lambda_values = scheduler.estimate_lambda(incomplete_data)
        assert isinstance(lambda_values, dict)

if __name__ == "__main__":
    pytest.main([__file__])
