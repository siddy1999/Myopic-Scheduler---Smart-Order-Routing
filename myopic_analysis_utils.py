# myopic_analysis_utils.py
"""
Utility functions for analyzing and visualizing myopic scheduling results.

This module provides tools for:
1. Performance analysis of myopic vs traditional approaches
2. Visualization of trading schedules and impacts
3. Parameter sensitivity analysis
4. Integration with existing metrics framework
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Dict, List, Tuple, Optional
import seaborn as sns
from datetime import datetime, timedelta
import logging


class MyopicAnalyzer:
    """Analyzer for myopic scheduling results."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def analyze_schedule_performance(self, myopic_results: Dict, 
                                   traditional_results: Dict) -> Dict:
        """
        Analyze performance differences between myopic and traditional approaches.
        
        Args:
            myopic_results: Results from myopic scheduling
            traditional_results: Results from traditional SOR
            
        Returns:
            Dictionary with performance analysis
        """
        analysis = {}
        
        # Cost analysis
        myopic_cost = myopic_results.get('avg_cost_per_share', 0)
        traditional_cost = traditional_results.get('avg_cost_per_share', 0)
        
        if traditional_cost > 0:
            cost_improvement = ((traditional_cost - myopic_cost) / traditional_cost) * 100
            analysis['cost_improvement_pct'] = cost_improvement
        else:
            analysis['cost_improvement_pct'] = 0
            
        # Execution analysis
        analysis['myopic_decisions'] = myopic_results.get('num_decisions', 0)
        analysis['traditional_decisions'] = len(traditional_results.get('results', []))
        
        # Size and fill analysis
        myopic_size = myopic_results.get('total_size', 0)
        traditional_size = traditional_results.get('total_size', 0)
        
        analysis['size_efficiency'] = myopic_size / traditional_size if traditional_size > 0 else 0
        
        # Lambda parameter used
        analysis['lambda_used'] = myopic_results.get('lambda_used', 0)
        
        return analysis
    
    def create_schedule_visualization(self, results: Dict, save_path: Optional[str] = None):
        """
        Create visualization of the trading schedule and impacts.
        
        Args:
            results: Myopic scheduling results
            save_path: Optional path to save the plot
        """
        if 'results' not in results or not results['results']:
            self.logger.warning("No trading results to visualize")
            return
            
        # Convert results to DataFrame
        df = pd.DataFrame(results['results'])
        df['time'] = pd.to_datetime(df['time'])
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Myopic Trading Schedule Analysis', fontsize=16)
        
        # Plot 1: Trading quantities over time
        ax1 = axes[0, 0]
        ax1.plot(df['time'], df['market_v'], 'bo-', label='Market Orders', alpha=0.7)
        ax1.plot(df['time'], df['limit_v'], 'ro-', label='Limit Orders', alpha=0.7)
        ax1.set_title('Order Quantities Over Time')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Quantity')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Price impact signals
        ax2 = axes[0, 1]
        if 'myopic_impact' in df.columns:
            ax2.plot(df['time'], df['myopic_impact'], 'g-', label='Price Impact', alpha=0.7)
        if 'optimal_impact' in df.columns:
            ax2.plot(df['time'], df['optimal_impact'], 'purple', label='Optimal Impact', alpha=0.7)
        ax2.set_title('Price Impact Evolution')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Impact')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Market vs Limit price levels
        ax3 = axes[1, 0]
        ax3.plot(df['time'], df['market_p'], 'b-', label='Market Price', alpha=0.7)
        ax3.plot(df['time'], df['limit_p'], 'r-', label='Limit Price', alpha=0.7)
        ax3.set_title('Price Levels')
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Price ($)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Myopic signals
        ax4 = axes[1, 1]
        if 'myopic_signal' in df.columns:
            ax4.plot(df['time'], df['myopic_signal'], 'orange', label='Alpha Signal', alpha=0.7)
        ax4.set_title('Myopic Alpha Signal')
        ax4.set_xlabel('Time')
        ax4.set_ylabel('Signal Value')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Format time axis
        for ax in axes.flat:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        plt.show()
    
    def parameter_sensitivity_analysis(self, base_results: Dict, 
                                     lambda_values: List[float]) -> pd.DataFrame:
        """
        Analyze sensitivity to lambda parameter values.
        
        Args:
            base_results: Base case results
            lambda_values: List of lambda values to test
            
        Returns:
            DataFrame with sensitivity analysis results
        """
        sensitivity_data = []
        
        base_cost = base_results.get('avg_cost_per_share', 0)
        base_lambda = base_results.get('lambda_used', 0)
        
        for lambda_val in lambda_values:
            # Calculate relative change
            lambda_change = (lambda_val - base_lambda) / base_lambda * 100 if base_lambda > 0 else 0
            
            # Estimate cost change (simplified relationship)
            # In practice, you'd re-run the optimization with different lambda
            estimated_cost_change = -0.1 * lambda_change  # Simplified relationship
            estimated_cost = base_cost * (1 + estimated_cost_change / 100)
            
            sensitivity_data.append({
                'lambda_value': lambda_val,
                'lambda_change_pct': lambda_change,
                'estimated_cost': estimated_cost,
                'estimated_cost_change_pct': estimated_cost_change
            })
        
        return pd.DataFrame(sensitivity_data)
    
    def generate_performance_report(self, comparison_results: Dict, 
                                  output_path: Optional[str] = None) -> str:
        """
        Generate a comprehensive performance report.
        
        Args:
            comparison_results: Dictionary of comparison results
            output_path: Optional path to save the report
            
        Returns:
            Report string
        """
        report_lines = []
        report_lines.append("MYOPIC SCHEDULING PERFORMANCE REPORT")
        report_lines.append("=" * 50)
        report_lines.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Summary statistics
        improvements = []
        for key, result in comparison_results.items():
            improvement = result.get('improvement', {}).get('cost_improvement_pct', 0)
            improvements.append(improvement)
        
        if improvements:
            report_lines.append("SUMMARY STATISTICS:")
            report_lines.append(f"Average cost improvement: {np.mean(improvements):.2f}%")
            report_lines.append(f"Median cost improvement: {np.median(improvements):.2f}%")
            report_lines.append(f"Best improvement: {np.max(improvements):.2f}%")
            report_lines.append(f"Worst improvement: {np.min(improvements):.2f}%")
            report_lines.append(f"Standard deviation: {np.std(improvements):.2f}%")
            report_lines.append("")
        
        # Detailed results by stock/day
        report_lines.append("DETAILED RESULTS:")
        for key, result in comparison_results.items():
            stock = result.get('stock', 'Unknown')
            date = result.get('comparison_date', 'Unknown')
            
            myopic = result.get('myopic', {})
            traditional = result.get('traditional', {})
            improvement = result.get('improvement', {})
            
            report_lines.append(f"\n{stock} on {date}:")
            report_lines.append(f"  Myopic approach:")
            report_lines.append(f"    Cost per share: ${myopic.get('avg_cost_per_share', 0):.4f}")
            report_lines.append(f"    Decisions made: {myopic.get('num_decisions', 0)}")
            report_lines.append(f"    Lambda used: {myopic.get('lambda_used', 0):.2f}")
            
            report_lines.append(f"  Traditional approach:")
            report_lines.append(f"    Cost per share: ${traditional.get('avg_cost_per_share', 0):.4f}")
            report_lines.append(f"    Decisions made: {traditional.get('num_decisions', 0)}")
            
            report_lines.append(f"  Improvement: {improvement.get('cost_improvement_pct', 0):.2f}%")
        
        # Technical details
        report_lines.append("\n\nTECHNICAL DETAILS:")
        report_lines.append("The myopic scheduling algorithm uses:")
        report_lines.append("- Market impact modeling with exponential decay")
        report_lines.append("- Alpha prediction for future price movements")
        report_lines.append("- Optimal control theory for trade scheduling")
        report_lines.append("- SOR optimization for venue allocation")
        
        report = "\n".join(report_lines)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)
            self.logger.info(f"Report saved to {output_path}")
        
        return report


def create_lambda_comparison_plot(lambda_stats_df: pd.DataFrame, save_path: Optional[str] = None):
    """
    Create visualization comparing different lambda values and their performance.
    
    Args:
        lambda_stats_df: DataFrame with lambda statistics (from myopic notebook analysis)
        save_path: Optional path to save the plot
    """
    
    # Extract periods and convert to numeric values
    periods = np.array([int(p.replace('s', '')) for p in lambda_stats_df.index])
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Lambda Parameter Analysis', fontsize=16)
    
    # Plot 1: Final PnL vs Period
    ax1 = axes[0, 0]
    ax1.plot(periods, lambda_stats_df['final_pnl'], 'bo-', alpha=0.7)
    ax1.set_xlabel('Aggregation Period (seconds)')
    ax1.set_ylabel('Final PnL')
    ax1.set_title('Final PnL vs Aggregation Period')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Lambda value vs Period
    ax2 = axes[0, 1]
    ax2.plot(periods, lambda_stats_df['lambda'], 'ro-', alpha=0.7)
    ax2.set_xlabel('Aggregation Period (seconds)')
    ax2.set_ylabel('Lambda Value')
    ax2.set_title('Lambda Value vs Aggregation Period')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Sharpe ratio vs Period
    ax3 = axes[1, 0]
    ax3.plot(periods, lambda_stats_df['sharpe'], 'go-', alpha=0.7)
    ax3.set_xlabel('Aggregation Period (seconds)')
    ax3.set_ylabel('Sharpe Ratio')
    ax3.set_title('Sharpe Ratio vs Aggregation Period')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Standard deviation vs Period
    ax4 = axes[1, 1]
    ax4.plot(periods, lambda_stats_df['std_pnl'], 'mo-', alpha=0.7)
    ax4.set_xlabel('Aggregation Period (seconds)')
    ax4.set_ylabel('PnL Standard Deviation')
    ax4.set_title('Risk vs Aggregation Period')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    plt.show()


def integrate_myopic_with_existing_metrics(myopic_results: Dict, 
                                         traditional_metrics: pd.DataFrame) -> pd.DataFrame:
    """
    Integrate myopic results with existing metrics framework.
    
    Args:
        myopic_results: Results from myopic scheduling
        traditional_metrics: Traditional metrics DataFrame
        
    Returns:
        Combined metrics DataFrame
    """
    
    # Create myopic metrics row
    myopic_metrics = {
        'approach': 'myopic',
        'avg_cost_per_share': myopic_results.get('avg_cost_per_share', 0),
        'total_decisions': myopic_results.get('num_decisions', 0),
        'total_quantity': myopic_results.get('total_size', 0),
        'lambda_parameter': myopic_results.get('lambda_used', 0),
        'cost_improvement_vs_baseline': 0  # Will be calculated relative to traditional
    }
    
    # Add traditional baseline if available
    if not traditional_metrics.empty and 'avg_cost_per_share' in traditional_metrics.columns:
        baseline_cost = traditional_metrics['avg_cost_per_share'].mean()
        if baseline_cost > 0:
            improvement = ((baseline_cost - myopic_metrics['avg_cost_per_share']) / 
                          baseline_cost) * 100
            myopic_metrics['cost_improvement_vs_baseline'] = improvement
    
    # Create combined DataFrame
    myopic_df = pd.DataFrame([myopic_metrics])
    
    # Combine with existing metrics
    if not traditional_metrics.empty:
        # Add approach column to traditional metrics if not present
        if 'approach' not in traditional_metrics.columns:
            traditional_metrics['approach'] = 'traditional'
        
        combined_df = pd.concat([traditional_metrics, myopic_df], ignore_index=True)
    else:
        combined_df = myopic_df
    
    return combined_df


# Example usage and testing functions
def example_analysis():
    """Example of how to use the analysis utilities."""
    
    # Mock data for demonstration
    myopic_results = {
        'avg_cost_per_share': 234.5678,
        'num_decisions': 25,
        'total_size': 2500,
        'lambda_used': 15000.0,
        'results': [
            {
                'time': datetime.now(),
                'market_v': 50,
                'limit_v': 50,
                'market_p': 234.50,
                'limit_p': 234.45,
                'myopic_signal': 0.002,
                'myopic_impact': 0.001
            }
        ]
    }
    
    traditional_results = {
        'avg_cost_per_share': 234.6000,
        'num_decisions': 30,
        'total_size': 2500,
        'results': []
    }
    
    # Create analyzer
    analyzer = MyopicAnalyzer()
    
    # Analyze performance
    analysis = analyzer.analyze_schedule_performance(myopic_results, traditional_results)
    print("Performance Analysis:", analysis)
    
    # Generate report
    comparison_data = {
        'AAPL_2025-04-02': {
            'myopic': myopic_results,
            'traditional': traditional_results,
            'improvement': analysis,
            'stock': 'AAPL',
            'comparison_date': '2025-04-02'
        }
    }
    
    report = analyzer.generate_performance_report(comparison_data)
    print("\nPerformance Report:")
    print(report)


if __name__ == "__main__":
    example_analysis()
