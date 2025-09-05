# sor_visualizations.py
"""
Comprehensive Visualization Module for Cont-Kukanov SOR Backtesting Results
Clean version with all syntax errors fixed
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
from datetime import datetime, timedelta
from pathlib import Path
import json
from typing import Dict, List, Optional, Tuple

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
warnings.filterwarnings('ignore')

class SORVisualizationSuite:
    """Comprehensive visualization suite for Cont-Kukanov SOR results"""
    
    def __init__(self, results_path: str = "results/", output_path: str = "visualizations/"):
        self.results_path = Path(results_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(exist_ok=True)
        
        print("🎨 SOR Visualization Suite Initialized")
        print(f"📁 Results path: {self.results_path}")
        print(f"🖼️ Output path: {self.output_path}")
    
    def load_backtest_results(self, pattern: str = "*result*.csv") -> Dict[str, pd.DataFrame]:
        """Load all backtest result files"""
        result_files = list(self.results_path.glob(pattern))
        
        if not result_files:
            print(f"❌ No result files found with pattern: {pattern}")
            return {}
        
        results = {}
        for file_path in result_files:
            try:
                df = pd.read_csv(file_path)
                if 'time' in df.columns:
                    df['time'] = pd.to_datetime(df['time'])
                results[file_path.stem] = df
                print(f"✅ Loaded: {file_path.name} ({len(df)} records)")
            except Exception as e:
                print(f"⚠️ Error loading {file_path.name}: {e}")
        
        return results
    
    def create_simple_performance_dashboard(self, results: Dict[str, pd.DataFrame]) -> None:
        """Create a simple performance dashboard using matplotlib"""
        
        if not results:
            print("❌ No results to visualize")
            return
        
        print("🎨 Creating Simple Performance Dashboard...")
        
        # Combine all results
        all_data = []
        for name, df in results.items():
            df_copy = df.copy()
            df_copy['strategy'] = name
            all_data.append(df_copy)
        
        if not all_data:
            return
            
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Cont-Kukanov SOR Performance Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Market vs Limit Orders Over Time
        for strategy in combined_df['strategy'].unique():
            strategy_data = combined_df[combined_df['strategy'] == strategy]
            if 'time' in strategy_data.columns:
                axes[0, 0].plot(strategy_data['time'], strategy_data['market_v'], 
                              label=f'{strategy} - Market', linewidth=2)
                axes[0, 0].plot(strategy_data['time'], strategy_data['limit_v'], 
                              label=f'{strategy} - Limit', linewidth=2, linestyle='--')
        
        axes[0, 0].set_title('Order Allocation Over Time')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Order Size')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Fill Rate Analysis
        fill_rates = {}
        for strategy in combined_df['strategy'].unique():
            strategy_data = combined_df[combined_df['strategy'] == strategy]
            total_ordered = (strategy_data['market_v'] + strategy_data['limit_v']).sum()
            total_filled = (strategy_data['market_v'] + strategy_data['limit_fill']).sum()
            fill_rate = total_filled / total_ordered if total_ordered > 0 else 0
            fill_rates[strategy] = fill_rate
        
        strategies = list(fill_rates.keys())
        rates = list(fill_rates.values())
        
        bars = axes[0, 1].bar(strategies, rates, color='lightblue', alpha=0.7)
        axes[0, 1].set_title('Fill Rate by Strategy')
        axes[0, 1].set_ylabel('Fill Rate')
        axes[0, 1].set_ylim(0, 1)
        
        # Add percentage labels
        for bar, rate in zip(bars, rates):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                          f'{rate:.1%}', ha='center', va='bottom', fontweight='bold')
        
        # 3. Execution Price Distribution
        for strategy in combined_df['strategy'].unique():
            strategy_data = combined_df[combined_df['strategy'] == strategy]
            axes[0, 2].hist(strategy_data['market_p'], bins=20, alpha=0.6, 
                          label=f'{strategy}', density=True)
        
        axes[0, 2].set_title('Execution Price Distribution')
        axes[0, 2].set_xlabel('Price ($)')
        axes[0, 2].set_ylabel('Density')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Queue Analysis (if available)
        if 'queue' in combined_df.columns:
            for strategy in combined_df['strategy'].unique():
                strategy_data = combined_df[combined_df['strategy'] == strategy]
                if 'time' in strategy_data.columns:
                    axes[1, 0].plot(strategy_data['time'], strategy_data['queue'], 
                                  label=f'{strategy}', linewidth=2, alpha=0.7)
            
            axes[1, 0].set_title('Queue Depth Over Time')
            axes[1, 0].set_xlabel('Time')
            axes[1, 0].set_ylabel('Queue Depth')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'Queue data\nnot available', 
                          ha='center', va='center', transform=axes[1, 0].transAxes, fontsize=12)
            axes[1, 0].set_title('Queue Analysis')
        
        # 5. Market vs Limit Ratio
        market_totals = combined_df.groupby('strategy')['market_v'].sum()
        limit_totals = combined_df.groupby('strategy')['limit_v'].sum()
        
        total_market = market_totals.sum()
        total_limit = limit_totals.sum()
        
        labels = ['Market Orders', 'Limit Orders']
        sizes = [total_market, total_limit]
        colors = ['lightcoral', 'lightgreen']
        
        wedges, texts, autotexts = axes[1, 1].pie(sizes, labels=labels, colors=colors, 
                                                autopct='%1.1f%%', startangle=90)
        axes[1, 1].set_title('Overall Market vs Limit Order Distribution')
        
        # 6. Performance Summary Table
        axes[1, 2].axis('tight')
        axes[1, 2].axis('off')
        
        # Create summary statistics
        summary_data = []
        for strategy in combined_df['strategy'].unique():
            strategy_data = combined_df[combined_df['strategy'] == strategy]
            
            total_orders = len(strategy_data)
            avg_market_size = strategy_data['market_v'].mean()
            avg_limit_size = strategy_data['limit_v'].mean()
            avg_price = strategy_data['market_p'].mean()
            
            summary_data.append([
                strategy,
                f"{total_orders}",
                f"{avg_market_size:.0f}",
                f"{avg_limit_size:.0f}",
                f"${avg_price:.4f}"
            ])
        
        headers = ['Strategy', 'Orders', 'Avg Market', 'Avg Limit', 'Avg Price']
        
        table = axes[1, 2].table(cellText=summary_data, colLabels=headers, 
                               cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        # Style the table
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        axes[1, 2].set_title('Performance Summary')
        
        plt.tight_layout()
        
        # Save plot
        plt.savefig(self.output_path / "sor_performance_dashboard.png", dpi=300, bbox_inches='tight')
        plt.show()
        print(f"✅ Dashboard saved to: {self.output_path / 'sor_performance_dashboard.png'}")
    
    def create_cost_analysis(self, results: Dict[str, pd.DataFrame]) -> None:
        """Create cost analysis visualization"""
        
        print("💰 Creating Cost Analysis...")
        
        # Combine results
        all_data = []
        for name, df in results.items():
            df_copy = df.copy()
            df_copy['strategy'] = name
            all_data.append(df_copy)
        
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Calculate costs
        combined_df['market_cost'] = combined_df['market_v'] * combined_df['market_p']
        combined_df['limit_revenue'] = combined_df['limit_fill'] * combined_df['limit_p']
        combined_df['net_cost'] = combined_df['market_cost'] - combined_df['limit_revenue']
        
        # Create cost visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Cost Function Analysis', fontsize=16, fontweight='bold')
        
        # 1. Cost components over time
        for strategy in combined_df['strategy'].unique():
            strategy_data = combined_df[combined_df['strategy'] == strategy]
            if 'time' in strategy_data.columns:
                axes[0, 0].plot(strategy_data['time'], strategy_data['market_cost'], 
                              label=f'{strategy} - Market Cost', linewidth=2)
                axes[0, 0].plot(strategy_data['time'], strategy_data['limit_revenue'], 
                              label=f'{strategy} - Limit Revenue', linewidth=2, linestyle='--')
        
        axes[0, 0].set_title('Cost Components Over Time')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Cost ($)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Net cost distribution
        for strategy in combined_df['strategy'].unique():
            strategy_data = combined_df[combined_df['strategy'] == strategy]
            axes[0, 1].hist(strategy_data['net_cost'], bins=20, alpha=0.6, 
                          label=f'{strategy}', density=True)
        
        axes[0, 1].set_title('Net Cost Distribution')
        axes[0, 1].set_xlabel('Net Cost ($)')
        axes[0, 1].set_ylabel('Density')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Market cost vs limit revenue scatter
        scatter = axes[1, 0].scatter(combined_df['market_cost'], combined_df['limit_revenue'], 
                                   c=combined_df['net_cost'], cmap='RdYlBu', alpha=0.6)
        axes[1, 0].set_title('Market Cost vs Limit Revenue')
        axes[1, 0].set_xlabel('Market Cost ($)')
        axes[1, 0].set_ylabel('Limit Revenue ($)')
        plt.colorbar(scatter, ax=axes[1, 0], label='Net Cost ($)')
        
        # 4. Cumulative cost by strategy
        for strategy in combined_df['strategy'].unique():
            strategy_data = combined_df[combined_df['strategy'] == strategy]
            cumulative_cost = strategy_data['net_cost'].cumsum()
            axes[1, 1].plot(range(len(cumulative_cost)), cumulative_cost, 
                          label=f'{strategy}', linewidth=2)
        
        axes[1, 1].set_title('Cumulative Cost by Strategy')
        axes[1, 1].set_xlabel('Order Sequence')
        axes[1, 1].set_ylabel('Cumulative Cost ($)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plt.savefig(self.output_path / "cost_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()
        print(f"✅ Cost analysis saved to: {self.output_path / 'cost_analysis.png'}")
    
    def create_strategy_comparison(self, results: Dict[str, pd.DataFrame]) -> None:
        """Create strategy comparison visualization"""
        
        if len(results) < 2:
            print("⚠️ Need at least 2 strategies for comparison")
            return
        
        print("📈 Creating Strategy Comparison...")
        
        # Combine all results
        all_data = []
        for name, df in results.items():
            df_copy = df.copy()
            df_copy['strategy'] = name
            all_data.append(df_copy)
        
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Calculate performance metrics
        performance_metrics = {}
        
        for strategy in combined_df['strategy'].unique():
            strategy_data = combined_df[combined_df['strategy'] == strategy]
            
            total_orders = len(strategy_data)
            total_market_orders = strategy_data['market_v'].sum()
            total_limit_orders = strategy_data['limit_v'].sum()
            total_fills = (strategy_data['market_v'] + strategy_data['limit_fill']).sum()
            total_size = (strategy_data['market_v'] + strategy_data['limit_v']).sum()
            
            fill_rate = total_fills / total_size if total_size > 0 else 0
            market_ratio = total_market_orders / total_size if total_size > 0 else 0
            
            avg_execution_price = ((strategy_data['market_v'] * strategy_data['market_p']).sum() + 
                                 (strategy_data['limit_fill'] * strategy_data['limit_p']).sum()) / total_fills if total_fills > 0 else 0
            
            total_cost = (strategy_data['market_v'] * strategy_data['market_p']).sum()
            
            performance_metrics[strategy] = {
                'Total Orders': total_orders,
                'Total Size': int(total_size),
                'Fill Rate': fill_rate,
                'Market Ratio': market_ratio,
                'Avg Price': avg_execution_price,
                'Total Cost': total_cost
            }
        
        # Create comparison visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Strategy Comparison Report', fontsize=16, fontweight='bold')
        
        strategies = list(performance_metrics.keys())
        
        # 1. Fill Rate Comparison
        fill_rates = [performance_metrics[s]['Fill Rate'] for s in strategies]
        bars1 = axes[0, 0].bar(strategies, fill_rates, color='skyblue', alpha=0.7)
        axes[0, 0].set_title('Fill Rate Comparison')
        axes[0, 0].set_ylabel('Fill Rate')
        axes[0, 0].set_ylim(0, 1)
        
        for bar, rate in zip(bars1, fill_rates):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                          f'{rate:.1%}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Market vs Limit Order Distribution
        market_ratios = [performance_metrics[s]['Market Ratio'] for s in strategies]
        limit_ratios = [1 - ratio for ratio in market_ratios]
        
        x = np.arange(len(strategies))
        width = 0.35
        
        bars2 = axes[0, 1].bar(x - width/2, market_ratios, width, label='Market Orders', color='lightcoral', alpha=0.7)
        bars3 = axes[0, 1].bar(x + width/2, limit_ratios, width, label='Limit Orders', color='lightgreen', alpha=0.7)
        
        axes[0, 1].set_title('Market vs Limit Order Distribution')
        axes[0, 1].set_ylabel('Proportion')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(strategies, rotation=45)
        axes[0, 1].legend()
        
        # 3. Average Execution Price
        avg_prices = [performance_metrics[s]['Avg Price'] for s in strategies]
        bars4 = axes[0, 2].bar(strategies, avg_prices, color='gold', alpha=0.7)
        axes[0, 2].set_title('Average Execution Price')
        axes[0, 2].set_ylabel('Price ($)')
        
        for bar, price in zip(bars4, avg_prices):
            axes[0, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(avg_prices) * 0.001,
                          f'${price:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        # 4. Total Cost Comparison
        total_costs = [performance_metrics[s]['Total Cost'] for s in strategies]
        bars5 = axes[1, 0].bar(strategies, total_costs, color='mediumpurple', alpha=0.7)
        axes[1, 0].set_title('Total Execution Cost')
        axes[1, 0].set_ylabel('Total Cost ($)')
        
        for bar, cost in zip(bars5, total_costs):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(total_costs) * 0.01,
                          f'${cost:,.0f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        # 5. Order Size Distribution
        for strategy in strategies:
            strategy_data = combined_df[combined_df['strategy'] == strategy]
            total_order_sizes = strategy_data['market_v'] + strategy_data['limit_v']
            axes[1, 1].hist(total_order_sizes, bins=20, alpha=0.6, label=strategy, density=True)
        
        axes[1, 1].set_title('Order Size Distribution')
        axes[1, 1].set_xlabel('Order Size')
        axes[1, 1].set_ylabel('Density')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Performance Summary Table
        axes[1, 2].axis('tight')
        axes[1, 2].axis('off')
        
        table_data = []
        headers = ['Strategy', 'Orders', 'Fill Rate', 'Market %', 'Avg Price']
        
        for strategy in strategies:
            metrics = performance_metrics[strategy]
            table_data.append([
                strategy,
                str(metrics['Total Orders']),
                f"{metrics['Fill Rate']:.1%}",
                f"{metrics['Market Ratio']:.1%}",
                f"${metrics['Avg Price']:.4f}"
            ])
        
        table = axes[1, 2].table(cellText=table_data, colLabels=headers, 
                               cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        
        # Style the table
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        axes[1, 2].set_title('Performance Summary')
        
        plt.tight_layout()
        
        # Save plot
        plt.savefig(self.output_path / "strategy_comparison.png", dpi=300, bbox_inches='tight')
        plt.show()
        print(f"✅ Strategy comparison saved to: {self.output_path / 'strategy_comparison.png'}")
    
    def create_summary_report(self, results: Dict[str, pd.DataFrame]) -> None:
        """Create a comprehensive summary report"""
        
        print("📋 Creating Summary Report...")
        
        if not results:
            print("❌ No results to summarize")
            return
        
        # Combine all results
        all_data = []
        for name, df in results.items():
            df_copy = df.copy()
            df_copy['strategy'] = name
            all_data.append(df_copy)
        
        combined_df = pd.concat(all_data, ignore_index=True)
        
        print("\n" + "="*60)
        print("📊 CONT-KUKANOV SOR ANALYSIS SUMMARY")
        print("="*60)
        
        # Overall statistics
        total_strategies = len(results)
        total_orders = len(combined_df)
        total_volume = (combined_df['market_v'] + combined_df['limit_v']).sum()
        total_fills = (combined_df['market_v'] + combined_df['limit_fill']).sum()
        overall_fill_rate = total_fills / total_volume if total_volume > 0 else 0
        
        print(f"📈 OVERALL STATISTICS:")
        print(f"  • Total Strategies Analyzed: {total_strategies}")
        print(f"  • Total Orders Executed: {total_orders:,}")
        print(f"  • Total Volume Processed: {total_volume:,} shares")
        print(f"  • Overall Fill Rate: {overall_fill_rate:.1%}")
        
        # Strategy-specific performance
        print(f"\n🎯 STRATEGY PERFORMANCE:")
        
        for strategy in combined_df['strategy'].unique():
            strategy_data = combined_df[combined_df['strategy'] == strategy]
            
            strategy_orders = len(strategy_data)
            strategy_volume = (strategy_data['market_v'] + strategy_data['limit_v']).sum()
            strategy_fills = (strategy_data['market_v'] + strategy_data['limit_fill']).sum()
            strategy_fill_rate = strategy_fills / strategy_volume if strategy_volume > 0 else 0
            
            market_ratio = strategy_data['market_v'].sum() / strategy_volume if strategy_volume > 0 else 0
            avg_price = strategy_data['market_p'].mean()
            
            print(f"\n  📊 {strategy}:")
            print(f"    • Orders: {strategy_orders:,}")
            print(f"    • Volume: {strategy_volume:,} shares")
            print(f"    • Fill Rate: {strategy_fill_rate:.1%}")
            print(f"    • Market Order Ratio: {market_ratio:.1%}")
            print(f"    • Average Price: ${avg_price:.4f}")
        
        # Best performing strategy
        strategy_scores = {}
        for strategy in combined_df['strategy'].unique():
            strategy_data = combined_df[combined_df['strategy'] == strategy]
            
            strategy_volume = (strategy_data['market_v'] + strategy_data['limit_v']).sum()
            strategy_fills = (strategy_data['market_v'] + strategy_data['limit_fill']).sum()
            fill_rate = strategy_fills / strategy_volume if strategy_volume > 0 else 0
            
            market_ratio = strategy_data['market_v'].sum() / strategy_volume if strategy_volume > 0 else 0
            
            # Simple scoring: high fill rate, balanced market/limit ratio
            score = fill_rate * 0.6 + (1 - abs(market_ratio - 0.5)) * 0.4
            strategy_scores[strategy] = score
        
        best_strategy = max(strategy_scores, key=strategy_scores.get)
        best_score = strategy_scores[best_strategy]
        
        print(f"\n🏆 BEST PERFORMING STRATEGY: {best_strategy}")
        print(f"📈 Performance Score: {best_score:.3f}")
        
        print(f"\n💡 RECOMMENDATIONS:")
        print(f"  • Best overall strategy: {best_strategy}")
        print(f"  • Focus on optimizing fill rates")
        print(f"  • Balance market vs limit order allocation")
        print(f"  • Monitor queue depth for better timing")
        
        return combined_df

def main():
    """Main function to run visualizations"""
    
    print("🎨 Starting SOR Visualization Suite")
    print("="*50)
    
    # Initialize visualization suite
    viz_suite = SORVisualizationSuite()
    
    # Load data
    print("\n📁 Loading backtest results...")
    results = viz_suite.load_backtest_results()
    
    if not results:
        print("❌ No results found. Please run backtests first.")
        print("💡 Expected files: *result*.csv in results/ directory")
        return
    
    print(f"\n🎯 Creating visualizations for {len(results)} strategy results...")
    
    try:
        # Create visualizations
        viz_suite.create_simple_performance_dashboard(results)
        viz_suite.create_cost_analysis(results)
        
        if len(results) > 1:
            viz_suite.create_strategy_comparison(results)
        
        # Create summary report
        summary_df = viz_suite.create_summary_report(results)
        
        print(f"\n🎉 ALL VISUALIZATIONS COMPLETED!")
        print(f"📁 Output directory: {viz_suite.output_path}")
        
        output_files = list(viz_suite.output_path.glob("*.png"))
        if output_files:
            print(f"🖼️ Generated files:")
            for file in output_files:
                print(f"  • {file.name}")
        else:
            print("📊 Visualizations displayed but no files saved")
            
    except Exception as e:
        print(f"❌ Error creating visualizations: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()