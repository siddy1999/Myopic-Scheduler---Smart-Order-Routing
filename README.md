# Myopic Scheduler - Smart Order Routing

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A sophisticated algorithmic trading system that integrates **myopic scheduling** with **Smart Order Routing (SOR)** to optimize trade execution timing and sizing. This system makes short-term optimal decisions for order execution, adapting to real-time market conditions to minimize trading costs and market impact.

## 🚀 Key Features

- **Myopic Market Impact Modeling**: Advanced mathematical models for optimal trade timing
- **Smart Order Routing Integration**: Seamless integration with existing SOR frameworks
- **Real-time Market Analysis**: Dynamic parameter estimation from live market data
- **Multi-venue Optimization**: Intelligent venue selection and order allocation
- **Comprehensive Backtesting**: Robust testing framework with performance analytics
- **Risk Management**: Built-in risk controls and position limits
- **Visualization Suite**: Advanced plotting and analysis tools

## 🏗️ System Architecture

```
Market Data Input → Myopic Scheduler → SOR Optimizer → Execution Engine
     ↓                    ↓                ↓              ↓
Tick Data, Order Book  Optimal Timing   Venue Allocation  Order Placement
Volume, Volatility     & Sizing         Market/Limit Split  & Management
```

## 📊 Mathematical Foundation

The system implements advanced market impact modeling:

- **Market Impact Model**: `I(t+1) = I(t) + λ*Q(t) - β*I(t)`
- **Alpha Prediction**: `α(t) = S(T) - S(t)`
- **Optimal Control**: `Q*(t) = (α'(t) + β*I*(t)) / λ`
- **Lambda Estimation**: `Δp = λ * ΔQ + ε`

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- Market data access (MBP-10 format supported)
- Required Python packages

### Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/myopic-scheduler.git
cd myopic-scheduler

# Install dependencies
pip install -r requirements.txt

# Run example
python examples/basic_usage.py
```

### Development Setup

```bash
# Install in development mode
pip install -e .

# Run tests
pytest tests/

# Run with specific configuration
python main_myopic_integration.py --config config/myopic_config.yaml
```

## 📈 Usage Examples

### Basic Usage

```python
from myopic_sor_scheduler import MyopicScheduler, MyopicParameters
from main_myopic_integration import run_myopic_backtest

# Configure parameters
strategy_params = {
    'S': 1000,          # Total order size
    'T': 15,            # Time horizon (minutes)
    'f': 0.003,         # Fee rate
    'r': [0.002],       # Rebate rates
    'lambda_u': 0.05,   # Underfill penalty
    'lambda_o': 0.05,   # Overfill penalty
    'N': 1000           # Monte Carlo simulations
}

# Run backtest
results = run_myopic_backtest(
    stock="AAPL",
    day="2025-04-02",
    strategy_params=strategy_params,
    data_path="./data/",
    use_myopic=True
)
```

### Advanced Configuration

```python
# Custom myopic parameters
myopic_params = MyopicParameters(
    lambda_value=25000.0,  # Market impact coefficient
    beta=0.693,           # Impact decay (1-hour half-life)
    volatility=0.01,      # Asset volatility
    adv=1000000.0,       # Average Daily Volume
    T=6.5,               # Trading session length
    Q_0=0.01             # Position normalization
)

# Create scheduler
scheduler = MyopicScheduler(myopic_params)

# Generate trading schedule
schedule = scheduler.generate_trading_schedule(
    df=market_data,
    total_quantity=1000,
    time_horizon=30
)
```

## 📁 Project Structure

```
myopic-scheduler/
├── src/
│   ├── myopic_sor_scheduler.py      # Core myopic scheduling logic
│   ├── myopic_analysis_utils.py     # Analysis and visualization tools
│   ├── benchmark_implementation.py  # Benchmarking framework
│   └── sor_visualizations.py        # Visualization suite
├── examples/
│   ├── basic_usage.py               # Basic usage example
│   ├── advanced_configuration.py    # Advanced configuration
│   └── batch_processing.py          # Batch processing example
├── tests/
│   ├── test_myopic_scheduler.py     # Unit tests
│   ├── test_integration.py          # Integration tests
│   └── test_analysis_utils.py       # Analysis tests
├── config/
│   ├── myopic_config.yaml           # Configuration file
│   └── parameters.json              # Parameter presets
├── docs/
│   ├── api_reference.md             # API documentation
│   ├── mathematical_foundation.md   # Mathematical details
│   └── performance_analysis.md      # Performance metrics
├── data/                            # Market data directory
├── results/                         # Backtest results
├── requirements.txt                 # Python dependencies
├── setup.py                        # Package setup
├── README.md                       # This file
└── LICENSE                         # MIT License
```

## 🔧 Configuration

### Strategy Parameters
```yaml
strategy:
  S: 1000                    # Total order size (shares)
  T: 15                      # Time horizon (minutes)
  f: 0.003                   # Fee rate (30 bps)
  r: [0.002]                 # Rebate rates per venue
  lambda_u: 0.05             # Underfill penalty
  lambda_o: 0.05             # Overfill penalty
  N: 1000                    # Monte Carlo simulations

myopic:
  lambda_value: 25000.0      # Market impact coefficient
  beta: 0.693                # Impact decay parameter
  volatility: 0.01           # Asset volatility
  adv: 1000000.0            # Average Daily Volume
  T: 6.5                     # Trading session length
  Q_0: 0.01                  # Position normalization
```

## 📊 Performance Metrics

The system provides comprehensive performance analysis:

- **Cost per Share**: Average execution cost
- **Market Impact**: Price impact of trading activity
- **Fill Rate**: Percentage of target quantity executed
- **Decision Efficiency**: Optimization of trading decisions
- **Risk Metrics**: Volatility and drawdown analysis

## 🧪 Testing

```bash
# Run all tests
pytest

# Run specific test suite
pytest tests/test_myopic_scheduler.py

# Run with coverage
pytest --cov=src --cov-report=html

# Run integration tests
pytest tests/test_integration.py -v
```

## 📈 Benchmarking

Compare performance against traditional approaches:

```python
from benchmark_implementation import BenchmarkRunner, BenchmarkConfig

# Configure benchmark
config = BenchmarkConfig(
    ticker="AAPL",
    days=["2025-04-01", "2025-04-02"],
    algorithms=["VWAP", "TWAP", "Myopic"]
)

# Run benchmark
runner = BenchmarkRunner(config)
results = runner.run_benchmark()
```

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📚 Documentation

- [API Reference](docs/api_reference.md)
- [Mathematical Foundation](docs/mathematical_foundation.md)
- [Performance Analysis](docs/performance_analysis.md)
- [Configuration Guide](docs/configuration_guide.md)

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **Data Format Issues**: Check that market data has required columns
3. **Memory Issues**: Use chunked processing for large datasets
4. **Performance Issues**: Enable parallel processing

See [Troubleshooting Guide](docs/troubleshooting.md) for detailed solutions.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Market impact modeling based on academic research
- Integration with existing SOR frameworks
- Community contributions and feedback

## 📞 Support

- Create an issue for bug reports
- Start a discussion for questions
- Check documentation for detailed guides

## 🔗 Related Projects

- [Smart Order Routing Framework](https://github.com/example/sor-framework)
- [Market Data Processing](https://github.com/example/market-data)
- [Algorithmic Trading Tools](https://github.com/example/algo-trading)

---

**Note**: This system is designed for educational and research purposes. Please ensure compliance with applicable regulations when using in production environments.
