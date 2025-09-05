# Contributing to Myopic Scheduler

Thank you for your interest in contributing to the Myopic Scheduler project! This document provides guidelines and information for contributors.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contributing Guidelines](#contributing-guidelines)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Issue Reporting](#issue-reporting)

## Code of Conduct

This project adheres to a code of conduct that we expect all contributors to follow. Please be respectful, inclusive, and constructive in all interactions.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/yourusername/Myopic-Scheduler---Smart-Order-Routing.git
   cd Myopic-Scheduler---Smart-Order-Routing
   ```
3. **Add the upstream remote**:
   ```bash
   git remote add upstream https://github.com/siddy1999/Myopic-Scheduler---Smart-Order-Routing.git
   ```

## Development Setup

### Prerequisites

- Python 3.8 or higher
- Git
- Virtual environment (recommended)

### Setup Steps

1. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install -e .[dev]  # Install in development mode with dev dependencies
   ```

3. **Install pre-commit hooks** (optional but recommended):
   ```bash
   pip install pre-commit
   pre-commit install
   ```

4. **Run tests** to ensure everything works:
   ```bash
   pytest
   ```

## Contributing Guidelines

### Types of Contributions

We welcome several types of contributions:

- **Bug fixes**: Fix issues in the codebase
- **Feature additions**: Add new functionality
- **Documentation**: Improve or add documentation
- **Performance improvements**: Optimize existing code
- **Tests**: Add or improve test coverage
- **Examples**: Add usage examples or tutorials

### Before You Start

1. **Check existing issues** to see if your contribution is already being worked on
2. **Create an issue** for significant changes to discuss the approach
3. **For small fixes**, you can proceed directly with a pull request

### Workflow

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/issue-description
   ```

2. **Make your changes** following the coding standards below

3. **Test your changes**:
   ```bash
   pytest tests/
   pytest tests/test_your_new_code.py  # Run specific tests
   ```

4. **Update documentation** if needed

5. **Commit your changes**:
   ```bash
   git add .
   git commit -m "Add: Brief description of your changes"
   ```

6. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Create a Pull Request** on GitHub

## Pull Request Process

### Before Submitting

- [ ] Code follows the project's coding standards
- [ ] Tests pass locally
- [ ] New functionality is covered by tests
- [ ] Documentation is updated if needed
- [ ] Commit messages are clear and descriptive
- [ ] Branch is up to date with main

### Pull Request Template

When creating a pull request, please include:

1. **Description**: What changes were made and why
2. **Type of change**: Bug fix, feature, documentation, etc.
3. **Testing**: How the changes were tested
4. **Breaking changes**: Any breaking changes (if applicable)
5. **Related issues**: Link to any related issues

### Review Process

- All pull requests require review before merging
- Address feedback promptly and constructively
- Keep pull requests focused and reasonably sized
- Update your branch if requested

## Coding Standards

### Python Code Style

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guidelines
- Use [Black](https://black.readthedocs.io/) for code formatting
- Use [flake8](https://flake8.pycqa.org/) for linting
- Use [mypy](https://mypy.readthedocs.io/) for type checking

### Code Formatting

```bash
# Format code with Black
black src/ tests/

# Check code style
flake8 src/ tests/

# Type checking
mypy src/
```

### Naming Conventions

- **Functions and variables**: `snake_case`
- **Classes**: `PascalCase`
- **Constants**: `UPPER_SNAKE_CASE`
- **Private methods**: `_leading_underscore`
- **Module names**: `snake_case`

### Documentation

- Use docstrings for all public functions, classes, and methods
- Follow [Google style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) docstrings
- Include type hints for function parameters and return values

Example:
```python
def calculate_market_impact(
    volume: float, 
    volatility: float, 
    adv: float
) -> float:
    """
    Calculate market impact using the myopic model.
    
    Args:
        volume: Trading volume
        volatility: Asset volatility
        adv: Average daily volume
        
    Returns:
        Calculated market impact value
        
    Raises:
        ValueError: If any parameter is negative
    """
    if volume < 0 or volatility < 0 or adv <= 0:
        raise ValueError("Parameters must be non-negative")
    
    return (volume * volatility) / adv
```

## Testing

### Test Structure

- Place tests in the `tests/` directory
- Mirror the source code structure in tests
- Use descriptive test names
- Group related tests in classes

### Writing Tests

```python
import pytest
from myopic_scheduler import MyopicScheduler, MyopicParameters

class TestMyopicScheduler:
    """Test cases for MyopicScheduler class."""
    
    def test_initialization(self):
        """Test scheduler initialization."""
        params = MyopicParameters(
            lambda_value=25000.0,
            beta=0.693,
            volatility=0.01,
            adv=1000000.0
        )
        scheduler = MyopicScheduler(params)
        assert scheduler.params.lambda_value == 25000.0
    
    def test_lambda_estimation(self):
        """Test lambda parameter estimation."""
        # Test implementation
        pass
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_myopic_scheduler.py

# Run with verbose output
pytest -v

# Run only fast tests
pytest -m "not slow"
```

## Documentation

### Code Documentation

- All public APIs must have docstrings
- Include examples in docstrings where helpful
- Document complex algorithms and mathematical formulas
- Keep comments up to date with code changes

### User Documentation

- Update README.md for user-facing changes
- Add examples for new features
- Update API documentation in `docs/`
- Include migration guides for breaking changes

### Mathematical Documentation

For mathematical components, include:

- Formula definitions
- Parameter explanations
- Derivation references
- Implementation notes

Example:
```python
def calculate_optimal_quantity(
    alpha_prime: float, 
    beta: float, 
    lambda_value: float, 
    impact: float
) -> float:
    """
    Calculate optimal trading quantity using myopic model.
    
    Implements the formula:
    Q*(t) = (α'(t) + β * I*(t)) / λ
    
    Where:
    - α'(t): Rate of change of alpha
    - β: Impact decay parameter
    - I*(t): Optimal market impact
    - λ: Market impact coefficient
    
    Args:
        alpha_prime: Rate of change of alpha signal
        beta: Impact decay parameter
        lambda_value: Market impact coefficient
        impact: Current market impact
        
    Returns:
        Optimal trading quantity
    """
    return (alpha_prime + beta * impact) / lambda_value
```

## Issue Reporting

### Before Creating an Issue

1. **Search existing issues** to avoid duplicates
2. **Check if it's already fixed** in the latest version
3. **Gather information** about your environment and the issue

### Issue Template

When creating an issue, please include:

1. **Bug Report**:
   - Clear description of the bug
   - Steps to reproduce
   - Expected vs actual behavior
   - Environment details (OS, Python version, etc.)
   - Error messages and logs

2. **Feature Request**:
   - Clear description of the feature
   - Use case and motivation
   - Proposed implementation (if any)
   - Alternatives considered

3. **Documentation Issue**:
   - What documentation is unclear or missing
   - Where the issue is located
   - Suggested improvements

### Good Issue Examples

**Bug Report**:
```
Title: Lambda estimation fails with small datasets

Description:
The lambda estimation function throws an error when processing datasets with fewer than 100 observations.

Steps to reproduce:
1. Load a dataset with 50 observations
2. Call scheduler.estimate_lambda(df)
3. Error occurs: "ValueError: Insufficient data for estimation"

Expected behavior:
Should handle small datasets gracefully or provide a clear error message.

Environment:
- Python 3.9.7
- pandas 1.5.0
- numpy 1.21.0
```

**Feature Request**:
```
Title: Add support for multiple time horizons in lambda estimation

Description:
Currently, lambda estimation only supports single time horizons. It would be useful to estimate lambda values for multiple time horizons simultaneously.

Use case:
When analyzing different trading strategies, we need lambda values for various time horizons (1min, 5min, 15min, 1hour) to optimize execution.

Proposed implementation:
Modify the estimate_lambda method to accept a list of time horizons and return a dictionary mapping horizons to lambda values.
```

## Release Process

### Version Numbering

We follow [Semantic Versioning](https://semver.org/):
- **MAJOR**: Incompatible API changes
- **MINOR**: New functionality (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Release Checklist

- [ ] All tests pass
- [ ] Documentation is updated
- [ ] Version number is incremented
- [ ] CHANGELOG.md is updated
- [ ] Release notes are prepared

## Getting Help

- **Documentation**: Check the `docs/` directory
- **Issues**: Search existing issues or create a new one
- **Discussions**: Use GitHub Discussions for questions
- **Email**: Contact the maintainers directly

## Recognition

Contributors will be recognized in:
- CONTRIBUTORS.md file
- Release notes
- Project documentation

Thank you for contributing to Myopic Scheduler! 🚀
