#!/usr/bin/env python3
"""
Setup script for Myopic Scheduler - Smart Order Routing
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

# Get version
def get_version():
    version_file = os.path.join("src", "myopic_scheduler", "__init__.py")
    if os.path.exists(version_file):
        with open(version_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("__version__"):
                    return line.split("=")[1].strip().strip('"').strip("'")
    return "1.0.0"

setup(
    name="myopic-scheduler",
    version=get_version(),
    author="Siddharth",
    author_email="siddharth@example.com",  # Replace with your email
    description="Advanced Algorithmic Trading System: Myopic Scheduling with Smart Order Routing",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/siddy1999/Myopic-Scheduler---Smart-Order-Routing",
    project_urls={
        "Bug Reports": "https://github.com/siddy1999/Myopic-Scheduler---Smart-Order-Routing/issues",
        "Source": "https://github.com/siddy1999/Myopic-Scheduler---Smart-Order-Routing",
        "Documentation": "https://github.com/siddy1999/Myopic-Scheduler---Smart-Order-Routing/tree/main/docs",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Science/Research",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "pytest-mock>=3.8.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=0.950",
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
        "viz": [
            "plotly>=5.10.0",
            "dash>=2.6.0",
            "streamlit>=1.12.0",
            "ipywidgets>=7.7.0",
        ],
        "ml": [
            "xgboost>=1.6.0",
            "lightgbm>=3.3.0",
            "arch>=5.2.0",
            "pmdarima>=2.0.0",
        ],
        "optimization": [
            "cvxpy>=1.2.0",
            "gurobipy>=9.5.0",
        ],
        "backtesting": [
            "backtrader>=1.9.76",
            "zipline-reloaded>=2.2.0",
        ],
        "risk": [
            "quantlib>=1.29",
            "pyfolio>=0.9.2",
        ],
    },
    entry_points={
        "console_scripts": [
            "myopic-scheduler=myopic_scheduler.cli:main",
            "myopic-backtest=myopic_scheduler.backtest:main",
            "myopic-analyze=myopic_scheduler.analysis:main",
        ],
    },
    include_package_data=True,
    package_data={
        "myopic_scheduler": [
            "config/*.yaml",
            "config/*.json",
            "data/*.csv",
            "data/*.parquet",
        ],
    },
    zip_safe=False,
    keywords=[
        "algorithmic-trading",
        "smart-order-routing",
        "market-impact",
        "myopic-scheduling",
        "financial-modeling",
        "quantitative-finance",
        "trading-algorithms",
        "execution-optimization",
        "market-microstructure",
        "order-routing",
    ],
)
