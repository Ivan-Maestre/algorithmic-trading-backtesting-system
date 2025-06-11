# Algorithmic Trading Backtesting System

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![TFG](https://img.shields.io/badge/TFG-Universidad%20La%20Salle-red.svg)
![Status](https://img.shields.io/badge/status-completed-green.svg)
![Academic](https://img.shields.io/badge/academic-project-orange.svg)

## Project Overview
A comprehensive backtesting system for evaluating algorithmic trading strategies based on academic research and efficient system design principles. This system provides a robust framework for testing various trading strategies with real market data.

## Features
- **Multiple Strategy Support**: EMA Crossover, Momentum, and Reinforcement Learning strategies
- **Comprehensive Analysis**: Performance metrics, risk analysis, and visualization tools
- **Data Pipeline**: Automated data fetching and preprocessing
- **Interactive Dashboard**: HTML-based performance visualization
- **Modular Architecture**: Easy to extend with new strategies and indicators

## Objective
Develop an operative, simple, and effective backtesting system capable of evaluating algorithmic trading strategies with focus on getting to results efficiently.

## Priority Strategies
1. **EMA Crossover Strategy**: 12-period and 25-period EMAs on 4-hour timeframe
2. **Momentum Strategy**: As defined in the TFG research
3. **Reinforcement Learning Strategy**: Q-learning algorithm implementation

## Backtesting Period
- **Start Date**: January 1, 2018
- **End Date**: December 31, 2023

## Project Structure
```
/
├── docs/                           # Documentation
│   ├── PRD.md                     # Product Requirements Document
│   ├── development_checklist.md   # Development Checklist
│   ├── progress_log.md            # Progress Log (Development Journal)
│   └── strategy_results_log.md    # Strategy & Results Log
├── src/                           # Source code
│   ├── data/                      # Data handling modules
│   ├── strategies/                # Trading strategy implementations
│   ├── backtesting/              # Backtesting engine
│   ├── visualization/            # Data visualization tools
│   ├── config.py                 # Configuration settings
│   └── main.py                   # Main application entry point
├── results/                      # Backtesting results and reports
├── venv/                         # Virtual environment (excluded from git)
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup
1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/AlgoSystem.git
   cd AlgoSystem
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   ```bash
   # Copy the environment template
   cp env.example .env
   
   # Edit .env with your API keys (if needed for live data)
   # Note: The system includes sample data and can run without API keys
   ```

5. **Verify installation**
   ```bash
   python src/main.py
   ```

## Usage

### Running the Main Analysis
```bash
python src/main.py
```

### Running Specific Tests
```bash
# Test EMA strategy
python src/test_simple_ema_backtest.py

# Test backtest engine
python src/test_backtest_engine.py

# Test data pipeline
python src/test_data_pipeline.py
```

### Creating Performance Dashboard
```bash
python src/create_simple_dashboard.py
```

### Full Analysis
```bash
python src/run_full_analysis.py
```

## Results
The system generates comprehensive analysis including:
- Performance metrics (Returns, Sharpe Ratio, Maximum Drawdown)
- Risk analysis and volatility measurements
- Interactive HTML dashboard for visualization
- JSON reports with detailed statistics

## Configuration
Edit `src/config.py` to customize:
- Data sources and symbols
- Strategy parameters
- Backtesting periods
- Risk management settings

## Contributing
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-strategy`)
3. Commit your changes (`git commit -am 'Add new strategy'`)
4. Push to the branch (`git push origin feature/new-strategy`)
5. Create a Pull Request

## License
This project is licensed under the MIT License - see the LICENSE file for details.

### Project Objectives
This project aims to develop and evaluate algorithmic trading strategies through a comprehensive backtesting framework, comparing:
- Traditional technical analysis (EMA Crossover)
- Quantitative momentum strategies
- Machine learning approaches (Q-Learning Reinforcement Learning)

### Key Contributions
- **Comprehensive Strategy Comparison**: Side-by-side analysis of three distinct algorithmic approaches
- **Robust Backtesting Framework**: Professional-grade system with realistic market conditions
- **Performance Analytics**: Detailed risk-adjusted performance metrics and visualizations
- **Academic Rigor**: Systematic methodology with proper statistical validation
