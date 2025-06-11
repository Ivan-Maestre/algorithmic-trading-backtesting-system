# Repository Guide - Algorithmic Trading Backtesting System

## 📖 About This Repository

This repository contains the complete implementation of an algorithmic trading backtesting system developed as part of a Final Degree Project (TFG) for the Bachelor's in Big Data and Intelligence Analytics (BIDA) program at Universidad La Salle.

## 🎓 Academic Context

**Project Title:** Algorithmic Trading Strategy Evaluation Through Comprehensive Backtesting  
**Author:** Ivan Maestre Ivanov  
**Institution:** Universidad La Salle  
**Program:** Bachelor's in Big Data and Intelligence Analytics (BIDA)  
**Academic Year:** 2023-2024  

## 🎯 Project Objectives

1. **Develop a Professional Backtesting System**: Create a robust, extensible framework for evaluating trading strategies
2. **Strategy Comparison**: Implement and compare three distinct algorithmic approaches:
   - Traditional Technical Analysis (EMA Crossover)
   - Quantitative Momentum Strategy
   - Machine Learning (Q-Learning Reinforcement Learning)
3. **Performance Analysis**: Provide comprehensive risk-adjusted performance metrics
4. **Academic Validation**: Ensure statistical rigor and proper methodology

## 📁 Repository Structure

```
AlgoSystem/
├── src/                          # Source code
│   ├── backtesting/             # Core backtesting engine
│   │   ├── backtest_engine.py   # Main backtesting logic
│   │   ├── portfolio.py         # Portfolio management
│   │   ├── order.py             # Order execution system
│   │   └── trade.py             # Trade tracking
│   ├── strategies/              # Trading strategy implementations
│   │   ├── base_strategy.py     # Base strategy interface
│   │   ├── ema_crossover.py     # EMA crossover strategy
│   │   ├── momentum.py          # Momentum strategy
│   │   └── q_learning.py        # Q-learning RL strategy
│   ├── data/                    # Data handling modules
│   │   ├── binance_client.py    # Data fetching from Binance
│   │   ├── data_processor.py    # Technical indicators
│   │   └── data_validator.py    # Data quality validation
│   └── visualization/           # Dashboard and charts
├── docs/                        # Documentation
├── results/                     # Analysis results
├── requirements.txt             # Python dependencies
├── LICENSE                      # MIT License
└── README.md                    # Main documentation
```

## 🚀 Key Features

### Backtesting Engine
- **Order Management**: Realistic order execution with slippage and commissions
- **Portfolio Tracking**: Real-time portfolio value and position management
- **Risk Management**: Stop-loss, take-profit, and position sizing
- **Performance Metrics**: Comprehensive analytics (Sharpe ratio, drawdown, etc.)

### Trading Strategies
- **EMA Crossover**: Classic technical analysis with 12/25 period EMAs
- **Momentum Strategy**: Multi-timeframe momentum with volume confirmation
- **Q-Learning RL**: Reinforcement learning with adaptive decision making

### Data Pipeline
- **Real-time Data**: Binance API integration for live market data
- **Technical Indicators**: EMA, RSI, MACD, Bollinger Bands, ATR
- **Data Validation**: Quality checks and outlier detection

### Visualization & Analysis
- **Interactive Dashboard**: HTML-based performance visualization
- **Performance Reports**: JSON exports with detailed metrics
- **Comparison Charts**: Side-by-side strategy analysis

## 📊 Research Results

The system was tested over a 6-year period (2018-2023) with the following key findings:

### Performance Summary
- **Best Strategy**: Momentum Strategy (56.78% cumulative return)
- **Most Consistent**: Q-Learning RL (best risk-adjusted returns)
- **Market Adaptation**: All strategies showed varying performance across market cycles

### Market Condition Analysis
- **2018 (Bear Market)**: All strategies showed defensive performance
- **2019-2021 (Bull Market)**: Peak performance period for momentum strategies
- **2022-2023 (Mixed Conditions)**: RL strategy showed better adaptation

## 🛠️ Technical Implementation

### Technologies Used
- **Python 3.8+**: Core programming language
- **Pandas/NumPy**: Data manipulation and numerical computing
- **Scikit-learn/TensorFlow**: Machine learning components
- **Plotly**: Interactive visualizations
- **Binance API**: Real-time market data

### Design Patterns
- **Strategy Pattern**: Modular strategy implementation
- **Observer Pattern**: Event-driven backtesting
- **Factory Pattern**: Strategy and indicator creation
- **Singleton Pattern**: Configuration management

## 📈 Running the Analysis

### Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Run main analysis
python src/main.py

# Generate comprehensive report
python src/run_full_analysis.py

# Create performance dashboard
python src/create_simple_dashboard.py
```

### Test Components
```bash
# Test backtesting engine
python src/test_backtest_engine.py

# Test data pipeline
python src/test_data_pipeline.py

# Test EMA strategy
python src/test_simple_ema_backtest.py
```

## 📋 Academic Methodology

### Research Approach
1. **Literature Review**: Analysis of existing algorithmic trading research
2. **System Design**: Professional-grade backtesting framework
3. **Strategy Implementation**: Three distinct algorithmic approaches
4. **Empirical Testing**: 6-year historical data analysis
5. **Statistical Validation**: Risk-adjusted performance metrics
6. **Comparative Analysis**: Cross-strategy performance evaluation

### Validation Methods
- **Walk-forward Analysis**: Out-of-sample testing
- **Statistical Significance**: Proper hypothesis testing
- **Risk Metrics**: Comprehensive risk-adjusted returns
- **Robustness Testing**: Multiple market conditions

## 🔒 Security & Privacy

- **API Keys**: Secured through environment variables
- **No Sensitive Data**: No personal or financial information stored
- **Open Source**: Full transparency of methodology and implementation

## 📚 Documentation

- **README.md**: Main project documentation
- **EMA_Strategy_Comparison_Analysis.md**: Detailed strategy analysis
- **Code Comments**: Comprehensive inline documentation
- **Docstrings**: Full API documentation

## 🎯 Future Work

- **Additional Strategies**: Implementation of more sophisticated algorithms
- **Real-time Trading**: Live trading capability
- **Extended Asset Coverage**: Multi-asset portfolio strategies
- **Advanced ML**: Deep learning and ensemble methods

## 📞 Contact

For academic inquiries or technical questions:
- **Author:** Ivan Maestre Ivanov
- **Institution:** Universidad La Salle
- **Program:** BIDA (Big Data and Intelligence Analytics)

---

*This repository represents original academic work and follows proper software engineering practices for reproducible research in quantitative finance.* 