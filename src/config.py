"""
Configuration module for the Algorithmic Trading Backtesting System.
Centralizes all configuration parameters and settings.
"""

import os
from datetime import datetime
from typing import Dict, Any
import yaml


class Config:
    """Main configuration class for the backtesting system."""
    
    # Project Information
    PROJECT_NAME = "Algorithmic Trading Backtesting System"
    VERSION = "1.0.0"
    
    # Data Configuration
    DATA_START_DATE = "2018-01-01"
    DATA_END_DATE = "2023-12-31"
    TIMEFRAME = "4h"  # 4-hour intervals
    
    # Default Asset Configuration
    DEFAULT_ASSET = "BTCUSDT"  # Binance symbol format
    
    # Directory Paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    RESULTS_DIR = os.path.join(BASE_DIR, "results")
    LOGS_DIR = os.path.join(BASE_DIR, "logs")
    
    # Data Storage
    RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
    PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
    
    # Backtesting Configuration
    INITIAL_CAPITAL = 100000.0  # Starting capital in USD
    TRANSACTION_COST = 0.001  # 0.1% transaction cost
    SLIPPAGE = 0.0005  # 0.05% slippage
    
    # Risk Management Defaults
    DEFAULT_POSITION_SIZE = 0.1  # 10% of portfolio per trade
    DEFAULT_STOP_LOSS = 0.02  # 2% stop loss
    MAX_POSITION_SIZE = 0.2  # Maximum 20% of portfolio per trade
    
    # Binance API Configuration
    BINANCE_CONFIG = {
        "api_key": os.getenv("BINANCE_API_KEY", ""),
        "secret_key": os.getenv("BINANCE_SECRET_KEY", ""),
        "base_url": "https://api.binance.com",
        "testnet": False,
        "timeout": 30,
        "recv_window": 5000,
    }
    
    # EMA Crossover Strategy Configuration
    EMA_STRATEGY = {
        "fast_period": 12,
        "slow_period": 25,
        "timeframe": TIMEFRAME,
        "position_size": DEFAULT_POSITION_SIZE,
        "stop_loss": DEFAULT_STOP_LOSS,
    }
    
    # Momentum Strategy Configuration (to be filled based on TFG)
    MOMENTUM_STRATEGY = {
        "lookback_period": 20,  # Period for momentum calculation
        "threshold": 0.02,  # 2% momentum threshold for signal generation
        "timeframe": TIMEFRAME,
        "position_size": DEFAULT_POSITION_SIZE,
        "stop_loss": DEFAULT_STOP_LOSS,
        "min_volume_ratio": 1.3,  # Minimum volume confirmation ratio
        "volatility_adjustment": True,  # Enable volatility-based position sizing
        "momentum_acceleration": True,  # Consider momentum acceleration
    }
    
    # Q-Learning RL Strategy Configuration (to be filled based on TFG)
    RL_STRATEGY = {
        "learning_rate": 0.01,  # Q-learning rate
        "discount_factor": 0.95,  # Future rewards discount
        "exploration_rate": 0.1,  # Epsilon for epsilon-greedy
        "exploration_decay": 0.995,  # Exploration rate decay
        "min_exploration_rate": 0.01,  # Minimum exploration rate
        "state_features": ["price_trend", "volume_trend", "volatility_level", "momentum_level"],
        "n_price_levels": 5,  # Discrete price trend levels
        "n_volume_levels": 3,  # Discrete volume levels
        "n_volatility_levels": 3,  # Discrete volatility levels
        "n_momentum_levels": 5,  # Discrete momentum levels
        "position_size": DEFAULT_POSITION_SIZE,
        "stop_loss": DEFAULT_STOP_LOSS,
        "timeframe": TIMEFRAME,
    }
    
    # Performance Metrics Configuration
    PERFORMANCE_METRICS = [
        "total_return",
        "annualized_return",
        "sharpe_ratio",
        "sortino_ratio",
        "max_drawdown",
        "calmar_ratio",
        "win_rate",
        "profit_factor",
        "average_win",
        "average_loss",
        "total_trades",
    ]
    
    # Data Source Configuration
    DATA_SOURCES = {
        "binance": {
            "enabled": True,
            "priority": 1,
            "config": BINANCE_CONFIG,
        },
        "yahoo": {
            "enabled": False,  # Disabled in favor of Binance
            "priority": 2,
        },
        "alpha_vantage": {
            "enabled": False,
            "priority": 3,
            "api_key": os.getenv("ALPHA_VANTAGE_API_KEY", ""),
        }
    }
    
    # Binance Symbol Mapping
    SYMBOL_MAPPING = {
        "BTC-USD": "BTCUSDT",
        "ETH-USD": "ETHUSDT", 
        "ADA-USD": "ADAUSDT",
        "DOT-USD": "DOTUSDT",
    }
    
    # Binance Timeframe Mapping
    TIMEFRAME_MAPPING = {
        "1m": "1m",
        "5m": "5m", 
        "15m": "15m",
        "30m": "30m",
        "1h": "1h",
        "4h": "4h",
        "1d": "1d",
        "1w": "1w"
    }
    
    # Logging Configuration
    LOGGING = {
        "level": "INFO",
        "format": "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name} | {message}",
        "file_rotation": "10 MB",
        "file_retention": "30 days",
    }
    
    # Testing Configuration
    TESTING = {
        "quick_test_period": "2022-01-01",  # For quick testing
        "sample_size": 1000,  # Sample size for testing
        "tolerance": 1e-6,  # Numerical tolerance for tests
    }
    
    @classmethod
    def get_strategy_config(cls, strategy_name: str) -> Dict[str, Any]:
        """Get configuration for a specific strategy."""
        strategy_configs = {
            "ema_crossover": cls.EMA_STRATEGY,
            "momentum": cls.MOMENTUM_STRATEGY,
            "q_learning": cls.RL_STRATEGY,
        }
        return strategy_configs.get(strategy_name, {})
    
    @classmethod
    def get_binance_symbol(cls, symbol: str) -> str:
        """Convert symbol to Binance format."""
        return cls.SYMBOL_MAPPING.get(symbol, symbol)
    
    @classmethod
    def get_binance_timeframe(cls, timeframe: str) -> str:
        """Convert timeframe to Binance format."""
        return cls.TIMEFRAME_MAPPING.get(timeframe, timeframe)
    
    @classmethod
    def create_directories(cls) -> None:
        """Create necessary directories if they don't exist."""
        directories = [
            cls.DATA_DIR,
            cls.RAW_DATA_DIR,
            cls.PROCESSED_DATA_DIR,
            cls.RESULTS_DIR,
            cls.LOGS_DIR,
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    @classmethod
    def load_from_file(cls, config_file: str) -> None:
        """Load configuration from YAML file."""
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config_data = yaml.safe_load(f)
                
            # Update class attributes with loaded config
            for key, value in config_data.items():
                if hasattr(cls, key.upper()):
                    setattr(cls, key.upper(), value)
    
    @classmethod
    def save_to_file(cls, config_file: str) -> None:
        """Save current configuration to YAML file."""
        config_data = {
            "data_start_date": cls.DATA_START_DATE,
            "data_end_date": cls.DATA_END_DATE,
            "timeframe": cls.TIMEFRAME,
            "initial_capital": cls.INITIAL_CAPITAL,
            "transaction_cost": cls.TRANSACTION_COST,
            "default_asset": cls.DEFAULT_ASSET,
            "ema_strategy": cls.EMA_STRATEGY,
            "momentum_strategy": cls.MOMENTUM_STRATEGY,
            "rl_strategy": cls.RL_STRATEGY,
        }
        
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False)
    
    @classmethod
    def validate_config(cls) -> bool:
        """Validate configuration parameters."""
        try:
            # Validate dates
            datetime.strptime(cls.DATA_START_DATE, "%Y-%m-%d")
            datetime.strptime(cls.DATA_END_DATE, "%Y-%m-%d")
            
            # Validate numerical parameters
            assert cls.INITIAL_CAPITAL > 0, "Initial capital must be positive"
            assert 0 <= cls.TRANSACTION_COST <= 1, "Transaction cost must be between 0 and 1"
            assert 0 <= cls.DEFAULT_POSITION_SIZE <= 1, "Position size must be between 0 and 1"
            
            # Validate Binance configuration
            assert cls.BINANCE_CONFIG["api_key"], "Binance API key is required"
            assert cls.BINANCE_CONFIG["secret_key"], "Binance secret key is required"
            
            return True
        except (ValueError, AssertionError) as e:
            print(f"Configuration validation error: {e}")
            return False


# Initialize directories on import
Config.create_directories()

# Load custom configuration if exists
custom_config_path = os.path.join(Config.BASE_DIR, "config.yaml")
if os.path.exists(custom_config_path):
    Config.load_from_file(custom_config_path) 