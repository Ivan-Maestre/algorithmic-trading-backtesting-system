"""
Data Processor Module

Handles OHLCV data processing, technical indicator calculations, 
and data preparation for backtesting strategies.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

from ..config import Config


class DataProcessor:
    """
    Processes OHLCV data and calculates technical indicators.
    
    This class provides functionality for data cleaning, technical analysis,
    and data preparation for trading strategies.
    """
    
    def __init__(self):
        """Initialize the data processor."""
        self.logger = logging.getLogger(__name__)
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean OHLCV data by handling missing values and outliers.
        
        Args:
            df: Raw OHLCV DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        self.logger.info("Starting data cleaning process")
        
        # Create a copy to avoid modifying original data
        cleaned_df = df.copy()
        
        # Remove rows with all NaN values
        cleaned_df = cleaned_df.dropna(how='all')
        
        # Forward fill missing values for price data
        price_columns = ['open', 'high', 'low', 'close']
        cleaned_df[price_columns] = cleaned_df[price_columns].ffill()
        
        # Fill volume with 0 if missing
        if 'volume' in cleaned_df.columns:
            cleaned_df['volume'] = cleaned_df['volume'].fillna(0)
        
        # Ensure OHLC consistency (High >= max(Open, Close), Low <= min(Open, Close))
        cleaned_df['high'] = np.maximum(cleaned_df['high'], 
                                       np.maximum(cleaned_df['open'], cleaned_df['close']))
        cleaned_df['low'] = np.minimum(cleaned_df['low'], 
                                      np.minimum(cleaned_df['open'], cleaned_df['close']))
        
        # Remove extreme outliers (more than 10x price change in one period)
        for col in price_columns:
            pct_change = cleaned_df[col].pct_change().abs()
            outlier_mask = pct_change > 10.0  # 1000% change
            cleaned_df.loc[outlier_mask, col] = np.nan
        
        # Forward fill any NaN values created by outlier removal
        cleaned_df[price_columns] = cleaned_df[price_columns].ffill()
        
        # Remove any remaining rows with NaN values
        cleaned_df = cleaned_df.dropna()
        
        self.logger.info(f"Data cleaning completed. Shape: {cleaned_df.shape}")
        return cleaned_df
    
    def calculate_ema(self, data: pd.Series, period: int) -> pd.Series:
        """
        Calculate Exponential Moving Average (EMA).
        
        Args:
            data: Price series (typically close prices)
            period: EMA period
            
        Returns:
            EMA series
        """
        return data.ewm(span=period, adjust=False).mean()
    
    def calculate_sma(self, data: pd.Series, period: int) -> pd.Series:
        """
        Calculate Simple Moving Average (SMA).
        
        Args:
            data: Price series
            period: SMA period
            
        Returns:
            SMA series
        """
        return data.rolling(window=period).mean()
    
    def calculate_rsi(self, data: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI).
        
        Args:
            data: Price series (typically close prices)
            period: RSI period (default 14)
            
        Returns:
            RSI series
        """
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_macd(self, data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence).
        
        Args:
            data: Price series (typically close prices)
            fast: Fast EMA period (default 12)
            slow: Slow EMA period (default 26)
            signal: Signal line EMA period (default 9)
            
        Returns:
            Dictionary with MACD line, signal line, and histogram
        """
        ema_fast = self.calculate_ema(data, fast)
        ema_slow = self.calculate_ema(data, slow)
        
        macd_line = ema_fast - ema_slow
        signal_line = self.calculate_ema(macd_line, signal)
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }
    
    def calculate_bollinger_bands(self, data: pd.Series, period: int = 20, std_dev: float = 2) -> Dict[str, pd.Series]:
        """
        Calculate Bollinger Bands.
        
        Args:
            data: Price series
            period: Period for moving average (default 20)
            std_dev: Standard deviation multiplier (default 2)
            
        Returns:
            Dictionary with upper band, middle band (SMA), and lower band
        """
        sma = self.calculate_sma(data, period)
        std = data.rolling(window=period).std()
        
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        return {
            'upper': upper_band,
            'middle': sma,
            'lower': lower_band
        }
    
    def calculate_volatility(self, data: pd.Series, period: int = 20) -> pd.Series:
        """
        Calculate rolling volatility.
        
        Args:
            data: Price series
            period: Rolling window period
            
        Returns:
            Volatility series
        """
        returns = data.pct_change()
        volatility = returns.rolling(window=period).std() * np.sqrt(period)
        return volatility
    
    def detect_ema_crossover(self, fast_ema: pd.Series, slow_ema: pd.Series) -> pd.Series:
        """
        Detect EMA crossover signals.
        
        Args:
            fast_ema: Fast EMA series
            slow_ema: Slow EMA series
            
        Returns:
            Series with crossover signals (1 for bullish, -1 for bearish, 0 for no signal)
        """
        # Calculate crossover points
        crossover = pd.Series(0, index=fast_ema.index, dtype=int)
        
        # Current and previous period comparison
        fast_above_slow = fast_ema > slow_ema
        fast_above_slow_prev = fast_above_slow.shift(1)
        
        # Handle NaN values from shift operation and ensure boolean type
        valid_mask = fast_above_slow.notna() & fast_above_slow_prev.notna()
        
        # Convert to boolean explicitly to avoid float issues
        fast_above_slow = fast_above_slow.astype(bool)
        fast_above_slow_prev = fast_above_slow_prev.fillna(False).astype(bool)
        
        # Bullish crossover: fast EMA crosses above slow EMA
        bullish_crossover = valid_mask & (~fast_above_slow_prev) & fast_above_slow
        crossover[bullish_crossover] = 1
        
        # Bearish crossover: fast EMA crosses below slow EMA  
        bearish_crossover = valid_mask & fast_above_slow_prev & (~fast_above_slow)
        crossover[bearish_crossover] = -1
        
        return crossover
    
    def add_technical_indicators(self, df: pd.DataFrame, ema_periods: List[int] = None) -> pd.DataFrame:
        """
        Add technical indicators to OHLCV DataFrame.
        
        Args:
            df: OHLCV DataFrame
            ema_periods: List of EMA periods to calculate (default from config)
            
        Returns:
            DataFrame with technical indicators added
        """
        if ema_periods is None:
            ema_config = Config.EMA_STRATEGY
            ema_periods = [ema_config['fast_period'], ema_config['slow_period']]
        
        self.logger.info(f"Adding technical indicators with EMA periods: {ema_periods}")
        
        # Create a copy to avoid modifying original data
        result_df = df.copy()
        
        # Calculate EMAs
        for period in ema_periods:
            result_df[f'ema_{period}'] = self.calculate_ema(result_df['close'], period)
        
        # Calculate RSI
        result_df['rsi'] = self.calculate_rsi(result_df['close'])
        
        # Calculate MACD
        macd_data = self.calculate_macd(result_df['close'])
        result_df['macd'] = macd_data['macd']
        result_df['macd_signal'] = macd_data['signal']
        result_df['macd_histogram'] = macd_data['histogram']
        
        # Calculate Bollinger Bands
        bb_data = self.calculate_bollinger_bands(result_df['close'])
        result_df['bb_upper'] = bb_data['upper']
        result_df['bb_middle'] = bb_data['middle']
        result_df['bb_lower'] = bb_data['lower']
        
        # Calculate volatility
        result_df['volatility'] = self.calculate_volatility(result_df['close'])
        
        # Calculate EMA crossover signals (if we have fast and slow EMAs)
        if len(ema_periods) >= 2:
            fast_period = min(ema_periods)
            slow_period = max(ema_periods)
            result_df['ema_crossover'] = self.detect_ema_crossover(
                result_df[f'ema_{fast_period}'], 
                result_df[f'ema_{slow_period}']
            )
        
        self.logger.info(f"Technical indicators added. New shape: {result_df.shape}")
        return result_df
    
    def resample_data(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        Resample OHLCV data to different timeframe.
        
        Args:
            df: OHLCV DataFrame with datetime index
            timeframe: Target timeframe (e.g., '4H', '1D')
            
        Returns:
            Resampled DataFrame
        """
        self.logger.info(f"Resampling data to {timeframe}")
        
        # Define aggregation rules
        agg_rules = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }
        
        # Add number_of_trades if present
        if 'number_of_trades' in df.columns:
            agg_rules['number_of_trades'] = 'sum'
        
        # Resample data
        resampled = df.resample(timeframe).agg(agg_rules)
        
        # Remove periods with no data
        resampled = resampled.dropna()
        
        self.logger.info(f"Resampling completed. New shape: {resampled.shape}")
        return resampled
    
    def prepare_strategy_data(self, df: pd.DataFrame, strategy_name: str) -> pd.DataFrame:
        """
        Prepare data specifically for a trading strategy.
        
        Args:
            df: Raw OHLCV DataFrame
            strategy_name: Name of the strategy
            
        Returns:
            Processed DataFrame ready for strategy
        """
        self.logger.info(f"Preparing data for {strategy_name} strategy")
        
        # Clean the data
        processed_df = self.clean_data(df)
        
        # Add technical indicators based on strategy
        if strategy_name.lower() == 'ema_crossover':
            ema_config = Config.get_strategy_config('ema_crossover')
            ema_periods = [ema_config['fast_period'], ema_config['slow_period']]
            processed_df = self.add_technical_indicators(processed_df, ema_periods)
        
        elif strategy_name.lower() == 'momentum':
            # Add momentum-specific indicators
            processed_df = self.add_technical_indicators(processed_df)
            # Add momentum calculation
            momentum_config = Config.get_strategy_config('momentum')
            lookback = momentum_config.get('lookback_period', 20)
            processed_df['momentum'] = processed_df['close'].pct_change(periods=lookback)
        
        elif strategy_name.lower() == 'q_learning':
            # Add all indicators for RL feature engineering
            processed_df = self.add_technical_indicators(processed_df)
            # Add additional features for RL
            processed_df['returns'] = processed_df['close'].pct_change()
            processed_df['log_returns'] = np.log(processed_df['close'] / processed_df['close'].shift(1))
        
        else:
            # Default: add all indicators
            processed_df = self.add_technical_indicators(processed_df)
        
        # Remove any NaN values created by indicators
        processed_df = processed_df.dropna()
        
        self.logger.info(f"Data preparation completed for {strategy_name}. Final shape: {processed_df.shape}")
        return processed_df
    
    def save_processed_data(self, df: pd.DataFrame, symbol: str, strategy_name: str) -> str:
        """
        Save processed data to CSV file.
        
        Args:
            df: Processed DataFrame
            symbol: Trading symbol
            strategy_name: Strategy name
            
        Returns:
            Path to saved file
        """
        filename = f"{symbol}_{strategy_name}_processed_{Config.DATA_START_DATE}_{Config.DATA_END_DATE}.csv"
        filepath = Config.PROCESSED_DATA_DIR + "/" + filename
        
        df.to_csv(filepath)
        self.logger.info(f"Processed data saved to {filepath}")
        
        return filepath 