"""
Simple EMA Crossover Strategy (12/26) with ATR-based position sizing and risk management

Implements a pure EMA crossover strategy similar to the Backtrader example,
using 12/26 EMA periods with ATR-based position sizing and stop-loss management.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import logging

from .base_strategy import BaseStrategy, TradingSignal, SignalType, Position
from ..config import Config


class SimpleEMACrossoverStrategy(BaseStrategy):
    """
    Simple EMA Crossover Trading Strategy (12/26)
    
    Strategy Logic:
    - BUY: When EMA(12) crosses above EMA(26) 
    - SELL: When EMA(12) crosses below EMA(26) (exit long positions)
    - Long-only strategy (no short positions)
    
    Risk Management:
    - ATR-based position sizing
    - ATR-based stop-loss
    - Risk per trade percentage of portfolio
    """
    
    def __init__(self, parameters: Dict[str, Any] = None):
        """
        Initialize the Simple EMA Crossover strategy.
        
        Args:
            parameters: Strategy parameters (optional)
        """
        # Default parameters similar to Backtrader example
        default_params = {
            'fast_period': 12,      # Fast EMA period
            'slow_period': 26,      # Slow EMA period  
            'risk_per_trade': 0.02, # Risk per trade (2% of portfolio)
            'atr_period': 14,       # ATR period
            'atr_multiplier': 2.0,  # ATR multiplier for stop loss
        }
        
        # Merge with provided parameters
        if parameters:
            default_params.update(parameters)
            
        super().__init__("Simple EMA Crossover (12/26)", default_params)
        
        # Strategy-specific parameters
        self.fast_period = self.get_parameter('fast_period', 12)
        self.slow_period = self.get_parameter('slow_period', 26)
        self.risk_per_trade = self.get_parameter('risk_per_trade', 0.02)
        self.atr_period = self.get_parameter('atr_period', 14)
        self.atr_multiplier = self.get_parameter('atr_multiplier', 2.0)
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        
        # Strategy state
        self.data = None
        self.is_initialized = False
        
        # Performance tracking
        self.trade_count = 0
        self.win_count = 0
        self.trades_history = []
        
        self.logger.info(f"Simple EMA Crossover strategy initialized with periods {self.fast_period}/{self.slow_period}")
    
    def initialize(self, data: pd.DataFrame) -> None:
        """
        Initialize the strategy with historical data.
        
        Args:
            data: DataFrame with OHLCV data
        """
        self.logger.info("Initializing Simple EMA Crossover strategy")
        
        # Store data reference
        self.data = data.copy()
        
        # Calculate EMAs if not present
        if f'ema_{self.fast_period}' not in self.data.columns:
            self.data[f'ema_{self.fast_period}'] = self._calculate_ema(self.data['close'], self.fast_period)
        
        if f'ema_{self.slow_period}' not in self.data.columns:
            self.data[f'ema_{self.slow_period}'] = self._calculate_ema(self.data['close'], self.slow_period)
            
        # Calculate ATR if not present
        if 'atr' not in self.data.columns:
            self.data['atr'] = self._calculate_atr(self.data, self.atr_period)
        
        # Calculate crossover signals
        self.data['ema_crossover'] = self._detect_crossover(
            self.data[f'ema_{self.fast_period}'], 
            self.data[f'ema_{self.slow_period}']
        )
        
        # Validate sufficient data
        min_periods = max(self.fast_period, self.slow_period, self.atr_period) + 10
        if len(data) < min_periods:
            raise ValueError(f"Insufficient data: {len(data)} periods (minimum: {min_periods})")
        
        # Reset strategy state
        self.reset()
        self.is_initialized = True
        
        self.logger.info(f"Strategy initialized with {len(data)} data points")
    
    def _calculate_ema(self, data: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average."""
        return data.ewm(span=period, adjust=False).mean()
    
    def _calculate_atr(self, data: pd.DataFrame, period: int) -> pd.Series:
        """
        Calculate Average True Range (ATR).
        
        Args:
            data: OHLC DataFrame
            period: ATR period
            
        Returns:
            ATR series
        """
        high = data['high']
        low = data['low']
        close = data['close']
        prev_close = close.shift(1)
        
        # True Range components
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        
        # True Range is the maximum of the three
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # ATR is the rolling mean of True Range
        atr = true_range.rolling(window=period).mean()
        
        return atr
    
    def _detect_crossover(self, fast_ema: pd.Series, slow_ema: pd.Series) -> pd.Series:
        """
        Detect EMA crossover signals.
        
        Args:
            fast_ema: Fast EMA series
            slow_ema: Slow EMA series
            
        Returns:
            Series with crossover signals (1 for bullish, -1 for bearish, 0 for no signal)
        """
        crossover = pd.Series(0, index=fast_ema.index, dtype=int)
        
        # Current and previous period comparison
        fast_above_slow = fast_ema > slow_ema
        fast_above_slow_prev = fast_above_slow.shift(1)
        
        # Handle NaN values
        valid_mask = fast_above_slow.notna() & fast_above_slow_prev.notna()
        
        # Bullish crossover: fast EMA crosses above slow EMA
        bullish_crossover = valid_mask & (~fast_above_slow_prev) & fast_above_slow
        crossover[bullish_crossover] = 1
        
        # Bearish crossover: fast EMA crosses below slow EMA  
        bearish_crossover = valid_mask & fast_above_slow_prev & (~fast_above_slow)
        crossover[bearish_crossover] = -1
        
        return crossover
    
    def generate_signal(self, data: pd.DataFrame, current_time: pd.Timestamp) -> TradingSignal:
        """
        Generate trading signal based on EMA crossover.
        
        Args:
            data: Current data window
            current_time: Current timestamp
            
        Returns:
            TradingSignal with buy/sell/hold recommendation
        """
        if not self.is_initialized:
            raise RuntimeError("Strategy not initialized. Call initialize() first.")
        
        try:
            # Get current data point
            if current_time not in data.index:
                self.logger.warning(f"Current time {current_time} not found in data")
                return TradingSignal(current_time, SignalType.HOLD, confidence=0.0)
            
            current_data = data.loc[current_time]
            current_price = current_data['close']
            
            # Get crossover signal
            crossover_signal = current_data.get('ema_crossover', 0)
            
            # Simple logic: only take positions on crossover
            if self.current_position is None:  # No position
                if crossover_signal == 1:  # Bullish crossover - BUY
                    signal_type = SignalType.BUY
                    confidence = 1.0  # Full confidence on crossover
                else:
                    signal_type = SignalType.HOLD
                    confidence = 0.0
            else:  # Have position
                if crossover_signal == -1:  # Bearish crossover - SELL (exit)
                    signal_type = SignalType.SELL
                    confidence = 1.0  # Full confidence on crossover
                else:
                    signal_type = SignalType.HOLD
                    confidence = 0.0
            
            # Create trading signal
            signal = TradingSignal(
                timestamp=current_time,
                signal_type=signal_type,
                confidence=confidence,
                price=current_price,
                metadata={
                    'fast_ema': current_data.get(f'ema_{self.fast_period}', 0),
                    'slow_ema': current_data.get(f'ema_{self.slow_period}', 0),
                    'atr': current_data.get('atr', 0),
                    'crossover': crossover_signal,
                    'strategy': 'simple_ema_crossover'
                }
            )
            
            # Log signal generation
            if signal_type != SignalType.HOLD:
                self.logger.info(f"Generated {signal_type.name} signal at {current_time}")
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Error generating signal at {current_time}: {e}")
            return TradingSignal(current_time, SignalType.HOLD, confidence=0.0)
    
    def calculate_position_size(self, data: pd.DataFrame, signal: TradingSignal, portfolio_value: float) -> float:
        """
        Calculate position size based on ATR and risk per trade.
        
        Args:
            data: Market data
            signal: Trading signal
            portfolio_value: Current portfolio value
            
        Returns:
            Position size in base currency
        """
        try:
            current_price = signal.price
            atr = signal.metadata.get('atr', 0)
            
            if atr <= 0:
                self.logger.warning("ATR is zero or negative, using fallback position sizing")
                # Fallback to percentage-based sizing
                return (portfolio_value * 0.1) / current_price
            
            # Risk amount (2% of portfolio by default)
            risk_amount = portfolio_value * self.risk_per_trade
            
            # Stop loss distance based on ATR
            stop_distance = atr * self.atr_multiplier
            
            # Position size = risk_amount / stop_distance
            position_size_usd = risk_amount / stop_distance
            position_quantity = position_size_usd / current_price
            
            self.logger.info(f"ATR-based position sizing: ${position_size_usd:.2f} = {position_quantity:.6f} units")
            self.logger.info(f"Risk: ${risk_amount:.2f}, Stop distance: ${stop_distance:.2f}, ATR: {atr:.4f}")
            
            return position_quantity
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            # Fallback to simple percentage-based sizing
            return (portfolio_value * 0.05) / signal.price
    
    def calculate_stop_loss(self, entry_price: float, position_type: SignalType) -> Optional[float]:
        """
        Calculate ATR-based stop loss price.
        
        Args:
            entry_price: Entry price for the position
            position_type: Type of position (BUY for long)
            
        Returns:
            Stop-loss price or None if no stop-loss configured
        """
        if self.data is None:
            return None
            
        try:
            # Get the latest ATR value
            latest_atr = self.data['atr'].iloc[-1]
            
            if pd.isna(latest_atr) or latest_atr <= 0:
                return None
            
            if position_type == SignalType.BUY:
                # Long position: stop-loss below entry price
                stop_loss = entry_price - (latest_atr * self.atr_multiplier)
            else:
                # This strategy is long-only, but keeping the logic for completeness
                stop_loss = entry_price + (latest_atr * self.atr_multiplier)
            
            return stop_loss
            
        except Exception as e:
            self.logger.error(f"Error calculating ATR-based stop loss: {e}")
            return None
    
    def should_exit_position(self, current_price: float, current_time: pd.Timestamp) -> bool:
        """
        Check if current position should be exited.
        
        Args:
            current_price: Current market price
            current_time: Current timestamp
            
        Returns:
            True if position should be exited
        """
        if not self.current_position:
            return False
        
        try:
            # Check ATR-based stop-loss
            if self.current_position.stop_loss:
                if current_price <= self.current_position.stop_loss:
                    self.logger.info(f"ATR Stop-loss triggered: {current_price} <= {self.current_position.stop_loss}")
                    return True
            
            # Check for bearish crossover (main exit signal)
            if self.data is not None and current_time in self.data.index:
                current_signal = self.generate_signal(self.data, current_time)
                if current_signal.signal_type == SignalType.SELL:
                    self.logger.info("Exit signal: Bearish EMA crossover")
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking exit conditions: {e}")
            return False
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """
        Get comprehensive strategy information and statistics.
        
        Returns:
            Dictionary with strategy details and performance metrics
        """
        base_info = super().get_strategy_info()
        
        # Add strategy-specific information
        simple_ema_info = {
            'strategy_type': 'Simple EMA Crossover (12/26)',
            'fast_ema_period': self.fast_period,
            'slow_ema_period': self.slow_period,
            'risk_per_trade_pct': self.risk_per_trade * 100,
            'atr_period': self.atr_period,
            'atr_multiplier': self.atr_multiplier,
            'performance': {
                'total_trades': self.trade_count,
                'winning_trades': self.win_count,
                'win_rate': (self.win_count / max(self.trade_count, 1)) * 100,
                'strategy_approach': 'Long-only, ATR-based risk management'
            },
            'current_state': {
                'is_initialized': self.is_initialized,
                'has_position': self.current_position is not None,
                'data_points': len(self.data) if self.data is not None else 0
            }
        }
        
        # Merge with base info
        base_info.update(simple_ema_info)
        
        return base_info
    
    def validate_parameters(self) -> bool:
        """
        Validate strategy parameters.
        
        Returns:
            True if parameters are valid
        """
        try:
            # Check EMA periods
            if self.fast_period <= 0 or self.slow_period <= 0:
                self.logger.error("EMA periods must be positive")
                return False
            
            if self.fast_period >= self.slow_period:
                self.logger.error("Fast EMA period must be less than slow EMA period")
                return False
            
            # Check risk per trade
            if not (0 < self.risk_per_trade <= 0.1):  # Max 10% risk per trade
                self.logger.error("Risk per trade must be between 0 and 0.1 (10%)")
                return False
            
            # Check ATR parameters
            if self.atr_period <= 0:
                self.logger.error("ATR period must be positive")
                return False
                
            if self.atr_multiplier <= 0:
                self.logger.error("ATR multiplier must be positive")
                return False
            
            self.logger.info("Strategy parameters validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Parameter validation error: {e}")
            return False
    
    def reset(self) -> None:
        """Reset strategy state for new backtest."""
        super().reset()
        self.trade_count = 0
        self.win_count = 0
        self.trades_history = []
        self.logger.info("Strategy state reset")
    
    def __str__(self) -> str:
        """String representation of the strategy."""
        return f"SimpleEMACrossoverStrategy(fast={self.fast_period}, slow={self.slow_period}, risk={self.risk_per_trade:.1%})" 