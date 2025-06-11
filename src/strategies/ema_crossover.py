"""
EMA Crossover Strategy

Implements the Exponential Moving Average crossover trading strategy using 12-period and 25-period EMAs.
This strategy generates buy signals when the fast EMA (12) crosses above the slow EMA (25),
and sell signals when the fast EMA crosses below the slow EMA.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import logging

from .base_strategy import BaseStrategy, TradingSignal, SignalType, Position
from ..config import Config


class EMACrossoverStrategy(BaseStrategy):
    """
    EMA Crossover Trading Strategy
    
    Strategy Logic:
    - BUY: When EMA(12) crosses above EMA(25) 
    - SELL: When EMA(12) crosses below EMA(25)
    - HOLD: When no crossover occurs
    
    Risk Management:
    - Position sizing based on portfolio percentage
    - Stop-loss orders at configurable percentage
    - Maximum position size limits
    """
    
    def __init__(self, parameters: Dict[str, Any] = None):
        """
        Initialize the EMA Crossover strategy.
        
        Args:
            parameters: Strategy parameters (optional, uses config defaults)
        """
        # Set default parameters from config
        default_params = Config.get_strategy_config('ema_crossover')
        super().__init__("EMA Crossover", parameters or default_params)
        
        # Strategy-specific parameters
        self.fast_period = self.get_parameter('fast_period', 12)
        self.slow_period = self.get_parameter('slow_period', 25)
        self.position_size = self.get_parameter('position_size', 0.1)
        self.stop_loss_pct = self.get_parameter('stop_loss', 0.02)
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        
        # Strategy state
        self.data = None
        self.is_initialized = False
        
        # Performance tracking
        self.total_signals = 0
        self.profitable_signals = 0
        self.total_return = 0.0
        
        self.logger.info(f"EMA Crossover strategy initialized with periods {self.fast_period}/{self.slow_period}")
    
    def initialize(self, data: pd.DataFrame) -> None:
        """
        Initialize the strategy with historical data.
        
        Args:
            data: DataFrame with OHLCV data and technical indicators
        """
        self.logger.info("Initializing EMA Crossover strategy")
        
        # Validate required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume', 
                          f'ema_{self.fast_period}', f'ema_{self.slow_period}']
        
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns for EMA strategy: {missing_columns}")
        
        # Store data reference
        self.data = data.copy()
        
        # Validate sufficient data
        min_periods = max(self.fast_period, self.slow_period) * 2
        if len(data) < min_periods:
            raise ValueError(f"Insufficient data: {len(data)} periods (minimum: {min_periods})")
        
        # Reset strategy state
        self.reset()
        self.is_initialized = True
        
        self.logger.info(f"Strategy initialized with {len(data)} data points")
        
        # Log data quality info
        fast_ema_valid = data[f'ema_{self.fast_period}'].notna().sum()
        slow_ema_valid = data[f'ema_{self.slow_period}'].notna().sum()
        self.logger.info(f"EMA validity: Fast({self.fast_period}): {fast_ema_valid}, Slow({self.slow_period}): {slow_ema_valid}")
    
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
            
            # Get EMA values
            fast_ema = current_data[f'ema_{self.fast_period}']
            slow_ema = current_data[f'ema_{self.slow_period}']
            current_price = current_data['close']
            
            # Check for NaN values
            if pd.isna(fast_ema) or pd.isna(slow_ema):
                self.logger.warning(f"NaN EMA values at {current_time}")
                return TradingSignal(current_time, SignalType.HOLD, confidence=0.0)
            
            # Check for crossover signal (if we have crossover column)
            if 'ema_crossover' in data.columns:
                crossover_signal = current_data['ema_crossover']
                
                if crossover_signal == 1:  # Bullish crossover
                    signal_type = SignalType.BUY
                    confidence = self._calculate_signal_confidence(data, current_time, signal_type)
                    
                elif crossover_signal == -1:  # Bearish crossover  
                    signal_type = SignalType.SELL
                    confidence = self._calculate_signal_confidence(data, current_time, signal_type)
                    
                else:  # No crossover
                    signal_type = SignalType.HOLD
                    confidence = 0.0
            
            else:
                # Manual crossover detection if column not available
                signal_type, confidence = self._detect_crossover_manual(data, current_time)
            
            # Create trading signal
            signal = TradingSignal(
                timestamp=current_time,
                signal_type=signal_type,
                confidence=confidence,
                price=current_price,
                metadata={
                    'fast_ema': fast_ema,
                    'slow_ema': slow_ema,
                    'ema_diff': fast_ema - slow_ema,
                    'ema_diff_pct': ((fast_ema - slow_ema) / slow_ema) * 100,
                    'strategy': 'ema_crossover'
                }
            )
            
            # Track signal generation
            if signal_type != SignalType.HOLD:
                self.total_signals += 1
                self.logger.info(f"Generated {signal_type.name} signal at {current_time} (confidence: {confidence:.2f})")
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Error generating signal at {current_time}: {e}")
            return TradingSignal(current_time, SignalType.HOLD, confidence=0.0)
    
    def _detect_crossover_manual(self, data: pd.DataFrame, current_time: pd.Timestamp) -> tuple:
        """
        Manually detect EMA crossover if crossover column is not available.
        
        Args:
            data: Market data
            current_time: Current timestamp
            
        Returns:
            Tuple of (signal_type, confidence)
        """
        try:
            # Get current and previous timestamps
            time_index = data.index.get_loc(current_time)
            if time_index == 0:
                return SignalType.HOLD, 0.0
            
            previous_time = data.index[time_index - 1]
            
            # Current EMA values
            current_fast = data.loc[current_time, f'ema_{self.fast_period}']
            current_slow = data.loc[current_time, f'ema_{self.slow_period}']
            
            # Previous EMA values
            previous_fast = data.loc[previous_time, f'ema_{self.fast_period}']
            previous_slow = data.loc[previous_time, f'ema_{self.slow_period}']
            
            # Check for crossover
            if previous_fast <= previous_slow and current_fast > current_slow:
                # Bullish crossover
                signal_type = SignalType.BUY
                confidence = self._calculate_signal_confidence(data, current_time, signal_type)
                
            elif previous_fast >= previous_slow and current_fast < current_slow:
                # Bearish crossover
                signal_type = SignalType.SELL  
                confidence = self._calculate_signal_confidence(data, current_time, signal_type)
                
            else:
                # No crossover
                signal_type = SignalType.HOLD
                confidence = 0.0
            
            return signal_type, confidence
            
        except Exception as e:
            self.logger.error(f"Error in manual crossover detection: {e}")
            return SignalType.HOLD, 0.0
    
    def _calculate_signal_confidence(self, data: pd.DataFrame, current_time: pd.Timestamp, signal_type: SignalType) -> float:
        """
        Calculate confidence level for the trading signal.
        
        Args:
            data: Market data
            current_time: Current timestamp
            signal_type: Type of signal (BUY/SELL)
            
        Returns:
            Confidence level between 0.0 and 1.0
        """
        try:
            current_data = data.loc[current_time]
            
            # Base confidence from EMA separation
            fast_ema = current_data[f'ema_{self.fast_period}']
            slow_ema = current_data[f'ema_{self.slow_period}']
            price = current_data['close']
            
            # EMA separation as confidence factor
            ema_diff_pct = abs((fast_ema - slow_ema) / slow_ema) * 100
            separation_confidence = min(ema_diff_pct / 2.0, 1.0)  # Cap at 100%
            
            # Volume confirmation (if available)
            volume_confidence = 0.5  # Default
            if 'volume' in current_data.index:
                # Use recent volume average for confirmation
                time_index = data.index.get_loc(current_time)
                recent_data = data.iloc[max(0, time_index-10):time_index+1]
                avg_volume = recent_data['volume'].mean()
                current_volume = current_data['volume']
                
                if current_volume > avg_volume:
                    volume_confidence = min((current_volume / avg_volume - 1.0) * 0.5 + 0.5, 1.0)
            
            # Price momentum confirmation
            momentum_confidence = 0.5  # Default
            if len(data) > 5:
                time_index = data.index.get_loc(current_time)
                if time_index >= 5:
                    price_5_ago = data.iloc[time_index-5]['close']
                    price_momentum = (price - price_5_ago) / price_5_ago
                    
                    if signal_type == SignalType.BUY and price_momentum > 0:
                        momentum_confidence = min(abs(price_momentum) * 10, 1.0)
                    elif signal_type == SignalType.SELL and price_momentum < 0:
                        momentum_confidence = min(abs(price_momentum) * 10, 1.0)
            
            # Combine confidence factors
            confidence = (separation_confidence * 0.5 + 
                         volume_confidence * 0.3 + 
                         momentum_confidence * 0.2)
            
            return min(max(confidence, 0.1), 0.95)  # Keep between 0.1 and 0.95
            
        except Exception as e:
            self.logger.error(f"Error calculating signal confidence: {e}")
            return 0.5  # Default confidence
    
    def calculate_position_size(self, data: pd.DataFrame, signal: TradingSignal, portfolio_value: float) -> float:
        """
        Calculate position size based on strategy parameters and risk management.
        
        Args:
            data: Market data
            signal: Trading signal
            portfolio_value: Current portfolio value
            
        Returns:
            Position size in base currency
        """
        try:
            # Base position size from configuration
            base_position_value = portfolio_value * self.position_size
            
            # Adjust based on signal confidence
            confidence_multiplier = signal.confidence
            adjusted_position_value = base_position_value * confidence_multiplier
            
            # Apply maximum position size limit
            max_position_value = portfolio_value * Config.MAX_POSITION_SIZE
            final_position_value = min(adjusted_position_value, max_position_value)
            
            # Convert to quantity based on current price
            current_price = signal.price or data.loc[signal.timestamp, 'close']
            position_quantity = final_position_value / current_price
            
            self.logger.info(f"Position size calculation: {final_position_value:.2f} USD = {position_quantity:.6f} units")
            
            return position_quantity
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return 0.0
    
    def should_exit_position(self, current_price: float, current_time: pd.Timestamp) -> bool:
        """
        Check if current position should be exited based on strategy rules.
        
        Args:
            current_price: Current market price
            current_time: Current timestamp
            
        Returns:
            True if position should be exited
        """
        if not self.current_position:
            return False
        
        try:
            # Check stop-loss
            if self.current_position.stop_loss:
                if self.current_position.position_type == SignalType.BUY:
                    if current_price <= self.current_position.stop_loss:
                        self.logger.info(f"Stop-loss triggered: {current_price} <= {self.current_position.stop_loss}")
                        return True
                else:  # SHORT position
                    if current_price >= self.current_position.stop_loss:
                        self.logger.info(f"Stop-loss triggered: {current_price} >= {self.current_position.stop_loss}")
                        return True
            
            # Check take-profit
            if self.current_position.take_profit:
                if self.current_position.position_type == SignalType.BUY:
                    if current_price >= self.current_position.take_profit:
                        self.logger.info(f"Take-profit triggered: {current_price} >= {self.current_position.take_profit}")
                        return True
                else:  # SHORT position
                    if current_price <= self.current_position.take_profit:
                        self.logger.info(f"Take-profit triggered: {current_price} <= {self.current_position.take_profit}")
                        return True
            
            # Check for opposite signal (EMA crossover in opposite direction)
            if self.data is not None and current_time in self.data.index:
                current_signal = self.generate_signal(self.data, current_time)
                
                if (self.current_position.position_type == SignalType.BUY and 
                    current_signal.signal_type == SignalType.SELL):
                    self.logger.info("Exit signal: EMA crossover in opposite direction (SELL)")
                    return True
                    
                elif (self.current_position.position_type == SignalType.SELL and 
                      current_signal.signal_type == SignalType.BUY):
                    self.logger.info("Exit signal: EMA crossover in opposite direction (BUY)")
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
        
        # Add EMA-specific information
        ema_info = {
            'strategy_type': 'EMA Crossover',
            'fast_ema_period': self.fast_period,
            'slow_ema_period': self.slow_period,
            'position_size_pct': self.position_size * 100,
            'stop_loss_pct': self.stop_loss_pct * 100,
            'performance': {
                'total_signals': self.total_signals,
                'profitable_signals': self.profitable_signals,
                'win_rate': (self.profitable_signals / max(self.total_signals, 1)) * 100,
                'total_return': self.total_return
            },
            'current_state': {
                'is_initialized': self.is_initialized,
                'has_position': self.current_position is not None,
                'data_points': len(self.data) if self.data is not None else 0
            }
        }
        
        # Merge with base info
        base_info.update(ema_info)
        
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
            
            # Check position size
            if not (0 < self.position_size <= 1):
                self.logger.error("Position size must be between 0 and 1")
                return False
            
            # Check stop loss
            if not (0 <= self.stop_loss_pct <= 1):
                self.logger.error("Stop loss percentage must be between 0 and 1")
                return False
            
            self.logger.info("Strategy parameters validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Parameter validation error: {e}")
            return False
    
    def reset(self) -> None:
        """Reset strategy state for new backtest."""
        super().reset()
        self.total_signals = 0
        self.profitable_signals = 0
        self.total_return = 0.0
        self.logger.info("Strategy state reset")
    
    def __str__(self) -> str:
        """String representation of the strategy."""
        return f"EMACrossoverStrategy(fast={self.fast_period}, slow={self.slow_period}, pos_size={self.position_size:.1%})"
    
    def calculate_stop_loss(self, entry_price: float, position_type: SignalType) -> Optional[float]:
        """
        Calculate stop-loss price based on entry price and position type.
        
        Args:
            entry_price: Entry price for the position
            position_type: Type of position (BUY for long, SELL for short)
            
        Returns:
            Stop-loss price or None if no stop-loss configured
        """
        if self.stop_loss_pct <= 0:
            return None
        
        if position_type == SignalType.BUY:
            # Long position: stop-loss below entry price
            stop_loss = entry_price * (1 - self.stop_loss_pct)
        elif position_type == SignalType.SELL:
            # Short position: stop-loss above entry price
            stop_loss = entry_price * (1 + self.stop_loss_pct)
        else:
            return None
        
        return stop_loss
    
    def calculate_take_profit(self, entry_price: float, position_type: SignalType) -> Optional[float]:
        """
        Calculate take-profit price based on entry price and position type.
        
        Args:
            entry_price: Entry price for the position
            position_type: Type of position (BUY for long, SELL for short)
            
        Returns:
            Take-profit price or None if no take-profit configured
        """
        # EMA strategy typically doesn't use fixed take-profit
        # Instead relies on opposite crossover signals for exits
        return None 