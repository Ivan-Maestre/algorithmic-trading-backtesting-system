"""
Momentum Strategy Implementation

Implements a momentum-based trading strategy that generates signals based on
price momentum and trend strength. The strategy uses rolling returns and
momentum indicators to identify trading opportunities.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional
from datetime import datetime

from .base_strategy import BaseStrategy, TradingSignal, SignalType
from ..config import Config


class MomentumStrategy(BaseStrategy):
    """
    Momentum Strategy implementation for algorithmic trading.
    
    This strategy generates trading signals based on price momentum over a specified
    lookback period. It uses momentum thresholds to determine entry and exit points,
    with additional confirmation from volume and volatility indicators.
    
    Key Features:
    - Momentum calculation over configurable lookback period
    - Threshold-based signal generation
    - Volume confirmation for signal strength
    - Volatility-adjusted position sizing
    - Multiple exit conditions (stop-loss, momentum reversal)
    """
    
    def __init__(self, parameters: Dict[str, Any] = None):
        """
        Initialize the Momentum strategy.
        
        Args:
            parameters: Strategy parameters (optional, uses config defaults)
        """
        # Set default parameters from config
        default_params = Config.get_strategy_config('momentum')
        super().__init__("Momentum Strategy", parameters or default_params)
        
        # Strategy-specific parameters
        self.lookback_period = self.get_parameter('lookback_period', 20)
        self.threshold = self.get_parameter('threshold', 0.01)  # 1% momentum threshold
        self.position_size = self.get_parameter('position_size', 0.1)  # 10% of portfolio
        self.stop_loss_pct = self.get_parameter('stop_loss', 0.02)  # 2% stop loss
        self.min_volume_ratio = self.get_parameter('min_volume_ratio', 1.2)  # Volume confirmation
        
        # Ensure the base strategy has the correct parameter name
        self.parameters['stop_loss_pct'] = self.stop_loss_pct
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        
        # Strategy state
        self.data = None
        self.is_initialized = False
        
        # Performance tracking
        self.total_signals = 0
        self.profitable_signals = 0
        self.total_return = 0.0
        self.momentum_history = []
        
        self.logger.info(f"Momentum strategy initialized with {self.lookback_period}-period lookback, {self.threshold:.1%} threshold")
    
    def initialize(self, data: pd.DataFrame) -> None:
        """
        Initialize the strategy with historical data.
        
        Args:
            data: DataFrame with OHLCV data and technical indicators
        """
        self.logger.info("Initializing Momentum strategy")
        
        # Validate required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns for Momentum strategy: {missing_columns}")
        
        # Store data reference
        self.data = data.copy()
        
        # Validate sufficient data
        min_periods = self.lookback_period * 3  # Need 3x lookback for reliable calculations
        if len(data) < min_periods:
            raise ValueError(f"Insufficient data: {len(data)} periods (minimum: {min_periods})")
        
        # Calculate momentum and additional indicators if not present
        self._calculate_momentum_indicators()
        
        # Reset strategy state
        self.reset()
        self.is_initialized = True
        
        self.logger.info(f"Strategy initialized with {len(data)} data points")
        
        # Log data quality info
        momentum_valid = self.data['momentum'].notna().sum()
        self.logger.info(f"Momentum validity: {momentum_valid}/{len(data)} periods")
    
    def _calculate_momentum_indicators(self) -> None:
        """Calculate momentum and related indicators."""
        if 'momentum' not in self.data.columns:
            # Price momentum over lookback period
            self.data['momentum'] = self.data['close'].pct_change(periods=self.lookback_period)
        
        # Rolling returns for different periods
        self.data['return_1'] = self.data['close'].pct_change(1)
        self.data['return_5'] = self.data['close'].pct_change(5)
        self.data['return_10'] = self.data['close'].pct_change(10)
        
        # Rolling volatility
        self.data['volatility'] = self.data['return_1'].rolling(window=20).std()
        
        # Volume moving average for confirmation
        self.data['volume_ma'] = self.data['volume'].rolling(window=20).mean()
        self.data['volume_ratio'] = self.data['volume'] / self.data['volume_ma']
        
        # Momentum strength indicator (normalized momentum)
        momentum_std = self.data['momentum'].rolling(window=50).std()
        self.data['momentum_strength'] = self.data['momentum'] / momentum_std
        
        # Trend consistency (percentage of positive returns in lookback)
        self.data['trend_consistency'] = (
            self.data['return_1'].rolling(window=self.lookback_period)
            .apply(lambda x: (x > 0).sum() / len(x))
        )
    
    def generate_signal(self, data: pd.DataFrame, current_time: pd.Timestamp) -> TradingSignal:
        """
        Generate a trading signal based on momentum analysis.
        
        Args:
            data: Market data DataFrame
            current_time: Current timestamp for signal generation
            
        Returns:
            TradingSignal with BUY, SELL, or HOLD
        """
        try:
            if current_time not in data.index:
                self.logger.warning(f"Timestamp {current_time} not in data index")
                return self._create_hold_signal(current_time, data.iloc[-1]['close'])
            
            current_data = data.loc[current_time]
            
            # Get momentum value
            momentum = current_data.get('momentum', 0.0)
            if pd.isna(momentum):
                return self._create_hold_signal(current_time, current_data['close'])
            
            # Determine signal type based on momentum threshold
            signal_type = SignalType.HOLD
            confidence = 0.0
            
            if momentum > self.threshold:
                # Strong positive momentum - BUY signal
                signal_type = SignalType.BUY
                confidence = self._calculate_signal_confidence(data, current_time, signal_type)
            elif momentum < -self.threshold:
                # Strong negative momentum - SELL signal  
                signal_type = SignalType.SELL
                confidence = self._calculate_signal_confidence(data, current_time, signal_type)
            
            # Create and return signal
            signal = TradingSignal(
                timestamp=current_time,
                signal_type=signal_type,
                confidence=confidence,
                price=current_data['close'],
                metadata={
                    'momentum': momentum,
                    'threshold': self.threshold,
                    'lookback_period': self.lookback_period,
                    'volume_ratio': current_data.get('volume_ratio', 1.0),
                    'volatility': current_data.get('volatility', 0.0),
                    'trend_consistency': current_data.get('trend_consistency', 0.5)
                }
            )
            
            # Track signal
            self.signals_history.append(signal)
            self.total_signals += 1
            
            if signal_type != SignalType.HOLD:
                self.logger.info(f"Momentum signal generated: {signal_type.name} at {current_time} "
                               f"(momentum: {momentum:.3f}, confidence: {confidence:.3f})")
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Error generating momentum signal at {current_time}: {e}")
            return self._create_hold_signal(current_time, data.iloc[-1]['close'])
    
    def _calculate_signal_confidence(self, data: pd.DataFrame, current_time: pd.Timestamp, 
                                   signal_type: SignalType) -> float:
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
            
            # Base confidence from momentum strength
            momentum = current_data.get('momentum', 0.0)
            momentum_strength = current_data.get('momentum_strength', 0.0)
            
            # Momentum magnitude confidence
            momentum_confidence = min(abs(momentum) / (self.threshold * 3), 1.0)  # Normalize by 3x threshold
            
            # Volume confirmation
            volume_ratio = current_data.get('volume_ratio', 1.0)
            volume_confidence = min((volume_ratio - 1.0) / 0.5 + 0.5, 1.0) if volume_ratio >= self.min_volume_ratio else 0.3
            
            # Trend consistency confirmation
            trend_consistency = current_data.get('trend_consistency', 0.5)
            if signal_type == SignalType.BUY:
                trend_confidence = trend_consistency
            else:  # SELL signal
                trend_confidence = 1.0 - trend_consistency
            
            # Volatility adjustment (lower volatility = higher confidence)
            volatility = current_data.get('volatility', 0.02)
            volatility_confidence = max(0.2, 1.0 - (volatility / 0.05))  # Normalize by 5% volatility
            
            # Recent momentum confirmation (check if momentum is accelerating)
            momentum_accel_confidence = 0.5  # Default
            time_index = data.index.get_loc(current_time)
            if time_index >= 5:
                prev_momentum = data.iloc[time_index-5].get('momentum', 0.0)
                if not pd.isna(prev_momentum) and not pd.isna(momentum):
                    if signal_type == SignalType.BUY and momentum > prev_momentum:
                        momentum_accel_confidence = 0.8
                    elif signal_type == SignalType.SELL and momentum < prev_momentum:
                        momentum_accel_confidence = 0.8
                    else:
                        momentum_accel_confidence = 0.3
            
            # Combine confidence factors with weights
            confidence = (momentum_confidence * 0.35 +       # Primary momentum signal
                         volume_confidence * 0.25 +          # Volume confirmation
                         trend_confidence * 0.20 +           # Trend consistency  
                         volatility_confidence * 0.10 +      # Volatility environment
                         momentum_accel_confidence * 0.10)   # Momentum acceleration
            
            return min(max(confidence, 0.1), 0.95)  # Keep between 0.1 and 0.95
            
        except Exception as e:
            self.logger.error(f"Error calculating signal confidence: {e}")
            return 0.5  # Default confidence
    
    def calculate_position_size(self, data: pd.DataFrame, signal: TradingSignal, 
                              portfolio_value: float) -> float:
        """
        Calculate position size based on signal confidence and risk management.
        
        Args:
            data: Current market data
            signal: Trading signal
            portfolio_value: Current portfolio value
            
        Returns:
            Position size (quantity to trade)
        """
        try:
            if signal.signal_type == SignalType.HOLD:
                return 0.0
            
            # Base position size from configuration
            base_size_pct = self.position_size
            
            # Adjust by signal confidence
            confidence_multiplier = signal.confidence
            
            # Volatility adjustment (reduce size in high volatility)
            volatility = signal.metadata.get('volatility', 0.02)
            volatility_multiplier = max(0.5, 1.0 - (volatility / 0.05))  # Reduce if vol > 5%
            
            # Trend consistency adjustment
            trend_consistency = signal.metadata.get('trend_consistency', 0.5)
            if signal.signal_type == SignalType.BUY:
                trend_multiplier = trend_consistency
            else:  # SELL
                trend_multiplier = 1.0 - trend_consistency
            
            # Combined position size percentage
            final_size_pct = (base_size_pct * confidence_multiplier * 
                             volatility_multiplier * trend_multiplier)
            
            # Apply limits
            final_size_pct = min(final_size_pct, 0.2)  # Max 20% position
            final_size_pct = max(final_size_pct, 0.01)  # Min 1% position
            
            # Calculate quantity
            position_value = portfolio_value * final_size_pct
            quantity = position_value / signal.price
            
            self.logger.debug(f"Position size calculation: {final_size_pct:.1%} of portfolio "
                            f"= {quantity:.6f} units (confidence: {confidence_multiplier:.3f}, "
                            f"volatility: {volatility:.3f}, trend: {trend_multiplier:.3f})")
            
            return quantity
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return 0.0
    
    def should_exit_position(self, current_price: float, current_time: pd.Timestamp) -> bool:
        """
        Determine if current position should be exited.
        
        Args:
            current_price: Current market price
            current_time: Current timestamp
            
        Returns:
            True if position should be exited
        """
        if not self.current_position:
            return False
        
        try:
            # Check stop loss
            stop_loss_price = self.current_position.stop_loss
            if stop_loss_price:
                if self.current_position.position_type == SignalType.BUY and current_price <= stop_loss_price:
                    self.logger.info(f"Stop loss triggered for BUY position: {current_price} <= {stop_loss_price}")
                    return True
                elif self.current_position.position_type == SignalType.SELL and current_price >= stop_loss_price:
                    self.logger.info(f"Stop loss triggered for SELL position: {current_price} >= {stop_loss_price}")
                    return True
            
            # Check momentum reversal if data is available
            if self.data is not None and current_time in self.data.index:
                current_momentum = self.data.loc[current_time].get('momentum')
                if not pd.isna(current_momentum):
                    # Exit if momentum reverses significantly
                    if (self.current_position.position_type == SignalType.BUY and 
                        current_momentum < -self.threshold):
                        self.logger.info(f"Momentum reversal exit for BUY: momentum = {current_momentum:.3f}")
                        return True
                    elif (self.current_position.position_type == SignalType.SELL and 
                          current_momentum > self.threshold):
                        self.logger.info(f"Momentum reversal exit for SELL: momentum = {current_momentum:.3f}")
                        return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking exit conditions: {e}")
            return False
    
    def _create_hold_signal(self, timestamp: pd.Timestamp, price: float) -> TradingSignal:
        """Create a HOLD signal with default values."""
        return TradingSignal(
            timestamp=timestamp,
            signal_type=SignalType.HOLD,
            confidence=0.0,
            price=price,
            metadata={'reason': 'insufficient_data_or_error'}
        )
    
    def validate_parameters(self) -> bool:
        """
        Validate strategy parameters.
        
        Returns:
            True if parameters are valid
        """
        try:
            # Check lookback period
            if self.lookback_period <= 0:
                self.logger.error("Lookback period must be positive")
                return False
            
            # Check threshold
            if not (0 < self.threshold <= 1):
                self.logger.error("Momentum threshold must be between 0 and 1")
                return False
            
            # Check position size
            if not (0 < self.position_size <= 1):
                self.logger.error("Position size must be between 0 and 1")
                return False
            
            # Check stop loss
            if not (0 <= self.stop_loss_pct <= 1):
                self.logger.error("Stop loss percentage must be between 0 and 1")
                return False
            
            # Check minimum volume ratio
            if self.min_volume_ratio < 1.0:
                self.logger.error("Minimum volume ratio should be >= 1.0")
                return False
            
            self.logger.info("Strategy parameters validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Parameter validation error: {e}")
            return False
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get detailed strategy information."""
        base_info = super().get_strategy_info()
        
        momentum_info = {
            'lookback_period': self.lookback_period,
            'momentum_threshold': self.threshold,
            'min_volume_ratio': self.min_volume_ratio,
            'total_signals_generated': self.total_signals,
            'momentum_history_size': len(self.momentum_history)
        }
        
        base_info.update(momentum_info)
        return base_info
    
    def reset(self) -> None:
        """Reset strategy state for new backtest."""
        super().reset()
        self.total_signals = 0
        self.profitable_signals = 0
        self.total_return = 0.0
        self.momentum_history.clear()
        self.logger.info("Momentum strategy state reset")
    
    def __str__(self) -> str:
        """String representation of the strategy."""
        return (f"MomentumStrategy(lookback={self.lookback_period}, "
                f"threshold={self.threshold:.1%}, pos_size={self.position_size:.1%})")
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return (f"MomentumStrategy(lookback={self.lookback_period}, "
                f"threshold={self.threshold}, position_size={self.position_size}, "
                f"stop_loss={self.stop_loss_pct})") 