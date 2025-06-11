"""
Base Strategy Module for the Algorithmic Trading Backtesting System.
Provides abstract base class for all trading strategies.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass
from enum import Enum


class SignalType(Enum):
    """Enumeration for different signal types."""
    BUY = 1
    SELL = -1
    HOLD = 0


@dataclass
class TradingSignal:
    """Data class for trading signals."""
    timestamp: pd.Timestamp
    signal_type: SignalType
    confidence: float = 1.0  # Signal confidence (0-1)
    price: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class Position:
    """Data class for position information."""
    entry_time: pd.Timestamp
    entry_price: float
    quantity: float
    position_type: SignalType  # BUY (long) or SELL (short)
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.
    
    This class defines the interface that all trading strategies must implement.
    It provides common functionality and ensures consistency across different
    strategy implementations.
    """
    
    def __init__(self, name: str, parameters: Dict[str, Any] = None):
        """
        Initialize the base strategy.
        
        Args:
            name: Strategy name
            parameters: Strategy-specific parameters
        """
        self.name = name
        self.parameters = parameters or {}
        self.is_initialized = False
        self.current_position: Optional[Position] = None
        self.signals_history: list[TradingSignal] = []
        self.performance_metrics: Dict[str, float] = {}
        
    @abstractmethod
    def initialize(self, data: pd.DataFrame) -> None:
        """
        Initialize the strategy with historical data.
        
        This method should prepare any indicators, lookback periods,
        or other strategy-specific setup requirements.
        
        Args:
            data: Historical OHLCV data
        """
        pass
    
    @abstractmethod
    def generate_signal(self, data: pd.DataFrame, current_time: pd.Timestamp) -> TradingSignal:
        """
        Generate a trading signal based on current market data.
        
        Args:
            data: Historical OHLCV data up to current_time
            current_time: Current timestamp for signal generation
            
        Returns:
            TradingSignal object with the generated signal
        """
        pass
    
    @abstractmethod
    def calculate_position_size(self, data: pd.DataFrame, signal: TradingSignal, 
                              portfolio_value: float) -> float:
        """
        Calculate the position size for a given signal.
        
        Args:
            data: Current market data
            signal: Generated trading signal
            portfolio_value: Current portfolio value
            
        Returns:
            Position size (number of shares/units)
        """
        pass
    
    def set_risk_management(self, stop_loss_pct: float = None, 
                           take_profit_pct: float = None) -> None:
        """
        Set risk management parameters.
        
        Args:
            stop_loss_pct: Stop loss percentage (e.g., 0.02 for 2%)
            take_profit_pct: Take profit percentage (e.g., 0.05 for 5%)
        """
        if stop_loss_pct is not None:
            self.parameters['stop_loss_pct'] = stop_loss_pct
        if take_profit_pct is not None:
            self.parameters['take_profit_pct'] = take_profit_pct
    
    def calculate_stop_loss(self, entry_price: float, position_type: SignalType) -> Optional[float]:
        """
        Calculate stop loss price based on entry price and position type.
        
        Args:
            entry_price: Entry price of the position
            position_type: Type of position (BUY/SELL)
            
        Returns:
            Stop loss price or None if not configured
        """
        stop_loss_pct = self.parameters.get('stop_loss_pct')
        if stop_loss_pct is None:
            return None
            
        if position_type == SignalType.BUY:
            return entry_price * (1 - stop_loss_pct)
        elif position_type == SignalType.SELL:
            return entry_price * (1 + stop_loss_pct)
        return None
    
    def calculate_take_profit(self, entry_price: float, position_type: SignalType) -> Optional[float]:
        """
        Calculate take profit price based on entry price and position type.
        
        Args:
            entry_price: Entry price of the position
            position_type: Type of position (BUY/SELL)
            
        Returns:
            Take profit price or None if not configured
        """
        take_profit_pct = self.parameters.get('take_profit_pct')
        if take_profit_pct is None:
            return None
            
        if position_type == SignalType.BUY:
            return entry_price * (1 + take_profit_pct)
        elif position_type == SignalType.SELL:
            return entry_price * (1 - take_profit_pct)
        return None
    
    def open_position(self, signal: TradingSignal, quantity: float, price: float) -> Position:
        """
        Open a new position based on a trading signal.
        
        Args:
            signal: Trading signal that triggered the position
            quantity: Position size
            price: Entry price
            
        Returns:
            Position object
        """
        position = Position(
            entry_time=signal.timestamp,
            entry_price=price,
            quantity=quantity,
            position_type=signal.signal_type,
            stop_loss=self.calculate_stop_loss(price, signal.signal_type),
            take_profit=self.calculate_take_profit(price, signal.signal_type),
            metadata=signal.metadata
        )
        self.current_position = position
        return position
    
    def close_position(self, close_time: pd.Timestamp, close_price: float) -> Optional[Position]:
        """
        Close the current position.
        
        Args:
            close_time: Time of position closure
            close_price: Price at which position is closed
            
        Returns:
            Closed position or None if no position was open
        """
        if self.current_position is None:
            return None
            
        closed_position = self.current_position
        self.current_position = None
        return closed_position
    
    def should_exit_position(self, current_price: float, current_time: pd.Timestamp) -> bool:
        """
        Check if current position should be exited based on risk management rules.
        
        Args:
            current_price: Current market price
            current_time: Current timestamp
            
        Returns:
            True if position should be exited, False otherwise
        """
        if self.current_position is None:
            return False
            
        position = self.current_position
        
        # Check stop loss
        if position.stop_loss is not None:
            if position.position_type == SignalType.BUY and current_price <= position.stop_loss:
                return True
            elif position.position_type == SignalType.SELL and current_price >= position.stop_loss:
                return True
        
        # Check take profit
        if position.take_profit is not None:
            if position.position_type == SignalType.BUY and current_price >= position.take_profit:
                return True
            elif position.position_type == SignalType.SELL and current_price <= position.take_profit:
                return True
                
        return False
    
    def add_signal_to_history(self, signal: TradingSignal) -> None:
        """Add a signal to the strategy's signal history."""
        self.signals_history.append(signal)
    
    def get_signals_dataframe(self) -> pd.DataFrame:
        """
        Convert signals history to a pandas DataFrame.
        
        Returns:
            DataFrame with signals history
        """
        if not self.signals_history:
            return pd.DataFrame()
            
        data = []
        for signal in self.signals_history:
            data.append({
                'timestamp': signal.timestamp,
                'signal_type': signal.signal_type.value,
                'confidence': signal.confidence,
                'price': signal.price
            })
            
        return pd.DataFrame(data)
    
    def reset(self) -> None:
        """Reset the strategy state for a new backtesting run."""
        self.current_position = None
        self.signals_history = []
        self.performance_metrics = {}
        self.is_initialized = False
    
    def get_parameter(self, key: str, default: Any = None) -> Any:
        """Get a strategy parameter value."""
        return self.parameters.get(key, default)
    
    def set_parameter(self, key: str, value: Any) -> None:
        """Set a strategy parameter value."""
        self.parameters[key] = value
    
    def validate_parameters(self) -> bool:
        """
        Validate strategy parameters.
        
        Returns:
            True if parameters are valid, False otherwise
        """
        # Base validation - can be overridden in subclasses
        return True
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """
        Get strategy information and current state.
        
        Returns:
            Dictionary with strategy information
        """
        return {
            'name': self.name,
            'parameters': self.parameters,
            'is_initialized': self.is_initialized,
            'has_position': self.current_position is not None,
            'signals_count': len(self.signals_history),
            'performance_metrics': self.performance_metrics
        }
    
    def __str__(self) -> str:
        """String representation of the strategy."""
        return f"{self.name} Strategy (Parameters: {self.parameters})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the strategy."""
        return f"BaseStrategy(name='{self.name}', parameters={self.parameters})" 