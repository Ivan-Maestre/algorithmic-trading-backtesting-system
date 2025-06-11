"""
Trade Management Module

Represents completed trades with entry/exit information and performance calculations.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import pandas as pd
from datetime import datetime, timedelta


@dataclass
class Trade:
    """
    Represents a completed trade (entry + exit).
    
    Attributes:
        trade_id: Unique identifier for the trade
        symbol: Trading symbol
        strategy_id: Strategy that generated the trade
        entry_time: Trade entry timestamp
        exit_time: Trade exit timestamp
        side: 'buy' (long) or 'sell' (short)
        quantity: Number of units traded
        entry_price: Price at trade entry
        exit_price: Price at trade exit
        entry_commission: Commission paid on entry
        exit_commission: Commission paid on exit
        pnl: Profit and loss (before commissions)
        pnl_percent: Profit and loss as percentage
        net_pnl: Net profit and loss (after commissions)
        duration: Trade duration
        metadata: Additional trade information
    """
    
    trade_id: str
    symbol: str
    strategy_id: str
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    side: str  # 'buy' (long) or 'sell' (short)
    quantity: float
    entry_price: float
    exit_price: float
    entry_commission: float = 0.0
    exit_commission: float = 0.0
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Calculate derived metrics after initialization."""
        if self.metadata is None:
            self.metadata = {}
            
        # Validate inputs
        if self.quantity <= 0:
            raise ValueError("Trade quantity must be positive")
            
        if self.entry_price <= 0 or self.exit_price <= 0:
            raise ValueError("Trade prices must be positive")
            
        if self.side not in ['buy', 'sell']:
            raise ValueError("Trade side must be 'buy' or 'sell'")
            
        if self.exit_time <= self.entry_time:
            raise ValueError("Exit time must be after entry time")
    
    @property
    def duration(self) -> timedelta:
        """Get trade duration."""
        return self.exit_time - self.entry_time
    
    @property
    def duration_hours(self) -> float:
        """Get trade duration in hours."""
        return self.duration.total_seconds() / 3600
    
    @property
    def pnl(self) -> float:
        """Calculate profit and loss before commissions."""
        if self.side == 'buy':  # Long position
            return (self.exit_price - self.entry_price) * self.quantity
        else:  # Short position
            return (self.entry_price - self.exit_price) * self.quantity
    
    @property
    def pnl_percent(self) -> float:
        """Calculate profit and loss as percentage."""
        if self.side == 'buy':  # Long position
            return ((self.exit_price - self.entry_price) / self.entry_price) * 100
        else:  # Short position
            return ((self.entry_price - self.exit_price) / self.entry_price) * 100
    
    @property
    def net_pnl(self) -> float:
        """Calculate net profit and loss after commissions."""
        return self.pnl - self.total_commission
    
    @property
    def net_pnl_percent(self) -> float:
        """Calculate net profit and loss percentage after commissions."""
        if self.side == 'buy':  # Long position
            cost_basis = self.entry_price * self.quantity + self.entry_commission
            return ((self.exit_price * self.quantity - self.exit_commission - cost_basis) / cost_basis) * 100
        else:  # Short position
            # For short trades, calculate based on initial margin/capital required
            initial_value = self.entry_price * self.quantity
            return (self.net_pnl / initial_value) * 100
    
    @property
    def total_commission(self) -> float:
        """Get total commissions paid."""
        return self.entry_commission + self.exit_commission
    
    @property
    def commission_percent(self) -> float:
        """Get commission as percentage of trade value."""
        trade_value = self.entry_price * self.quantity
        return (self.total_commission / trade_value) * 100 if trade_value > 0 else 0
    
    @property
    def is_profitable(self) -> bool:
        """Check if trade was profitable (after commissions)."""
        return self.net_pnl > 0
    
    @property
    def is_winning(self) -> bool:
        """Alias for is_profitable."""
        return self.is_profitable
    
    @property
    def trade_value(self) -> float:
        """Get total trade value (entry price * quantity)."""
        return self.entry_price * self.quantity
    
    @property
    def exit_value(self) -> float:
        """Get exit value (exit price * quantity)."""
        return self.exit_price * self.quantity
    
    @property
    def return_on_investment(self) -> float:
        """Calculate return on investment (ROI) percentage."""
        return self.net_pnl_percent
    
    def get_metrics(self) -> Dict[str, float]:
        """
        Get comprehensive trade metrics.
        
        Returns:
            Dictionary with all calculated metrics
        """
        return {
            'pnl': self.pnl,
            'pnl_percent': self.pnl_percent,
            'net_pnl': self.net_pnl,
            'net_pnl_percent': self.net_pnl_percent,
            'total_commission': self.total_commission,
            'commission_percent': self.commission_percent,
            'duration_hours': self.duration_hours,
            'trade_value': self.trade_value,
            'exit_value': self.exit_value,
            'roi': self.return_on_investment,
            'is_winning': self.is_winning
        }
    
    def get_risk_metrics(self, portfolio_value: float) -> Dict[str, float]:
        """
        Get risk-related metrics for the trade.
        
        Args:
            portfolio_value: Portfolio value at trade entry
            
        Returns:
            Dictionary with risk metrics
        """
        return {
            'position_size_percent': (self.trade_value / portfolio_value) * 100,
            'risk_percent': abs(self.net_pnl / portfolio_value) * 100,
            'reward_to_risk': abs(self.net_pnl / self.trade_value) if self.trade_value > 0 else 0
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert trade to dictionary representation."""
        base_dict = {
            'trade_id': self.trade_id,
            'symbol': self.symbol,
            'strategy_id': self.strategy_id,
            'entry_time': self.entry_time,
            'exit_time': self.exit_time,
            'side': self.side,
            'quantity': self.quantity,
            'entry_price': self.entry_price,
            'exit_price': self.exit_price,
            'entry_commission': self.entry_commission,
            'exit_commission': self.exit_commission,
            'duration_hours': self.duration_hours,
        }
        
        # Add calculated metrics
        base_dict.update(self.get_metrics())
        
        # Add metadata
        if self.metadata:
            base_dict.update(self.metadata)
        
        return base_dict
    
    def add_metadata(self, key: str, value: Any) -> None:
        """
        Add metadata to the trade.
        
        Args:
            key: Metadata key
            value: Metadata value
        """
        if self.metadata is None:
            self.metadata = {}
        self.metadata[key] = value
    
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """
        Get metadata value.
        
        Args:
            key: Metadata key
            default: Default value if key not found
            
        Returns:
            Metadata value or default
        """
        if self.metadata is None:
            return default
        return self.metadata.get(key, default)
    
    @classmethod
    def from_orders(cls, entry_order, exit_order, trade_id: str = None):
        """
        Create a Trade from entry and exit orders.
        
        Args:
            entry_order: Order object for trade entry
            exit_order: Order object for trade exit
            trade_id: Custom trade ID (generated if None)
            
        Returns:
            Trade object
        """
        if trade_id is None:
            trade_id = f"{entry_order.order_id}_{exit_order.order_id}"
        
        # Determine trade side from entry order
        side = entry_order.side
        
        return cls(
            trade_id=trade_id,
            symbol=entry_order.symbol,
            strategy_id=entry_order.strategy_id,
            entry_time=entry_order.timestamp,
            exit_time=exit_order.timestamp,
            side=side,
            quantity=min(entry_order.filled_quantity, exit_order.filled_quantity),
            entry_price=entry_order.filled_price,
            exit_price=exit_order.filled_price,
            entry_commission=entry_order.commission,
            exit_commission=exit_order.commission,
            metadata={
                'entry_order_id': entry_order.order_id,
                'exit_order_id': exit_order.order_id,
                'entry_order_type': entry_order.order_type.value,
                'exit_order_type': exit_order.order_type.value
            }
        )
    
    def __str__(self) -> str:
        """String representation of the trade."""
        return (f"Trade({self.trade_id}: {self.side} {self.quantity} {self.symbol} "
                f"${self.entry_price:.2f} -> ${self.exit_price:.2f} "
                f"PnL: ${self.net_pnl:.2f} ({self.net_pnl_percent:.2f}%))")
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return (f"Trade(id={self.trade_id}, side={self.side}, qty={self.quantity}, "
                f"entry=${self.entry_price:.2f}, exit=${self.exit_price:.2f}, "
                f"pnl=${self.net_pnl:.2f})") 