"""
Order Management Module

Defines order types, statuses, and order execution logic for backtesting.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional
import pandas as pd
from datetime import datetime


class OrderType(Enum):
    """Order types supported by the backtesting engine."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderStatus(Enum):
    """Order execution statuses."""
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Order:
    """
    Represents a trading order in the backtesting system.
    
    Attributes:
        order_id: Unique identifier for the order
        timestamp: When the order was created
        symbol: Trading symbol (e.g., 'BTCUSDT')
        order_type: Type of order (market, limit, etc.)
        side: 'buy' or 'sell'
        quantity: Number of units to trade
        price: Order price (None for market orders)
        stop_price: Stop price for stop orders
        status: Current order status
        filled_quantity: Amount actually filled
        filled_price: Average fill price
        commission: Transaction costs
        strategy_id: ID of strategy that placed the order
        metadata: Additional order information
    """
    
    order_id: str
    timestamp: pd.Timestamp
    symbol: str
    order_type: OrderType
    side: str  # 'buy' or 'sell'
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    filled_price: Optional[float] = None
    commission: float = 0.0
    strategy_id: str = ""
    metadata: Optional[dict] = None
    
    def __post_init__(self):
        """Validate order parameters after initialization."""
        if self.quantity <= 0:
            raise ValueError("Order quantity must be positive")
        
        if self.side not in ['buy', 'sell']:
            raise ValueError("Order side must be 'buy' or 'sell'")
        
        if self.order_type == OrderType.LIMIT and self.price is None:
            raise ValueError("Limit orders must have a price")
        
        if self.order_type in [OrderType.STOP, OrderType.STOP_LIMIT] and self.stop_price is None:
            raise ValueError("Stop orders must have a stop price")
        
        if self.metadata is None:
            self.metadata = {}
    
    @property
    def is_filled(self) -> bool:
        """Check if order is completely filled."""
        return self.status == OrderStatus.FILLED
    
    @property
    def is_pending(self) -> bool:
        """Check if order is still pending."""
        return self.status == OrderStatus.PENDING
    
    @property
    def remaining_quantity(self) -> float:
        """Get remaining unfilled quantity."""
        return self.quantity - self.filled_quantity
    
    @property
    def fill_percentage(self) -> float:
        """Get percentage of order filled."""
        return (self.filled_quantity / self.quantity) * 100 if self.quantity > 0 else 0
    
    def can_fill(self, current_price: float) -> bool:
        """
        Check if order can be filled at current market price.
        
        Args:
            current_price: Current market price
            
        Returns:
            True if order can be filled
        """
        if self.status != OrderStatus.PENDING:
            return False
        
        if self.order_type == OrderType.MARKET:
            return True
        
        elif self.order_type == OrderType.LIMIT:
            if self.side == 'buy':
                return current_price <= self.price
            else:  # sell
                return current_price >= self.price
        
        elif self.order_type == OrderType.STOP:
            if self.side == 'buy':
                return current_price >= self.stop_price
            else:  # sell
                return current_price <= self.stop_price
        
        elif self.order_type == OrderType.STOP_LIMIT:
            # Stop condition must be triggered first
            if self.side == 'buy':
                return current_price >= self.stop_price
            else:  # sell
                return current_price <= self.stop_price
        
        return False
    
    def get_execution_price(self, current_price: float, slippage: float = 0.0) -> float:
        """
        Get the price at which order would be executed.
        
        Args:
            current_price: Current market price
            slippage: Slippage factor (0.001 = 0.1%)
            
        Returns:
            Execution price including slippage
        """
        if self.order_type == OrderType.MARKET:
            # Apply slippage to market orders
            if self.side == 'buy':
                return current_price * (1 + slippage)
            else:  # sell
                return current_price * (1 - slippage)
        
        elif self.order_type == OrderType.LIMIT:
            # Limit orders execute at limit price or better
            return self.price
        
        elif self.order_type in [OrderType.STOP, OrderType.STOP_LIMIT]:
            # Stop orders become market orders when triggered
            if self.side == 'buy':
                return current_price * (1 + slippage)
            else:  # sell
                return current_price * (1 - slippage)
        
        return current_price
    
    def fill(self, quantity: float, price: float, commission: float = 0.0) -> None:
        """
        Fill the order (partially or completely).
        
        Args:
            quantity: Quantity to fill
            price: Fill price
            commission: Commission for this fill
        """
        if quantity <= 0:
            raise ValueError("Fill quantity must be positive")
        
        if quantity > self.remaining_quantity:
            raise ValueError("Cannot fill more than remaining quantity")
        
        # Update filled quantities
        total_value = self.filled_quantity * (self.filled_price or 0) + quantity * price
        self.filled_quantity += quantity
        self.filled_price = total_value / self.filled_quantity
        self.commission += commission
        
        # Update status
        if self.filled_quantity >= self.quantity:
            self.status = OrderStatus.FILLED
        else:
            self.status = OrderStatus.PARTIALLY_FILLED
    
    def cancel(self) -> None:
        """Cancel the order."""
        if self.status == OrderStatus.PENDING:
            self.status = OrderStatus.CANCELLED
        elif self.status == OrderStatus.PARTIALLY_FILLED:
            self.status = OrderStatus.CANCELLED  # Partially filled orders can be cancelled
    
    def reject(self, reason: str = "") -> None:
        """
        Reject the order.
        
        Args:
            reason: Reason for rejection
        """
        self.status = OrderStatus.REJECTED
        if reason:
            self.metadata['rejection_reason'] = reason
    
    def to_dict(self) -> dict:
        """Convert order to dictionary representation."""
        return {
            'order_id': self.order_id,
            'timestamp': self.timestamp,
            'symbol': self.symbol,
            'order_type': self.order_type.value,
            'side': self.side,
            'quantity': self.quantity,
            'price': self.price,
            'stop_price': self.stop_price,
            'status': self.status.value,
            'filled_quantity': self.filled_quantity,
            'filled_price': self.filled_price,
            'commission': self.commission,
            'strategy_id': self.strategy_id,
            'fill_percentage': self.fill_percentage,
            'metadata': self.metadata
        }
    
    def __str__(self) -> str:
        """String representation of the order."""
        return (f"Order({self.order_id}: {self.side} {self.quantity} {self.symbol} "
                f"@ {self.price or 'market'} - {self.status.value})")
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return (f"Order(id={self.order_id}, side={self.side}, qty={self.quantity}, "
                f"price={self.price}, status={self.status.value}, "
                f"filled={self.filled_quantity}/{self.quantity})") 