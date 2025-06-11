"""
Portfolio Management Module

Manages portfolio state, positions, cash, and performance tracking for backtesting.
"""

from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from datetime import datetime
import logging

from .order import Order, OrderStatus
from .trade import Trade
from ..config import Config


class Portfolio:
    """
    Manages portfolio state during backtesting.
    
    Tracks cash, positions, orders, trades, and calculates performance metrics.
    """
    
    def __init__(self, initial_capital: float, commission_rate: float = None):
        """
        Initialize portfolio.
        
        Args:
            initial_capital: Starting cash amount
            commission_rate: Commission rate per transaction (optional, uses config)
        """
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate or Config.TRANSACTION_COST
        
        # Portfolio state
        self.cash = initial_capital
        self.positions = {}  # symbol -> quantity
        self.orders = []  # All orders
        self.pending_orders = []  # Orders waiting to be filled
        self.filled_orders = []  # Completed orders
        self.trades = []  # Completed trades
        
        # Performance tracking
        self.equity_curve = []  # (timestamp, portfolio_value)
        self.drawdown_curve = []  # (timestamp, drawdown)
        self.peak_value = initial_capital
        self.max_drawdown = 0.0
        
        # Statistics
        self.total_commission_paid = 0.0
        self.last_update_time = None
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"Portfolio initialized with ${initial_capital:,.2f} initial capital")
    
    @property
    def total_cash(self) -> float:
        """Get total cash available."""
        return self.cash
    
    @property
    def total_positions_value(self) -> float:
        """Get total value of all positions at current prices."""
        # This will be calculated when update_portfolio is called
        return getattr(self, '_positions_value', 0.0)
    
    @property
    def total_portfolio_value(self) -> float:
        """Get total portfolio value (cash + positions)."""
        return self.cash + self.total_positions_value
    
    @property
    def net_liquidation_value(self) -> float:
        """Alias for total portfolio value."""
        return self.total_portfolio_value
    
    @property
    def total_return(self) -> float:
        """Get total return in absolute terms."""
        return self.total_portfolio_value - self.initial_capital
    
    @property
    def total_return_percent(self) -> float:
        """Get total return as percentage."""
        return (self.total_return / self.initial_capital) * 100
    
    @property
    def current_drawdown(self) -> float:
        """Get current drawdown from peak."""
        if self.peak_value <= 0:
            return 0.0
        return ((self.peak_value - self.total_portfolio_value) / self.peak_value) * 100
    
    def has_position(self, symbol: str) -> bool:
        """Check if portfolio has position in symbol."""
        return symbol in self.positions and abs(self.positions[symbol]) > 1e-8
    
    def get_position(self, symbol: str) -> float:
        """Get position quantity for symbol."""
        return self.positions.get(symbol, 0.0)
    
    def get_position_value(self, symbol: str, current_price: float) -> float:
        """Get position value at current price."""
        quantity = self.get_position(symbol)
        return quantity * current_price
    
    def can_afford(self, symbol: str, quantity: float, price: float) -> bool:
        """
        Check if portfolio can afford to buy quantity at price.
        
        Args:
            symbol: Trading symbol
            quantity: Quantity to buy
            price: Price per unit
            
        Returns:
            True if affordable
        """
        cost = quantity * price
        commission = self.calculate_commission(cost)
        total_cost = cost + commission
        
        return self.cash >= total_cost
    
    def calculate_commission(self, trade_value: float) -> float:
        """
        Calculate commission for a trade.
        
        Args:
            trade_value: Value of the trade
            
        Returns:
            Commission amount
        """
        return trade_value * self.commission_rate
    
    def place_order(self, order: Order) -> bool:
        """
        Place an order in the portfolio.
        
        Args:
            order: Order to place
            
        Returns:
            True if order was accepted
        """
        try:
            # Validate order
            if order.side == 'buy':
                if not self.can_afford(order.symbol, order.quantity, order.price or 0):
                    order.reject("Insufficient funds")
                    self.logger.warning(f"Order rejected: insufficient funds for {order}")
                    return False
            
            elif order.side == 'sell':
                current_position = self.get_position(order.symbol)
                if current_position < order.quantity:
                    order.reject(f"Insufficient position: have {current_position}, need {order.quantity}")
                    self.logger.warning(f"Order rejected: insufficient position for {order}")
                    return False
            
            # Add to orders list
            self.orders.append(order)
            self.pending_orders.append(order)
            
            self.logger.info(f"Order placed: {order}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error placing order: {e}")
            order.reject(str(e))
            return False
    
    def process_pending_orders(self, current_data: pd.Series, symbol: str = None) -> List[Order]:
        """
        Process pending orders against current market data.
        
        Args:
            current_data: Current market data with OHLCV
            symbol: Trading symbol (required for order matching)
            
        Returns:
            List of filled orders
        """
        filled_orders = []
        # Use passed symbol or try to extract from data
        trading_symbol = symbol or getattr(current_data, 'name', None)
        
        if not trading_symbol:
            # Try to get symbol from order if available
            if self.pending_orders:
                trading_symbol = self.pending_orders[0].symbol
        
        current_price = current_data['close']
        
        # Process each pending order
        orders_to_remove = []
        
        for order in self.pending_orders:
            if trading_symbol and order.symbol != trading_symbol:
                continue  # Skip orders for other symbols
            
            if order.can_fill(current_price):
                # Calculate execution price with slippage
                execution_price = order.get_execution_price(current_price, Config.SLIPPAGE)
                commission = self.calculate_commission(order.quantity * execution_price)
                
                # Fill the order
                order.fill(order.quantity, execution_price, commission)
                
                # Update portfolio positions and cash
                self._execute_order(order)
                
                # Move to filled orders
                filled_orders.append(order)
                orders_to_remove.append(order)
                self.filled_orders.append(order)
                
                self.logger.info(f"Order filled: {order}")
        
        # Remove filled orders from pending
        for order in orders_to_remove:
            self.pending_orders.remove(order)
        
        return filled_orders
    
    def _execute_order(self, order: Order) -> None:
        """
        Execute a filled order by updating portfolio state.
        
        Args:
            order: Filled order to execute
        """
        trade_value = order.filled_quantity * order.filled_price
        commission = order.commission
        
        if order.side == 'buy':
            # Add to position, subtract from cash
            self.positions[order.symbol] = self.positions.get(order.symbol, 0.0) + order.filled_quantity
            self.cash -= (trade_value + commission)
            
        elif order.side == 'sell':
            # Subtract from position, add to cash
            self.positions[order.symbol] = self.positions.get(order.symbol, 0.0) - order.filled_quantity
            self.cash += (trade_value - commission)
            
            # Clean up zero positions
            if abs(self.positions[order.symbol]) < 1e-8:
                del self.positions[order.symbol]
        
        # Track total commission
        self.total_commission_paid += commission
    
    def update_portfolio(self, timestamp: pd.Timestamp, current_prices: Dict[str, float]) -> None:
        """
        Update portfolio value and performance metrics.
        
        Args:
            timestamp: Current timestamp
            current_prices: Dictionary of symbol -> current price
        """
        # Calculate positions value
        positions_value = 0.0
        for symbol, quantity in self.positions.items():
            if symbol in current_prices:
                positions_value += quantity * current_prices[symbol]
        
        self._positions_value = positions_value
        total_value = self.cash + positions_value
        
        # Update peak and drawdown
        if total_value > self.peak_value:
            self.peak_value = total_value
        
        current_dd = self.current_drawdown
        if current_dd > self.max_drawdown:
            self.max_drawdown = current_dd
        
        # Update curves
        self.equity_curve.append((timestamp, total_value))
        self.drawdown_curve.append((timestamp, current_dd))
        
        self.last_update_time = timestamp
    
    def create_trade_from_orders(self, entry_order: Order, exit_order: Order) -> Trade:
        """
        Create a trade record from entry and exit orders.
        
        Args:
            entry_order: Order that opened the position
            exit_order: Order that closed the position
            
        Returns:
            Trade object
        """
        trade = Trade.from_orders(entry_order, exit_order)
        self.trades.append(trade)
        return trade
    
    def get_equity_curve(self) -> pd.Series:
        """Get equity curve as pandas Series."""
        if not self.equity_curve:
            return pd.Series(dtype=float)
        
        timestamps, values = zip(*self.equity_curve)
        return pd.Series(values, index=timestamps, name='portfolio_value')
    
    def get_drawdown_curve(self) -> pd.Series:
        """Get drawdown curve as pandas Series."""
        if not self.drawdown_curve:
            return pd.Series(dtype=float)
        
        timestamps, values = zip(*self.drawdown_curve)
        return pd.Series(values, index=timestamps, name='drawdown')
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """
        Calculate comprehensive performance metrics.
        
        Returns:
            Dictionary with performance metrics
        """
        if not self.equity_curve:
            return {}
        
        equity_series = self.get_equity_curve()
        returns = equity_series.pct_change().dropna()
        
        # Basic metrics
        metrics = {
            'initial_capital': self.initial_capital,
            'final_value': self.total_portfolio_value,
            'total_return': self.total_return,
            'total_return_percent': self.total_return_percent,
            'max_drawdown': self.max_drawdown,
            'current_drawdown': self.current_drawdown,
            'total_commission': self.total_commission_paid,
            'commission_percent': (self.total_commission_paid / self.initial_capital) * 100
        }
        
        # Time-based metrics
        if len(equity_series) > 1:
            days = (equity_series.index[-1] - equity_series.index[0]).days
            if days > 0:
                metrics['annualized_return'] = ((self.total_portfolio_value / self.initial_capital) ** (365.25 / days) - 1) * 100
            else:
                metrics['annualized_return'] = 0.0
        else:
            metrics['annualized_return'] = 0.0
        
        # Risk metrics
        if len(returns) > 1:
            metrics['volatility'] = returns.std() * np.sqrt(252 * 6) * 100  # Annualized volatility for 4h data
            if metrics['volatility'] > 0:
                metrics['sharpe_ratio'] = metrics['annualized_return'] / metrics['volatility']
            else:
                metrics['sharpe_ratio'] = 0.0
        else:
            metrics['volatility'] = 0.0
            metrics['sharpe_ratio'] = 0.0
        
        # Trade statistics
        if self.trades:
            winning_trades = [t for t in self.trades if t.is_winning]
            losing_trades = [t for t in self.trades if not t.is_winning]
            
            metrics.update({
                'total_trades': len(self.trades),
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'win_rate': (len(winning_trades) / len(self.trades)) * 100,
                'average_win': np.mean([t.net_pnl for t in winning_trades]) if winning_trades else 0.0,
                'average_loss': np.mean([t.net_pnl for t in losing_trades]) if losing_trades else 0.0,
                'largest_win': max([t.net_pnl for t in winning_trades]) if winning_trades else 0.0,
                'largest_loss': min([t.net_pnl for t in losing_trades]) if losing_trades else 0.0,
            })
            
            # Profit factor
            total_wins = sum(t.net_pnl for t in winning_trades)
            total_losses = abs(sum(t.net_pnl for t in losing_trades))
            metrics['profit_factor'] = total_wins / total_losses if total_losses > 0 else float('inf')
        else:
            metrics.update({
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'average_win': 0.0,
                'average_loss': 0.0,
                'largest_win': 0.0,
                'largest_loss': 0.0,
                'profit_factor': 0.0
            })
        
        return metrics
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get comprehensive portfolio summary."""
        summary = {
            'portfolio_state': {
                'cash': self.cash,
                'positions': dict(self.positions),
                'total_value': self.total_portfolio_value,
                'positions_value': self.total_positions_value
            },
            'orders': {
                'total_orders': len(self.orders),
                'pending_orders': len(self.pending_orders),
                'filled_orders': len(self.filled_orders)
            },
            'trades': {
                'total_trades': len(self.trades),
                'completed_trades': len(self.trades)
            },
            'performance': self.get_performance_metrics()
        }
        
        return summary
    
    def reset(self) -> None:
        """Reset portfolio to initial state."""
        self.cash = self.initial_capital
        self.positions.clear()
        self.orders.clear()
        self.pending_orders.clear()
        self.filled_orders.clear()
        self.trades.clear()
        self.equity_curve.clear()
        self.drawdown_curve.clear()
        self.peak_value = self.initial_capital
        self.max_drawdown = 0.0
        self.total_commission_paid = 0.0
        self.last_update_time = None
        self._positions_value = 0.0
        
        self.logger.info("Portfolio reset to initial state")
    
    def __str__(self) -> str:
        """String representation of portfolio."""
        return (f"Portfolio(value=${self.total_portfolio_value:,.2f}, "
                f"cash=${self.cash:,.2f}, positions={len(self.positions)}, "
                f"return={self.total_return_percent:.2f}%)")
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return (f"Portfolio(initial=${self.initial_capital:,.2f}, "
                f"current=${self.total_portfolio_value:,.2f}, "
                f"positions={dict(self.positions)}, trades={len(self.trades)})") 