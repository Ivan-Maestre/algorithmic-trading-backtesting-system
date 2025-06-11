"""
Backtesting Engine Module

Main backtesting engine that coordinates strategy execution, portfolio management,
and performance tracking for algorithmic trading strategies.
"""

from typing import Dict, List, Optional, Any, Union
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import uuid

from .portfolio import Portfolio
from .order import Order, OrderType, OrderStatus
from .trade import Trade
from ..strategies.base_strategy import BaseStrategy
from ..config import Config


class BacktestEngine:
    """
    Main backtesting engine for algorithmic trading strategies.
    
    Coordinates strategy execution, portfolio management, order processing,
    and performance tracking in an event-driven simulation.
    """
    
    def __init__(self, 
                 initial_capital: float = None,
                 commission_rate: float = None,
                 slippage: float = None):
        """
        Initialize the backtesting engine.
        
        Args:
            initial_capital: Starting capital (uses config default if None)
            commission_rate: Commission rate per trade (uses config default if None)
            slippage: Slippage factor (uses config default if None)
        """
        self.initial_capital = initial_capital or Config.INITIAL_CAPITAL
        self.commission_rate = commission_rate or Config.TRANSACTION_COST
        self.slippage = slippage or Config.SLIPPAGE
        
        # Initialize portfolio
        self.portfolio = Portfolio(self.initial_capital, self.commission_rate)
        
        # Strategy and data
        self.strategy = None
        self.data = None
        self.symbol = None
        
        # Backtesting state
        self.current_time = None
        self.current_data = None
        self.is_running = False
        self.start_time = None
        self.end_time = None
        
        # Results tracking
        self.backtest_results = {}
        self.execution_log = []
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"BacktestEngine initialized with ${self.initial_capital:,.2f} capital")
    
    def add_strategy(self, strategy: BaseStrategy) -> None:
        """
        Add a trading strategy to the backtesting engine.
        
        Args:
            strategy: Strategy instance to backtest
        """
        self.strategy = strategy
        self.logger.info(f"Strategy added: {strategy.name}")
    
    def add_data(self, data: pd.DataFrame, symbol: str = None) -> None:
        """
        Add market data for backtesting.
        
        Args:
            data: DataFrame with OHLCV data and indicators
            symbol: Trading symbol (extracted from data if None)
        """
        self.data = data.copy()
        self.symbol = symbol or Config.DEFAULT_ASSET
        
        # Ensure data is sorted by timestamp
        self.data = self.data.sort_index()
        
        self.logger.info(f"Data added: {len(data)} periods for {self.symbol}")
        self.logger.info(f"Date range: {data.index[0]} to {data.index[-1]}")
    
    def run(self, 
            start_date: str = None, 
            end_date: str = None,
            save_results: bool = True) -> Dict[str, Any]:
        """
        Run the backtest.
        
        Args:
            start_date: Start date for backtesting (YYYY-MM-DD)
            end_date: End date for backtesting (YYYY-MM-DD)
            save_results: Whether to save results
            
        Returns:
            Dictionary with backtest results
        """
        if self.strategy is None:
            raise ValueError("No strategy added. Use add_strategy() first.")
        
        if self.data is None:
            raise ValueError("No data added. Use add_data() first.")
        
        self.logger.info("Starting backtest execution")
        self.is_running = True
        
        try:
            # Filter data by date range if specified
            test_data = self._filter_data_by_date(start_date, end_date)
            
            # Initialize strategy with data
            self.strategy.initialize(test_data)
            
            # Reset portfolio
            self.portfolio.reset()
            
            # Execute backtest
            self._execute_backtest(test_data)
            
            # Calculate final results
            results = self._calculate_results()
            
            if save_results:
                self.backtest_results = results
            
            self.logger.info("Backtest completed successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"Backtest execution failed: {e}")
            self.is_running = False
            raise
        
        finally:
            self.is_running = False
    
    def _filter_data_by_date(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Filter data by date range."""
        data = self.data.copy()
        
        if start_date:
            start_dt = pd.to_datetime(start_date)
            data = data[data.index >= start_dt]
            self.start_time = start_dt
        else:
            self.start_time = data.index[0]
        
        if end_date:
            end_dt = pd.to_datetime(end_date)
            data = data[data.index <= end_dt]
            self.end_time = end_dt
        else:
            self.end_time = data.index[-1]
        
        self.logger.info(f"Filtered data: {len(data)} periods from {self.start_time} to {self.end_time}")
        return data
    
    def _execute_backtest(self, data: pd.DataFrame) -> None:
        """Execute the main backtesting loop."""
        self.logger.info("Executing backtesting loop")
        
        for timestamp, row in data.iterrows():
            self.current_time = timestamp
            self.current_data = row
            
            # Update portfolio with current prices
            current_prices = {self.symbol: row['close']}
            self.portfolio.update_portfolio(timestamp, current_prices)
            
            # Process pending orders
            filled_orders = self.portfolio.process_pending_orders(row, self.symbol)
            
            # Log filled orders
            for order in filled_orders:
                self._log_execution(f"Order filled: {order}")
            
            # Generate trading signal
            signal = self.strategy.generate_signal(data, timestamp)
            
            # Process signal if not HOLD
            if signal.signal_type.name != 'HOLD':
                self._process_trading_signal(signal, row)
            
            # Check for position exits
            self._check_position_exits(row)
        
        self.logger.info(f"Backtesting loop completed. Processed {len(data)} periods.")
    
    def _process_trading_signal(self, signal, current_data: pd.Series) -> None:
        """
        Process a trading signal by creating and placing orders.
        
        Args:
            signal: Trading signal from strategy
            current_data: Current market data
        """
        current_price = current_data['close']
        
        try:
            # Calculate position size
            position_size = self.strategy.calculate_position_size(
                self.data, signal, self.portfolio.total_portfolio_value
            )
            
            if position_size <= 0:
                self.logger.warning(f"Invalid position size: {position_size}")
                return
            
            # Create order
            order = Order(
                order_id=self._generate_order_id(),
                timestamp=signal.timestamp,
                symbol=self.symbol,
                order_type=OrderType.MARKET,  # Use market orders for simplicity
                side=signal.signal_type.name.lower(),
                quantity=position_size,
                strategy_id=self.strategy.name,
                metadata={
                    'signal_confidence': signal.confidence,
                    'signal_price': signal.price,
                    'current_price': current_price
                }
            )
            
            # Place order
            if self.portfolio.place_order(order):
                self._log_execution(f"Signal processed: {signal.signal_type.name} order placed")
            else:
                self._log_execution(f"Signal rejected: {signal.signal_type.name} order failed")
                
        except Exception as e:
            self.logger.error(f"Error processing signal: {e}")
            self._log_execution(f"Signal processing error: {e}")
    
    def _check_position_exits(self, current_data: pd.Series) -> None:
        """
        Check if current positions should be exited based on strategy rules.
        
        Args:
            current_data: Current market data
        """
        if not self.portfolio.has_position(self.symbol):
            return
        
        current_price = current_data['close']
        
        try:
            # Check if strategy wants to exit position
            should_exit = self.strategy.should_exit_position(current_price, self.current_time)
            
            if should_exit:
                # Create exit order
                current_position = self.portfolio.get_position(self.symbol)
                
                # Determine exit order side
                exit_side = 'sell' if current_position > 0 else 'buy'
                exit_quantity = abs(current_position)
                
                exit_order = Order(
                    order_id=self._generate_order_id(),
                    timestamp=self.current_time,
                    symbol=self.symbol,
                    order_type=OrderType.MARKET,
                    side=exit_side,
                    quantity=exit_quantity,
                    strategy_id=self.strategy.name,
                    metadata={
                        'exit_reason': 'strategy_exit',
                        'current_price': current_price
                    }
                )
                
                if self.portfolio.place_order(exit_order):
                    self._log_execution(f"Exit order placed: {exit_side} {exit_quantity}")
                
        except Exception as e:
            self.logger.error(f"Error checking position exits: {e}")
    
    def _generate_order_id(self) -> str:
        """Generate unique order ID."""
        return f"{self.symbol}_{self.current_time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    
    def _log_execution(self, message: str) -> None:
        """Log execution event."""
        log_entry = {
            'timestamp': self.current_time,
            'message': message,
            'portfolio_value': self.portfolio.total_portfolio_value,
            'cash': self.portfolio.cash,
            'positions': dict(self.portfolio.positions)
        }
        self.execution_log.append(log_entry)
        self.logger.debug(f"{self.current_time}: {message}")
    
    def _calculate_results(self) -> Dict[str, Any]:
        """Calculate comprehensive backtest results."""
        self.logger.info("Calculating backtest results")
        
        # Get portfolio performance metrics
        performance_metrics = self.portfolio.get_performance_metrics()
        
        # Get equity and drawdown curves
        equity_curve = self.portfolio.get_equity_curve()
        drawdown_curve = self.portfolio.get_drawdown_curve()
        
        # Strategy-specific metrics
        strategy_info = self.strategy.get_strategy_info()
        
        # Compile results
        results = {
            'backtest_info': {
                'strategy_name': self.strategy.name,
                'symbol': self.symbol,
                'start_date': self.start_time,
                'end_date': self.end_time,
                'initial_capital': self.initial_capital,
                'commission_rate': self.commission_rate,
                'slippage': self.slippage,
                'data_points': len(self.data)
            },
            'performance': performance_metrics,
            'strategy': strategy_info,
            'portfolio': self.portfolio.get_portfolio_summary(),
            'equity_curve': equity_curve,
            'drawdown_curve': drawdown_curve,
            'trades': [trade.to_dict() for trade in self.portfolio.trades],
            'orders': [order.to_dict() for order in self.portfolio.orders],
            'execution_log': self.execution_log
        }
        
        # Add derived metrics
        results['summary'] = self._create_summary(results)
        
        return results
    
    def _create_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Create executive summary of backtest results."""
        perf = results['performance']
        
        summary = {
            'total_return_pct': perf.get('total_return_percent', 0.0),
            'annualized_return_pct': perf.get('annualized_return', 0.0),
            'max_drawdown_pct': perf.get('max_drawdown', 0.0),
            'sharpe_ratio': perf.get('sharpe_ratio', 0.0),
            'win_rate_pct': perf.get('win_rate', 0.0),
            'total_trades': perf.get('total_trades', 0),
            'profit_factor': perf.get('profit_factor', 0.0),
            'final_value': perf.get('final_value', 0.0),
            'total_commission': perf.get('total_commission', 0.0)
        }
        
        # Performance rating
        if summary['total_return_pct'] > 0 and summary['sharpe_ratio'] > 1.0:
            summary['performance_rating'] = 'Excellent'
        elif summary['total_return_pct'] > 0 and summary['sharpe_ratio'] > 0.5:
            summary['performance_rating'] = 'Good'
        elif summary['total_return_pct'] > 0:
            summary['performance_rating'] = 'Fair'
        else:
            summary['performance_rating'] = 'Poor'
        
        return summary
    
    def get_trade_analysis(self) -> pd.DataFrame:
        """Get detailed trade analysis as DataFrame."""
        if not self.portfolio.trades:
            return pd.DataFrame()
        
        trade_data = [trade.to_dict() for trade in self.portfolio.trades]
        df = pd.DataFrame(trade_data)
        
        # Set entry_time as index
        df.set_index('entry_time', inplace=True)
        
        return df
    
    def get_daily_returns(self) -> pd.Series:
        """Calculate daily returns from equity curve."""
        equity_curve = self.portfolio.get_equity_curve()
        
        if equity_curve.empty:
            return pd.Series(dtype=float)
        
        # Resample to daily frequency and calculate returns
        daily_equity = equity_curve.resample('D').last().ffill()
        daily_returns = daily_equity.pct_change().dropna()
        
        return daily_returns
    
    def save_results(self, filepath: str) -> None:
        """
        Save backtest results to file.
        
        Args:
            filepath: Path to save results (CSV, JSON, or pickle)
        """
        if not self.backtest_results:
            raise ValueError("No results to save. Run backtest first.")
        
        import json
        
        # Convert pandas objects to serializable format
        serializable_results = self._make_serializable(self.backtest_results)
        
        if filepath.endswith('.json'):
            with open(filepath, 'w') as f:
                json.dump(serializable_results, f, indent=2, default=str)
        elif filepath.endswith('.csv'):
            # Save key metrics as CSV
            summary_df = pd.DataFrame([serializable_results['summary']])
            summary_df.to_csv(filepath, index=False)
        else:
            # Default to pickle
            import pickle
            with open(filepath, 'wb') as f:
                pickle.dump(self.backtest_results, f)
        
        self.logger.info(f"Results saved to {filepath}")
    
    def _make_serializable(self, obj: Any) -> Any:
        """Convert pandas objects to serializable format."""
        if isinstance(obj, pd.Series):
            return obj.to_dict()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        else:
            return obj
    
    def reset(self) -> None:
        """Reset the backtesting engine to initial state."""
        self.portfolio.reset()
        if self.strategy:
            self.strategy.reset()
        
        self.current_time = None
        self.current_data = None
        self.backtest_results.clear()
        self.execution_log.clear()
        
        self.logger.info("BacktestEngine reset to initial state")
    
    def __str__(self) -> str:
        """String representation of the backtest engine."""
        return (f"BacktestEngine(capital=${self.initial_capital:,.2f}, "
                f"strategy={self.strategy.name if self.strategy else None}, "
                f"symbol={self.symbol})")
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return (f"BacktestEngine(initial_capital={self.initial_capital}, "
                f"commission_rate={self.commission_rate}, "
                f"strategy={self.strategy.name if self.strategy else None})") 