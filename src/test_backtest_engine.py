"""
Test Backtesting Engine

Test script to validate the complete backtesting engine with EMA Crossover strategy.
"""

import sys
import logging
from pathlib import Path
import pandas as pd

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.config import Config
from src.data import BinanceDataFetcher, DataProcessor
from src.strategies import EMACrossoverStrategy
from src.backtesting import BacktestEngine
from src.backtesting.order import OrderType
import matplotlib.pyplot as plt


def setup_logging():
    """Set up logging for the test."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def get_test_data(quick_test=True):
    """Get processed data for backtesting."""
    print("📊 Fetching data for backtesting...")
    
    try:
        # Fetch data from Binance
        fetcher = BinanceDataFetcher()
        
        if quick_test:
            # Quick test with 1 month of data
            raw_data = fetcher.fetch_ohlcv_data(
                symbol="BTCUSDT",
                timeframe="4h", 
                start_date="2023-06-01",
                end_date="2023-07-01"
            )
        else:
            # Full test with 6 months
            raw_data = fetcher.fetch_ohlcv_data(
                symbol="BTCUSDT",
                timeframe="4h",
                start_date="2023-01-01", 
                end_date="2023-07-01"
            )
        
        # Process data for EMA strategy
        processor = DataProcessor()
        processed_data = processor.prepare_strategy_data(raw_data, "ema_crossover")
        
        print(f"✅ Data ready: {processed_data.shape}")
        print(f"   Date range: {processed_data.index.min()} to {processed_data.index.max()}")
        print(f"   Price range: ${processed_data['close'].min():.2f} - ${processed_data['close'].max():.2f}")
        
        return processed_data
        
    except Exception as e:
        print(f"❌ Error getting test data: {e}")
        return None


def test_backtest_engine_initialization():
    """Test backtesting engine initialization."""
    print("\n🔧 Testing Backtest Engine Initialization...")
    
    try:
        # Test with default parameters
        engine = BacktestEngine()
        print(f"✅ Engine created: {engine}")
        
        # Test with custom parameters
        custom_engine = BacktestEngine(
            initial_capital=50000,
            commission_rate=0.001,
            slippage=0.0005
        )
        print(f"✅ Custom engine created: {custom_engine}")
        
        # Test strategy addition
        strategy = EMACrossoverStrategy()
        engine.add_strategy(strategy)
        print(f"✅ Strategy added: {strategy.name}")
        
        return engine
        
    except Exception as e:
        print(f"❌ Engine initialization error: {e}")
        return None


def test_data_addition(engine, data):
    """Test adding data to the backtesting engine."""
    print("\n📊 Testing Data Addition...")
    
    try:
        # Add data to engine
        engine.add_data(data, symbol="BTCUSDT")
        print(f"✅ Data added successfully")
        print(f"   Symbol: {engine.symbol}")
        print(f"   Data points: {len(engine.data)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data addition error: {e}")
        return False


def test_full_backtest(engine, quick_test=True):
    """Test complete backtesting execution."""
    print("\n🚀 Testing Full Backtest Execution...")
    
    try:
        # Run backtest
        if quick_test:
            # Test on subset of data
            results = engine.run(
                start_date="2023-06-01",
                end_date="2023-06-15"
            )
        else:
            # Run on full dataset
            results = engine.run()
        
        print(f"✅ Backtest completed successfully")
        
        # Display key results
        summary = results['summary']
        performance = results['performance']
        
        print(f"\n📈 Backtest Results Summary:")
        print(f"   • Strategy: {results['backtest_info']['strategy_name']}")
        print(f"   • Period: {results['backtest_info']['start_date']} to {results['backtest_info']['end_date']}")
        print(f"   • Initial Capital: ${results['backtest_info']['initial_capital']:,.2f}")
        print(f"   • Final Value: ${summary['final_value']:,.2f}")
        print(f"   • Total Return: {summary['total_return_pct']:.2f}%")
        print(f"   • Annualized Return: {summary['annualized_return_pct']:.2f}%")
        print(f"   • Max Drawdown: {summary['max_drawdown_pct']:.2f}%")
        print(f"   • Sharpe Ratio: {summary['sharpe_ratio']:.3f}")
        print(f"   • Total Trades: {summary['total_trades']}")
        print(f"   • Win Rate: {summary['win_rate_pct']:.1f}%")
        print(f"   • Profit Factor: {summary['profit_factor']:.2f}")
        print(f"   • Total Commission: ${summary['total_commission']:.2f}")
        print(f"   • Performance Rating: {summary['performance_rating']}")
        
        return results
        
    except Exception as e:
        print(f"❌ Backtest execution error: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_order_processing(engine):
    """Test order processing and trade execution."""
    print("\n📋 Testing Order Processing...")
    
    try:
        results = engine.backtest_results
        
        if not results:
            print("❌ No backtest results available")
            return False
        
        orders = results['orders']
        trades = results['trades']
        
        print(f"✅ Order processing analysis:")
        print(f"   • Total orders: {len(orders)}")
        
        if orders:
            filled_orders = [o for o in orders if o['status'] == 'filled']
            pending_orders = [o for o in orders if o['status'] == 'pending']
            rejected_orders = [o for o in orders if o['status'] == 'rejected']
            
            print(f"   • Filled orders: {len(filled_orders)}")
            print(f"   • Pending orders: {len(pending_orders)}")
            print(f"   • Rejected orders: {len(rejected_orders)}")
            
            # Show sample orders
            if filled_orders:
                sample_order = filled_orders[0]
                print(f"   • Sample filled order:")
                print(f"     - ID: {sample_order['order_id']}")
                print(f"     - Side: {sample_order['side']}")
                print(f"     - Quantity: {sample_order['quantity']:.6f}")
                print(f"     - Price: ${sample_order['filled_price']:.2f}")
                print(f"     - Commission: ${sample_order['commission']:.2f}")
        
        print(f"✅ Trade execution analysis:")
        print(f"   • Total trades: {len(trades)}")
        
        if trades:
            winning_trades = [t for t in trades if t['is_winning']]
            losing_trades = [t for t in trades if not t['is_winning']]
            
            print(f"   • Winning trades: {len(winning_trades)}")
            print(f"   • Losing trades: {len(losing_trades)}")
            
            if trades:
                avg_trade_duration = sum(t['duration_hours'] for t in trades) / len(trades)
                print(f"   • Average trade duration: {avg_trade_duration:.1f} hours")
                
                # Show sample trade
                sample_trade = trades[0]
                print(f"   • Sample trade:")
                print(f"     - ID: {sample_trade['trade_id']}")
                print(f"     - Side: {sample_trade['side']}")
                print(f"     - Entry: ${sample_trade['entry_price']:.2f}")
                print(f"     - Exit: ${sample_trade['exit_price']:.2f}")
                print(f"     - PnL: ${sample_trade['net_pnl']:.2f} ({sample_trade['net_pnl_percent']:.2f}%)")
                print(f"     - Duration: {sample_trade['duration_hours']:.1f} hours")
        
        return True
        
    except Exception as e:
        print(f"❌ Order processing analysis error: {e}")
        return False


def test_performance_metrics(engine):
    """Test performance metrics calculation."""
    print("\n📊 Testing Performance Metrics...")
    
    try:
        results = engine.backtest_results
        
        if not results:
            print("❌ No backtest results available")
            return False
        
        performance = results['performance']
        
        print(f"✅ Performance metrics validation:")
        
        # Validate key metrics exist
        required_metrics = [
            'initial_capital', 'final_value', 'total_return', 'total_return_percent',
            'max_drawdown', 'sharpe_ratio', 'total_trades', 'win_rate'
        ]
        
        missing_metrics = []
        for metric in required_metrics:
            if metric not in performance:
                missing_metrics.append(metric)
            else:
                print(f"   • {metric}: {performance[metric]}")
        
        if missing_metrics:
            print(f"❌ Missing metrics: {missing_metrics}")
            return False
        
        # Validate metric ranges
        validations = [
            ('total_return_percent', lambda x: -100 <= x <= 1000, "Should be between -100% and 1000%"),
            ('max_drawdown', lambda x: 0 <= x <= 100, "Should be between 0% and 100%"),
            ('win_rate', lambda x: 0 <= x <= 100, "Should be between 0% and 100%"),
            ('sharpe_ratio', lambda x: -10 <= x <= 10, "Should be reasonable range")
        ]
        
        for metric, validator, description in validations:
            if metric in performance:
                value = performance[metric]
                if not validator(value):
                    print(f"⚠️  {metric} value {value} may be invalid: {description}")
                else:
                    print(f"   ✅ {metric} validation passed")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance metrics error: {e}")
        return False


def test_equity_curve(engine):
    """Test equity curve generation."""
    print("\n📈 Testing Equity Curve...")
    
    try:
        results = engine.backtest_results
        
        if not results:
            print("❌ No backtest results available")
            return False
        
        equity_curve = results['equity_curve']
        drawdown_curve = results['drawdown_curve']
        
        if isinstance(equity_curve, dict):
            # Convert from dict format
            equity_values = list(equity_curve.values())
            drawdown_values = list(drawdown_curve.values())
        else:
            equity_values = equity_curve.tolist() if hasattr(equity_curve, 'tolist') else []
            drawdown_values = drawdown_curve.tolist() if hasattr(drawdown_curve, 'tolist') else []
        
        print(f"✅ Equity curve analysis:")
        print(f"   • Data points: {len(equity_values)}")
        
        if equity_values:
            print(f"   • Starting value: ${equity_values[0]:,.2f}")
            print(f"   • Ending value: ${equity_values[-1]:,.2f}")
            print(f"   • Peak value: ${max(equity_values):,.2f}")
            print(f"   • Minimum value: ${min(equity_values):,.2f}")
        
        if drawdown_values:
            print(f"   • Max drawdown observed: {max(drawdown_values):.2f}%")
        
        return True
        
    except Exception as e:
        print(f"❌ Equity curve error: {e}")
        return False


def test_portfolio_state(engine):
    """Test portfolio state management."""
    print("\n💼 Testing Portfolio State...")
    
    try:
        portfolio = engine.portfolio
        
        print(f"✅ Portfolio state analysis:")
        print(f"   • Initial capital: ${portfolio.initial_capital:,.2f}")
        print(f"   • Current cash: ${portfolio.cash:,.2f}")
        print(f"   • Current positions: {dict(portfolio.positions)}")
        print(f"   • Total portfolio value: ${portfolio.total_portfolio_value:,.2f}")
        print(f"   • Total return: {portfolio.total_return_percent:.2f}%")
        print(f"   • Max drawdown: {portfolio.max_drawdown:.2f}%")
        print(f"   • Total commission paid: ${portfolio.total_commission_paid:.2f}")
        
        # Portfolio summary
        summary = portfolio.get_portfolio_summary()
        print(f"   • Orders summary: {summary['orders']}")
        print(f"   • Trades summary: {summary['trades']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Portfolio state error: {e}")
        return False


def create_simple_visualizations(engine):
    """Create basic performance visualizations."""
    print("\n📊 Creating Visualizations...")
    
    try:
        results = engine.backtest_results
        
        if not results:
            print("❌ No results to visualize")
            return False
        
        # Get equity curve data
        equity_curve = results['equity_curve']
        
        if isinstance(equity_curve, dict):
            timestamps = list(equity_curve.keys())
            values = list(equity_curve.values())
            
            # Convert timestamps if they're strings
            if timestamps and isinstance(timestamps[0], str):
                timestamps = pd.to_datetime(timestamps)
            
            # Create simple plot
            plt.figure(figsize=(12, 6))
            
            # Equity curve
            plt.subplot(2, 1, 1)
            plt.plot(timestamps, values, 'b-', linewidth=2)
            plt.title('Portfolio Equity Curve')
            plt.ylabel('Portfolio Value ($)')
            plt.grid(True, alpha=0.3)
            
            # Performance summary
            plt.subplot(2, 1, 2)
            summary = results['summary']
            metrics = ['total_return_pct', 'max_drawdown_pct', 'win_rate_pct']
            values_metric = [summary[m] for m in metrics]
            labels = ['Total Return %', 'Max Drawdown %', 'Win Rate %']
            
            plt.bar(labels, values_metric, color=['green', 'red', 'blue'])
            plt.title('Key Performance Metrics')
            plt.ylabel('Percentage')
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            plt.savefig('backtest_results.png', dpi=300, bbox_inches='tight')
            print(f"✅ Visualization saved as 'backtest_results.png'")
            
            return True
        else:
            print("❌ Equity curve data format not supported for visualization")
            return False
        
    except Exception as e:
        print(f"❌ Visualization error: {e}")
        return False


def main(quick_test=True):
    """Main test function."""
    print("🧪 Backtest Engine Test Suite")
    if quick_test:
        print("   📝 Running in QUICK TEST mode")
    print("=" * 60)
    
    setup_logging()
    
    # Test 1: Get test data
    data = get_test_data(quick_test=quick_test)
    if data is None:
        print("\n❌ Failed to get test data")
        return False
    
    # Test 2: Engine initialization
    engine = test_backtest_engine_initialization()
    if engine is None:
        print("\n❌ Engine initialization failed")
        return False
    
    # Test 3: Data addition
    if not test_data_addition(engine, data):
        print("\n❌ Data addition failed")
        return False
    
    # Test 4: Full backtest execution
    results = test_full_backtest(engine, quick_test=quick_test)
    if results is None:
        print("\n❌ Backtest execution failed")
        return False
    
    # Test 5: Order processing analysis
    if not test_order_processing(engine):
        print("\n❌ Order processing analysis failed")
        return False
    
    # Test 6: Performance metrics
    if not test_performance_metrics(engine):
        print("\n❌ Performance metrics failed")
        return False
    
    # Test 7: Equity curve
    if not test_equity_curve(engine):
        print("\n❌ Equity curve test failed")
        return False
    
    # Test 8: Portfolio state
    if not test_portfolio_state(engine):
        print("\n❌ Portfolio state test failed")
        return False
    
    # Test 9: Visualizations
    try:
        create_simple_visualizations(engine)
    except Exception as e:
        print(f"⚠️  Visualization creation failed: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("🎉 Backtest Engine Test Results:")
    print(f"   • Engine initialization: ✅")
    print(f"   • Data processing: ✅")
    print(f"   • Strategy execution: ✅")
    print(f"   • Order management: ✅")
    print(f"   • Performance calculation: ✅")
    print(f"   • Portfolio tracking: ✅")
    
    if results:
        summary = results['summary']
        print(f"\n🎯 Final Performance Summary:")
        print(f"   • Total Return: {summary['total_return_pct']:.2f}%")
        print(f"   • Max Drawdown: {summary['max_drawdown_pct']:.2f}%")
        print(f"   • Sharpe Ratio: {summary['sharpe_ratio']:.3f}")
        print(f"   • Win Rate: {summary['win_rate_pct']:.1f}%")
        print(f"   • Total Trades: {summary['total_trades']}")
        print(f"   • Performance Rating: {summary['performance_rating']}")
    
    return True


if __name__ == "__main__":
    # Check command line argument for test mode
    quick_mode = "--full" not in sys.argv
    success = main(quick_test=quick_mode)
    
    if quick_mode:
        print("\n💡 Tip: Run with '--full' for complete backtest test with more data")
    
    sys.exit(0 if success else 1) 