"""
Main Application Entry Point for the Algorithmic Trading Backtesting System.

This module serves as the entry point for running backtests and provides
a simple interface for testing the system setup.
"""

import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.config import Config
from src.strategies import BaseStrategy, SignalType


def test_system_setup():
    """Test basic system setup and configuration."""
    print(f"🚀 {Config.PROJECT_NAME} v{Config.VERSION}")
    print("=" * 60)
    
    # Test configuration
    print("📋 Configuration Test:")
    print(f"   Data Period: {Config.DATA_START_DATE} to {Config.DATA_END_DATE}")
    print(f"   Timeframe: {Config.TIMEFRAME}")
    print(f"   Initial Capital: ${Config.INITIAL_CAPITAL:,.2f}")
    print(f"   Default Asset: {Config.DEFAULT_ASSET}")
    
    # Test configuration validation
    is_valid = Config.validate_config()
    print(f"   Configuration Valid: {'✅' if is_valid else '❌'}")
    
    # Test directory creation
    print("\n📁 Directory Setup:")
    directories = [Config.DATA_DIR, Config.RESULTS_DIR, Config.LOGS_DIR]
    for directory in directories:
        exists = os.path.exists(directory)
        print(f"   {directory}: {'✅' if exists else '❌'}")
    
    # Test strategy configuration
    print("\n🎯 Strategy Configuration:")
    strategies = ['ema_crossover', 'momentum', 'q_learning']
    for strategy_name in strategies:
        config = Config.get_strategy_config(strategy_name)
        print(f"   {strategy_name.replace('_', ' ').title()}: {'✅' if config else '❌'}")
    
    # Test imports
    print("\n🔧 Module Import Test:")
    try:
        from src.strategies.base_strategy import BaseStrategy, SignalType, TradingSignal, Position
        print("   Strategy modules: ✅")
    except ImportError as e:
        print(f"   Strategy modules: ❌ ({e})")
    
    print("\n" + "=" * 60)
    
    if is_valid and all(os.path.exists(d) for d in directories):
        print("✅ System setup complete and ready for development!")
        return True
    else:
        print("❌ System setup has issues that need to be resolved.")
        return False


def display_development_status():
    """Display current development status and next steps."""
    print("\n📊 Development Status:")
    print("=" * 40)
    
    # Phase 1 status
    phase1_tasks = [
        ("Project structure", True),
        ("Documentation setup", True),
        ("Configuration system", True),
        ("Base strategy class", True),
        ("Dependencies defined", True),
        ("Data management", False),
        ("Basic architecture", False),
    ]
    
    print("Phase 1 - Foundation Setup:")
    for task, completed in phase1_tasks:
        status = "✅" if completed else "⏳"
        print(f"  {status} {task}")
    
    completed_tasks = sum(1 for _, completed in phase1_tasks if completed)
    total_tasks = len(phase1_tasks)
    progress = (completed_tasks / total_tasks) * 100
    
    print(f"\nPhase 1 Progress: {progress:.1f}% ({completed_tasks}/{total_tasks})")
    
    print("\n🎯 Next Steps:")
    print("  1. Set up data ingestion module")
    print("  2. Implement EMA calculation utilities")
    print("  3. Create EMA Crossover strategy")
    print("  4. Develop basic backtesting engine")
    print("  5. Run initial strategy validation")


def main():
    """Main application function."""
    print("🔧 Testing System Setup...")
    print()
    
    # Test system setup
    setup_success = test_system_setup()
    
    if setup_success:
        # Display development status
        display_development_status()
        
        print("\n" + "=" * 60)
        print("🚀 Ready to begin EMA Crossover strategy implementation!")
        print("   Run development tasks from the checklist to continue.")
    else:
        print("\n❌ Please resolve setup issues before continuing.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 