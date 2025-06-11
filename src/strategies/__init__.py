"""
Trading Strategies Module

Contains all strategy implementations for the algorithmic trading system.
"""

from .base_strategy import BaseStrategy, TradingSignal, SignalType, Position
from .ema_crossover import EMACrossoverStrategy
from .simple_ema_crossover import SimpleEMACrossoverStrategy
from .momentum import MomentumStrategy
from .q_learning import QLearningStrategy

__all__ = [
    'BaseStrategy',
    'TradingSignal', 
    'SignalType',
    'Position',
    'EMACrossoverStrategy',
    'SimpleEMACrossoverStrategy',
    'MomentumStrategy',
    'QLearningStrategy'
] 