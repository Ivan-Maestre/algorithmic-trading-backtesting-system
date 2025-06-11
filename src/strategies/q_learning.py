"""
Q-Learning Reinforcement Learning Strategy Implementation

Implements a Q-learning based trading strategy that learns optimal trading decisions
through trial and error using state-action-reward cycles. The agent learns to
maximize cumulative rewards by exploring different trading actions.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List, Optional
from datetime import datetime
import pickle
import json

from .base_strategy import BaseStrategy, TradingSignal, SignalType
from ..config import Config


class QLearningStrategy(BaseStrategy):
    """
    Q-Learning Reinforcement Learning Strategy for algorithmic trading.
    
    This strategy uses Q-learning to learn optimal trading policies by:
    - Defining market states based on technical indicators
    - Learning Q-values for state-action pairs through experience
    - Balancing exploration vs exploitation during learning
    - Making trading decisions based on learned Q-table
    
    Key Features:
    - Discrete state space representation of market conditions
    - Three actions: BUY, SELL, HOLD
    - Reward function based on returns and risk
    - Configurable learning parameters
    - Q-table persistence for reuse
    """
    
    def __init__(self, parameters: Dict[str, Any] = None):
        """
        Initialize the Q-Learning strategy.
        
        Args:
            parameters: Strategy parameters (optional, uses config defaults)
        """
        # Set default parameters from config
        default_params = Config.get_strategy_config('q_learning')
        super().__init__("Q-Learning RL Strategy", parameters or default_params)
        
        # Q-Learning parameters
        self.learning_rate = self.get_parameter('learning_rate', 0.01)
        self.discount_factor = self.get_parameter('discount_factor', 0.95)
        self.exploration_rate = self.get_parameter('exploration_rate', 0.1)
        self.exploration_decay = self.get_parameter('exploration_decay', 0.995)
        self.min_exploration_rate = self.get_parameter('min_exploration_rate', 0.01)
        
        # Strategy parameters
        self.position_size = self.get_parameter('position_size', 0.1)
        self.stop_loss_pct = self.get_parameter('stop_loss', 0.02)
        
        # State space configuration
        self.state_features = self.get_parameter('state_features', 
            ['price_trend', 'volume_trend', 'volatility_level', 'momentum_level'])
        self.n_price_levels = self.get_parameter('n_price_levels', 5)
        self.n_volume_levels = self.get_parameter('n_volume_levels', 3)
        self.n_volatility_levels = self.get_parameter('n_volatility_levels', 3)
        self.n_momentum_levels = self.get_parameter('n_momentum_levels', 5)
        
        # Ensure the base strategy has the correct parameter name
        self.parameters['stop_loss_pct'] = self.stop_loss_pct
        
        # Q-Learning state
        self.q_table = {}
        self.state_space_size = (self.n_price_levels * self.n_volume_levels * 
                               self.n_volatility_levels * self.n_momentum_levels)
        self.action_space = [SignalType.BUY, SignalType.SELL, SignalType.HOLD]
        self.action_to_index = {action: i for i, action in enumerate(self.action_space)}
        
        # Training state
        self.is_training = True
        self.training_history = []
        self.last_state = None
        self.last_action = None
        self.last_reward = 0.0
        self.episode_rewards = []
        self.training_steps = 0
        
        # Performance tracking
        self.total_episodes = 0
        self.total_reward = 0.0
        self.exploration_actions = 0
        self.exploitation_actions = 0
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Q-Learning strategy initialized with {self.state_space_size} states, "
                        f"learning_rate={self.learning_rate}, exploration_rate={self.exploration_rate}")
    
    def initialize(self, data: pd.DataFrame) -> None:
        """
        Initialize the strategy with historical data.
        
        Args:
            data: DataFrame with OHLCV data and technical indicators
        """
        self.logger.info("Initializing Q-Learning strategy")
        
        # Validate required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns for Q-Learning strategy: {missing_columns}")
        
        # Store data reference
        self.data = data.copy()
        
        # Calculate additional features if not present
        self._calculate_state_features()
        
        # Initialize Q-table if empty
        if not self.q_table:
            self._initialize_q_table()
        
        # Reset strategy state
        self.reset()
        self.is_initialized = True
        
        self.logger.info(f"Strategy initialized with {len(data)} data points")
        self.logger.info(f"Q-table size: {len(self.q_table)} state-action pairs")
    
    def _calculate_state_features(self) -> None:
        """Calculate features needed for state representation."""
        # Price trend (returns over different periods)
        self.data['return_1'] = self.data['close'].pct_change(1)
        self.data['return_5'] = self.data['close'].pct_change(5)
        self.data['return_20'] = self.data['close'].pct_change(20)
        
        # Volume trend
        self.data['volume_ma'] = self.data['volume'].rolling(window=20).mean()
        self.data['volume_ratio'] = self.data['volume'] / self.data['volume_ma']
        
        # Volatility (rolling standard deviation of returns)
        self.data['volatility'] = self.data['return_1'].rolling(window=20).std()
        
        # Momentum (rate of change)
        self.data['momentum'] = self.data['close'].pct_change(periods=10)
        
        # RSI for additional momentum
        if 'rsi' not in self.data.columns:
            self.data['rsi'] = self._calculate_rsi(self.data['close'], window=14)
        
        # Moving averages for trend
        self.data['ma_short'] = self.data['close'].rolling(window=10).mean()
        self.data['ma_long'] = self.data['close'].rolling(window=30).mean()
        self.data['ma_trend'] = (self.data['ma_short'] > self.data['ma_long']).astype(int)
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _get_state(self, data: pd.DataFrame, timestamp: pd.Timestamp) -> Tuple[int, ...]:
        """
        Convert market conditions to discrete state representation.
        
        Args:
            data: Market data
            timestamp: Current timestamp
            
        Returns:
            Tuple representing the discrete state
        """
        try:
            if timestamp not in data.index:
                # Return default state if timestamp not found
                return (0, 0, 0, 0)
            
            row = data.loc[timestamp]
            
            # Price trend state (based on returns)
            return_5 = row.get('return_5', 0)
            if pd.isna(return_5):
                price_state = 2  # Neutral
            elif return_5 > 0.05:
                price_state = 4  # Strong up
            elif return_5 > 0.02:
                price_state = 3  # Up
            elif return_5 > -0.02:
                price_state = 2  # Neutral
            elif return_5 > -0.05:
                price_state = 1  # Down
            else:
                price_state = 0  # Strong down
            
            # Volume state
            volume_ratio = row.get('volume_ratio', 1.0)
            if pd.isna(volume_ratio):
                volume_state = 1  # Normal
            elif volume_ratio > 1.5:
                volume_state = 2  # High volume
            elif volume_ratio > 0.7:
                volume_state = 1  # Normal volume
            else:
                volume_state = 0  # Low volume
            
            # Volatility state
            volatility = row.get('volatility', 0.02)
            if pd.isna(volatility):
                volatility_state = 1  # Normal
            elif volatility > 0.05:
                volatility_state = 2  # High volatility
            elif volatility > 0.02:
                volatility_state = 1  # Normal volatility
            else:
                volatility_state = 0  # Low volatility
            
            # Momentum state (based on RSI and momentum)
            rsi = row.get('rsi', 50)
            momentum = row.get('momentum', 0)
            if pd.isna(rsi) or pd.isna(momentum):
                momentum_state = 2  # Neutral
            elif rsi > 70 and momentum > 0.03:
                momentum_state = 4  # Strong overbought
            elif rsi > 60 or momentum > 0.015:
                momentum_state = 3  # Overbought
            elif 40 <= rsi <= 60 and abs(momentum) <= 0.015:
                momentum_state = 2  # Neutral
            elif rsi < 40 or momentum < -0.015:
                momentum_state = 1  # Oversold
            else:
                momentum_state = 0  # Strong oversold
            
            return (price_state, volume_state, volatility_state, momentum_state)
            
        except Exception as e:
            self.logger.error(f"Error getting state for {timestamp}: {e}")
            return (0, 0, 0, 0)  # Default state
    
    def _initialize_q_table(self) -> None:
        """Initialize Q-table with default values."""
        self.logger.info("Initializing Q-table with default values")
        
        # Initialize with small random values to break symmetry
        for price_state in range(self.n_price_levels):
            for volume_state in range(self.n_volume_levels):
                for volatility_state in range(self.n_volatility_levels):
                    for momentum_state in range(self.n_momentum_levels):
                        state = (price_state, volume_state, volatility_state, momentum_state)
                        self.q_table[state] = np.random.normal(0, 0.01, len(self.action_space))
    
    def _get_q_values(self, state: Tuple[int, ...]) -> np.ndarray:
        """Get Q-values for a given state."""
        if state not in self.q_table:
            # Initialize new state with small random values
            self.q_table[state] = np.random.normal(0, 0.01, len(self.action_space))
        return self.q_table[state]
    
    def _choose_action(self, state: Tuple[int, ...]) -> SignalType:
        """
        Choose action using epsilon-greedy policy.
        
        Args:
            state: Current market state
            
        Returns:
            Action to take (BUY, SELL, HOLD)
        """
        if self.is_training and np.random.random() < self.exploration_rate:
            # Exploration: random action
            action = np.random.choice(self.action_space)
            self.exploration_actions += 1
        else:
            # Exploitation: best action based on Q-values
            q_values = self._get_q_values(state)
            best_action_index = np.argmax(q_values)
            action = self.action_space[best_action_index]
            self.exploitation_actions += 1
        
        return action
    
    def _calculate_reward(self, data: pd.DataFrame, current_time: pd.Timestamp, 
                         action: SignalType, last_price: float = None) -> float:
        """
        Calculate reward for the taken action.
        
        Args:
            data: Market data
            current_time: Current timestamp
            action: Action taken
            last_price: Price from previous period
            
        Returns:
            Reward value
        """
        try:
            if current_time not in data.index:
                return 0.0
            
            current_row = data.loc[current_time]
            current_price = current_row['close']
            
            if last_price is None:
                return 0.0
            
            # Calculate return
            price_return = (current_price - last_price) / last_price
            
            # Base reward is the directional correctness
            if action == SignalType.BUY:
                base_reward = price_return  # Positive if price went up
            elif action == SignalType.SELL:
                base_reward = -price_return  # Positive if price went down
            else:  # HOLD
                base_reward = -abs(price_return) * 0.1  # Small penalty for missed opportunities
            
            # Risk adjustment (penalize high volatility periods)
            volatility = current_row.get('volatility', 0.02)
            if not pd.isna(volatility):
                volatility_penalty = volatility * 0.5
                base_reward -= volatility_penalty
            
            # Volume confirmation bonus
            volume_ratio = current_row.get('volume_ratio', 1.0)
            if not pd.isna(volume_ratio) and volume_ratio > 1.2:
                if action != SignalType.HOLD:
                    base_reward += 0.001  # Small bonus for volume confirmation
            
            # Scale reward
            reward = base_reward * 100  # Scale to make rewards more significant
            
            return float(reward)
            
        except Exception as e:
            self.logger.error(f"Error calculating reward: {e}")
            return 0.0
    
    def _update_q_table(self, state: Tuple[int, ...], action: SignalType, 
                       reward: float, next_state: Tuple[int, ...]) -> None:
        """
        Update Q-table using Q-learning update rule.
        
        Args:
            state: Previous state
            action: Action taken
            reward: Reward received
            next_state: Current state
        """
        if state not in self.q_table:
            self.q_table[state] = np.random.normal(0, 0.01, len(self.action_space))
        
        action_index = self.action_to_index[action]
        current_q = self.q_table[state][action_index]
        
        # Get maximum Q-value for next state
        next_q_values = self._get_q_values(next_state)
        max_next_q = np.max(next_q_values)
        
        # Q-learning update rule
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        
        self.q_table[state][action_index] = new_q
        self.training_steps += 1
        
        # Decay exploration rate
        if self.is_training:
            self.exploration_rate = max(
                self.min_exploration_rate,
                self.exploration_rate * self.exploration_decay
            )
    
    def generate_signal(self, data: pd.DataFrame, current_time: pd.Timestamp) -> TradingSignal:
        """
        Generate a trading signal using Q-learning policy.
        
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
            
            # Get current state
            current_state = self._get_state(data, current_time)
            current_row = data.loc[current_time]
            current_price = current_row['close']
            
            # Update Q-table if this is a training step
            if self.is_training and self.last_state is not None and self.last_action is not None:
                # Calculate reward for last action
                last_price = None
                current_index = data.index.get_loc(current_time)
                if current_index > 0:
                    last_price = data.iloc[current_index - 1]['close']
                
                reward = self._calculate_reward(data, current_time, self.last_action, last_price)
                self._update_q_table(self.last_state, self.last_action, reward, current_state)
                self.total_reward += reward
            
            # Choose action
            action = self._choose_action(current_state)
            
            # Calculate confidence based on Q-value spread
            q_values = self._get_q_values(current_state)
            if len(q_values) > 1:
                confidence = (np.max(q_values) - np.mean(q_values)) / (np.std(q_values) + 1e-6)
                confidence = min(max(confidence / 5, 0.1), 0.9)  # Normalize to 0.1-0.9
            else:
                confidence = 0.5
            
            # Store current state and action for next update
            self.last_state = current_state
            self.last_action = action
            
            # Create signal
            signal = TradingSignal(
                timestamp=current_time,
                signal_type=action,
                confidence=confidence,
                price=current_price,
                metadata={
                    'state': current_state,
                    'q_values': q_values.tolist(),
                    'exploration_rate': self.exploration_rate,
                    'training_step': self.training_steps,
                    'is_training': self.is_training
                }
            )
            
            # Track signal
            self.signals_history.append(signal)
            
            if action != SignalType.HOLD:
                self.logger.info(f"Q-Learning signal: {action.name} at {current_time} "
                               f"(confidence: {confidence:.3f}, state: {current_state})")
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Error generating Q-learning signal at {current_time}: {e}")
            return self._create_hold_signal(current_time, data.iloc[-1]['close'])
    
    def calculate_position_size(self, data: pd.DataFrame, signal: TradingSignal, 
                              portfolio_value: float) -> float:
        """
        Calculate position size based on signal confidence and Q-learning confidence.
        
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
            
            # Base position size
            base_size_pct = self.position_size
            
            # Adjust by signal confidence
            confidence_multiplier = signal.confidence
            
            # Adjust by Q-value confidence
            q_values = signal.metadata.get('q_values', [])
            if len(q_values) >= 3:
                q_array = np.array(q_values)
                q_spread = np.max(q_array) - np.min(q_array)
                q_multiplier = min(q_spread / 2, 1.5)  # Scale Q-value spread
            else:
                q_multiplier = 1.0
            
            # Training phase adjustment (more conservative)
            training_multiplier = 0.5 if self.is_training else 1.0
            
            # Combined position size
            final_size_pct = (base_size_pct * confidence_multiplier * 
                             q_multiplier * training_multiplier)
            
            # Apply limits
            final_size_pct = min(final_size_pct, 0.15)  # Max 15% position
            final_size_pct = max(final_size_pct, 0.01)  # Min 1% position
            
            # Calculate quantity
            position_value = portfolio_value * final_size_pct
            quantity = position_value / signal.price
            
            return quantity
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return 0.0
    
    def _create_hold_signal(self, timestamp: pd.Timestamp, price: float) -> TradingSignal:
        """Create a HOLD signal with default values."""
        return TradingSignal(
            timestamp=timestamp,
            signal_type=SignalType.HOLD,
            confidence=0.0,
            price=price,
            metadata={'reason': 'insufficient_data_or_error'}
        )
    
    def set_training_mode(self, training: bool) -> None:
        """Set training mode on/off."""
        self.is_training = training
        self.logger.info(f"Training mode: {'ON' if training else 'OFF'}")
    
    def save_q_table(self, filepath: str) -> None:
        """Save Q-table to file."""
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(self.q_table, f)
            self.logger.info(f"Q-table saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Error saving Q-table: {e}")
    
    def load_q_table(self, filepath: str) -> bool:
        """Load Q-table from file."""
        try:
            with open(filepath, 'rb') as f:
                self.q_table = pickle.load(f)
            self.logger.info(f"Q-table loaded from {filepath}")
            return True
        except Exception as e:
            self.logger.error(f"Error loading Q-table: {e}")
            return False
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics."""
        total_actions = self.exploration_actions + self.exploitation_actions
        return {
            'training_steps': self.training_steps,
            'total_reward': self.total_reward,
            'exploration_rate': self.exploration_rate,
            'exploration_actions': self.exploration_actions,
            'exploitation_actions': self.exploitation_actions,
            'exploration_ratio': self.exploration_actions / max(total_actions, 1),
            'q_table_size': len(self.q_table),
            'average_reward': self.total_reward / max(self.training_steps, 1)
        }
    
    def validate_parameters(self) -> bool:
        """Validate strategy parameters."""
        try:
            # Check learning parameters
            if not (0 < self.learning_rate <= 1):
                self.logger.error("Learning rate must be between 0 and 1")
                return False
            
            if not (0 < self.discount_factor <= 1):
                self.logger.error("Discount factor must be between 0 and 1")
                return False
            
            if not (0 <= self.exploration_rate <= 1):
                self.logger.error("Exploration rate must be between 0 and 1")
                return False
            
            # Check position parameters
            if not (0 < self.position_size <= 1):
                self.logger.error("Position size must be between 0 and 1")
                return False
            
            self.logger.info("Q-Learning strategy parameters validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Parameter validation error: {e}")
            return False
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get detailed strategy information."""
        base_info = super().get_strategy_info()
        
        ql_info = {
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor,
            'exploration_rate': self.exploration_rate,
            'state_space_size': self.state_space_size,
            'q_table_size': len(self.q_table),
            'training_steps': self.training_steps,
            'is_training': self.is_training
        }
        
        base_info.update(ql_info)
        return base_info
    
    def reset(self) -> None:
        """Reset strategy state for new backtest."""
        super().reset()
        self.last_state = None
        self.last_action = None
        self.last_reward = 0.0
        self.training_steps = 0
        self.exploration_actions = 0
        self.exploitation_actions = 0
        self.logger.info("Q-Learning strategy state reset")
    
    def __str__(self) -> str:
        """String representation of the strategy."""
        return (f"QLearningStrategy(lr={self.learning_rate}, "
                f"exploration={self.exploration_rate:.3f}, "
                f"states={self.state_space_size})")
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return (f"QLearningStrategy(learning_rate={self.learning_rate}, "
                f"discount_factor={self.discount_factor}, "
                f"exploration_rate={self.exploration_rate}, "
                f"state_space_size={self.state_space_size})") 