"""
Provides a reinforcement learning environment for training and testing RL models that are compatible with rqt.rl.data_provider.DataProvider.
"""

import logging
from typing import Any
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from rqt.rl.data_provider import DataProvider
from rqt.rl.tm.trade_manager import TradeManager

logger = logging.getLogger(__name__)


class Environment(gym.Env):
    def __init__(self, data_provider: DataProvider, action_space: spaces.Space, observation_space: spaces.Space, initial_capital: float = 1000):
        self.dp = data_provider
        self.action_space = action_space
        self.observation_space = observation_space
        self.trade_manager = TradeManager(initial_capital=initial_capital, data_provider=self.dp)

        # Initialize generator
        self._step_generator = None
        self._current_date = None
        self._current_state = None

        logger.debug("🎮 Initialized RL Environment with data window size: %d", self.dp.window_size)

    def open_trade(self, stake_amount: float) -> bool:
        """
        Open a new trade with the given stake amount.
        
        Args:
            stake_amount: Amount to stake in the trade
            
        Returns:
            bool: True if trade was opened successfully, False otherwise
        """
        try:
            current_price = self.dp.get_pre_normalized_data(self._current_date)['close']
            self.trade_manager.open_trade(stake_amount, current_price, self._current_date)
            logger.debug("🔓 Opened trade: stake=%.2f, price=%.2f", stake_amount, current_price)
            return True
        except ValueError as e:
            logger.debug("❌ Failed to open trade: %s", str(e))
            return False

    def close_trade(self, trade_id: int) -> bool:
        """
        Close a specific trade.
        
        Args:
            trade_id: ID of the trade to close
            
        Returns:
            bool: True if trade was closed successfully, False otherwise
        """
        try:
            current_price = self.dp.get_pre_normalized_data(self._current_date)['close']
            self.trade_manager.close_trade(trade_id, current_price, self._current_date)
            # Get the closed trade details from trade manager
            closed_trade = self.trade_manager.last_completed_trade
            profit_emoji = "📈" if closed_trade.profit > 0 else "📉"
            logger.debug(
                "%s Closed trade #%d: date=%s, price=%.2f | profit=%.2f (%.1f%%)", 
                profit_emoji,
                trade_id,
                closed_trade.close_date,
                current_price,
                closed_trade.profit,
                (closed_trade.profit / closed_trade.stake_amount) * 100
            )
            return True
        except ValueError as e:
            logger.debug("❌ Failed to close trade: %s", str(e))
            return False

    def reset(self) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset()
        
        # Reset the generator
        self._step_generator = self.dp.step_generator_npy(start_index=self.dp.window_size)

        # Get initial state
        self._current_date, self._current_state = next(self._step_generator)

        # Reset the trade manager
        self.trade_manager.reset()
        
        logger.debug("🔄 Environment reset - starting from date: %s", self._current_date)
        
        return self._current_state, {}

    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one step in the environment."""
        try:
            reward = 0.0
            # Get next state from generator
            self._current_date, next_state = next(self._step_generator)

            match action:
                case 0:
                    pass
                case 1:
                    if not self.trade_manager.current_trade:
                        self.open_trade(stake_amount=100)
                case 2:
                    if self.trade_manager.current_trade:
                        closed = self.close_trade(trade_id=self.trade_manager.current_trade.id)
                        if closed:
                            reward = self.trade_manager.last_completed_trade.profit

            # Update current state
            self._current_state = next_state
            
            # Check if we've reached the end of the data
            done = False
            truncated = False
            
            return next_state, reward, done, truncated, {
                "date": self._current_date,
                "raw_state": self.dp.get_pre_normalized_data(self._current_date)
            }
            
        except StopIteration:
            # We've reached the end of our data
            logger.debug("📈 Episode complete - reached end of data at: %s", self._current_date)
            return (
                self._current_state,
                0.0,
                True,  # done
                False,  # truncated
                {"date": self._current_date}
            )