import logging
import numpy as np
import pandas as pd
import pytest
from gymnasium import spaces

from rqt.rl.environment import Environment
from rqt.rl.data_provider import DataProvider
from test_utils import create_mock_data

# Set logging to debug for tests
logging.getLogger('rqt.rl.environment').setLevel(logging.DEBUG)

def test_environment_reset():
    """Test that environment reset works correctly."""
    # Create test data and data provider
    n_periods = 20
    window_size = 5
    
    # Create mock data with features and prices
    features = create_mock_data(n_periods=n_periods)
    normalized_features = features.copy()
    prices = features.copy()
    
    # Add some technical indicators to features
    features['sma_5'] = features['close'].rolling(window=5).mean()
    features['rsi_14'] = 50 + np.random.randn(n_periods) * 10  # Mock RSI values
    features['macd'] = features['close'].ewm(span=12).mean() - features['close'].ewm(span=26).mean()
    
    # Add normalized versions to normalized_features
    normalized_features['sma_5'] = normalized_features['close'].rolling(window=5).mean()
    normalized_features['rsi_14'] = 50 + np.random.randn(n_periods) * 10
    normalized_features['macd'] = normalized_features['close'].ewm(span=12).mean() - normalized_features['close'].ewm(span=26).mean()
    
    dp = DataProvider(features=features,
                     normalized_features=normalized_features,
                     prices=prices,
                     window_size=window_size)

    # Create environment with simple spaces
    n_features = len(features.columns)  # Account for all features including technical indicators
    action_space = spaces.Discrete(3)  # Example: Buy, Sell, Hold
    observation_space = spaces.Box(
        low=-np.inf, 
        high=np.inf, 
        shape=(window_size, n_features)  # All features including technical indicators
    )
    env = Environment(dp, action_space, observation_space)

    # Test reset
    initial_state, info = env.reset()

    # Verify state shape and type
    assert isinstance(initial_state, np.ndarray)
    assert initial_state.shape == (window_size, n_features)

    # Verify internal state
    assert env._current_date == features.index[window_size]
    assert env._step_generator is not None
    np.testing.assert_array_equal(env._current_state, features.iloc[0:window_size].to_numpy())

    # Test multiple resets
    for _ in range(3):
        new_state, info = env.reset()
        assert isinstance(new_state, np.ndarray)
        assert new_state.shape == (window_size, n_features)
        np.testing.assert_array_equal(new_state, features.iloc[0:window_size].to_numpy())
