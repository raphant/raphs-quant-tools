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
    data = create_mock_data(n_periods=n_periods)
    dp = DataProvider(normalized_data=data,
                     pre_normalized_data=data,
                     window_size=window_size)

    # Create environment with simple spaces
    action_space = spaces.Discrete(3)  # Example: Buy, Sell, Hold
    observation_space = spaces.Box(
        low=-np.inf, 
        high=np.inf, 
        shape=(window_size, 5)  # 5 features: open, high, low, close, volume
    )
    env = Environment(dp, action_space, observation_space)

    # Test reset
    initial_state, info = env.reset()

    # Verify state shape and type
    assert isinstance(initial_state, np.ndarray)
    assert initial_state.shape == (window_size, 5)

    # Verify internal state
    assert env._current_date == data.index[window_size]
    assert env._step_generator is not None
    np.testing.assert_array_equal(env._current_state, data.iloc[0:window_size].to_numpy())

    # Test multiple resets
    for _ in range(3):
        new_state, info = env.reset()
        assert isinstance(new_state, np.ndarray)
        assert new_state.shape == (window_size, 5)
        np.testing.assert_array_equal(new_state, data.iloc[0:window_size].to_numpy())
