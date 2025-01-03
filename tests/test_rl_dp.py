import numpy as np
import pandas as pd
import datetime

import pytest

from rqt.rl.data_provider import DataProvider
from test_utils import create_mock_data



def test_data_provider_non_datetime_index():
    # Create sample data with non-datetime index
    data = create_mock_data()
    data.index = data.index.astype(int)

    # Test that it raises ValueError when creating DataProvider
    with pytest.raises(ValueError):
        DataProvider(normalized_data=data, 
                    pre_normalized_data=data,
                    window_size=5)

def test_data_provider_insufficient_data_before_start_date():
    # Create sample data
    data = create_mock_data(n_periods=10)
    window_size = 5
    start_date = data.index[0]

    # Test that it raises ValueError when not enough data before start_date
    with pytest.raises(ValueError):
        DataProvider(normalized_data=data,
                    pre_normalized_data=data, 
                    window_size=window_size,
                    start_date=start_date)

def test_get_date_from_index():
    # Create sample data
    data = create_mock_data(n_periods=10)
    window_size = 5
    dp = DataProvider(normalized_data=data,
                     pre_normalized_data=data,
                     window_size=window_size)

    # Test getting date from index
    test_index = 3
    expected_date = data.index[test_index]
    actual_date = dp._get_date_from_index(test_index)

    # Confirm instance types
    assert isinstance(actual_date, pd.Timestamp), f"Expected pd.Timestamp but got {type(actual_date)}"
    assert isinstance(expected_date, pd.Timestamp), f"Expected pd.Timestamp but got {type(expected_date)}"

    assert actual_date == expected_date, f"Expected {expected_date}, but got {actual_date}"

def test_get_step_npy():
    # Create sample data with known values
    n_periods = 20
    window_size = 5
    data = create_mock_data(n_periods=n_periods)
    dp = DataProvider(normalized_data=data, 
                     pre_normalized_data=data,
                     window_size=window_size)

    # Test getting step by index
    test_index = 10
    date, state = dp.get_step_npy(test_index)

    # Verify return types
    assert isinstance(date, pd.Timestamp), f"Expected date to be pd.Timestamp but got {type(date)}"
    assert isinstance(state, np.ndarray), f"Expected state to be np.ndarray but got {type(state)}"
    
    # Verify date matches index
    expected_date = data.index[test_index]
    assert date == expected_date, f"Expected date {expected_date}, but got {date}"

    # Verify state shape and content
    assert state.shape == (window_size, 5), f"Expected shape (5,5) but got {state.shape}"
    expected_state = data.iloc[test_index-window_size:test_index].to_numpy()
    np.testing.assert_array_equal(state, expected_state)

    # Test getting step by datetime
    test_date = data.index[test_index]
    date2, state2 = dp.get_step_npy(test_date)
    
    # Verify results match when using datetime vs index
    assert date2 == date, f"Date mismatch using datetime vs index"
    np.testing.assert_array_equal(state2, state)

    # Test edge cases
    # First valid index
    date_first, state_first = dp.get_step_npy(window_size)
    assert len(state_first) == window_size
    
    # Last index
    date_last, state_last = dp.get_step_npy(n_periods-1)
    assert len(state_last) == window_size

    # Test invalid index raises error
    with pytest.raises(IndexError):
        dp.get_step_npy(n_periods)  # Index out of bounds
        
    with pytest.raises(KeyError):
        dp.get_step_npy(pd.Timestamp('2025-01-01'))  # Date not in index

def test_get_step_df():
    # Create sample data with known values
    n_periods = 20
    window_size = 5
    data = create_mock_data(n_periods=n_periods)
    dp = DataProvider(normalized_data=data,
                     pre_normalized_data=data, 
                     window_size=window_size)

    # Test getting step by index
    test_index = 10
    date, state = dp.get_step_df(test_index)

    # Verify return types
    assert isinstance(date, pd.Timestamp), f"Expected date to be pd.Timestamp but got {type(date)}"
    assert isinstance(state, pd.DataFrame), f"Expected state to be pd.DataFrame but got {type(state)}"
    
    # Verify date matches index
    expected_date = data.index[test_index]
    assert date == expected_date, f"Expected date {expected_date}, but got {date}"

    # Verify state shape and content
    assert state.shape == (window_size, 5), f"Expected shape (5,5) but got {state.shape}"
    expected_state = data.iloc[test_index-window_size:test_index]
    pd.testing.assert_frame_equal(state, expected_state)

    # Test getting step by datetime
    test_date = data.index[test_index]
    date2, state2 = dp.get_step_df(test_date)
    
    # Verify results match when using datetime vs index
    assert date2 == date, f"Date mismatch using datetime vs index"
    pd.testing.assert_frame_equal(state2, state)

    # Test edge cases
    # First valid index
    _, state_first = dp.get_step_df(window_size)
    assert len(state_first) == window_size
    
    # Last index
    _, state_last = dp.get_step_df(n_periods-1)
    assert len(state_last) == window_size

    # Test invalid index raises error
    with pytest.raises(IndexError):
        dp.get_step_df(n_periods)  # Index out of bounds
        
    with pytest.raises(KeyError):
        dp.get_step_df(pd.Timestamp('2025-01-01'))  # Date not in index

def test_step_generator_npy():
    # Create sample data
    n_periods = 20
    window_size = 5
    data = create_mock_data(n_periods=n_periods)
    dp = DataProvider(normalized_data=data,
                     pre_normalized_data=data,
                     window_size=window_size)

    # Test default start index (window_size)
    generator = dp.step_generator_npy()
    first_date, first_state = next(generator)
    
    # Verify first step
    assert isinstance(first_date, pd.Timestamp)
    assert isinstance(first_state, np.ndarray)
    assert first_date == data.index[window_size]
    assert first_state.shape == (window_size, 5)
    np.testing.assert_array_equal(first_state, data.iloc[0:window_size].to_numpy())

    # Test with custom start index
    start_idx = 10
    generator = dp.step_generator_npy(start_index=start_idx)
    first_date, first_state = next(generator)
    
    assert first_date == data.index[start_idx]
    np.testing.assert_array_equal(first_state, data.iloc[start_idx-window_size:start_idx].to_numpy())

    # Test full iteration
    generator = dp.step_generator_npy()
    steps = list(generator)
    
    # Verify number of steps
    expected_steps = n_periods - window_size
    assert len(steps) == expected_steps, f"Expected {expected_steps} steps but got {len(steps)}"

    # Verify each step
    for i, (date, state) in enumerate(steps):
        current_idx = i + window_size
        assert date == data.index[current_idx]
        expected_state = data.iloc[current_idx-window_size:current_idx].to_numpy()
        np.testing.assert_array_equal(state, expected_state)

    # Test empty iteration when start_index is at end
    generator = dp.step_generator_npy(start_index=n_periods)
    steps = list(generator)
    assert len(steps) == 0, "Expected no steps when starting at end of data"

    # Test invalid start_index
    with pytest.raises(IndexError):
        next(dp.step_generator_npy(start_index=n_periods + 1))

def test_missing_dates():
    """Test that missing dates in the data are detected and raise an error."""
    # Create mock data with missing dates
    n_periods = 20
    data = create_mock_data(n_periods=n_periods)
    
    # Remove some dates to create gaps
    data = data.drop(data.index[5:8])  # Remove 3 dates in the middle
    
    # Attempt to create DataProvider with data containing gaps
    with pytest.raises(ValueError):
        DataProvider(normalized_data=data,
                    pre_normalized_data=data, 
                    window_size=5)
        

def test_get_pre_normalized_series():
    """Test getting pre-normalized data as a pandas Series."""
    # Create sample data with known values
    n_periods = 20
    window_size = 5
    
    # Create mock data with technical indicators
    data = create_mock_data(n_periods=n_periods)
    
    # Add some technical indicators
    data['sma_5'] = data['close'].rolling(window=5).mean()
    data['rsi_14'] = 50 + np.random.randn(n_periods) * 10  # Mock RSI values
    data['macd'] = data['close'].ewm(span=12).mean() - data['close'].ewm(span=26).mean()
    
    dp = DataProvider(normalized_data=data,
                     pre_normalized_data=data, 
                     window_size=window_size)

    # Test getting data by index
    test_index = 10
    series = dp.get_pre_normalized_series(test_index)
    
    # Verify it returns a Series
    assert isinstance(series, pd.Series), f"Expected pd.Series but got {type(series)}"
    
    # Verify values match original data
    assert series['close'] == data.iloc[test_index]['close']
    assert series['sma_5'] == data.iloc[test_index]['sma_5']
    assert series['rsi_14'] == data.iloc[test_index]['rsi_14']
    assert series['macd'] == data.iloc[test_index]['macd']
    
    # Test getting data by date
    test_date = data.index[test_index]
    series_by_date = dp.get_pre_normalized_series(test_date)
    
    # Verify values match when accessing by date
    pd.testing.assert_series_equal(series, series_by_date)
    
    # Test invalid index
    with pytest.raises(IndexError):
        dp.get_pre_normalized_series(len(data) + 1)
        
    # Test invalid date
    invalid_date = pd.Timestamp('2025-01-01')
    with pytest.raises(KeyError):
        dp.get_pre_normalized_series(invalid_date)