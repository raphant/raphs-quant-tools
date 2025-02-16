# Reinforcement Learning Components

## DataProvider (`data_provider.py`)

The DataProvider class serves as a robust data provider for reinforcement learning based on preprocessed datasets. It is primarily used in the training environment to provide time-series financial data in a format suitable for RL algorithms.

### Key Features

- Retrieves data in steps of configurable window size
- Supports both DataFrame and NumPy array output formats
- Maintains access to both normalized and pre-normalized data
- Validates data requirements and integrity
- Provides convenient access methods by date or index
- Handles missing data validation and suggestions
- Includes built-in data validation and error handling

### Main Methods

#### Data Access

- `get_step_npy()` - Get step data as NumPy array
- `get_step_df()` - Get step data as DataFrame  
- `step_generator_npy()` - Generate sequential steps
- `get_pre_normalized_series()` - Access original data values
- `get_pre_normalized_data()` - Get pre-normalized data for specific date/index
- `get_pre_normalized_data_npy()` - Get pre-normalized data as numpy array

#### Properties

- `data_length` - Total length of the dataset
- `end_date` - Last date in the dataset
- `window_size` - Current window size
- `start_date` - Start date of the dataset

### Data Requirements

- Dataset must have lowercase OHLCV columns:
  - open
  - high
  - low
  - close
  - volume
- DateTime index is required
- No missing dates allowed in the time series
- Sufficient historical data before start date (at least window_size days)
- Both normalized and pre-normalized data must have identical indices

### Usage Examples

```python
import pandas as pd
from datetime import datetime
from rqt.reinforcement_learning.data_provider import DataProvider

# Prepare your data (example)
normalized_data = pd.DataFrame(...)  # Your normalized OHLCV data
pre_normalized_data = pd.DataFrame(...)  # Your original OHLCV data

# Initialize DataProvider
dp = DataProvider(
    normalized_data=normalized_data,
    pre_normalized_data=pre_normalized_data,
    window_size=30,
    start_date=datetime(2023, 1, 1)
)

# Get data step as numpy array
date, state = dp.get_step_npy(datetime(2023, 2, 1))

# Get data step as DataFrame
date, df_state = dp.get_step_df(100)  # Using index instead of date

# Generate sequential steps
for date, state in dp.step_generator_npy():
    # Process each step
    print(f"Processing data for {date}")

# Access pre-normalized data
original_values = dp.get_pre_normalized_series(datetime(2023, 2, 1))
```

### Error Handling

The DataProvider includes comprehensive error handling for common issues:

- Missing required columns
- Invalid index types
- Missing dates in the time series
- Insufficient historical data
- Index out of bounds
- Date not found in dataset

When encountering missing dates, the provider suggests interpolation methods:

```python
# Example fix for missing dates
df = df.reindex(pd.date_range(start=df.index[0], end=df.index[-1], freq='D'))
df = df.interpolate(method='linear')
```

### Best Practices

1. Always validate your data before initializing the DataProvider
2. Use appropriate window sizes based on your RL model's requirements
3. Consider the trade-off between window size and available training data
4. Handle both normalized and pre-normalized data appropriately in your RL environment
5. Use the step generator for efficient sequential data processing

## Environment (`environment.py`)

The Environment class provides a Gymnasium-compatible reinforcement learning environment for training and testing RL models. It works seamlessly with the DataProvider to create a complete RL training pipeline.

### Key Features

- Fully compatible with OpenAI Gymnasium interface
- Integrated with DataProvider for time-series financial data
- Built-in trade management through TradeManager
- Customizable action and observation spaces
- Configurable initial capital
- Step-by-step environment progression with state management

### Main Methods

#### Core Environment Methods

- `reset()` - Reset the environment to initial state
- `step(action)` - Execute one step with given action
- `_calculate_reward()` - Calculate reward for current step (customizable)

#### Properties

- `action_space` - Defines possible actions
- `observation_space` - Defines state observation structure
- `trade_manager` - Manages trading state and execution

### Usage Examples

```python
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from rqt.rl.environment import Environment
from rqt.rl.data_provider import DataProvider

# Prepare your DataProvider
dp = DataProvider(...)

# Define your action and observation spaces
action_space = spaces.Discrete(3)  # Example: Buy, Hold, Sell
observation_space = spaces.Box(
    low=-np.inf, 
    high=np.inf, 
    shape=(dp.window_size, 5)  # OHLCV data
)

# Initialize environment
env = Environment(
    data_provider=dp,
    action_space=action_space,
    observation_space=observation_space,
    initial_capital=10000
)

# Use in training loop
observation, info = env.reset()
for _ in range(1000):
    action = env.action_space.sample()  # Your agent's action here
    observation, reward, done, truncated, info = env.step(action)
    
    if done:
        observation, info = env.reset()
```

### Best Practices

1. Customize the reward function (`_calculate_reward`) based on your trading strategy
2. Choose appropriate action and observation spaces for your use case
3. Consider using the trade manager's state in your reward calculation
4. Monitor the environment's performance through logging
5. Handle episode termination conditions appropriately
