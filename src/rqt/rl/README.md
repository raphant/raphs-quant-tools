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
