"""
Data Provider for Reinforcement Learning

This class serves as a data provider for reinforcement learning based on a provided preprocessed dataset. It is primarily used in the training environment.

Features:
- Retrieves data in steps, each of length `self.window_size`.
- Offers two methods to fetch steps:
  - **DataFrame Method**: Returns the data as a pandas DataFrame.
  - **NumPy Array Method**: Returns a tuple containing `(date, state)`.
- Provides a convenience method that accepts (date/index, numpy_array) and returns a named tuple,
  allowing easy access to array data through column names.
- Maintains access to pre-normalized data:
  - Stores original, non-normalized DataFrame internally
  - Provides convenience method to retrieve pre-normalized values by date or index
  - Enables result interpretation and visualization

Functionality:
- Supports fetching steps by numerical index or date for both DataFrame and NumPy array methods.
- Accepts datasets with lowercase OHLCV (Open, High, Low, Close, Volume) columns.
- Requires datetime to be set as the index of the dataset.
- Ensures the first step includes `self.window_size` days of data.
- Allows specifying a custom start date, with the requirement that `self.window_size` days of data precede it.
- Utilizes assertions to enforce data requirements.
- Returns data in named tuples for improved code readability and maintainability.
"""

from typing import Generator, NamedTuple, Optional, Tuple, Union
import numpy as np
import pandas as pd
import datetime
import logging

logger = logging.getLogger(__name__)

# Type representing either a datetime or an integer index for time series data
TimeIndexType = Union[datetime.datetime, int]
StepType = Tuple[pd.Timestamp, Union[np.ndarray, pd.DataFrame]]

REQUIRED_COLUMNS = {"open", "high", "low", "close", "volume"}
PRICE_COLUMNS = {"open", "high", "low", "close"}

class DataProvider:
    """
    A data provider for reinforcement learning that handles both normalized and pre-normalized data.

    Attributes:
        window_size (int): The size of the data window to return in each step
        start_date (pd.Timestamp): The starting date for data provision
    """

    def __init__(
        self,
        normalized_data: pd.DataFrame,
        pre_normalized_data: pd.DataFrame,
        window_size: int,
        start_date: Optional[datetime.datetime] = None,
    ) -> None:
        """
        Initialize the DataProvider with normalized and pre-normalized data.

        Args:
            normalized_data: DataFrame with normalized OHLCV data
            pre_normalized_data: DataFrame with original OHLCV data
            window_size: Number of time steps in each data window
            start_date: Optional starting date for data provision

        Raises:
            ValueError: If data requirements are not met
        """
        self._validate_data(normalized_data, pre_normalized_data)

        self._normalized_data = normalized_data
        self._normalized_data_npy = normalized_data.to_numpy()
        self._pre_normalized_data = pre_normalized_data
        self._pre_normalized_data_npy = pre_normalized_data.to_numpy()
        self._window_size = window_size

        logger.debug("✨ Initialized DataProvider with window_size=%d", window_size)

        if start_date:
            self._validate_start_date(start_date)
            self._start_date = start_date
        else:
            self._start_date = self._normalized_data.index[0 + self._window_size]

        logger.debug("📅 Start date set to %s", self._start_date)

    def _validate_data(
        self, normalized_data: pd.DataFrame, pre_normalized_data: pd.DataFrame
    ) -> None:
        """Validate input data meets requirements."""
        # Check index type
        if not isinstance(normalized_data.index, pd.DatetimeIndex):
            raise ValueError("The index of normalized_data must be a DatetimeIndex")

        # Check required columns
        missing_cols = REQUIRED_COLUMNS - set(normalized_data.columns.str.lower())
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Check data alignment
        if not normalized_data.index.equals(pre_normalized_data.index):
            raise ValueError(
                "Normalized and pre-normalized data must have identical indices"
            )

        # Check for missing dates
        date_range = pd.date_range(start=normalized_data.index[0], end=normalized_data.index[-1], freq='D')
        missing_dates = date_range.difference(normalized_data.index)
        if len(missing_dates) > 0:
            logger.debug("❌ Found %d missing dates in data", len(missing_dates))
            suggestion = (
                "Consider filling missing dates with random values based on the mean and "
                "standard deviation of nearby data points. Example:\n"
                "df = df.reindex(pd.date_range(start=df.index[0], end=df.index[-1], freq='D'))\n"
                "df = df.interpolate(method='linear')"
            )
            raise ValueError(
                f"Data has {len(missing_dates)} missing dates between {missing_dates[0]} "
                f"and {missing_dates[-1]}. {suggestion}"
            )

        logger.debug("✅ Data validation passed")

    def _validate_start_date(self, start_date: datetime.datetime) -> None:
        """Validate start date has sufficient preceding data."""
        try:
            start_idx = self._get_index_from_date(start_date)
        except KeyError:
            raise ValueError(f"Start date {start_date} not found in data")

        if start_idx < self._window_size:
            raise ValueError(
                f"Not enough data before start_date. Need at least {self._window_size} days, "
                f"but only found {start_idx} days before {start_date}"
            )

    @property
    def data_length(self) -> int:
        """Get the total length of the dataset."""
        return len(self._normalized_data)

    @property
    def end_date(self) -> pd.Timestamp:
        """Get the last date in the dataset."""
        return self._normalized_data.index[-1]

    @property
    def window_size(self) -> int:
        """Get the window size."""
        return self._window_size

    @property
    def start_date(self) -> pd.Timestamp:
        """Get the start date."""
        return self._start_date

    def _get_index_from_date(self, date: datetime.datetime) -> int:
        """Convert datetime to data index location."""
        try:
            return self._normalized_data.index.get_loc(date)
        except KeyError:
            raise KeyError(f"Date {date} not found in data")

    def _get_date_from_index(self, index: int) -> pd.Timestamp:
        """Convert data index to datetime."""
        try:
            return self._normalized_data.index[index]
        except IndexError:
            raise IndexError(
                f"Index {index} out of bounds for data length {len(self._normalized_data)}"
            )

    def get_step_npy(self, index: TimeIndexType) -> Tuple[pd.Timestamp, np.ndarray]:
        """
        Get a data step as numpy array.

        Args:
            index: Either a datetime or integer index

        Returns:
            Tuple of (timestamp, numpy array) for the step
        """
        if isinstance(index, datetime.datetime):
            index = self._get_index_from_date(index)
        return (
            self._get_date_from_index(index),
            self._normalized_data_npy[index - self._window_size : index],
        )

    def get_step_df(self, index: TimeIndexType) -> Tuple[pd.Timestamp, pd.DataFrame]:
        """
        Get a data step as DataFrame.

        Args:
            index: Either a datetime or integer index

        Returns:
            Tuple of (timestamp, DataFrame) for the step
        """
        if isinstance(index, datetime.datetime):
            index = self._get_index_from_date(index)
        return (
            self._get_date_from_index(index),
            self._normalized_data.iloc[index - self._window_size : index],
        )

    def get_pre_normalized_data(self, index: TimeIndexType) -> pd.DataFrame:
        """
        Get pre-normalized data for a specific date/index.

        Args:
            index: Either a datetime or integer index

        Returns:
            DataFrame row of pre-normalized data
        """
        if isinstance(index, datetime.datetime):
            return self._pre_normalized_data.loc[index]
        return self._pre_normalized_data.iloc[index]

    def get_pre_normalized_data_npy(self, index: TimeIndexType) -> np.ndarray:
        """
        Get pre-normalized data as numpy array for a specific date/index.

        Args:
            index: Either a datetime or integer index

        Returns:
            Numpy array of pre-normalized data
        """
        if isinstance(index, datetime.datetime):
            return self._pre_normalized_data_npy[self._get_index_from_date(index)]
        return self._pre_normalized_data_npy[index]

    def step_generator_npy(
        self, start_index: Optional[int] = None
    ) -> Generator[Tuple[pd.Timestamp, np.ndarray], None, None]:
        """
        Generate steps from the data starting at start_index.

        Args:
            start_index: Optional starting index (defaults to window_size)

        Yields:
            Tuples of (timestamp, numpy array) for each step

        Raises:
            IndexError: If start_index is out of bounds
        """
        if start_index is None:
            start_index = self._window_size

        if start_index > len(self._normalized_data):
            raise IndexError(
                f"start_index {start_index} is out of bounds for data length {len(self._normalized_data)}"
            )

        logger.debug("🔄 Starting step generator from index %d", start_index)

        for i in range(start_index, len(self._normalized_data)):
            yield self.get_step_npy(i)

    def get_pre_normalized_series(self, index: TimeIndexType) -> pd.Series:
        """
        Get pre-normalized data as a pandas Series for a specific date/index.
        """
        if isinstance(index, datetime.datetime):
            return self._pre_normalized_data.loc[index]
        return self._pre_normalized_data.iloc[index]

    def get_prices(self, column: str = "close") -> dict[pd.Timestamp, float]:
        """
        Get a dictionary mapping timestamps to close prices from the pre-normalized data.

        Returns:
            dict[pd.Timestamp, float]: Dictionary mapping timestamps to close prices
        """
        assert column in PRICE_COLUMNS, f"Column {column} is not in the required columns: {PRICE_COLUMNS}"
        logger.debug("📊 Getting %s prices dictionary from pre-normalized data", column)
        return self._pre_normalized_data[column].to_dict()
