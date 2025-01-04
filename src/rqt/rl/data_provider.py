"""
Data Provider for Reinforcement Learning

This class serves as a data provider for reinforcement learning based on provided feature and price datasets.
It is primarily used in the training environment.

Features:
- Handles separate feature data and price (OHLCV) data
- Retrieves data in steps, each of length `self.window_size`
- Offers two methods to fetch steps:
  - **DataFrame Method**: Returns the data as a pandas DataFrame
  - **NumPy Array Method**: Returns a tuple containing `(date, state)`
- Provides a convenience method that accepts (date/index, numpy_array) and returns a named tuple,
  allowing easy access to array data through column names
- Maintains access to non-normalized features and raw price data for result interpretation

Functionality:
- Supports fetching steps by numerical index or date for both DataFrame and NumPy array methods
- Accepts price datasets with lowercase OHLCV (Open, High, Low, Close, Volume) columns
- Requires datetime to be set as the index of all datasets
- Ensures the first step includes `self.window_size` days of data
- Allows specifying a custom start date, with the requirement that `self.window_size` days of data precede it
- Utilizes assertions to enforce data requirements
- Returns data in named tuples for improved code readability and maintainability
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
    A data provider for reinforcement learning that handles features and price data separately.

    Attributes:
        window_size (int): The size of the data window to return in each step
        start_date (pd.Timestamp): The starting date for data provision
    """

    def __init__(
        self,
        features: pd.DataFrame,
        normalized_features: pd.DataFrame,
        prices: pd.DataFrame,
        window_size: int,
        start_date: Optional[datetime.datetime] = None,
    ) -> None:
        """
        Initialize the DataProvider with feature and price data.

        Args:
            features: DataFrame with normalized feature data
            normalized_features: DataFrame with original feature data
            prices: DataFrame with OHLCV price data
            window_size: Number of time steps in each data window
            start_date: Optional starting date for data provision

        Raises:
            ValueError: If data requirements are not met
        """
        self._validate_data(features, normalized_features, prices)

        self._features = features
        self._features_npy = features.to_numpy()
        self._normalized_features = normalized_features
        self._normalized_features_npy = normalized_features.to_numpy()
        self._prices = prices
        self._prices_npy = prices.to_numpy()
        self._window_size = window_size

        logger.debug("✨ Initialized DataProvider with window_size=%d", window_size)

        if start_date:
            self._validate_start_date(start_date)
            self._start_date = start_date
        else:
            self._start_date = self._features.index[0 + self._window_size]

        logger.debug("📅 Start date set to %s", self._start_date)

    def _validate_data(
        self, 
        features: pd.DataFrame, 
        normalized_features: pd.DataFrame,
        prices: pd.DataFrame
    ) -> None:
        """Validate input data meets requirements."""
        # Check index types
        for df, name in [(features, "features"), 
                        (normalized_features, "normalized_features"),
                        (prices, "prices")]:
            if not isinstance(df.index, pd.DatetimeIndex):
                raise ValueError(f"The index of {name} must be a DatetimeIndex")

        # Check required price columns
        missing_cols = REQUIRED_COLUMNS - set(prices.columns.str.lower())
        if missing_cols:
            raise ValueError(f"Missing required price columns: {missing_cols}")

        # Check data alignment
        if not (features.index.equals(normalized_features.index) and 
                features.index.equals(prices.index)):
            raise ValueError(
                "Features, normalized features, and prices must have identical indices"
            )

        # Check for missing dates
        date_range = pd.date_range(start=features.index[0], end=features.index[-1], freq='D')
        missing_dates = date_range.difference(features.index)
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
        return len(self._features)

    @property
    def end_date(self) -> pd.Timestamp:
        """Get the last date in the dataset."""
        return self._features.index[-1]

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
            return self._features.index.get_loc(date)
        except KeyError:
            raise KeyError(f"Date {date} not found in data")

    def _get_date_from_index(self, index: int) -> pd.Timestamp:
        """Convert data index to datetime."""
        try:
            return self._features.index[index]
        except IndexError:
            raise IndexError(
                f"Index {index} out of bounds for data length {len(self._features)}"
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
            self._features_npy[index - self._window_size : index],
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
            self._features.iloc[index - self._window_size : index],
        )

    def get_normalized_features(self, index: TimeIndexType) -> pd.DataFrame:
        """
        Get normalized features for a specific date/index.

        Args:
            index: Either a datetime or integer index

        Returns:
            DataFrame row of normalized features
        """
        if isinstance(index, datetime.datetime):
            return self._normalized_features.loc[index]
        return self._normalized_features.iloc[index]

    def get_normalized_features_npy(self, index: TimeIndexType) -> np.ndarray:
        """
        Get normalized features as numpy array for a specific date/index.

        Args:
            index: Either a datetime or integer index

        Returns:
            Numpy array of normalized features
        """
        if isinstance(index, datetime.datetime):
            return self._normalized_features_npy[self._get_index_from_date(index)]
        return self._normalized_features_npy[index]

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

        if start_index > len(self._features):
            raise IndexError(
                f"start_index {start_index} is out of bounds for data length {len(self._features)}"
            )

        logger.debug("🔄 Starting step generator from index %d", start_index)

        for i in range(start_index, len(self._features)):
            yield self.get_step_npy(i)

    def get_normalized_series(self, index: TimeIndexType) -> pd.Series:
        """
        Get normalized features as a pandas Series for a specific date/index.
        """
        if isinstance(index, datetime.datetime):
            return self._normalized_features.loc[index]
        return self._normalized_features.iloc[index]

    def get_prices(self, column: str = "close") -> dict[pd.Timestamp, float]:
        """
        Get a dictionary mapping timestamps to prices from the price data.

        Args:
            column: The price column to retrieve (one of open, high, low, close)

        Returns:
            dict[pd.Timestamp, float]: Dictionary mapping timestamps to prices
        """
        assert column in PRICE_COLUMNS, f"Column {column} is not in the price columns: {PRICE_COLUMNS}"
        logger.debug("📊 Getting %s prices dictionary from price data", column)
        return self._prices[column].to_dict()
