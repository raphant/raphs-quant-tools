import numpy as np
import pandas as pd


def create_mock_data(n_periods: int = 100) -> pd.DataFrame:
    return pd.DataFrame({
        'open': np.random.rand(n_periods),
        'high': np.random.rand(n_periods),
        'low': np.random.rand(n_periods), 
        'close': np.random.rand(n_periods),
        'volume': np.random.rand(n_periods),
    }, index=pd.date_range(start='2024-01-01', periods=n_periods, freq='D'))