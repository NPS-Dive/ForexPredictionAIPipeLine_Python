# Utilities (Infrastructure Layer)
# Single Responsibility: Provide reusable helper functions

import pandas as pd
import numpy as np

def normalize(series: pd.Series) -> pd.Series:
    """Normalize a series using a rolling window."""
    min_val = series.rolling(20).min()
    max_val = series.rolling(20).max()
    return (series - min_val) / (max_val - min_val + 1e-6)