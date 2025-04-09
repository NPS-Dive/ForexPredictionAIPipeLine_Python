# Macro Data Loader (Infrastructure Layer)
# Single Responsibility: Fetch macro-economic data

import pandas as pd
import pandas_datareader as pdr
from datetime import datetime

class MacroLoader:
    def __init__(self, start_date: str, end_date: str):
        """Initialize macro data loader."""
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d")
        self.end_date = datetime.strptime(end_date, "%Y-%m-%d")

    def load_data(self) -> pd.DataFrame:
        """Load macro-economic data from FRED."""
        try:
            df = pdr.get_data_fred(['FEDFUNDS'], start=self.start_date, end=self.end_date)
            df.index = pd.to_datetime(df.index)
            return df.rename(columns={'FEDFUNDS': 'Interest_Rate'})
        except Exception as e:
            print(f"Error loading macro data: {e}")
            return pd.DataFrame()