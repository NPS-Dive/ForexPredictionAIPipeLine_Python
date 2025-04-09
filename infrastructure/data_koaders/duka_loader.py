# Dukascopy Data Loader (Infrastructure Layer)
# Single Responsibility: Fetch price data from Dukascopy

import pandas as pd
from duka.app import app as duka_app
from duka.core.utils import TimeFrame
from datetime import datetime
import os

class DukaLoader:
    def __init__(self, symbol: str, start_date: str, end_date: str, timeframe: str):
        """Initialize Dukascopy data loader."""
        self.symbol = symbol
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
        self.end_date = datetime.strptime(end_date, "%Y-%m-%d").date()
        try:
            self.timeframe = TimeFrame[timeframe.upper()]
        except KeyError:
            raise ValueError(f"Invalid timeframe: {timeframe}. Use M1, M5, M15, M30, H1, D1, W1.")
        self.output_dir = os.path.abspath(f"data/{self.timeframe.name.lower()}")
        os.makedirs(self.output_dir, exist_ok=True)
        self.file_path = os.path.join(self.output_dir, f"{symbol.lower()}_{timeframe.lower()}_raw.csv")

    def load_data(self) -> pd.DataFrame:
        """Load and process Dukascopy data."""
        try:
            duka_app(
                symbols=[self.symbol],
                start=self.start_date,
                end=self.end_date,
                timeframe=self.timeframe,
                folder=self.output_dir,
                threads=4,
                header=False
            )
            df = pd.read_csv(self.file_path, names=['Time', 'Bid', 'Ask', 'Volume'], header=None)
            df['Time'] = pd.to_datetime(df['Time'], unit='ms')
            df.set_index('Time', inplace=True)
            df['Open'] = df['Bid'].shift(1)
            df['High'] = df[['Bid', 'Ask']].max(axis=1)
            df['Low'] = df[['Bid', 'Ask']].min(axis=1)
            df['Close'] = df['Bid']
            return df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
        except Exception as e:
            print(f"Error loading Dukascopy data: {e}")
            return pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume'])