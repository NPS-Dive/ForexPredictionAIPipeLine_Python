# Data loading using Dukascopy via duka (Infrastructure Layer)
# Single Responsibility: Fetch and clean raw data from Dukascopy
import pandas as pd
from duka.app import app as duka_app
from duka.core.utils import TimeFrame
from datetime import datetime
import os

class DataLoader:
    def __init__(self, symbol: str = "EURUSD", start_date: str = "2023-01-01", end_date: str = "2023-12-31", output_dir: str = "data"):
        """Initialize with symbol and date range for Dukascopy data."""
        self.symbol = symbol
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d")
        self.end_date = datetime.strptime(end_date, "%Y-%m-%d")
        self.output_dir = os.path.abspath(output_dir)  # Ensure absolute path
        os.makedirs(self.output_dir, exist_ok=True)    # Create directory if missing
        self.file_path = os.path.join(self.output_dir, f"{symbol.lower()}_raw.csv")

    def load_data(self) -> pd.DataFrame:
        """Fetch historical EUR/USD data from Dukascopy and load into DataFrame."""
        # Simplified duka_app call without 'flatten'
        duka_app(
            [self.symbol],         # Positional: symbols
            self.start_date,       # Positional: start_date
            self.end_date,         # Positional: end_date
            TimeFrame.H1,          # Positional: timeframe
            self.output_dir,       # Positional: folder
            4                      # Positional: threads
        )
        
        # Load the downloaded CSV (duka format: time, bid, ask, volume)
        df = pd.read_csv(self.file_path, names=['Time', 'Bid', 'Ask', 'Volume'], header=None)
        df['Time'] = pd.to_datetime(df['Time'], unit='ms')  # Convert timestamp to datetime
        df.set_index('Time', inplace=True)
        
        # Simulate OHLC from bid/ask (simple approximation)
        df['Open'] = df['Bid'].shift(1)  # Previous close as open
        df['High'] = df[['Bid', 'Ask']].max(axis=1)
        df['Low'] = df[['Bid', 'Ask']].min(axis=1)
        df['Close'] = df['Bid']  # Use bid as close for simplicity
        return df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()