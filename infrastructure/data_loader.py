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
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d").date()  # Convert to date object
        self.end_date = datetime.strptime(end_date, "%Y-%m-%d").date()      # Convert to date object
        self.output_dir = os.path.abspath(output_dir)  # Ensure absolute path
        os.makedirs(self.output_dir, exist_ok=True)    # Create directory if missing
        self.file_path = os.path.join(self.output_dir, f"{symbol.lower()}_raw.csv")

    def load_data(self) -> pd.DataFrame:
        """Fetch historical EUR/USD data from Dukascopy and load into DataFrame."""
        # Call duka_app with named parameters to avoid confusion
        duka_app(
            symbols=[self.symbol],
            start=self.start_date,
            end=self.end_date,
            timeframe=TimeFrame.H1,
            folder=self.output_dir,
            threads=4,
            header=False
        )
        
        # Load the downloaded CSV (duka format: time, bid, ask, volume)
        try:
            df = pd.read_csv(self.file_path, names=['Time', 'Bid', 'Ask', 'Volume'], header=None)
            df['Time'] = pd.to_datetime(df['Time'], unit='ms')  # Convert timestamp to datetime
            df.set_index('Time', inplace=True)
            
            # Simulate OHLC from bid/ask (simple approximation)
            df['Open'] = df['Bid'].shift(1)  # Previous close as open
            df['High'] = df[['Bid', 'Ask']].max(axis=1)
            df['Low'] = df[['Bid', 'Ask']].min(axis=1)
            df['Close'] = df['Bid']  # Use bid as close for simplicity
            return df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
        except FileNotFoundError:
            print(f"Warning: Data file not found at {self.file_path}")
            print("Creating an empty DataFrame with expected columns")
            return pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume'])