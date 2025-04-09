# Command-Line Interface (Presentation Layer)
# Single Responsibility: Handle user input and display results

from application.short_predictor import ShortTrendPredictor
from application.medium_predictor import MediumTrendPredictor
from application.long_predictor import LongTrendPredictor
from config import PAIRS, TIMEFRAMES, DEFAULT_START_DATE, DEFAULT_END_DATE

class ForexCLI:
    def __init__(self):
        """Initialize CLI with default dates."""
        self.start_date = DEFAULT_START_DATE
        self.end_date = DEFAULT_END_DATE

    def run(self):
        """Run the CLI application."""
        print("Welcome to Forex Prediction!")
        try:
            self.start_date = input(f"Enter start date (YYYY-MM-DD, default {self.start_date}): ") or self.start_date
            self.end_date = input(f"Enter end date (YYYY-MM-DD, default {self.end_date}): ") or self.end_date
            symbol = self._select_symbol()
            timeframe = self._select_timeframe()
            predictor = self._get_predictor(symbol, timeframe)

            X, y, df_processed = predictor.preprocess_data()
            df_processed.to_csv(f"data/{timeframe.lower()}/{symbol.lower()}_processed.csv", index=True)
            print(f"Data processed. Shape: {X.shape}")
            
            predictor.train(X, y)
            print("Models trained.")
            
            latest = X[-1:]
            price = df_processed['Close'].iloc[-1]
            volatility = df_processed['Close'].pct_change().std() * 100
            result = predictor.predict(latest, price, volatility)
            
            print(f"\nPrediction: {result['pred']:.5f}")
            print(f"Trade Signal: {result['signal']}")
            print(f"Monte Carlo Results: {result['monte_carlo']}")
        except Exception as e:
            print(f"Error: {e}")

    def _select_symbol(self) -> str:
        """Prompt user to select a currency pair."""
        print("\nAvailable Pairs:")
        for i, pair in enumerate(PAIRS, 1):
            print(f"{i}. {pair}")
        while True:
            try:
                choice = int(input(f"Select a pair (1-{len(PAIRS)}): ")) - 1
                if 0 <= choice < len(PAIRS):
                    return PAIRS[choice]
                print("Invalid selection. Try again.")
            except ValueError:
                print("Please enter a number.")

    def _select_timeframe(self) -> str:
        """Prompt user to select a timeframe."""
        print("\nAvailable Timeframes:")
        all_timeframes = TIMEFRAMES['short'] + TIMEFRAMES['medium'] + TIMEFRAMES['long']
        for i, tf in enumerate(all_timeframes, 1):
            print(f"{i}. {tf}")
        while True:
            try:
                choice = int(input(f"Select a timeframe (1-{len(all_timeframes)}): ")) - 1
                if 0 <= choice < len(all_timeframes):
                    return all_timeframes[choice]
                print("Invalid selection. Try again.")
            except ValueError:
                print("Please enter a number.")

    def _get_predictor(self, symbol: str, timeframe: str):
        """Return the appropriate predictor based on timeframe."""
        if timeframe in TIMEFRAMES['short']:
            return ShortTrendPredictor(symbol, self.start_date, self.end_date, timeframe)
        elif timeframe in TIMEFRAMES['medium']:
            return MediumTrendPredictor(symbol, self.start_date, self.end_date, timeframe)
        else:
            return LongTrendPredictor(symbol, self.start_date, self.end_date, timeframe)