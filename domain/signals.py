# Signal Generation (Domain Layer)
# Single Responsibility: Convert predictions to trade signals

class SignalGenerator:
    @staticmethod
    def generate_short(prediction: float, price: float) -> str:
        """Generate trade signals for short timeframes."""
        if prediction > 0.8:
            return f"BUY @ {price:.5f}"
        elif prediction < 0.2:
            return f"SELL @ {price:.5f}"
        return "HOLD"

    @staticmethod
    def generate_medium(prediction: float, price: float) -> str:
        """Generate trade signals for medium timeframes."""
        if prediction > 0.7:
            return f"BUY @ {price:.5f}"
        elif prediction < 0.3:
            return f"SELL @ {price:.5f}"
        return "HOLD"

    @staticmethod
    def generate_long(prediction: float, price: float, threshold: float = 0.02) -> str:
        """Generate trade signals for long timeframes."""
        if prediction > price * (1 + threshold):
            return f"BUY @ {price:.5f}"
        elif prediction < price * (1 - threshold):
            return f"SELL @ {price:.5f}"
        return "HOLD"