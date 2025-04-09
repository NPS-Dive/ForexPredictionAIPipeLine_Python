# Application Configuration
class Config:
    DATA_DIR = "data"
    PAIRS = ['EURUSD', 'USDJPY', 'GBPUSD', 'AUDUSD', 'USDCHF', 'XAUUSD']
    TIMEFRAMES = {
        'short': ['M1', 'M5', 'M15', 'M30'],
        'medium': ['H1'],
        'long': ['D1', 'W1']
    }
    RANDOM_SEED = 42
    LOG_LEVEL = "INFO"
    MODEL_PARAMS = {
        'short': {'input_shape': (10, 4), 'dropout': 0.3},
        'medium': {'input_shape': (20, 6), 'dropout': 0.2},
        'long': {'input_shape': (50, 8), 'dropout': 0.1}
    }