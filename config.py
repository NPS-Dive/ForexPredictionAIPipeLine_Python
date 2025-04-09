# Configuration (Cross-Cutting Concern)
# Single Responsibility: Define global constants

PAIRS = ['EURUSD', 'USDJPY', 'GBPUSD', 'AUDUSD', 'USDCHF', 'XAUUSD']
TIMEFRAMES = {
    'short': ['M1', 'M5', 'M15', 'M30'],
    'medium': ['H1'],
    'long': ['D1', 'W1']
}
DEFAULT_START_DATE = "2023-01-01"
DEFAULT_END_DATE = "2023-12-31"