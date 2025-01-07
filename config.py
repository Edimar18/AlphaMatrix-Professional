from datetime import datetime, time

# MT5 Configuration
MT5_TIMEFRAMES = {
    'H1': 'mt5.TIMEFRAME_H1',
    'M5': 'mt5.TIMEFRAME_M5'
}

# Data Configuration
START_DATE = datetime(2024, 9, 1)
END_DATE = datetime(2024, 12, 10)
SYMBOLS = ['EURUSD']  # Add more symbols as needed

# Market Sessions (UTC)
MARKET_SESSIONS = {
    'ASIAN': {'start': time(0, 0), 'end': time(8, 0)},
    'LONDON': {'start': time(8, 0), 'end': time(16, 0)},
    'NEW_YORK': {'start': time(13, 0), 'end': time(21, 0)}
}

# Feature Configuration
H1_EMA_LONG_PERIOD = 10
H1_EMA_SHORT_PERIOD = 20
M5_EMA_LONG_PERIOD = 10
M5_EMA_SHORT_PERIOD = 20

# Bollinger Bands
BOLLINGER_PERIOD = 20
BOLLINGER_STD = 2

# RSI Configuration
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30

# MACD Configuration
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# Volume Configuration
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5  # For high volume detection

# Sequence Lengths
H1_SEQUENCE_LENGTH = 10
M5_SEQUENCE_LENGTH = 120

# Model Configuration
MODEL_CONFIG = {
    'lstm_units': [128, 64, 32],
    'conv_filters': [64, 32],
    'dense_units': [64, 32, 16],  # Added one more layer
    'dropout_rate': 0.2,
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 100
}

# Trading Thresholds
BASE_SIGNIFICANT_MOVE = 0.002000  # Reduced from 0.002751 (~20 pips)
BASE_MINIMAL_MOVE = 0.0003  # Reduced from 0.0005 (3 pips)

# Session-specific thresholds (multipliers)
SESSION_THRESHOLDS = {
    'ASIAN': 0.9,      # Slightly less conservative
    'LONDON': 1.1,     # Slightly less aggressive
    'NEW_YORK': 1.1,   # Slightly less aggressive
    'OVERLAP': 1.3     # Slightly less aggressive
}

# Risk/Reward Configuration
MIN_RISK_REWARD_RATIO = 1.3  # Reduced from 1.5 for more trade opportunities
STOP_LOSS_MULTIPLIER = 1.3  # Reduced from 1.5

# Volatility Configuration
VOLATILITY_WINDOW = 20
VOLATILITY_THRESHOLD = 1.1  # Reduced from 1.2 for more sensitive volatility detection

# Paths
MODEL_SAVE_PATH = 'models/'
LOG_PATH = 'logs/'
DATA_PATH = 'data/'

# Training Configuration
TRAIN_TEST_SPLIT = 0.8
VALIDATION_SPLIT = 0.2
RANDOM_SEED = 42 