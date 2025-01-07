from datetime import datetime

# MT5 Configuration
MT5_TIMEFRAMES = {
    'H1': 'mt5.TIMEFRAME_H1',
    'M5': 'mt5.TIMEFRAME_M5'
}

# Data Configuration
START_DATE = datetime(2024, 9, 1)
END_DATE = datetime(2024, 12, 31)
SYMBOLS = ['EURUSD']  # Add more symbols as needed

# Feature Configuration
H1_EMA_LONG_PERIOD = 10
H1_EMA_SHORT_PERIOD = 20
M5_EMA_LONG_PERIOD = 10
M5_EMA_SHORT_PERIOD = 20
BOLLINGER_PERIOD = 20
BOLLINGER_STD = 2

# Sequence Lengths
H1_SEQUENCE_LENGTH = 10
M5_SEQUENCE_LENGTH = 120

# Model Configuration
MODEL_CONFIG = {
    'lstm_units': [128, 64, 32],
    'conv_filters': [64, 32],
    'dense_units': [32, 16],
    'dropout_rate': 0.2,
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 100
}

# Trading Thresholds
SIGNIFICANT_MOVE_THRESHOLD = 0.002751  # ~27.5 pips
MINIMAL_MOVE_THRESHOLD = 0.0005  # 5 pips

# Paths
MODEL_SAVE_PATH = 'models/'
LOG_PATH = 'logs/'
DATA_PATH = 'data/'

# Training Configuration
TRAIN_TEST_SPLIT = 0.8
VALIDATION_SPLIT = 0.2
RANDOM_SEED = 42 