import os
import numpy as np
import pandas as pd
from mt5_util import MT5DataFetcher
from config import *

class TrainingDataPreparator:
    def __init__(self):
        self.mt5_fetcher = MT5DataFetcher()
        os.makedirs(DATA_PATH, exist_ok=True)
        
    def prepare_training_data(self, symbol):
        """Prepare and save training data"""
        print(f"Fetching data for {symbol}...")
        
        # Fetch and prepare data
        h1_data, m5_data = self.mt5_fetcher.prepare_data(symbol, START_DATE, END_DATE)
        
        print("Creating sequences...")
        features, labels = self.mt5_fetcher.create_sequences(h1_data, m5_data)
        
        # Convert features to numpy arrays
        processed_features = {
            # H1 Features
            'h1_ema_long': np.array([f['h1_ema_long'] for f in features]),
            'h1_ema_short': np.array([f['h1_ema_short'] for f in features]),
            'h1_bb_upper': np.array([f['h1_bb_upper'] for f in features]),
            'h1_bb_middle': np.array([f['h1_bb_middle'] for f in features]),
            'h1_bb_lower': np.array([f['h1_bb_lower'] for f in features]),
            'h1_rsi': np.array([f['h1_rsi'] for f in features]),
            'h1_macd': np.array([f['h1_macd'] for f in features]),
            'h1_macd_signal': np.array([f['h1_macd_signal'] for f in features]),
            'h1_macd_diff': np.array([f['h1_macd_diff'] for f in features]),
            'h1_volume_ratio': np.array([f['h1_volume_ratio'] for f in features]),
            'h1_volatility_ratio': np.array([f['h1_volatility_ratio'] for f in features]),
            'h1_ohlc': np.array([f['h1_ohlc'] for f in features]),
            
            # M5 Features
            'm5_ema_long': np.array([f['m5_ema_long'] for f in features]),
            'm5_ema_short': np.array([f['m5_ema_short'] for f in features]),
            'm5_bb_upper': np.array([f['m5_bb_upper'] for f in features]),
            'm5_bb_middle': np.array([f['m5_bb_middle'] for f in features]),
            'm5_bb_lower': np.array([f['m5_bb_lower'] for f in features]),
            'm5_rsi': np.array([f['m5_rsi'] for f in features]),
            'm5_macd': np.array([f['m5_macd'] for f in features]),
            'm5_macd_signal': np.array([f['m5_macd_signal'] for f in features]),
            'm5_macd_diff': np.array([f['m5_macd_diff'] for f in features]),
            'm5_volume_ratio': np.array([f['m5_volume_ratio'] for f in features]),
            'm5_volatility_ratio': np.array([f['m5_volatility_ratio'] for f in features]),
            'm5_ohlc': np.array([f['m5_ohlc'] for f in features])
        }
        
        labels = np.array(labels)
        
        # Save data
        print("Saving processed data...")
        save_path = os.path.join(DATA_PATH, f"{symbol}_processed.npz")
        np.savez(
            save_path,
            **processed_features,  # Unpack all features
            labels=labels
        )
        
        print(f"Data saved to {save_path}")
        print(f"Total sequences: {len(labels)}")
        print("\nLabel distribution:")
        print(f"Strong Buy (1): {np.sum(labels == 1)}")
        print(f"Strong Sell (0): {np.sum(labels == 0)}")
        print(f"No Trade (-1): {np.sum(labels == -1)}")
        print(f"Partial Signals: {np.sum((labels > 0) & (labels < 1))} (Buy), "
              f"{np.sum((labels < 0) & (labels > -1))} (Sell)")

if __name__ == "__main__":
    preparator = TrainingDataPreparator()
    for symbol in SYMBOLS:
        preparator.prepare_training_data(symbol) 