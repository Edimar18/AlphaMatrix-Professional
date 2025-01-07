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
            'h1_ema_long': np.array([f['h1_ema_long'] for f in features]),
            'h1_ema_short': np.array([f['h1_ema_short'] for f in features]),
            'h1_bb_upper': np.array([f['h1_bb_upper'] for f in features]),
            'h1_bb_middle': np.array([f['h1_bb_middle'] for f in features]),
            'h1_bb_lower': np.array([f['h1_bb_lower'] for f in features]),
            'h1_ohlc': np.array([f['h1_ohlc'] for f in features]),
            'm5_ema_long': np.array([f['m5_ema_long'] for f in features]),
            'm5_ema_short': np.array([f['m5_ema_short'] for f in features]),
            'm5_bb_upper': np.array([f['m5_bb_upper'] for f in features]),
            'm5_bb_middle': np.array([f['m5_bb_middle'] for f in features]),
            'm5_bb_lower': np.array([f['m5_bb_lower'] for f in features]),
            'm5_ohlc': np.array([f['m5_ohlc'] for f in features])
        }
        
        labels = np.array(labels)
        
        # Save data
        print("Saving processed data...")
        save_path = os.path.join(DATA_PATH, f"{symbol}_processed")
        np.save(
            save_path,
            h1_ema_long=processed_features['h1_ema_long'],
            h1_ema_short=processed_features['h1_ema_short'],
            h1_bb_upper=processed_features['h1_bb_upper'],
            h1_bb_middle=processed_features['h1_bb_middle'],
            h1_bb_lower=processed_features['h1_bb_lower'],
            h1_ohlc=processed_features['h1_ohlc'],
            m5_ema_long=processed_features['m5_ema_long'],
            m5_ema_short=processed_features['m5_ema_short'],
            m5_bb_upper=processed_features['m5_bb_upper'],
            m5_bb_middle=processed_features['m5_bb_middle'],
            m5_bb_lower=processed_features['m5_bb_lower'],
            m5_ohlc=processed_features['m5_ohlc'],
            labels=labels
        )
        
        print(f"Data saved to {save_path}.npz")
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