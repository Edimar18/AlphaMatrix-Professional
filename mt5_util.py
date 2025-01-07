import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import ta
from config import *

class MT5DataFetcher:
    def __init__(self):
        if not mt5.initialize():
            raise Exception("MT5 initialization failed")
            
    def __del__(self):
        mt5.shutdown()
    
    def _fetch_ohlc(self, symbol, timeframe, start_date, end_date):
        """Fetch OHLC data from MT5"""
        rates = mt5.copy_rates_range(symbol, eval(timeframe), start_date, end_date)
        if rates is None:
            raise Exception(f"Failed to fetch data for {symbol}")
        
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        return df.set_index('time')
    
    def _add_indicators(self, df, ema_long, ema_short, bb_period):
        """Add technical indicators to dataframe"""
        # EMAs
        df['ema_long'] = ta.trend.ema_indicator(df['close'], window=ema_long)
        df['ema_short'] = ta.trend.ema_indicator(df['close'], window=ema_short)
        
        # Bollinger Bands
        bb_ind = ta.volatility.BollingerBands(df['close'], bb_period, BOLLINGER_STD)
        df['bb_upper'] = bb_ind.bollinger_hband()
        df['bb_middle'] = bb_ind.bollinger_mavg()
        df['bb_lower'] = bb_ind.bollinger_lband()
        
        return df
    
    def prepare_data(self, symbol, start_date, end_date):
        """Prepare data for both timeframes with indicators"""
        # Extend start date to account for indicator calculation
        extended_start = start_date - timedelta(days=30)
        
        # Fetch H1 data
        h1_data = self._fetch_ohlc(symbol, MT5_TIMEFRAMES['H1'], extended_start, end_date)
        h1_data = self._add_indicators(h1_data, H1_EMA_LONG_PERIOD, H1_EMA_SHORT_PERIOD, BOLLINGER_PERIOD)
        
        # Fetch M5 data
        m5_data = self._fetch_ohlc(symbol, MT5_TIMEFRAMES['M5'], extended_start, end_date)
        m5_data = self._add_indicators(m5_data, M5_EMA_LONG_PERIOD, M5_EMA_SHORT_PERIOD, BOLLINGER_PERIOD)
        
        # Remove NaN values from indicator calculation
        h1_data = h1_data.dropna()
        m5_data = m5_data.dropna()
        
        return h1_data, m5_data
    
    def create_sequences(self, h1_data, m5_data):
        """Create sequences for model input"""
        features = []
        labels = []
        
        for i in range(len(h1_data) - H1_SEQUENCE_LENGTH - 1):
            # H1 sequences
            h1_seq = h1_data.iloc[i:i+H1_SEQUENCE_LENGTH]
            
            # Find corresponding M5 data
            h1_end_time = h1_seq.index[-1]
            m5_end_idx = m5_data.index.get_loc(h1_end_time, method='nearest')
            m5_start_idx = m5_end_idx - M5_SEQUENCE_LENGTH + 1
            m5_seq = m5_data.iloc[m5_start_idx:m5_end_idx+1]
            
            if len(m5_seq) != M5_SEQUENCE_LENGTH:
                continue
                
            # Create feature vector
            feature = {
                'h1_ema_long': h1_seq['ema_long'].values,
                'h1_ema_short': h1_seq['ema_short'].values,
                'h1_bb_upper': h1_seq['bb_upper'].values,
                'h1_bb_middle': h1_seq['bb_middle'].values,
                'h1_bb_lower': h1_seq['bb_lower'].values,
                'h1_ohlc': h1_seq[['open', 'high', 'low', 'close']].values,
                'm5_ema_long': m5_seq['ema_long'].values,
                'm5_ema_short': m5_seq['ema_short'].values,
                'm5_bb_upper': m5_seq['bb_upper'].values,
                'm5_bb_middle': m5_seq['bb_middle'].values,
                'm5_bb_lower': m5_seq['bb_lower'].values,
                'm5_ohlc': m5_seq[['open', 'high', 'low', 'close']].values
            }
            
            # Calculate label
            next_candle = h1_data.iloc[i+H1_SEQUENCE_LENGTH+1]
            price_change = next_candle['close'] - next_candle['open']
            
            if abs(price_change) < MINIMAL_MOVE_THRESHOLD:
                label = -1  # Not worth trading
            elif price_change >= SIGNIFICANT_MOVE_THRESHOLD:
                label = 1   # Strong buy
            elif price_change <= -SIGNIFICANT_MOVE_THRESHOLD:
                label = 0   # Strong sell
            else:
                # Calculate normalized strength
                label = price_change / SIGNIFICANT_MOVE_THRESHOLD
            
            features.append(feature)
            labels.append(label)
        
        return features, labels
    
    def get_latest_data(self, symbol):
        """Get latest data for live prediction"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=5)  # Fetch more than needed
        
        h1_data, m5_data = self.prepare_data(symbol, start_date, end_date)
        
        # Get latest sequences
        features, _ = self.create_sequences(h1_data, m5_data)
        return features[-1] if features else None 