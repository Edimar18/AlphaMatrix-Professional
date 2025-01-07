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
        
        # RSI
        df['rsi'] = ta.momentum.rsi(df['close'], window=RSI_PERIOD)
        
        # MACD
        macd = ta.trend.MACD(
            df['close'], 
            window_slow=MACD_SLOW,
            window_fast=MACD_FAST,
            window_sign=MACD_SIGNAL
        )
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        
        # Volume Analysis
        df['volume_ma'] = ta.trend.sma_indicator(df['tick_volume'], window=VOLUME_MA_PERIOD)
        df['volume_ratio'] = df['tick_volume'] / df['volume_ma']
        df['high_volume'] = df['volume_ratio'] > VOLUME_THRESHOLD
        
        # Volatility
        df['volatility'] = df['close'].rolling(window=VOLATILITY_WINDOW).std()
        df['volatility_ratio'] = df['volatility'] / df['volatility'].rolling(window=VOLATILITY_WINDOW).mean()
        
        return df
    
    def _get_market_session(self, timestamp):
        """Determine current market session"""
        # Convert pandas timestamp to time
        current_time = timestamp.time()
        sessions = []
        
        for session, hours in MARKET_SESSIONS.items():
            if hours['start'] <= current_time <= hours['end']:
                sessions.append(session)
        
        if len(sessions) > 1:
            return 'OVERLAP'
        return sessions[0] if sessions else 'ASIAN'  # Default to ASIAN if no session found
    
    def _calculate_dynamic_thresholds(self, df, time):
        """Calculate dynamic thresholds based on session and volatility"""
        session = self._get_market_session(time)
        session_multiplier = SESSION_THRESHOLDS[session]
        
        # Adjust thresholds based on volatility
        volatility_multiplier = 1.0
        if df['volatility_ratio'].iloc[-1] > VOLATILITY_THRESHOLD:
            volatility_multiplier = df['volatility_ratio'].iloc[-1]
        
        significant_move = BASE_SIGNIFICANT_MOVE * session_multiplier * volatility_multiplier
        minimal_move = BASE_MINIMAL_MOVE * session_multiplier * volatility_multiplier
        
        return significant_move, minimal_move
    
    def _calculate_risk_reward(self, df, price_change):
        """Calculate risk/reward ratio for potential trade"""
        if price_change == 0:
            return 0
            
        # Calculate potential profit (2x the price change)
        potential_profit = abs(price_change) * 2
        
        # Calculate potential loss (using ATR or volatility-based stop loss)
        stop_loss = df['volatility'].iloc[-1] * STOP_LOSS_MULTIPLIER
        
        return potential_profit / stop_loss if stop_loss != 0 else 0
    
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
        
        # Convert timestamps to nanoseconds at once for better performance
        m5_timestamps = pd.to_datetime(m5_data.index).values.astype(np.int64)
        h1_timestamps = pd.to_datetime(h1_data.index).values.astype(np.int64)
        
        print(f"Processing {len(h1_data) - H1_SEQUENCE_LENGTH - 1} potential sequences...")
        
        for i in range(len(h1_data) - H1_SEQUENCE_LENGTH - 1):
            try:
                # H1 sequences
                h1_seq = h1_data.iloc[i:i+H1_SEQUENCE_LENGTH]
                h1_end_time_int = h1_timestamps[i + H1_SEQUENCE_LENGTH - 1]
                
                # Find the closest M5 timestamp
                m5_end_idx = np.searchsorted(m5_timestamps, h1_end_time_int)
                if m5_end_idx >= len(m5_timestamps):
                    m5_end_idx = len(m5_timestamps) - 1
                
                m5_start_idx = m5_end_idx - M5_SEQUENCE_LENGTH + 1
                if m5_start_idx < 0:
                    continue
                    
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
                    'h1_rsi': h1_seq['rsi'].values,
                    'h1_macd': h1_seq['macd'].values,
                    'h1_macd_signal': h1_seq['macd_signal'].values,
                    'h1_macd_diff': h1_seq['macd_diff'].values,
                    'h1_volume_ratio': h1_seq['volume_ratio'].values,
                    'h1_volatility_ratio': h1_seq['volatility_ratio'].values,
                    'h1_ohlc': h1_seq[['open', 'high', 'low', 'close']].values,
                    'm5_ema_long': m5_seq['ema_long'].values,
                    'm5_ema_short': m5_seq['ema_short'].values,
                    'm5_bb_upper': m5_seq['bb_upper'].values,
                    'm5_bb_middle': m5_seq['bb_middle'].values,
                    'm5_bb_lower': m5_seq['bb_lower'].values,
                    'm5_rsi': m5_seq['rsi'].values,
                    'm5_macd': m5_seq['macd'].values,
                    'm5_macd_signal': m5_seq['macd_signal'].values,
                    'm5_macd_diff': m5_seq['macd_diff'].values,
                    'm5_volume_ratio': m5_seq['volume_ratio'].values,
                    'm5_volatility_ratio': m5_seq['volatility_ratio'].values,
                    'm5_ohlc': m5_seq[['open', 'high', 'low', 'close']].values
                }
                
                # Calculate dynamic thresholds
                significant_move, minimal_move = self._calculate_dynamic_thresholds(
                    h1_data.iloc[i:i+H1_SEQUENCE_LENGTH+1],
                    h1_data.index[i + H1_SEQUENCE_LENGTH - 1]  # Use the index directly
                )
                
                # Calculate label
                next_candle = h1_data.iloc[i+H1_SEQUENCE_LENGTH+1]
                price_change = next_candle['close'] - next_candle['open']
                risk_reward = self._calculate_risk_reward(
                    h1_data.iloc[i:i+H1_SEQUENCE_LENGTH+1],
                    price_change
                )
                
                # Determine label based on price change and risk/reward
                if abs(price_change) < minimal_move or risk_reward < MIN_RISK_REWARD_RATIO:
                    label = -1  # Not worth trading
                elif price_change >= significant_move and risk_reward >= MIN_RISK_REWARD_RATIO:
                    label = 1   # Strong buy
                elif price_change <= -significant_move and risk_reward >= MIN_RISK_REWARD_RATIO:
                    label = 0   # Strong sell
                else:
                    # Calculate normalized strength considering risk/reward
                    base_strength = price_change / significant_move
                    label = base_strength * min(risk_reward / MIN_RISK_REWARD_RATIO, 1.0)
                
                features.append(feature)
                labels.append(label)
                
                # Print progress every 500 sequences
                if len(features) % 500 == 0:
                    print(f"Processed {len(features)} valid sequences...")
                
            except Exception as e:
                print(f"Warning: Skipping sequence at index {i} due to error: {str(e)}")
                continue
        
        if not features:
            raise Exception("No valid sequences could be created. Check data alignment and sequence lengths.")
        
        print(f"Successfully created {len(features)} sequences.")
        return features, labels
    
    def get_latest_data(self, symbol):
        """Get latest data for live prediction"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=5)  # Fetch more than needed
        
        h1_data, m5_data = self.prepare_data(symbol, start_date, end_date)
        
        # Get latest sequences
        features, _ = self.create_sequences(h1_data, m5_data)
        return features[-1] if features else None 