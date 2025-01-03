import MetaTrader5 as mt5
from datetime import datetime
import pandas as pd

class MT5Util:
    def __init__(self):
        # Initialize connection to MetaTrader 5
        if not mt5.initialize():
            print("initialize() failed")
            mt5.shutdown()

    def get_ohlc(self, symbol, timeframe, n_candles):
        """
        Get OHLC data for the specified symbol and timeframe.

        :param symbol: Trading symbol (e.g., 'EURUSD')
        :param timeframe: Timeframe (e.g., mt5.TIMEFRAME_H1)
        :param n_candles: Number of candles to retrieve
        :return: DataFrame containing OHLC data
        """
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n_candles)
        if rates is None:
            print(f"Failed to get rates for {symbol}")
            return None
        
        # Convert to DataFrame and localize to UTC
        ohlc = pd.DataFrame(rates)
        ohlc['time'] = pd.to_datetime(ohlc['time'], unit='s').dt.tz_localize('UTC')
        return ohlc[['time', 'open', 'high', 'low', 'close']]

    def get_previous_n_candle_ohlc(self, symbol, timeframe, n):
        """
        Get OHLC data of the previous nth candle.

        :param symbol: Trading symbol (e.g., 'EURUSD')
        :param timeframe: Timeframe (e.g., mt5.TIMEFRAME_H1)
        :param n: The nth candle back from the current candle
        :return: A dictionary containing OHLC values of the nth candle
        """
        ohlc_data = self.get_ohlc(symbol, timeframe, n + 1)  # Get n+1 candles to access the nth one
        if ohlc_data is None or len(ohlc_data) < n + 1:
            return None
        
        nth_candle = ohlc_data.iloc[n]
        return {
            'time': nth_candle['time'],
            'open': nth_candle['open'],
            'high': nth_candle['high'],
            'low': nth_candle['low'],
            'close': nth_candle['close']
        }

    def shutdown(self):
        """Shutdown the MetaTrader 5 connection."""
        mt5.shutdown()


