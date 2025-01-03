import MetaTrader5 as mt5
from datetime import datetime, timedelta

class MT5Util:
    def __init__(self):
        if not mt5.initialize():
            print("Initialization failed")
            mt5.shutdown()

    def get_last_n_candles(self, symbol, timeframe, n):
        """
        Retrieve the last n candles for a given symbol and timeframe.
        
        :param symbol: Trading symbol (e.g., 'EURUSD').
        :param timeframe: Timeframe (e.g., mt5.TIMEFRAME_H1).
        :param n: Number of candles to retrieve.
        :return: List of candle data.
        """
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n)
        if rates is None:
            print(f"Failed to get rates for {symbol} on timeframe {timeframe}")
            return []
        return rates

    def get_candle_by_time(self, symbol, timeframe, time):
        """
        Retrieve candle data at a specific time.
        
        :param symbol: Trading symbol (e.g., 'EURUSD').
        :param timeframe: Timeframe (e.g., mt5.TIMEFRAME_H1).
        :param time: Specific datetime to retrieve the candle.
        :return: Candle data or None if not found.
        """
        timestamp = int(time.timestamp())
        rates = mt5.copy_rates_range(symbol, timeframe, time - timedelta(minutes=1), time + timedelta(minutes=1))
        
        if rates is None or len(rates) == 0:
            print(f"No candle found for {symbol} at {time}")
            return None
        return rates[-1]  # Return the last candle within the range

    def get_candles_in_range(self, symbol, timeframe, start_time, end_time):
        """
        Retrieve candle data within a specific date range.
        
        :param symbol: Trading symbol (e.g., 'EURUSD').
        :param timeframe: Timeframe (e.g., mt5.TIMEFRAME_H1).
        :param start_time: Start datetime.
        :param end_time: End datetime.
        :return: List of candle data.
        """
        rates = mt5.copy_rates_range(symbol, timeframe, start_time, end_time)
        
        if rates is None:
            print(f"Failed to get rates for {symbol} between {start_time} and {end_time}")
            return []
        
        return rates

    def shutdown(self):
        """Shutdown MT5 connection."""
        mt5.shutdown()

# Example usage
if __name__ == "__main__":
    util = MT5Util()
    
    # Get last 10 hourly candles for EURUSD
    cand
