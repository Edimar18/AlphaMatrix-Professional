from mt5_util import MT5Util
import MetaTrader5 as mt5
from MovingAverage import MVS

util = MT5Util()
data = util.get_ohlc("EURUSD", mt5.TIMEFRAME_M5, 5)

## MOVING AVERAGES
mvs = MVS()
SMA_H4_LONG = mvs.calculate_sma(data, 50)