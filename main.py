from mt5_util import MT5Util
import MetaTrader5 as mt5
from MovingAverage import MVS

util = MT5Util()
M5_data = util.get_ohlc("EURUSD", mt5.TIMEFRAME_M5, 200)
M30_data = util.get_ohlc("EURUSD", mt5.TIMEFRAME_M30, 200)
H1_data = util.get_ohlc("EURUSD", mt5.TIMEFRAME_H1, 200)
H4_data = util.get_ohlc("EURUSD", mt5.TIMEFRAME_H4, 200)
## MOVING AVERAGES
mvs = MVS()

## SMA
SMA_H4_LONG = mvs.calculate_sma(H4_data, 200)
SMA_H4_SHORT = mvs.calculate_sma(H4_data,50)
SMA_H1_LONG = mvs.calculate_sma(H1_data, 100)
SMA_H1_SHORT = mvs.calculate_sma(H1_data, 20)
SMA_M30_LONG = mvs.calculate_sma(M30_data, 50)
SMA_H30_SHORT = mvs.calculate_sma(M30_data, 10)
SMA_M5_LONG = mvs.calculate_sma(M5_data, 20)
SMA_M5_SHORT = mvs.calculate_sma(M5_data, 5)

### EMA
EMA_H4_LONG = mvs.calculate_ema(H4_data, 100)
EMA_H4_SHORT = mvs.calculate_ema(H4_data, 21)
EMA_H1_LONG = mvs.calculate_ema(H1_data, 50)
EMA_H1_SHORT = mvs.calculate_ema(H1_data, 9)
EMA_M30_LONG = mvs.calculate_ema(M30_data, 21)
EMA_M30_SHORT = mvs.calculate_ema(M30_data, 9)
EMA_M5_LONG = mvs.calculate_ema(M5_data, 10)
EMA_M5_SHORT = mvs.calculate_ema(M5_data, 5)

### WMA
WMA_H4_LONG = mvs.calculate_wma(H4_data, 50)
WMA_H4_SHORT = mvs.calculate_wma(H4_data, 21)
WMA_H1_LONG = mvs.calculate_wma(H1_data, 30)
WMA_H1_SHORT = mvs.calculate_wma(H1_data, 14)
WMA_M30_LONG = mvs.calculate_wma(M30_data, 14)
WMA_M30_SHORT = mvs.calculate_wma(M30_data, 7)
WMA_M5_LONG = mvs.calculate_wma(M5_data, 10)
WMA_M5_SHORT = mvs.calculate_wma(M5_data, 5)

### HMA
HMA_H4_LONG = mvs.calculate_hma(H4_data, 55)
HMA_H4_SHORT = mvs.calculate_hma(H4_data, 21)
HMA_H1_LONG = mvs.calculate_hma(H1_data, 21)
HMA_H1_SHORT = mvs.calculate_hma(H1_data, 13)
HMA_M30_LONG = mvs.calculate_hma(M30_data, 13)
HMA_M30_SHORT = mvs.calculate_hma(M30_data, 9)
HMA_M5_LONG = mvs.calculate_hma(M5_data, 9)
HMA_M5_SHORT = mvs.calculate_hma(M5_data, 5)