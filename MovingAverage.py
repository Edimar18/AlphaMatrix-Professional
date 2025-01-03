import MetaTrader5 as mt5
import mt5_util
import pandas as pd
import numpy as np


class MVS():
    def calculate_sma(data: pd.DataFrame, period: int) -> pd.Series:
        """
        Calculate Simple Moving Average (SMA) for a given DataFrame and period.

        Args:
            data (pd.DataFrame): DataFrame containing the data to calculate SMA for.
            period (int): Period for the SMA calculation.

        Returns:
            pd.Series: SMA values for the given DataFrame and period.
        """
        return data['Close'].rolling(window=period).mean()
def calculate_ema(data: pd.DataFrame, period: int) -> pd.Series:
    """
    Calculate Exponential Moving Average (EMA) for a given DataFrame and period.

    Args:
        data (pd.DataFrame): DataFrame containing the data to calculate EMA for.
        period (int): Period for the EMA calculation.

    Returns:
        pd.Series: EMA values for the given DataFrame and period.
    """
    return data['Close'].ewm(span=period, adjust=False).mean()
def calculate_wma(data: pd.DataFrame, period: int) -> pd.Series:
    """
    Calculate Weighted Moving Average (WMA) for a given DataFrame and period.

    Args:
        data (pd.DataFrame): DataFrame containing the data to calculate WMA for.
        period (int): Period for the WMA calculation.

    Returns:
        pd.Series: WMA values for the given DataFrame and period.
    """
    weights = np.arange(1, period + 1)
    weights = weights / weights.sum()
    return data['Close'].rolling(window=period).apply(lambda x: np.dot(x, weights))
def calculate_hma(self, data: pd.DataFrame, period: int) -> pd.Series:
    """
    Calculate Hull Moving Average (HMA) for a given DataFrame and period.

    Args:
        data (pd.DataFrame): DataFrame containing the data to calculate HMA for.
        period (int): Period for the HMA calculation.

    Returns:
        pd.Series: HMA values for the given DataFrame and period.
    """
    wma = self.calculate_wma(data, period)
    hma = 2 * wma - wma.rolling(window=period // 2).mean()
    return hma