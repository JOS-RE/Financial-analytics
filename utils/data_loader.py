import yfinance as yf
import pandas as pd
import numpy as np

def get_price_data(tickers, start, end):
    """
    Fetch adjusted close prices for given tickers
    """
    data = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False
    )["Close"]

    return data.dropna()


def get_returns(price_data, log_returns=False):
    """
    Compute returns from price data
    """
    if log_returns:
        returns = (price_data / price_data.shift(1)).apply(
            lambda x: pd.Series(np.log(x))
        )
    else:
        returns = price_data.pct_change()

    return returns.dropna()
