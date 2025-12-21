import numpy as np
import pandas as pd

# =====================================================
# ================= INDICATORS ========================
# =====================================================

def SMA(series, window):
    return series.rolling(window).mean()


# =====================================================
# ================= METRICS ===========================
# =====================================================

def compute_metrics(algo_returns, rf=0.065):
    algo_returns = algo_returns.dropna()
    ann_return = ((algo_returns.mean() + 1) ** 252 - 1) * 100
    std_dev = algo_returns.std() * np.sqrt(252) * 100
    sharpe = (ann_return / 100 - rf) / (std_dev / 100)
    return ann_return, std_dev, sharpe


# =====================================================
# =============== SMA STRATEGIES ======================
# =====================================================

def sma_long_only(prices, returns, fast=7, slow=14, rf=0.065):
    df = pd.DataFrame({
        "Price": prices,
        "Returns": returns,
        "SMA_Fast": SMA(prices, fast),
        "SMA_Slow": SMA(prices, slow)
    }).dropna()

    df["Position"] = (df["SMA_Fast"] >= df["SMA_Slow"]).astype(int)
    df["Signal"] = df["Position"].diff().fillna(0)   # ✅ Added
    df["Trade"] = np.where(df["Position"] == 1, "Long", "Cash")
    df["AlgoRet"] = df["Returns"] * df["Position"]

    ann_ret, sd, sharpe = compute_metrics(df["AlgoRet"], rf)
    trades = df["Position"].diff().abs().sum()

    return {
        "Strategy": f"SMA({fast},{slow}) Long-Only",
        "Data": df,
        "AnnualizedReturn": ann_ret,
        "StdDev": sd,
        "SharpeRatio": sharpe,
        "Trades": trades
    }


def sma_short_only(prices, returns, fast=7, slow=14, rf=0.065):
    df = pd.DataFrame({
        "Price": prices,
        "Returns": returns,
        "SMA_Fast": SMA(prices, fast),
        "SMA_Slow": SMA(prices, slow)
    }).dropna()

    df["Position"] = np.where(df["SMA_Fast"] < df["SMA_Slow"], -1, 0)
    df["Signal"] = df["Position"].diff().fillna(0)   # ✅ Added
    df["Trade"] = np.where(df["Position"] == -1, "Short", "Cash")
    df["AlgoRet"] = df["Returns"] * df["Position"]

    ann_ret, sd, sharpe = compute_metrics(df["AlgoRet"], rf)
    trades = df["Position"].diff().abs().sum()

    return {
        "Strategy": f"SMA({fast},{slow}) Short-Only",
        "Data": df,
        "AnnualizedReturn": ann_ret,
        "StdDev": sd,
        "SharpeRatio": sharpe,
        "Trades": trades
    }


def sma_long_short(prices, returns, fast=7, slow=14, rf=0.065):
    df = pd.DataFrame({
        "Price": prices,
        "Returns": returns,
        "SMA_Fast": SMA(prices, fast),
        "SMA_Slow": SMA(prices, slow)
    }).dropna()

    df["Position"] = np.where(df["SMA_Fast"] >= df["SMA_Slow"], 1, -1)
    df["Signal"] = df["Position"].diff().fillna(0)   # ✅ Added
    df["Trade"] = np.where(df["Position"] == 1, "Long", "Short")
    df["AlgoRet"] = df["Returns"] * df["Position"]

    ann_ret, sd, sharpe = compute_metrics(df["AlgoRet"], rf)
    trades = df["Position"].diff().abs().sum() / 2

    return {
        "Strategy": f"SMA({fast},{slow}) Long-Short",
        "Data": df,
        "AnnualizedReturn": ann_ret,
        "StdDev": sd,
        "SharpeRatio": sharpe,
        "Trades": trades
    }


# =====================================================
# ================= RSI STRATEGIES ====================
# =====================================================

def RSI(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def rsi_long_only(prices, returns, period=14, oversold=30, overbought=70, rf=0.065):
    df = pd.DataFrame({
        "Price": prices,
        "Returns": returns,
        "RSI": RSI(prices, period)
    }).dropna()

    df["Position"] = 0
    for i in range(1, len(df)):
        if df["RSI"].iloc[i] <= oversold:
            df.iloc[i, df.columns.get_loc("Position")] = 1
        elif df["RSI"].iloc[i] >= overbought:
            df.iloc[i, df.columns.get_loc("Position")] = 0
        else:
            df.iloc[i, df.columns.get_loc("Position")] = df["Position"].iloc[i-1]

    df["Signal"] = df["Position"].diff().fillna(0)   # ✅ Added
    df["Trade"] = np.where(df["Position"] == 1, "Long", "Cash")
    df["AlgoRet"] = df["Returns"] * df["Position"]

    ann_ret, sd, sharpe = compute_metrics(df["AlgoRet"], rf)
    trades = df["Position"].diff().abs().sum()

    return {
        "Strategy": f"RSI({period})[{oversold},{overbought}] Long-Only",
        "Data": df,
        "AnnualizedReturn": ann_ret,
        "StdDev": sd,
        "SharpeRatio": sharpe,
        "Trades": trades
    }


def rsi_short_only(prices, returns, period=14, oversold=30, overbought=70, rf=0.065):
    df = pd.DataFrame({
        "Price": prices,
        "Returns": returns,
        "RSI": RSI(prices, period)
    }).dropna()

    df["Position"] = 0
    for i in range(1, len(df)):
        if df["RSI"].iloc[i] >= overbought:
            df.iloc[i, df.columns.get_loc("Position")] = -1
        elif df["RSI"].iloc[i] <= oversold:
            df.iloc[i, df.columns.get_loc("Position")] = 0
        else:
            df.iloc[i, df.columns.get_loc("Position")] = df["Position"].iloc[i-1]

    df["Signal"] = df["Position"].diff().fillna(0)   # ✅ Added
    df["Trade"] = np.where(df["Position"] == -1, "Short", "Cash")
    df["AlgoRet"] = df["Returns"] * df["Position"]

    ann_ret, sd, sharpe = compute_metrics(df["AlgoRet"], rf)
    trades = df["Position"].diff().abs().sum()

    return {
        "Strategy": f"RSI({period})[{oversold},{overbought}] Short-Only",
        "Data": df,
        "AnnualizedReturn": ann_ret,
        "StdDev": sd,
        "SharpeRatio": sharpe,
        "Trades": trades
    }


def rsi_long_short(prices, returns, period=14, oversold=30, overbought=70, rf=0.065):
    df = pd.DataFrame({
        "Price": prices,
        "Returns": returns,
        "RSI": RSI(prices, period)
    }).dropna()

    df["Position"] = 0
    for i in range(1, len(df)):
        if df["RSI"].iloc[i] <= oversold:
            df.iloc[i, df.columns.get_loc("Position")] = 1
        elif df["RSI"].iloc[i] >= overbought:
            df.iloc[i, df.columns.get_loc("Position")] = -1
        else:
            df.iloc[i, df.columns.get_loc("Position")] = df["Position"].iloc[i-1]

    df["Signal"] = df["Position"].diff().fillna(0)   # ✅ Added
    df["Trade"] = np.where(
        df["Position"] == 1, "Long",
        np.where(df["Position"] == -1, "Short", "Neutral")
    )
    df["AlgoRet"] = df["Returns"] * df["Position"]

    ann_ret, sd, sharpe = compute_metrics(df["AlgoRet"], rf)
    trades = df["Position"].diff().abs().sum() / 2

    return {
        "Strategy": f"RSI({period})[{oversold},{overbought}] Long-Short",
        "Data": df,
        "AnnualizedReturn": ann_ret,
        "StdDev": sd,
        "SharpeRatio": sharpe,
        "Trades": trades
    }


def custom_triple_sma(prices, returns, rf=0.065):
    sma7 = SMA(prices, 7)
    sma14 = SMA(prices, 14)
    sma21 = SMA(prices, 21)

    df = pd.DataFrame({
        "Price": prices,
        "Returns": returns,
        "SMA7": sma7,
        "SMA14": sma14,
        "SMA21": sma21
    }).dropna()

    df["Position"] = 0
    df["Signal"] = 0   # ✅ Numeric signal added

    for i in range(1, len(df)):
        prev_pos = df["Position"].iloc[i - 1]

        if (
            df["SMA7"].iloc[i] > df["SMA14"].iloc[i]
            and df["SMA14"].iloc[i] > df["SMA21"].iloc[i]
            and prev_pos == 0
        ):
            df.iloc[i, df.columns.get_loc("Position")] = 1
            df.iloc[i, df.columns.get_loc("Signal")] = 1   # Entry

        elif (
            df["SMA14"].iloc[i] < df["SMA21"].iloc[i]
            and prev_pos == 1
        ):
            df.iloc[i, df.columns.get_loc("Position")] = 0
            df.iloc[i, df.columns.get_loc("Signal")] = -1  # Exit

        else:
            df.iloc[i, df.columns.get_loc("Position")] = prev_pos

    df["Trade"] = np.where(df["Position"] == 1, "Long", "Cash")
    df["AlgoRet"] = df["Returns"] * df["Position"]

    ann_ret, sd, sharpe = compute_metrics(df["AlgoRet"], rf)
    trades = (df["Signal"] != 0).sum()

    return {
        "Strategy": "Custom Triple SMA (7,14,21)",
        "Data": df,
        "AnnualizedReturn": ann_ret,
        "StdDev": sd,
        "SharpeRatio": sharpe,
        "Trades": trades
    }
