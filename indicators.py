import pandas as pd


def add_daily_return(df):
    """Calculate daily percentage return."""
    df["Daily_Return"] = df["Close"].pct_change() * 100
    return df


def add_sma(df, window):
    """Calculate Simple Moving Average."""
    df[f"SMA_{window}"] = df["Close"].rolling(window).mean()
    return df


def add_ema(df, window):
    """Calculate Exponential Moving Average."""
    df[f"EMA_{window}"] = df["Close"].ewm(span=window, adjust=False).mean()
    return df


def add_rsi(df, window=14):
    """Calculate Relative Strength Index."""
    delta = df["Close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()

    rs = avg_gain / avg_loss

    df[f"RSI_{window}"] = 100 - (100 / (1 + rs))

    return df


def add_all_indicators(df):
    """Add all technical indicators used by the project."""
    df = add_daily_return(df)
    df = add_sma(df, 20)
    df = add_sma(df, 50)
    df = add_ema(df, 20)
    df = add_ema(df, 50)
    df = add_rsi(df, 14)

    return df
