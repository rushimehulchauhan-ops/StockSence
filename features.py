import pandas as pd


FEATURE_COLUMNS = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "Daily_Return",
    "SMA_20",
    "SMA_50",
    "EMA_20",
    "EMA_50",
    "RSI_14"
]


def create_target(df):
    """Create next-day direction target: 1 = UP, 0 = DOWN."""
    df = df.copy()

    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

    # Last row has no next-day value.
    df = df.iloc[:-1].copy()

    return df


def prepare_ml_data(df):
    """Prepare features and target for machine learning."""
    df = df.copy()

    df = create_target(df)

    df = df.dropna(subset=FEATURE_COLUMNS + ["Target"]).copy()

    X = df[FEATURE_COLUMNS]
    y = df["Target"]

    return df, X, y
