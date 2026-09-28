import pandas as pd


def load_stock_data(file_path):
    """Load stock data from a CSV file."""
    df = pd.read_csv(file_path)

    df["Date"] = pd.to_datetime(df["Date"])

    return df


def load_ml_data(file_path):
    """Load the processed machine-learning dataset."""
    df = pd.read_csv(file_path)

    df["Date"] = pd.to_datetime(df["Date"])

    return df
