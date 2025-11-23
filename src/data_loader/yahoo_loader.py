import yfinance as yf
import pandas as pd
import os

class YahooDataLoader:

    def __init__(self, save_path="data/raw/"):
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)

    def fetch(self, symbol, interval="1d", period="5y"):
        df = yf.download(
            symbol,
            interval=interval,
            period=period,
            auto_adjust=True,
            progress=False
        )

        # Fix multi-index (sometimes happens)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Drop NaN rows
        df = df.dropna().reset_index(drop=True)

        # Force numeric dtypes
        numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna().reset_index(drop=True)
        return df
