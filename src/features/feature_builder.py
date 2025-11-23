import numpy as np
import pandas as pd


class FeatureBuilder:
    def _rsi(self, series: pd.Series, period: int = 14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _macd(self, series: pd.Series):
        ema12 = series.ewm(span=12).mean()
        ema26 = series.ewm(span=26).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9).mean()
        return macd, signal

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # EMAs
        df["ema20"] = df["Close"].ewm(span=20).mean()
        df["ema50"] = df["Close"].ewm(span=50).mean()
        df["ema200"] = df["Close"].ewm(span=200).mean()

        # RSI & MACD
        df["rsi"] = self._rsi(df["Close"])
        df["macd"], df["macd_signal"] = self._macd(df["Close"])

        # ATR (normalized)
        high = df["High"]
        low = df["Low"]
        close_prev = df["Close"].shift(1)

        tr1 = high - low
        tr2 = (high - close_prev).abs()
        tr3 = (low - close_prev).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        df["atr"] = true_range.rolling(14).mean()
        df["atr_norm"] = df["atr"] / df["Close"]

        # Body & range normalized
        df["body_norm"] = (df["Close"] - df["Open"]).abs() / df["Close"]
        df["range_norm"] = (df["High"] - df["Low"]) / df["Close"]

        # Trend strength (slope over N candles)
        window = 30
        trend_vals = []
        closes = df["Close"].values

        for i in range(len(df)):
            if i < window:
                trend_vals.append(0.0)
                continue

            y = closes[i-window:i]
            x = np.arange(window)

            if len(y) < window:
                trend_vals.append(0.0)
                continue

            slope = np.polyfit(x, y, 1)[0]
            trend_vals.append(float(slope))

        df["trend_strength"] = trend_vals

        # Compression (low ATR environment)
        df["compression"] = df["atr_norm"].rolling(20).mean()

        return df
