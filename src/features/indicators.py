import pandas as pd
import numpy as np

# ------------- EMA / SMA -------------------

def ema(df, length=20):
    return df['Close'].ewm(span=length).mean()

def sma(df, length=20):
    return df['Close'].rolling(length).mean()


# ------------- RSI -------------------------

def rsi(df, length=14):
    delta = df['Close'].diff()
    gain = (delta.clip(lower=0)).rolling(length).mean()
    loss = (-delta.clip(upper=0)).rolling(length).mean()

    rs = gain / loss
    return 100 - (100 / (1 + rs))


# ------------- MACD -------------------------

def macd(df):
    ema12 = df['Close'].ewm(span=12).mean()
    ema26 = df['Close'].ewm(span=26).mean()
    macd_line = ema12 - ema26
    signal = macd_line.ewm(span=9).mean()
    hist = macd_line - signal
    return macd_line, signal, hist


# ------------- ATR (Volatility) -------------

def atr(df, length=14):
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close_prev = df["Close"].shift(1).astype(float)

    tr1 = high - low
    tr2 = (high - close_prev).abs()
    tr3 = (low - close_prev).abs()

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return true_range.rolling(length).mean()

