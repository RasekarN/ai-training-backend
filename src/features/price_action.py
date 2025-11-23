import pandas as pd
import numpy as np

def candle_body(df):
    return (df['Close'] - df['Open']).abs()

def candle_range(df):
    return df['High'] - df['Low']

def upper_wick(df):
    return df['High'] - df[['Close', 'Open']].max(axis=1)

def lower_wick(df):
    return df[['Close', 'Open']].min(axis=1) - df['Low']

def wick_ratio(df):
    body = candle_body(df).replace(0, 1)
    return (upper_wick(df) + lower_wick(df)) / body

def range_compression(df, window=20):
    rng = (df['High'] - df['Low']).rolling(window).mean()
    return rng / df['Close']
