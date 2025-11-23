import pandas as pd

class StructureEngine:
    def detect_bos(self, df):
        df = df.copy()
        df['BOS'] = None

        last_high = None
        last_low = None

        for i in range(len(df)):
            row = df.iloc[i]

            if row['swing_high']:
                last_high = row['High']
            if row['swing_low']:
                last_low = row['Low']

            if last_high and row['Close'] > last_high:
                df.at[df.index[i], 'BOS'] = "bullish"

            if last_low and row['Close'] < last_low:
                df.at[df.index[i], 'BOS'] = "bearish"

        return df
