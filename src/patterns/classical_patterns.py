import pandas as pd
import numpy as np

class ClassicalPatterns:
    def detect_double_top(self, df, tol=0.005):
        df = df.copy()
        df["double_top"] = False
        df["double_top_breakout"] = False

        peaks = df[df["swing_high"]].index.tolist()

        for i in range(1, len(peaks)):
            p1, p2 = peaks[i-1], peaks[i]

            if abs(df.loc[p1, "High"] - df.loc[p2, "High"]) / df.loc[p1, "High"] <= tol:
                df.at[p2, "double_top"] = True

                neckline = df["Low"].loc[p1:p2].min()
                if df["Close"].iloc[-1] < neckline:
                    df.at[df.index[-1], "double_top_breakout"] = True

        return df
