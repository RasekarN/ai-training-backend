import pandas as pd
import numpy as np

class PatternTester:

    def __init__(self, target_factor=1.5, stop_factor=1.0):
        self.target_factor = target_factor
        self.stop_factor = stop_factor

    def compute_pnl(self, df, signal_col):
        df = df.copy()

        entries = df[df[signal_col] == True].index

        results = []

        for idx in entries:
            entry_price = df.loc[idx, 'Close']

            target = entry_price * (1 + self.target_factor / 100)
            stop = entry_price * (1 - self.stop_factor / 100)

            # forward scan
            for i in range(df.index.get_loc(idx) + 1, len(df)):
                price = df['Close'].iloc[i]

                if price >= target:
                    results.append(1)
                    break
                elif price <= stop:
                    results.append(0)
                    break
            else:
                results.append(0)

        if len(results) == 0:
            return None

        win_rate = np.mean(results)
        return win_rate
