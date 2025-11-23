import numpy as np

class CombinedTester:

    def backtest(self, df, prob_col="ml_prob_up", threshold=0.65):
        df = df.copy()
        signals = df[prob_col] > threshold

        entries = df[signals].index
        results = []

        for idx in entries:
            entry = df.loc[idx, 'Close']
            target = entry * 1.01
            stop = entry * 0.99

            for i in range(df.index.get_loc(idx)+1, len(df)):
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

        return np.mean(results)
