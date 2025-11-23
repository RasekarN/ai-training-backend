import pandas as pd

class StatsReport:

    def create_pattern_table(self, pattern_results):
        df = pd.DataFrame(pattern_results)
        df.to_csv("data/backtest/pattern_stats.csv", index=False)
        return df

    def create_model_report(self, model_stats):
        df = pd.DataFrame([model_stats])
        df.to_csv("data/backtest/model_stats.csv", index=False)
        return df

    def create_combined_report(self, combined_stats):
        df = pd.DataFrame([combined_stats])
        df.to_csv("data/backtest/combined_stats.csv", index=False)
        return df
