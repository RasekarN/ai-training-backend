from src.features.feature_builder import FeatureBuilder
from src.models.predict import Predictor

class LiveEngine:

    def __init__(self, hybrid_model):
        self.feature_builder = FeatureBuilder()
        self.predictor = Predictor(hybrid_model)

    def analyze(self, df):
        df = df.dropna().reset_index(drop=True)
        df = self.feature_builder.build(df)
        df = df.dropna().reset_index(drop=True)

        result = self.predictor.predict_next_move(df)

        final_signal = self.classify_signal(result["ml_prob_up"])

        return {
            "signal": final_signal,
            "ml_prob_up": float(result["ml_prob_up"]),
            "ml_prob_down": float(result["ml_prob_down"]),
            "latest_close": float(df["Close"].iloc[-1])
        }

    def classify_signal(self, prob):
        if prob > 0.85: return "STRONG BUY"
        if prob > 0.70: return "BUY"
        if prob < 0.15: return "STRONG SELL"
        if prob < 0.30: return "SELL"
        return "NO TRADE"
