import torch

class Predictor:

    def __init__(self, hybrid_model):
        self.model = hybrid_model

    def predict_next_move(self, df):
        last_row = df.iloc[-1]

        # Tabular feature values
        tab_features = last_row[
            ['ema20','ema50','ema200','rsi','macd',
             'macd_signal','atr_norm','body_norm',
             'range_norm','trend_strength','compression']
        ].values.reshape(1, -1)

        # Sequence for Transformer
        seq_features = df[
            ['ema20','ema50','ema200','rsi','macd',
             'macd_signal','atr_norm','body_norm',
             'range_norm','trend_strength','compression']
        ].tail(30).values

        seq_tensor = torch.tensor(seq_features).unsqueeze(0).float()

        return self.model.predict(tab_features, seq_tensor)
