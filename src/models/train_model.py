import numpy as np
import pandas as pd
import torch

from src.models.ml_models import HybridModel
from src.features.feature_builder import FeatureBuilder


class ModelTrainer:
    def __init__(self):
        self.model = HybridModel()
        self.feature_builder = FeatureBuilder()

    # ============================================================
    #                     DATA PREPARATION
    # ============================================================
    def prepare_data(self, df: pd.DataFrame):
        df = df.dropna().reset_index(drop=True)

        # Force numeric
        numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna().reset_index(drop=True)

        # Build features
        df = self.feature_builder.build(df)
        df = df.dropna().reset_index(drop=True)

        # Target: next candle direction
        df["future"] = df["Close"].shift(-1)
        df = df.dropna().reset_index(drop=True)

        df["target"] = (df["future"] > df["Close"]).astype(int)

        feature_cols = [
            "ema20","ema50","ema200",
            "rsi","macd","macd_signal",
            "atr_norm","body_norm",
            "range_norm","trend_strength",
            "compression"
        ]

        df = df.dropna(subset=feature_cols).reset_index(drop=True)

        X_tab = df[feature_cols].values
        y_tab = df["target"].values

        # Sequence data for transformer
        seq_len = 30
        X_seq = []
        y_seq = []

        for i in range(len(df) - seq_len):
            X_seq.append(df[feature_cols].iloc[i:i+seq_len].values)
            y_seq.append(y_tab[i + seq_len])

        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)

        X_seq_t = torch.tensor(X_seq, dtype=torch.float32)
        y_seq_t = torch.tensor(y_seq, dtype=torch.long)

        return X_tab[:-seq_len], y_tab[:-seq_len], X_seq_t, y_seq_t, df, feature_cols

    # ============================================================
    #                      TRAIN MODEL
    # ============================================================
    def train(self, df: pd.DataFrame):
        X_tab, y_tab, X_seq, y_seq, df_proc, feature_cols = self.prepare_data(df)

        print("[Trainer] Training XGBoost...")
        self.model.train_xgb(X_tab, y_tab)

        print("[Trainer] Training Transformer...")
        self.model.train_transformer(X_seq, y_seq)

        # Store last df & feature_cols to reuse in prediction (if needed)
        self._last_df = df_proc
        self._last_features = feature_cols

        return self.model

    # ============================================================
    #                PREDICT ON LATEST CANDLE
    # ============================================================
    def predict_latest_from_df(self, model: HybridModel, df: pd.DataFrame):
        """
        Rebuilds features on df, takes last 30 bars and last row, and predicts next move.
        """
        df = df.dropna().reset_index(drop=True)

        numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna().reset_index(drop=True)
        df = self.feature_builder.build(df)
        df = df.dropna().reset_index(drop=True)

        feature_cols = [
            "ema20","ema50","ema200",
            "rsi","macd","macd_signal",
            "atr_norm","body_norm",
            "range_norm","trend_strength",
            "compression"
        ]

        df = df.dropna(subset=feature_cols).reset_index(drop=True)

        if len(df) < 31:
            raise ValueError("Not enough data after feature building for prediction.")

        last_seq = df[feature_cols].tail(30).values           # (30, features)
        last_tab = df[feature_cols].iloc[-1].values           # (features,)

        probs = model.predict(last_tab, last_seq)
        p_down, p_up = float(probs[0]), float(probs[1])

        if p_up > 0.6:
            signal = "BUY"
        elif p_down > 0.6:
            signal = "SELL"
        else:
            signal = "NO TRADE"

        latest_price = float(df["Close"].iloc[-1])

        return signal, p_up, p_down, latest_price
