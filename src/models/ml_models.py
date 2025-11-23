import torch
import torch.nn as nn
import torch.optim as optim
import xgboost as xgb
import numpy as np


# =========================================================
#               TRANSFORMER SEQUENCE MODEL
# =========================================================
class TransformerClassifier(nn.Module):
    def __init__(self, feature_dim=11, hidden=128, heads=4, layers=2):
        super().__init__()

        self.embedding = nn.Linear(feature_dim, hidden)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=heads,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.fc = nn.Linear(hidden, 2)

    def forward(self, x):
        # x: (batch, seq_len, feature_dim)
        x = self.embedding(x)
        x = self.transformer(x)
        x = x[:, -1, :]          # last timestep
        return self.fc(x)


# =========================================================
#                   HYBRID MODEL
# =========================================================
class HybridModel:
    def __init__(self):
        self.xgb_model = None
        self.transformer = TransformerClassifier()

    # ----------------- XGBOOST TRAIN -----------------
    def train_xgb(self, X, y):
        model = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            max_depth=5,
            n_estimators=120,
            learning_rate=0.07,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
        )
        model.fit(X, y)
        self.xgb_model = model

    # ----------------- TRANSFORMER TRAIN -----------------
    def train_transformer(self, X_seq, y_seq, epochs=5):
        device = "mps" if torch.backends.mps.is_available() else "cpu"

        self.transformer.to(device)
        X_seq = X_seq.to(device)
        y_seq = y_seq.to(device)

        optimizer = optim.Adam(self.transformer.parameters(), lr=0.001)
        loss_fn = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            optimizer.zero_grad()
            out = self.transformer(X_seq)
            loss = loss_fn(out, y_seq)
            loss.backward()
            optimizer.step()
            print(f"[Transformer] Epoch {epoch+1}/{epochs}  Loss={loss.item():.4f}")

    # ----------------- PREDICT COMBINED -----------------
    def predict(self, X_tab_last, X_seq_last):
        """
        X_tab_last: (features,)
        X_seq_last: (seq_len, features)
        Returns: array [p_down, p_up]
        """
        # XGBoost
        p_xgb = self.xgb_model.predict_proba(X_tab_last.reshape(1, -1))[0]

        # Transformer
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.transformer.to(device)

        X_seq_last = torch.tensor(
            X_seq_last.reshape(1, X_seq_last.shape[0], X_seq_last.shape[1]),
            dtype=torch.float32,
        ).to(device)

        p_tr = torch.softmax(self.transformer(X_seq_last), dim=1)
        p_tr = p_tr.detach().cpu().numpy()[0]

        # Simple average ensemble
        p_final = (p_xgb + p_tr) / 2.0
        return p_final
