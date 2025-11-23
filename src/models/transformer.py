import torch
import torch.nn as nn

class MarketTransformer(nn.Module):
    def __init__(self, feature_dim=11, hidden_dim=128, num_layers=3, heads=4):
        super().__init__()

        self.embedding = nn.Linear(feature_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.fc = nn.Linear(hidden_dim, 2)  # up/down probability

    def forward(self, x):
        x = self.embedding(x)
        out = self.transformer(x)
        out = out[:, -1, :]   # last timestep
        return torch.softmax(self.fc(out), dim=1)
