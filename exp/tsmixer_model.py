
import torch
import torch.nn as nn


class MixerBlock(nn.Module):
    def __init__(self, seq_len, n_features, hidden_dim=32, dropout=0.1):
        super().__init__()

        self.time_mixing = nn.Sequential(
            nn.Linear(seq_len, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, seq_len),
        )

        self.feature_mixing = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_features),
        )

        self.time_norm = nn.LayerNorm(n_features)
        self.feature_norm = nn.LayerNorm(n_features)

    def forward(self, x):
        # x: [batch, seq_len, n_features]
        time_out = self.time_mixing(x.transpose(1, 2))
        x = self.time_norm(x + time_out.transpose(1, 2))

        feature_out = self.feature_mixing(x)
        return self.feature_norm(x + feature_out)


class TSMixerClassifier(nn.Module):
    """TSMixer 기반 5등급 분류 모델."""

    def __init__(self, seq_len=20, n_features=5, gap_dim=1,
                 hidden_dim=32, n_blocks=2, dropout=0.1):
        super().__init__()

        self.blocks = nn.Sequential(*[
            MixerBlock(seq_len, n_features, hidden_dim, dropout)
            for _ in range(n_blocks)
        ])

        self.classifier = nn.Sequential(
            nn.Linear(seq_len * n_features + gap_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 5),
        )

    def forward(self, x, gap):
        x = self.blocks(x)
        x = x.flatten(start_dim=1)
        combined = torch.cat([x, gap], dim=1)
        return self.classifier(combined)
