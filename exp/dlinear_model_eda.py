
import torch
import torch.nn as nn
import torch.nn.functional as F


class DLinearEDAClassifier(nn.Module):
    """과거 20일 시계열 + EDA-selected features → 5등급 분류."""

    def __init__(
        self,
        seq_len=20,
        n_features=5,
        n_eda_features=11,
        kernel_size=5,
        hidden_dim=32,
    ):
        super().__init__()

        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")

        self.seq_len = seq_len
        self.n_features = n_features
        self.n_eda_features = n_eda_features
        self.kernel_size = kernel_size

        # DLinear: Trend / Seasonal decomposition
        self.trend_linear = nn.Linear(seq_len, 1)
        self.seasonal_linear = nn.Linear(seq_len, 1)

        # 시계열 표현 + EDA Feature 결합
        self.classifier = nn.Sequential(
            nn.Linear(
                n_features + n_eda_features,
                hidden_dim,
            ),
            nn.ReLU(),
            nn.Linear(hidden_dim, 5),
        )

    def forward(self, sequence, eda_features):
        # sequence: [batch, 20, 5]
        # eda_features: [batch, n_eda_features]

        if sequence.ndim != 3:
            raise ValueError("sequence must be 3D")

        if sequence.shape[1:] != (
            self.seq_len, self.n_features
        ):
            raise ValueError("Unexpected sequence shape")

        if eda_features.ndim != 2:
            raise ValueError("eda_features must be 2D")

        if eda_features.shape[1] != self.n_eda_features:
            raise ValueError("Unexpected EDA feature dimension")

        z = sequence.transpose(1, 2)

        pad = self.kernel_size // 2

        trend = F.avg_pool1d(
            F.pad(z, (pad, pad), mode="replicate"),
            kernel_size=self.kernel_size,
            stride=1,
        )

        seasonal = z - trend

        trend_repr = self.trend_linear(trend).squeeze(-1)
        seasonal_repr = self.seasonal_linear(seasonal).squeeze(-1)

        sequence_repr = trend_repr + seasonal_repr

        combined = torch.cat(
            [sequence_repr, eda_features],
            dim=1,
        )

        return self.classifier(combined)
