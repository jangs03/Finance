
import torch
import torch.nn as nn
import torch.nn.functional as F


class DLinearResidual(nn.Module):
    """과거 주가 시계열에서 잔여 수익률 신호를 예측."""

    def __init__(self, seq_len=20, n_features=5, kernel_size=5):
        super().__init__()

        self.kernel_size = kernel_size

        self.trend_linear = nn.Linear(seq_len, 1)
        self.seasonal_linear = nn.Linear(seq_len, 1)

        self.head = nn.Sequential(
            nn.Linear(n_features, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        # x: [batch, 20, 5]
        z = x.transpose(1, 2)

        pad = self.kernel_size // 2

        trend = F.avg_pool1d(
            F.pad(z, (pad, pad), mode="replicate"),
            kernel_size=self.kernel_size,
            stride=1,
        )

        seasonal = z - trend

        trend_out = self.trend_linear(trend).squeeze(-1)
        seasonal_out = self.seasonal_linear(seasonal).squeeze(-1)

        residual = self.head(trend_out + seasonal_out)

        return residual.squeeze(-1)
