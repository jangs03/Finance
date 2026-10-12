
import torch
import torch.nn as nn
import torch.nn.functional as F


class DLinearClassifier(nn.Module):
    """DLinear 기반 5등급 주가 분류 모델."""

    def __init__(self, seq_len=20, n_features=5, gap_dim=1, kernel_size=5):
        super().__init__()

        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")

        self.seq_len = seq_len
        self.n_features = n_features

        # 추세(trend)와 계절/잔차(seasonal) 분리
        self.moving_avg = nn.AvgPool1d(
            kernel_size=kernel_size,
            stride=1,
        )

        self.trend_linear = nn.Linear(seq_len, 1)
        self.seasonal_linear = nn.Linear(seq_len, 1)

        # 시계열 정보 + 당일 프리마켓 갭
        self.classifier = nn.Sequential(
            nn.Linear(n_features + gap_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 5),
        )

    def forward(self, x, gap):
        # x: [batch, 20, features]
        # gap: [batch, 1]
        if x.ndim != 3 or x.shape[1:] != (
            self.seq_len, self.n_features
        ):
            raise ValueError("Unexpected time-series input shape")

        # 이동평균으로 trend 추출
        z = x.transpose(1, 2)
        pad = self.moving_avg.kernel_size[0] // 2
        trend = self.moving_avg(
            F.pad(z, (pad, pad), mode="replicate")
        )

        seasonal = z - trend

        # 시점별 Linear projection
        trend_out = self.trend_linear(trend).squeeze(-1)
        seasonal_out = self.seasonal_linear(seasonal).squeeze(-1)

        features = trend_out + seasonal_out

        combined = torch.cat([features, gap], dim=1)

        # 5개 클래스 logits 반환
        return self.classifier(combined)
