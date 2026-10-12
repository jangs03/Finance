
"""EDA-selected features + historical OHLCV sequence."""

import numpy as np
import pandas as pd

from exp.timeseries_data import build_windows
from exp.panel import universe_panel


EDA_GROUPS = (
    "core", "scale", "cross",
    "news", "event", "analyst",
)

EDA_FEATURES = [
    "gap_pct",
    "resid_gap",
    "vol5",
    "news_n",
    "news_abn",
    "earn",
    "surp_abs",
    "an_n",
    "an_down",
]


def build_eda_training_data(universe):
    """DEV-only feature panel matched to 20-day sequences."""

    windows = build_windows()

    panel = universe_panel(
        universe,
        groups=EDA_GROUPS,
        research=(),
        allow_lockbox=False,
    )

    required = [
        "symbol", "target", "ret_pct", "label",
        *EDA_FEATURES,
    ]

    missing = [c for c in required if c not in panel.columns]
    if missing:
        raise ValueError(f"Missing features: {missing}")

    windows["target"] = pd.to_datetime(
        windows["target"]
    ).dt.normalize()
    panel["target"] = pd.to_datetime(
        panel["target"]
    ).dt.normalize()

    result = windows.merge(
        panel[required],
        on=["symbol", "target"],
        how="inner",
        validate="one_to_one",
    )

    return result


def prepare_eda_features(fit, calib, test):
    """Train(Fit) 기준으로 결측 처리 및 정규화."""

    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    features = EDA_FEATURES.copy()

    # 결측 여부를 추가 Feature로 사용
    def make_features(df):
        x = df[features].copy()
        x["news_abn_missing"] = x["news_abn"].isna().astype(float)
        x["surp_abs_missing"] = x["surp_abs"].isna().astype(float)
        return x

    x_fit = make_features(fit)
    x_cal = make_features(calib)
    x_test = make_features(test)

    columns = x_fit.columns.tolist()

    # 학습 데이터에서만 결측 대체값 계산
    imputer = SimpleImputer(
        strategy="constant",
        fill_value=0.0,
    )

    x_fit = imputer.fit_transform(x_fit)
    x_cal = imputer.transform(x_cal)
    x_test = imputer.transform(x_test)

    # 학습 데이터의 평균/표준편차만 사용
    scaler = StandardScaler()

    x_fit = scaler.fit_transform(x_fit)
    x_cal = scaler.transform(x_cal)
    x_test = scaler.transform(x_test)

    return (
        x_fit.astype(np.float32),
        x_cal.astype(np.float32),
        x_test.astype(np.float32),
        columns,
    )
