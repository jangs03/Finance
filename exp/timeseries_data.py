
import numpy as np
import pandas as pd

WINDOW = 20
FEATURES = ["ret_1d", "open_rel", "high_rel", "low_rel", "log_volume"]


def build_windows(daily_path="dataset/daily.parquet", window=WINDOW):
    daily = pd.read_parquet(daily_path).copy()
    daily["date_et"] = pd.to_datetime(daily["date_et"])
    daily = daily.sort_values(["symbol", "date_et"])

    samples = []

    for symbol, group in daily.groupby("symbol", sort=True):
        group = group.reset_index(drop=True)

        prev = group["close"].shift(1)
        feat = pd.DataFrame({
            "ret_1d": group["close"] / prev - 1,
            "open_rel": group["open"] / prev - 1,
            "high_rel": group["high"] / prev - 1,
            "low_rel": group["low"] / prev - 1,
            "log_volume": np.log1p(group["volume"]),
        })

        values = feat.to_numpy(dtype=np.float32)

        for i in range(window + 1, len(group)):
            # 대상일 i를 예측할 때 과거 i-window ... i-1만 사용
            x = values[i-window:i]

            if not np.isfinite(x).all():
                continue

            samples.append({
                "symbol": symbol,
                "target": group.loc[i, "date_et"],
                "sequence": x,
            })

    return pd.DataFrame(samples)



def build_labeled_windows():
    from exp.panel import labels

    windows = build_windows()
    targets = labels()[["symbol", "target", "ret_pct", "label"]].copy()

    windows["target"] = pd.to_datetime(windows["target"]).dt.normalize()
    targets["target"] = pd.to_datetime(targets["target"]).dt.normalize()

    result = windows.merge(
        targets,
        on=["symbol", "target"],
        how="inner",
        validate="one_to_one",
    )

    return result


def build_training_data(universe):
    """기존 DEV 패널의 Gap/Label과 20일 시계열 결합."""
    from exp.panel import universe_panel

    windows = build_windows()
    panel = universe_panel(
        universe,
        groups=("core",),
        research=(),
        allow_lockbox=False,
    )

    panel = panel[
        ["symbol", "target", "gap_pct", "ret_pct", "label"]
    ].copy()

    windows["target"] = pd.to_datetime(
        windows["target"]
    ).dt.normalize()

    panel["target"] = pd.to_datetime(
        panel["target"]
    ).dt.normalize()

    result = windows.merge(
        panel,
        on=["symbol", "target"],
        how="inner",
        validate="one_to_one",
    )

    return result
