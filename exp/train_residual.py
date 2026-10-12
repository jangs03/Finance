
"""DLinear Residual: Fold A / Draw 0 smoke test."""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from exp.residual_model import DLinearResidual
from exp.timeseries_data import build_training_data
from exp.splits import universe_draws, fold_masks
from exp.train_dlinear import (
    SEED, SPLIT, FOLDS, EMBARGO_DAYS,
    split_calibration, prepare_arrays, set_seed,
)
from src.data import score
from src.model import KRule


def main():
    set_seed(SEED)

    symbols = sorted(
        pd.read_parquet(
            "dataset/daily.parquet",
            columns=["symbol"],
        )["symbol"].unique()
    )
    draw = universe_draws(symbols, SPLIT, SEED)[0]
    fold = FOLDS[0]

    print("[1/4] Preparing DEV data", flush=True)

    train_panel = pd.concat(
        [build_training_data(u) for u in draw["train_universes"]],
        ignore_index=True,
    )
    test_panel = build_training_data(draw["test_universe"])

    tr, te = fold_masks(
        train_panel, test_panel, fold, EMBARGO_DAYS
    )

    train = train_panel.loc[tr].dropna(
        subset=["gap_pct", "ret_pct", "label"]
    ).copy()
    test = test_panel.loc[te].dropna(
        subset=["gap_pct", "ret_pct", "label"]
    ).copy()

    fit, calib = split_calibration(train)

    assert not set(fit["symbol"]) & set(draw["unseen"])
    assert fit["target"].max() < calib["target"].min()
    assert test["target"].min() >= pd.Timestamp(fold["start"])

    xf, xc, xv, _, _, _ = prepare_arrays(
        fit, calib, test, gap_only=False
    )

    # Residual target, percentage points
    y_residual = (
        fit["ret_pct"].to_numpy(dtype=np.float32)
        - fit["gap_pct"].to_numpy(dtype=np.float32)
    )

    # 극단적인 수익률의 영향 제한
    yf = torch.from_numpy(
        np.clip(y_residual, -15, 15).copy()
    )

    print(
        f"Fit={len(fit)} Calibration={len(calib)} "
        f"Test={len(test)}",
        flush=True,
    )

    print("[2/4] Training residual model", flush=True)

    model = DLinearResidual()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(15):
        model.train()
        total = 0.0

        for idx in torch.randperm(len(xf)).split(128):
            prediction = model(xf[idx])
            loss = F.smooth_l1_loss(prediction, yf[idx])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total += loss.item() * len(idx)

        if epoch in (0, 14):
            print(
                f"Epoch {epoch+1:02d}/15 "
                f"Loss={total/len(xf):.4f}",
                flush=True,
            )

    model.eval()
    with torch.no_grad():
        residual_cal = model(xc).numpy()
        residual_test = model(xv).numpy()

    gap_cal = calib["gap_pct"].to_numpy(float)
    gap_test = test["gap_pct"].to_numpy(float)

    y_cal = calib["label"].to_numpy(int)
    y_test = test["label"].to_numpy(int)

    objective = lambda y, p: score(y, p)["score"]

    # B2와 Residual 모두 같은 Calibration 구간 사용
    b2 = KRule()
    b2.fit(gap_cal, y_cal, objective)

    baseline_score = objective(
        y_test, b2.predict(gap_test)
    )

    print("[3/4] Calibrating alpha and k", flush=True)

    # alpha=0이면 정확히 B2로 복원
    best_alpha = 0.0
    best_k = b2.k
    best_cal_score = objective(
        y_cal, b2.predict(gap_cal)
    )

    for alpha in [-2, -1, -0.5, -0.25,
                  0.25, 0.5, 1, 2]:
        signal = gap_cal + alpha * residual_cal

        rule = KRule()
        rule.fit(signal, y_cal, objective)

        cal_score = objective(
            y_cal, rule.predict(signal)
        )

        if cal_score > best_cal_score + 1e-12:
            best_cal_score = cal_score
            best_alpha = alpha
            best_k = rule.k

    # k는 Calibration에서 고정된 값만 적용
    from src.data import label_of

    residual_signal = (
        gap_test + best_alpha * residual_test
    )
    residual_pred = label_of(
        best_k * residual_signal
    ).to_numpy(dtype=int)

    candidate_score = objective(y_test, residual_pred)

    print("\n[4/4] DEV Fold A / Draw 0 Results")
    print("B2 k:", b2.k)
    print("Selected alpha:", best_alpha)
    print("Selected k:", best_k)
    print("B2 Score:", round(baseline_score, 4))
    print("Residual Score:", round(candidate_score, 4))
    print(
        "Delta:",
        round(candidate_score - baseline_score, 4)
    )
    print("Residual metrics:", score(y_test, residual_pred))


if __name__ == "__main__":
    main()
