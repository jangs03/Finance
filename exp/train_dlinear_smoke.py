
"""DLinear 학습 Smoke Test: DEV Fold A × Draw 0."""

import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import os

from exp.dlinear_model import DLinearClassifier
from exp.splits import universe_draws, fold_masks
from exp.timeseries_data import build_training_data
from src.data import score
from src.model import KRule
from src.model import CLASS_CENTERS

SEED = 20261005
GAP_ONLY = os.getenv("GAP_ONLY", "0") == "1"
EPOCHS = 15
BATCH_SIZE = 128

SPLIT = {
    "n_draws": 3,
    "n_unseen": 10,
    "n_seen": 10,
    "universe_size": 20,
}

FOLD = {
    "name": "A",
    "start": "2025-07-01",
    "end": "2025-11-01",
}


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # 기존 Universe Draw 방식 사용
    daily = pd.read_parquet(
        "dataset/daily.parquet",
        columns=["symbol"],
    )
    symbols = sorted(daily["symbol"].unique())

    draw = universe_draws(symbols, SPLIT, SEED)[0]

    print("[1/5] Universe 구성")
    print("Train groups:", len(draw["train_universes"]))
    print("Test symbols:", len(draw["test_universe"]))
    print("Unseen symbols:", len(draw["unseen"]))

    # Train universe와 Test universe의 패널은 따로 생성
    print("[2/5] 시계열 데이터 준비")

    train_parts = [
        build_training_data(universe)
        for universe in draw["train_universes"]
    ]
    train_panel = pd.concat(train_parts, ignore_index=True)
    test_panel = build_training_data(draw["test_universe"])

    tr_mask, te_mask = fold_masks(
        train_panel, test_panel, FOLD, embargo_days=14
    )

    train = train_panel.loc[tr_mask].copy()
    test = test_panel.loc[te_mask].copy()

    # Gap을 사용할 수 없는 행은 이번 테스트에서 제외
    train = train.dropna(subset=["gap_pct", "label"])
    test = test.dropna(subset=["gap_pct", "label"])

    print("Train samples:", len(train))
    print("Test samples:", len(test))

    if train.empty or test.empty:
        raise ValueError("Train 또는 Test 데이터가 비어 있음")

    assert not set(draw["unseen"]) & set(train["symbol"])
    assert train["target"].max() < (
        pd.Timestamp(FOLD["start"]) - pd.Timedelta(days=14)
    )
    assert (
        (test["target"] >= pd.Timestamp(FOLD["start"]))
        & (test["target"] < pd.Timestamp(FOLD["end"]))
    ).all()

    # 시계열 배열 생성
    x_train = np.stack(train["sequence"].to_numpy())
    x_test = np.stack(test["sequence"].to_numpy())
    if GAP_ONLY:
        x_train = np.zeros_like(x_train)
        x_test = np.zeros_like(x_test)
        print("[Ablation] Gap-only: 시계열 입력 제거")
    else:
        print("[Ablation] DLinear + Gap: 시계열 입력 사용")

    gap_train = train[["gap_pct"]].to_numpy(dtype=np.float32)
    gap_test = test[["gap_pct"]].to_numpy(dtype=np.float32)

    y_train = train["label"].to_numpy(dtype=np.int64)
    y_test = test["label"].to_numpy(dtype=np.int64)

    # 정규화 통계는 오직 Train에서 계산
    mean = x_train.mean(axis=(0, 1), keepdims=True)
    std = x_train.std(axis=(0, 1), keepdims=True)
    std = np.maximum(std, 1e-6)

    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std

    gap_mean = gap_train.mean(axis=0, keepdims=True)
    gap_std = np.maximum(
        gap_train.std(axis=0, keepdims=True), 1e-6
    )

    gap_train = (gap_train - gap_mean) / gap_std
    gap_test = (gap_test - gap_mean) / gap_std

    if not (
        np.isfinite(x_train).all()
        and np.isfinite(x_test).all()
        and np.isfinite(gap_train).all()
        and np.isfinite(gap_test).all()
    ):
        raise ValueError("입력 데이터에 NaN 또는 Inf 존재")

    xt = torch.from_numpy(x_train).float()
    xv = torch.from_numpy(x_test).float()
    gt = torch.from_numpy(gap_train).float()
    gv = torch.from_numpy(gap_test).float()
    yt = torch.from_numpy(y_train.copy())

    print("[3/5] DLinear 초기화")

    model = DLinearClassifier(
        seq_len=20,
        n_features=5,
        gap_dim=1,
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-3
    )

    print("[4/5] 학습 시작")

    for epoch in range(EPOCHS):
        model.train()
        indices = torch.randperm(len(xt))
        total_loss = 0.0

        for batch_idx in indices.split(BATCH_SIZE):
            logits = model(
                xt[batch_idx],
                gt[batch_idx],
            )
            loss = F.cross_entropy(logits, yt[batch_idx])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(batch_idx)

        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} "
            f"Loss: {total_loss / len(xt):.4f}",
            flush=True,
        )

    print("[5/5] Fold A 평가")

    model.eval()
    with torch.no_grad():
        logits = model(xv, gv)
        pred = logits.argmax(dim=1).numpy()

    accuracy = (pred == y_test).mean()
        # 기존 프로젝트 공식 평가 지표
    metrics = score(y_test, pred)

    print("\n[DLinear DEV Fold A / Draw 0]")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}" if isinstance(value, float)
              else f"{name}: {value}")

    print("Accuracy:", round(float(accuracy), 4))
    print(
        "Prediction distribution:",
        np.bincount(pred, minlength=5).tolist(),
    )
        # B2: 동일한 Train/Test 샘플에서 Gap + KRule 평가
    b2 = KRule()
    b2.fit(
        train["gap_pct"].to_numpy(dtype=float),
        y_train,
        lambda y, p: score(y, p)["score"],
    )

    b2_pred = b2.predict(
        test["gap_pct"].to_numpy(dtype=float)
    )
    b2_metrics = score(y_test, b2_pred)

    print("\n[B2 DEV Fold A / Draw 0]")
    print("Best k:", b2.k)

    for name, value in b2_metrics.items():
        print(
            f"{name}: {value:.4f}"
            if isinstance(value, float)
            else f"{name}: {value}"
        )

    print(
        "\nDelta Score (DLinear - B2):",
        round(metrics["score"] - b2_metrics["score"], 4),
    )

    # DLinear + KRule
    # KRule은 반드시 Train 예측값으로만 학습
    model.eval()

    with torch.no_grad():
        train_proba = torch.softmax(
            model(xt, gt), dim=1
        ).numpy()

        test_proba = torch.softmax(
            model(xv, gv), dim=1
        ).numpy()

    # B4와 동일한 클래스 중심값으로 스칼라 신호 생성
    train_signal = train_proba @ CLASS_CENTERS
    test_signal = test_proba @ CLASS_CENTERS

    # Train 데이터에서만 최적 k 선택
    dlinear_rule = KRule()
    dlinear_rule.fit(
        train_signal,
        y_train,
        lambda y, p: score(y, p)["score"],
    )

    calibrated_pred = dlinear_rule.predict(test_signal)
    calibrated_metrics = score(y_test, calibrated_pred)

    print("\n[DLinear + KRule / Fold A / Draw 0]")
    print("Best k:", dlinear_rule.k)

    for name, value in calibrated_metrics.items():
        print(
            f"{name}: {value:.4f}"
            if isinstance(value, float)
            else f"{name}: {value}"
        )

    print("\n[Score 비교]")
    print("B2:", round(b2_metrics["score"], 4))
    print("DLinear Argmax:", round(metrics["score"], 4))
    print("DLinear + KRule:", round(calibrated_metrics["score"], 4))
    print(
        "Delta vs B2:",
        round(
            calibrated_metrics["score"] - b2_metrics["score"], 4
        ),
    )

    print("Smoke Test 완료")


if __name__ == "__main__":
    main()
